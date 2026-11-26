"""V2: durable unique events; one process, GPU 2, ordered per-run requests."""
import os
# Device selection is configured by serve.py before importing torch.
from pathlib import Path
import sys, json, sqlite3, threading, hashlib, time, re,zlib
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('YOLO_CONFIG_DIR', str(ROOT/'test_house_detection/.cache/yolo'))
os.environ.setdefault('MPLCONFIGDIR', str(ROOT/'test_house_detection/.cache/matplotlib'))
import cv2, numpy as np, torch
from fastapi import FastAPI, HTTPException, Form, File, UploadFile,Request
from fastapi.responses import FileResponse
from starlette.concurrency import run_in_threadpool
from pydantic import BaseModel, Field, ValidationError
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from unique_target_core import UniqueTargetCore, export_event
from test_house_detection.detector import HouseDetector

torch.set_num_threads(4)
cv2.setNumThreads(2)

class RunRequest(BaseModel):
    run_id: str = Field(pattern=r'^[a-f0-9]{32}$')
    session_id: str = Field(min_length=1, max_length=128)
    detector_id: str = 'house'

class FrameSpec(BaseModel):
    frame_id: int = Field(ge=0)
    timestamp_ms: float = Field(ge=0,allow_inf_nan=False)

class Engine:
    def __init__(self, root=None, detector=None, core_factory=UniqueTargetCore):
        self.root = Path(root or ROOT/'outputs/integration_detection')
        self.root.mkdir(parents=True, exist_ok=True)
        self.db = self.root/'journal.sqlite'
        self.lock = threading.RLock()
        self.sessions = {}
        self.progress = threading.Condition(self.lock)
        self.batch_slots = threading.BoundedSemaphore(2)
        self.detector = detector
        self.core_factory = core_factory
        self.decoders = ThreadPoolExecutor(max_workers=4,thread_name_prefix="jpeg-decode")
        with self.connect() as c:
            c.executescript("""CREATE TABLE IF NOT EXISTS runs(run_id TEXT PRIMARY KEY,session_id TEXT,detector_id TEXT,status TEXT,summary TEXT);
            CREATE TABLE IF NOT EXISTS frames(run_id TEXT,frame_id INTEGER,digest TEXT,response TEXT,PRIMARY KEY(run_id,frame_id));
            CREATE TABLE IF NOT EXISTS events(run_id TEXT,target_id TEXT,payload TEXT,PRIMARY KEY(run_id,target_id));""")
    def connect(self):
        return sqlite3.connect(self.db, timeout=30)
    def begin(self, req):
        if req.detector_id != 'house': raise HTTPException(422,'Only detector_id=house is configured')
        with self.lock, self.connect() as c:
            row=c.execute('SELECT session_id,detector_id,status FROM runs WHERE run_id=?',(req.run_id,)).fetchone()
            if row:
                if row[:2] != (req.session_id,req.detector_id): raise HTTPException(409,'run identity conflict')
                if req.run_id not in self.sessions: raise HTTPException(409,'Run closed or process restarted; use a new run_id. Persisted events remain readable.')
                return {'run_id':req.run_id,'status':row[2]}
            if len(self.sessions)>=2: raise HTTPException(429,'At most two active runs')
            if self.detector is None: self.detector=HouseDetector(weights=os.environ.get('HOUSE_MODEL') or None,conf=.4,tiled=True,device=os.environ.get('HOUSE_DEVICE','cpu'))
            self.sessions[req.run_id]={'core':self.core_factory(stream_id='r'+req.run_id),'last':-1,'ms':-1.,'detection_s':0.,'core_s':0.,'jpeg_decode_s':0.,'batch_decode_wall_s':0.,'frames':0}
            c.execute('INSERT INTO runs VALUES(?,?,?,?,?)',(req.run_id,req.session_id,req.detector_id,'running','{}'))
            return {'run_id':req.run_id,'status':'running'}
    @staticmethod
    def decode(raw):
        t=time.perf_counter()
        image=cv2.imdecode(np.frombuffer(raw,np.uint8),cv2.IMREAD_COLOR)
        if image is None: raise HTTPException(415,'Invalid image')
        if image.shape[0]*image.shape[1]>40000000: raise HTTPException(413,'Image dimensions too large')
        return image,time.perf_counter()-t
    def batch(self,run_id,specs,raws,after_frame_id=None):
        if not 1<=len(specs)<=4 or len(specs)!=len(raws): raise HTTPException(422,'Batch must contain 1 to 4 matching frames/images')
        if any(b.frame_id<=a.frame_id or b.timestamp_ms<=a.timestamp_ms for a,b in zip(specs,specs[1:])): raise HTTPException(409,'Batch frame order must increase')
        if after_frame_id is not None and after_frame_id>=specs[0].frame_id:raise HTTPException(422,'Predecessor must precede batch')
        if not self.batch_slots.acquire(timeout=5):raise HTTPException(429,'Two batches are already in flight')
        try:
            # Reading immutable cache entries and decoding do not need the inference lock.
            with self.connect() as c:
                needed=[]
                for spec,raw in zip(specs,raws):
                    old=c.execute('SELECT digest FROM frames WHERE run_id=? AND frame_id=?',(run_id,spec.frame_id)).fetchone()
                    digest=hashlib.sha256(str(spec.timestamp_ms).encode()+b'|'+raw).hexdigest()
                    if old and old[0]!=digest:raise HTTPException(409,'Same frame ID with different image or timestamp')
                    needed.append(not bool(old))
            t=time.perf_counter()
            futures=[self.decoders.submit(self.decode,raw) if need else None for raw,need in zip(raws,needed)]
            decoded=[f.result() if f else None for f in futures]
            decode_wall=time.perf_counter()-t
            with self.progress:
                deadline=time.monotonic()+10
                while after_frame_id is not None and any(needed):
                    state=self.sessions.get(run_id)
                    if state is None:raise HTTPException(409,'Run not active')
                    if state['last']>=after_frame_id:break
                    if time.monotonic()>deadline:raise HTTPException(409,'Predecessor frame has not committed')
                    self.progress.wait(timeout=.1)
                state=self.sessions.get(run_id)
                # Validate the whole batch against the current state before any mutation.
                with self.connect() as c:
                    for spec,raw in zip(specs,raws):
                        old=c.execute('SELECT digest FROM frames WHERE run_id=? AND frame_id=?',(run_id,spec.frame_id)).fetchone()
                        digest=hashlib.sha256(str(spec.timestamp_ms).encode()+b'|'+raw).hexdigest()
                        if old and old[0]!=digest:raise HTTPException(409,'Frame conflict')
                        if not old and (state is None or spec.frame_id<=state['last'] or spec.timestamp_ms<=state['ms']):raise HTTPException(409,'Frame order conflict')
                if state is not None:state['batch_decode_wall_s']+=decode_wall
                results=[self.process(run_id,spec.frame_id,spec.timestamp_ms,raw,im) for spec,raw,im in zip(specs,raws,decoded)]
                return {'run_id':run_id,'results':results}
        finally:self.batch_slots.release()
    def process(self, run_id, frame_id, ms, raw, decoded=None,digest_override=None):
        if not re.fullmatch('[a-f0-9]{32}',run_id): raise HTTPException(422,'Invalid run ID')
        digest=digest_override or hashlib.sha256(str(ms).encode()+b'|'+raw).hexdigest()
        with self.lock, self.connect() as c:
            cached=c.execute('SELECT digest,response FROM frames WHERE run_id=? AND frame_id=?',(run_id,frame_id)).fetchone()
            if cached:
                if cached[0]!=digest: raise HTTPException(409,'Same frame ID with different image or timestamp')
                return json.loads(cached[1])
            state=self.sessions.get(run_id)
            if state is None: raise HTTPException(409,'Run not active; begin a new run or read persisted events')
            if frame_id<=state['last'] or ms<=state['ms']: raise HTTPException(409,'Frames and source timestamps must increase')
            image,decode_s = decoded if decoded is not None else self.decode(raw)
            state['jpeg_decode_s']+=decode_s
            try:
                t=time.perf_counter(); detections=self.detector.detect(image,frame_id,conf=.4)
                state['detection_s']+=time.perf_counter()-t
                t=time.perf_counter(); events=state['core'].push(image,frame_id,ms,detections)
                state['core_s']+=time.perf_counter()-t
                result=[]
                for event in events:
                    event['event_id']=run_id+':'+event['target_id']
                    event['run_id']=run_id
                    info=export_event(event,self.root/run_id/'targets')
                    info.pop('files',None)
                    info['assets']={name:f"/v2/runs/{run_id}/events/{info['target_id']}/files/{name}" for name in ['original.jpg','annotated.jpg','bbox.txt','bbox_yolo.txt','metadata.json']}
                    c.execute('INSERT INTO events VALUES(?,?,?)',(run_id,info['target_id'],json.dumps(info)))
                    result.append(info)
                payload={'run_id':run_id,'request_frame_id':frame_id,'events':result}
                c.execute('INSERT INTO frames VALUES(?,?,?,?)',(run_id,frame_id,digest,json.dumps(payload)))
                c.commit()
                state.update(last=frame_id,ms=ms,frames=state['frames']+1)
                self.progress.notify_all()
                return payload
            except Exception:
                c.rollback()
                self.sessions.pop(run_id,None)
                c.execute("UPDATE runs SET status='failed' WHERE run_id=?",(run_id,));c.commit()
                raise
    def local_batch(self,run_id,items,after_frame_id=None):
        if not re.fullmatch('[a-f0-9]{32}',run_id):raise HTTPException(422,'Invalid run ID')
        specs=[FrameSpec(**item) for item in items]
        if not 1<=len(specs)<=4:raise HTTPException(422,'Batch size must be 1..4')
        if any(b.frame_id<=a.frame_id or b.timestamp_ms<=a.timestamp_ms for a,b in zip(specs,specs[1:])):raise HTTPException(409,'Frame order conflict')
        if after_frame_id is not None and after_frame_id>=specs[0].frame_id:raise HTTPException(422,'Invalid predecessor')
        if not self.batch_slots.acquire(timeout=5):raise HTTPException(429,'Two batches are already in flight')
        arrays=[]
        try:
            with self.connect() as c:
                for spec,item in zip(specs,items):
                    digest=item.get('digest','')
                    if not isinstance(digest,str) or not re.fullmatch('[a-f0-9]{64}',digest):raise HTTPException(422,'Invalid image digest')
                    cached=c.execute('SELECT digest FROM frames WHERE run_id=? AND frame_id=?',(run_id,spec.frame_id)).fetchone()
                    if cached:
                        if cached[0]!=digest:raise HTTPException(409,'Frame conflict')
                        arrays.append(None);continue
                    path=(ROOT/'outputs/ipc_buffer'/run_id/f'{spec.frame_id}.npy')
                    expected=(ROOT/'outputs/ipc_buffer').resolve()/run_id
                    if path.resolve().parent!=expected or not path.is_file() or path.stat().st_size>128*1024*1024:raise HTTPException(422,'Missing or invalid local buffer')
                    try:
                        image=np.load(path,mmap_mode='r',allow_pickle=False)
                        arrays.append(image)
                        if image.dtype!=np.uint8 or image.ndim!=3 or image.shape[2]!=3 or image.shape[0]*image.shape[1]>40000000 or min(image.shape[:2])<=0 or list(image.shape)!=item.get('shape'):raise ValueError('Invalid local image shape')
                        if zlib.crc32(image)!=item.get('crc32'):raise ValueError('Local buffer checksum mismatch')
                    except (ValueError,OSError) as exc:raise HTTPException(422,str(exc))
            with self.progress:
                deadline=time.monotonic()+10
                while after_frame_id is not None and any(a is not None for a in arrays):
                    state=self.sessions.get(run_id)
                    if state is None:raise HTTPException(409,'Run not active')
                    if state['last']>=after_frame_id:break
                    if time.monotonic()>deadline:raise HTTPException(409,'Predecessor frame has not committed')
                    self.progress.wait(.1)
                state=self.sessions.get(run_id)
                with self.connect() as c:
                    for spec,item in zip(specs,items):
                        old=c.execute('SELECT digest FROM frames WHERE run_id=? AND frame_id=?',(run_id,spec.frame_id)).fetchone()
                        if old and old[0]!=item['digest']:raise HTTPException(409,'Frame conflict')
                        if not old and (state is None or spec.frame_id<=state['last'] or spec.timestamp_ms<=state['ms']):raise HTTPException(409,'Frame order conflict')
                results=[self.process(run_id,spec.frame_id,spec.timestamp_ms,b'',(im,0.) if im is not None else None,item['digest']) for spec,item,im in zip(specs,items,arrays)]
                return {'run_id':run_id,'results':results}
        finally:
            # Let NumPy views release mmap ownership; YOLO may retain its last orig_img.
            arrays.clear()
            self.batch_slots.release()
    def events(self,run_id):
        with self.connect() as c:
            return [json.loads(x[0]) for x in c.execute('SELECT payload FROM events WHERE run_id=? ORDER BY rowid',(run_id,))]
    def finish(self,run_id):
        with self.lock,self.connect() as c:
            state=self.sessions.get(run_id)
            if state:
                core=state['core'];core.flush()
                summary={k:v for k,v in state.items() if k!='core'}
                summary['stats']=core.stats()
                p=self.root/run_id;p.mkdir(exist_ok=True)
                (p/'pending_review.json').write_text(json.dumps(core.review_items,indent=2))
                (p/'summary.json').write_text(json.dumps(summary,indent=2))
                c.execute("UPDATE runs SET status='closed',summary=? WHERE run_id=?",(json.dumps(summary),run_id));c.commit()
                self.sessions.pop(run_id,None)
            row=c.execute('SELECT summary,status FROM runs WHERE run_id=?',(run_id,)).fetchone()
            if not row: raise HTTPException(404,'Unknown run')
            return {'run_id':run_id,'status':row[1],'summary':json.loads(row[0]),'events':self.events(run_id)}

def create_app(engine=None):
    engine=engine or Engine()
    @asynccontextmanager
    async def lifespan(app):
        yield
        engine.decoders.shutdown(wait=True)
    app=FastAPI(title='House unique event service',version='2.1',lifespan=lifespan)
    app.state.engine=engine
    @app.get('/health')
    def health(): return {'status':'ok','protocol':2,'device':os.environ.get('HOUSE_DEVICE','cpu'),'visible_gpu':os.environ.get('CUDA_VISIBLE_DEVICES',''),'active_runs':len(engine.sessions)}
    @app.post('/v2/runs')
    def begin(req:RunRequest): return engine.begin(req)
    @app.post('/v2/detect')
    async def detect(run_id:str=Form(...),frame_id:int=Form(...,ge=0),timestamp_ms:float=Form(...,ge=0,allow_inf_nan=False),image:UploadFile=File(...)):
        raw=await image.read(20*1024*1024+1)
        if not raw or len(raw)>20*1024*1024: raise HTTPException(413,'Upload limit 20 MiB')
        return await run_in_threadpool(engine.process,run_id,frame_id,timestamp_ms,raw)
    @app.post('/v2/detect-batch')
    async def batch(run_id:str=Form(...),frames:str=Form(...),images:list[UploadFile]=File(...),after_frame_id:int|None=Form(None,ge=-1)):
        try:
            items=json.loads(frames)
            if not isinstance(items,list) or not 1<=len(items)<=4: raise ValueError('Expected 1 to 4 frames')
            specs=[FrameSpec(**item) for item in items]
        except (ValueError,TypeError,ValidationError) as exc: raise HTTPException(422,str(exc))
        if len(specs)!=len(images):raise HTTPException(422,'Frame/image count mismatch')
        raws=[]
        for image in images:
            raw=await image.read(20*1024*1024+1)
            if not raw or len(raw)>20*1024*1024: raise HTTPException(413,'Image limit 20 MiB')
            raws.append(raw)
        if sum(map(len,raws))>64*1024*1024:raise HTTPException(413,'Batch limit 64 MiB')
        return await run_in_threadpool(engine.batch,run_id,specs,raws,after_frame_id)
    @app.post('/v2/detect-local')
    def local(payload:dict,request:Request):
        if request.client is None or request.client.host not in {'127.0.0.1','::1'}:raise HTTPException(403,'Local buffers are loopback-only')
        try:
            run=payload['run_id'];items=payload['frames'];after=payload.get('after_frame_id')
            if not isinstance(run,str) or not isinstance(items,list) or not 1<=len(items)<=4 or (after is not None and (not isinstance(after,int) or after < -1)):raise ValueError('Invalid local batch')
            return engine.local_batch(run,items,after)
        except (KeyError,TypeError,ValueError,ValidationError) as exc:raise HTTPException(422,str(exc))
    @app.get('/v2/runs/{run_id}/events')
    def events(run_id:str): return {'events':engine.events(run_id)}
    @app.post('/v2/runs/{run_id}/close')
    def close(run_id:str): return engine.finish(run_id)
    @app.get('/v2/runs/{run_id}/events/{target_id}/files/{name}')
    def asset(run_id:str,target_id:str,name:str):
        if name not in {'original.jpg','annotated.jpg','bbox.txt','bbox_yolo.txt','metadata.json'}: raise HTTPException(404)
        if not any(e['target_id']==target_id for e in engine.events(run_id)): raise HTTPException(404)
        p=(engine.root/run_id/'targets'/target_id/name).resolve()
        if not p.is_relative_to(engine.root.resolve()) or not p.is_file(): raise HTTPException(404)
        return FileResponse(p)
    return app
