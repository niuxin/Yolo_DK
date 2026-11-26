from collections import deque
from dataclasses import dataclass,field
from itertools import count
from types import SimpleNamespace
import time,math,re,cv2,numpy as np
from ultralytics.engine.results import Boxes
from ultralytics.trackers.bot_sort import BOTSORT
try:
 from .geometry import Geometry
 from .motion import BackgroundMotion,project,overlap
except ImportError:
 from geometry import Geometry
 from motion import BackgroundMotion,project,overlap

class StreamBOTSORT(BOTSORT):
 def __init__(self,args):
  super().__init__(args)
  self._ids=count(1)

 @staticmethod
 def reset_id():
  # Do not reset the library's global ID counter when another stream is created.
  pass

@dataclass
class Target:
 id:str
 cls:int
 name:str
 first_ms:float
 first_frame:int
 last_ms:float
 box:object
 emitted:bool=False
 high:object=field(default_factory=deque)
 first_high_ms:object=None
 best:object=None
 views:object=field(default_factory=list)
 appearance:object=None
 entry_appearance:object=None
 confirmed_view:object=None
 confirmed_appearance:object=None
 last_view_ms:float=-1e12
 hits:int=0
 max_high_hits:int=0
 last_quality:object=None
 last_confidence:float=0.
 relinks:int=0
 motion_box:object=None
 motion_quad:object=None
 extent_quad:object=None
 extent_box:object=None
 extent_ms:float=-1e12
 motion_valid:bool=False
 stationary:object=field(default_factory=deque)

class UniqueTargetCore:
 """One instance per ordered stream. push returns only newly confirmed, once-only business events."""
 def __init__(self,stream_id='demo',confirm_conf=.85,confirm_hits=5,window_ms=1500,archive_ms=1800000,max_targets=500,geometry_budget=6,assume_static=True,min_border_fraction=.015):
  if confirm_hits<1 or window_ms<=0 or max_targets<1:raise ValueError('Invalid confirmation settings')
  if not re.fullmatch(r'[A-Za-z0-9_-]{1,64}',stream_id):raise ValueError('Use a safe alphanumeric stream_id')
  self.assume_static=assume_static;self.min_border_fraction=min_border_fraction;self.review_items=[];self.review_ids=set()
  self.stream_id=stream_id;self.confirm_conf=confirm_conf;self.confirm_hits=confirm_hits;self.window_ms=window_ms;self.archive_ms=archive_ms;self.max_targets=max_targets;self.geometry_budget=geometry_budget
  args=SimpleNamespace(track_high_thresh=.55,track_low_thresh=.35,new_track_thresh=.55,track_buffer=8,match_thresh=.8,fuse_score=True,gmc_method='sparseOptFlow',proximity_thresh=.5,appearance_thresh=.25,with_reid=False,model='auto',device='cpu')
  self.tracker=StreamBOTSORT(args);self.geometry=Geometry();self.motion=BackgroundMotion();self.motion_relinks=0;self.moving_rejections=0;self.targets={};self.track_map={};self.sequence=0;self.last_frame=-1;self.last_ms=-1.;self.counter=0;self.events=0;self.relinks=0;self.expired=0;self.evicted=0;self.deferred=0;self.closed=False
 def _appearance(self,image,box):
  h,w=image.shape[:2];a,b,c,d=np.rint(box).astype(int);roi=image[max(0,b):min(h,d),max(0,a):min(w,c)]
  if not roi.size:return np.zeros(48,np.float32)
  hsv=cv2.cvtColor(cv2.resize(roi,(64,64)),cv2.COLOR_BGR2HSV);hist=cv2.calcHist([hsv],[0,1],None,[12,4],[0,180,0,256]).ravel();return hist/(hist.sum()+1e-8)
 def _quality(self,image,box,confidence):
  h,w=image.shape[:2];a,b,c,d=np.rint(box).astype(int);a=max(0,a);b=max(0,b);c=min(w,c);d=min(h,d)
  roi=image[b:d,a:c]
  if not roi.size:return None
  gray=cv2.cvtColor(cv2.resize(roi,(min(256,c-a),min(256,d-b))),cv2.COLOR_BGR2GRAY)
  sharp=float(cv2.Laplacian(gray,cv2.CV_64F).var());exposure=float(((gray>12)&(gray<244)).mean())
  center=max(0.,1.-np.linalg.norm(np.array([(a+c)/2-w/2,(b+d)/2-h/2]))/math.hypot(w/2,h/2))
  size=min(1.,math.sqrt((c-a)*(d-b)/(w*h))/.2);edge=min(a,b,w-c,h-d);border=float(edge>=8)
  score=.35*min(math.log1p(sharp)/math.log(600),1)+.2*center+.15*size+.15*exposure+.1*border+.05*confidence
  return {'score':float(score),'sharpness':sharp,'exposure':exposure,'centrality':float(center),'border_complete':bool(border),'eligible':bool(min(c-a,d-b)>=24 and sharp>=8 and exposure>=.35 and edge>=max(8.,self.min_border_fraction*min(w,h)))}
 def push(self,image,frame_id,timestamp_ms,detections):
  if self.closed:raise RuntimeError('Stream is closed')
  if frame_id==self.last_frame and timestamp_ms==self.last_ms:return []
  if frame_id<=self.last_frame or timestamp_ms<self.last_ms:raise ValueError('Frames/timestamps must be ordered; use a new core per stream')
  if image is None or image.ndim!=3:raise ValueError('Expected BGR image')
  gap=timestamp_ms-self.last_ms if self.last_ms>=0 else 200.
  if gap>2000:self.tracker.reset();self.track_map.clear();self.motion.reset()
  self.tracker.max_frames_lost=max(1,min(100,round(1500/max(gap,40))))
  self.last_frame=frame_id;self.last_ms=timestamp_ms;self.counter+=1
  h,w=image.shape[:2];valid=[]
  for d in detections:
   box=np.asarray(d['bbox_xyxy'],float);confidence=float(d['confidence'])
   if len(box)!=4 or not np.isfinite(box).all() or not np.isfinite(confidence):continue
   box[[0,2]]=np.clip(box[[0,2]],0,w);box[[1,3]]=np.clip(box[[1,3]],0,h)
   if min(box[2:]-box[:2])<8 or confidence<.35:continue
   valid.append({**d,'bbox_xyxy':box,'confidence':confidence})
  # A high-confidence whole-building box may coexist with a smaller partial roof box.
  # Prefer the whole box only for strong containment and a clear area difference.
  partial=set()
  for i,d in enumerate(valid):
   for j,other in enumerate(valid):
    if i==j or d.get('class_id',0)!=other.get('class_id',0) or other['confidence']<self.confirm_conf:continue
    b=d['bbox_xyxy'];large=other['bbox_xyxy'];area=np.prod(b[2:]-b[:2]);big=np.prod(large[2:]-large[:2])
    if area>=.75*big:continue
    _,contain,_=overlap(b,large);center=(b[:2]+b[2:])/2
    if contain>=.7 and np.all(center>=large[:2]) and np.all(center<=large[2:]):partial.add(i);break
  valid=[d for i,d in enumerate(valid) if i not in partial]
  # Same-frame duplicates are suppressed before evidence counting.
  dedup=[]
  for d in sorted(valid,key=lambda x:x['confidence'],reverse=True):
   b=d['bbox_xyxy'];duplicate=False
   for prev in dedup:
    p=prev['bbox_xyxy'];inter=np.maximum(0,np.minimum(b[2:],p[2:])-np.maximum(b[:2],p[:2])).prod();small=min(np.prod(b[2:]-b[:2]),np.prod(p[2:]-p[:2]))
    if d.get('class_id',0)==prev.get('class_id',0) and inter/max(small,1)>.9:duplicate=True;break
   if not duplicate:dedup.append(d)
  valid=dedup
  H=self.motion.update(image,[d['bbox_xyxy'] for d in valid])
  for target in self.targets.values():
   if H is not None and target.motion_box is not None and timestamp_ms-target.last_ms<=30000:
    if target.motion_quad is None:
     a,b,c,d=target.motion_box;target.motion_quad=np.float32([[a,b],[c,b],[c,d],[a,d]])
    target.motion_quad=cv2.perspectiveTransform(target.motion_quad[None].astype(np.float32),H)[0]
    if np.isfinite(target.motion_quad).all() and cv2.isContourConvex(target.motion_quad):target.motion_box=np.r_[target.motion_quad.min(0),target.motion_quad.max(0)]
    else:target.motion_box=None
    target.motion_valid=target.motion_valid and target.motion_box is not None
    if target.extent_quad is not None:
     target.extent_quad=cv2.perspectiveTransform(target.extent_quad[None].astype(np.float32),H)[0]
     if np.isfinite(target.extent_quad).all():target.extent_box=np.r_[target.extent_quad.min(0),target.extent_quad.max(0)]
     else:target.extent_box=None;target.extent_quad=None
   else:target.motion_valid=False
  s=min(1.,960/w);small=cv2.resize(image,(round(w*s),round(h*s)))
  arr=np.array([[*(d['bbox_xyxy']*s),d['confidence'],d.get('class_id',0)] for d in valid],np.float32).reshape(-1,6)
  self.tracker.update(Boxes(arr,small.shape[:2]),small)
  mapping={int(t.idx):int(t.track_id) for t in self.tracker.tracked_stracks if t.frame_id==self.tracker.frame_id and 0<=int(t.idx)<len(valid)}
  used=set();assign={};unknown=[]
  for i,d in enumerate(valid):
   track=mapping.get(i);tid=self.track_map.get(track);target=self.targets.get(tid)
   if target is not None:
    b=d['bbox_xyxy'];center=np.linalg.norm((b[:2]+b[2:]-target.box[:2]-target.box[2:])/2);diag=max(np.linalg.norm(target.box[2:]-target.box[:2]),40)
    if center>2*diag or timestamp_ms-target.last_ms>2000 or target.cls!=d.get('class_id',0):target=None
   if target is not None and target.id not in used:assign[i]=target;used.add(target.id)
   elif target is not None:continue
   elif d['confidence']>=self.confirm_conf:unknown.append(i)
  features=None;budget=self.geometry_budget;frame_jpeg=None
  def get_features():
   nonlocal features
   if features is None:features=self.geometry.extract(image,[d['bbox_xyxy'] for d in valid],frame_id)
   return features
  for i in unknown:
   d=valid[i];appearance=self._appearance(image,d['bbox_xyxy'])
   # Join overlapping fragments only when a recent whole-object extent supports their union.
   fragment_parent=None
   for j,t in assign.items():
    if t.cls!=d.get('class_id',0) or t.extent_box is None or not t.motion_valid or timestamp_ms-t.extent_ms>5000:continue
    existing=valid[j]['bbox_xyxy'];box=d['bbox_xyxy'];union=np.r_[np.minimum(existing[:2],box[:2]),np.maximum(existing[2:],box[2:])]
    iou,_,_=overlap(existing,box);history_iou,_,_=overlap(union,t.extent_box)
    inter=np.maximum(0,np.minimum(box[2:],t.extent_box[2:])-np.maximum(box[:2],t.extent_box[:2])).prod()/max(np.prod(box[2:]-box[:2]),1)
    if iou>=.25 and history_iou>=.65 and inter>=.75:
     valid[j]['bbox_xyxy']=union;valid[j]['confidence']=min(valid[j]['confidence'],d['confidence']);fragment_parent=t;break
   if fragment_parent is not None:
    if i in mapping:self.track_map[mapping[i]]=fragment_parent.id
    self.motion_relinks+=1;continue
   projected=[]
   for t in self.targets.values():
    if t.id in used or not t.motion_valid or t.motion_box is None or t.cls!=d.get('class_id',0) or timestamp_ms-t.last_ms>30000:continue
    iou,contain,ratio=overlap(t.motion_box,d['bbox_xyxy']);distance=np.linalg.norm((t.motion_box[:2]+t.motion_box[2:]-d['bbox_xyxy'][:2]-d['bbox_xyxy'][2:])/2)/max(np.linalg.norm(d['bbox_xyxy'][2:]-d['bbox_xyxy'][:2]),1)
    if (iou>=.35 or (contain>=.8 and ratio>=.25)) and distance<.45:projected.append((iou+.2*contain,t))
   projected.sort(key=lambda x:x[0],reverse=True)
   if projected and (len(projected)==1 or projected[0][0]-projected[1][0]>.2):
    target=projected[0][1];target.relinks+=1;self.relinks+=1;self.motion_relinks+=1;assign[i]=target;used.add(target.id)
    if i in mapping:self.track_map[mapping[i]]=target.id
    continue
   eligible=[t for t in self.targets.values() if t.id not in used and t.cls==d.get('class_id',0) and timestamp_ms-t.last_ms<=self.archive_ms and t.views]
   # Rank retrieval only; appearance alone NEVER merges identities.
   candidates=sorted(eligible,key=lambda t:min([float(np.abs(appearance-a).sum()) for a in [t.appearance,t.entry_appearance,t.confirmed_appearance] if a is not None] or [2.])+min((timestamp_ms-t.last_ms)/60000,.5))[:3]
   matches=[];fully_checked=True
   for t in candidates:
    if budget<=0:fully_checked=False;break
    best=None
    for old,old_box in ([t.views[-1],t.views[0]] if len(t.views)>1 else t.views):
     if budget<=0:fully_checked=False;break
     budget-=1;result=self.geometry.verify(old,get_features(),old_box,d['bbox_xyxy'])
     if result is not None:best=result;break
    if best:matches.append((best['iou'],t))
   matches.sort(key=lambda p:p[0],reverse=True)
   if matches and (len(matches)==1 or matches[0][0]-matches[1][0]>.15):
    target=matches[0][1];target.relinks+=1;self.relinks+=1
   elif matches or not fully_checked:
    self.deferred+=1;continue
   else:
    self.sequence+=1;tid=f'{self.stream_id}_{self.sequence:06d}'
    target=Target(tid,int(d.get('class_id',0)),d.get('class_name','house'),timestamp_ms,frame_id,timestamp_ms,d['bbox_xyxy'].copy());self.targets[tid]=target
   assign[i]=target;used.add(target.id)
   if i in mapping:self.track_map[mapping[i]]=target.id
  events=[]
  for i,target in assign.items():
   d=valid[i]
   static_ok=False
   if target.motion_valid and target.motion_box is not None:
    residual=np.linalg.norm((target.motion_box[:2]+target.motion_box[2:]-d['bbox_xyxy'][:2]-d['bbox_xyxy'][2:])/2)
    static_ok=residual<=max(12.,.12*np.linalg.norm(d['bbox_xyxy'][2:]-d['bbox_xyxy'][:2]))
   target.stationary.append((timestamp_ms,bool(static_ok)))
   while target.stationary and timestamp_ms-target.stationary[0][0]>self.window_ms:target.stationary.popleft()
   if target.extent_box is None or timestamp_ms-target.extent_ms>5000 or np.prod(d['bbox_xyxy'][2:]-d['bbox_xyxy'][:2])>=np.prod(target.extent_box[2:]-target.extent_box[:2]):
    a,b,c,e=d['bbox_xyxy'];target.extent_quad=np.float32([[a,b],[c,b],[c,e],[a,e]]);target.extent_box=d['bbox_xyxy'].copy();target.extent_ms=timestamp_ms
   target.last_ms=timestamp_ms;target.box=d['bbox_xyxy'].copy();target.motion_box=target.box.copy();target.motion_quad=None;target.motion_valid=True;target.hits+=1
   if i in mapping:self.track_map[mapping[i]]=target.id
   target.appearance=self._appearance(image,target.box);target.last_confidence=d['confidence']
   while target.high and timestamp_ms-target.high[0][0]>self.window_ms:target.high.popleft()
   if d['confidence']>=self.confirm_conf:
    if not target.high or target.high[-1][1]!=frame_id:target.high.append((timestamp_ms,frame_id))
    target.max_high_hits=max(target.max_high_hits,len(target.high))
    if target.first_high_ms is None:target.first_high_ms=timestamp_ms
    if timestamp_ms-target.last_view_ms>=1500 or not target.views:
     if not target.views:target.entry_appearance=target.appearance.copy()
     target.views.append((get_features(),target.box.copy()))
     # Retain the entry viewpoint for later revisits, plus two recent views.
     if target.confirmed_view is not None:target.views=[target.views[0],target.confirmed_view,target.views[-1]]
     elif len(target.views)>3:target.views=[target.views[0],*target.views[-2:]]
     target.last_view_ms=timestamp_ms
    if not target.emitted:
     q=self._quality(image,target.box,d['confidence']);target.last_quality=q
     if q and q['eligible'] and (target.best is None or q['score']>target.best['quality']['score']):
      if frame_jpeg is None:
       ok,jpg=cv2.imencode('.jpg',image,[cv2.IMWRITE_JPEG_QUALITY,95])
       if ok:frame_jpeg=jpg.tobytes()
      if frame_jpeg is not None:target.best={'jpeg':frame_jpeg,'frame_id':frame_id,'timestamp_ms':timestamp_ms,'bbox_xyxy':target.box.tolist(),'confidence':d['confidence'],'quality':q}
   if not target.emitted and target.best is not None and len(target.high)>=self.confirm_hits and (not self.assume_static or sum(ok for _,ok in target.stationary)>=3):
    # An entry fragment can become a new track before enough geometry exists
    # to recognize an old target. Recheck when the full target is confirmable.
    archived=[t for t in self.targets.values() if t.emitted and t.id not in used and t.cls==target.cls and t.confirmed_view is not None and timestamp_ms-t.last_ms<=self.archive_ms]
    archived=sorted(archived,key=lambda t:min([float(np.abs(target.appearance-a).sum()) for a in [t.appearance,t.entry_appearance,t.confirmed_appearance] if a is not None] or [2.]))[:3]
    revisits=[];complete=True
    for old_target in archived:
     for old_features,old_box in [old_target.confirmed_view,old_target.views[0]]:
      if budget<=0:complete=False;break
      budget-=1;verified=self.geometry.verify(old_features,get_features(),old_box,target.box)
      if verified is not None:
       revisits.append((verified['iou'],old_target));break
    revisits.sort(key=lambda v:v[0],reverse=True)
    if complete and revisits and (len(revisits)==1 or revisits[0][0]-revisits[1][0]>.15):
     canonical=revisits[0][1]
     for track,tid in list(self.track_map.items()):
      if tid==target.id:self.track_map[track]=canonical.id
     for attr in ['last_ms','box','appearance','motion_box','motion_quad','motion_valid','extent_box','extent_quad','extent_ms','stationary','last_confidence']:
      setattr(canonical,attr,getattr(target,attr))
     canonical.hits+=target.hits;canonical.relinks+=1;self.relinks+=1
     self.targets.pop(target.id,None);assign[i]=canonical;used.add(canonical.id)
     continue
    if revisits or not complete:
     self.deferred+=1;continue
    target.confirmed_view=(get_features(),target.box.copy());target.confirmed_appearance=target.appearance.copy()
    target.views=[target.views[0],target.confirmed_view,target.views[-1]]
    b=target.best;event={k:v for k,v in b.items() if k!='jpeg'}
    event.update({'target_id':target.id,'stream_id':self.stream_id,'class_id':target.cls,'class_name':target.name,'image':cv2.imdecode(np.frombuffer(b['jpeg'],np.uint8),cv2.IMREAD_COLOR),'confirmed_frame_id':frame_id,'confirmed_timestamp_ms':timestamp_ms,'first_seen_frame_id':target.first_frame,'first_high_timestamp_ms':target.first_high_ms,'confirmation_latency_ms':timestamp_ms-target.first_high_ms,'evidence_frames':[v[1] for v in target.high],'relinks':target.relinks,'stationary_evidence':sum(ok for _,ok in target.stationary)})
    target.emitted=True;target.best=None;self.events+=1;events.append(event)
  self.frame_assignments=[{'target_id':t.id,'track_id':mapping.get(i),'bbox_xyxy':valid[i]['bbox_xyxy'].tolist(),'confidence':valid[i]['confidence'],'emitted':t.emitted} for i,t in assign.items()]
  self._cleanup(timestamp_ms)
  return events
 def _review(self,target):
  if target.emitted or target.id in self.review_ids or target.hits<3:return
  reason='insufficient_recent_high_confidence'
  if target.max_high_hits>=self.confirm_hits:
   reason='representative_frame_quality_or_border' if target.best is None else 'insufficient_background_stationarity'
  self.review_items.append({'candidate_id':target.id,'class_id':target.cls,'class_name':target.name,'first_frame_id':target.first_frame,'last_timestamp_ms':target.last_ms,'last_bbox_xyxy':target.box.tolist(),'last_confidence':target.last_confidence,'hits':target.hits,'max_high_hits_in_window':target.max_high_hits,'reason':reason,'last_quality':target.last_quality})
  self.review_ids.add(target.id)
  if len(self.review_items)>self.max_targets:self.review_items=self.review_items[-self.max_targets:]
  self.review_ids={item["candidate_id"] for item in self.review_items}
 def _cleanup(self,timestamp_ms):
  remove=[k for k,t in self.targets.items() if timestamp_ms-t.last_ms>(self.archive_ms if t.emitted else 30000)]
  self.expired+=len(remove)
  for k in remove:self._review(self.targets[k]);del self.targets[k]
  if len(self.targets)>self.max_targets:
   keys=sorted(self.targets,key=lambda k:self.targets[k].last_ms)[:len(self.targets)-self.max_targets];self.evicted+=len(keys)
   for k in keys:self._review(self.targets[k]);del self.targets[k]
  self.track_map={k:v for k,v in self.track_map.items() if v in self.targets}
  # Best frames from stale unconfirmed candidates are bounded by the 30-second retention above.
 def tick(self,timestamp_ms):
  if timestamp_ms<self.last_ms:raise ValueError('Time cannot go backwards')
  self._cleanup(timestamp_ms);return []
 def flush(self):
  self.closed=True
  for t in self.targets.values():self._review(t);t.best=None
  return []
 def stats(self):
  return {'frames':self.counter,'business_targets_created':self.sequence,'emitted':self.events,'geometry_checks':self.geometry.matches,'geometry_accepted':self.geometry.accepted,'relinks':self.relinks,'motion_relinks':self.motion_relinks,'background_motion_frames':self.motion.accepted,'deferred_associations':self.deferred,'retained_targets':len(self.targets),'expired_targets':self.expired,'capacity_evictions':self.evicted,'review_candidates':len(self.review_items)}
