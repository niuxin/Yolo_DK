"""Runnable HTTP contract example. Repeating a sample tests the API, not real-world accuracy."""
import argparse,json,uuid
from pathlib import Path
from urllib.request import Request,build_opener,ProxyHandler
HTTP=build_opener(ProxyHandler({}))
def request(base,method,path,data=None,content_type='application/json'):
 raw=json.dumps(data).encode() if data is not None and not isinstance(data,bytes) else data
 with HTTP.open(Request(base.rstrip('/')+path,data=raw,method=method,headers={'Content-Type':content_type}),timeout=120) as r:return json.load(r)
def multipart(fields,image):
 boundary='house_'+uuid.uuid4().hex;parts=[]
 for k,v in fields.items():parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode())
 parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="image"; filename="frame.jpg"\r\nContent-Type: image/jpeg\r\n\r\n'.encode()+image+b'\r\n')
 parts.append(f'--{boundary}--\r\n'.encode());return b''.join(parts),'multipart/form-data; boundary='+boundary

def main():
 p=argparse.ArgumentParser();p.add_argument('--base',default='http://127.0.0.1:18081');p.add_argument('--image',type=Path,default=Path(__file__).with_name('sample.jpg'));p.add_argument('--frames',type=int,default=8);p.add_argument('--output',type=Path,default=Path(__file__).parent/'results');a=p.parse_args()
 if a.frames<1:p.error('frames must be positive')
 rid=uuid.uuid4().hex;raw=a.image.read_bytes();responses=[];closed=None
 request(a.base,'POST','/v2/runs',{'run_id':rid,'session_id':'example_'+rid[:8],'detector_id':'house'})
 try:
  for i in range(a.frames):
   body,ctype=multipart({'run_id':rid,'frame_id':i,'timestamp_ms':i*200},raw)
   r=request(a.base,'POST','/v2/detect',body,ctype);responses.append(r)
   retry=request(a.base,'POST','/v2/detect',body,ctype)
   assert r==retry,'Identical frame retry changed response'
   print('frame',i,'new events',len(r['events']),flush=True)
 finally:closed=request(a.base,'POST',f'/v2/runs/{rid}/close')
 dest=a.output/rid;dest.mkdir(parents=True,exist_ok=True)
 for event in closed['events']:
  folder=dest/event['target_id'];folder.mkdir(exist_ok=True)
  for name,url in event['assets'].items():
   if Path(name).name!=name or not url.startswith(f'/v2/runs/{rid}/events/'):raise ValueError('Invalid asset response')
   with HTTP.open(a.base.rstrip('/')+url,timeout=30) as r:(folder/name).write_bytes(r.read())
 (dest/'result.json').write_text(json.dumps({'run_id':rid,'responses':responses,'closed':closed,'retry_consistent':True},ensure_ascii=False,indent=2))
 print('Completed:',len(closed['events']),'unique business events;',dest,flush=True)
if __name__=='__main__':main()
