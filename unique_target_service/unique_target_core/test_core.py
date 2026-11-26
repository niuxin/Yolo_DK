from pathlib import Path
import os
ROOT=Path(__file__).resolve().parent.parent
os.environ['YOLO_CONFIG_DIR']=str(ROOT/'test_house_detection/.cache/yolo')
import cv2,numpy as np,json
from core import UniqueTargetCore
from geometry import Geometry
cv2.setNumThreads(2)
rng=np.random.default_rng(42)
im=rng.integers(0,256,(540,960,3),dtype=np.uint8)
im=cv2.GaussianBlur(im,(3,3),0)
for i in range(120):
 x,y=rng.integers([0,0],[960,540]);cv2.circle(im,(int(x),int(y)),int(rng.integers(3,12)),tuple(int(x) for x in rng.integers(0,255,3)),-1)
b=[300,200,420,320]
def det(box=b,conf=.94):return {'bbox_xyxy':box,'confidence':conf,'class_id':0,'class_name':'house'}
checks=[]
c=UniqueTargetCore('one');events=[]
for i in range(10):events+=c.push(im,i,i*200,[det()])
assert len(events)==1 and events[0]['confirmation_latency_ms']==800 and events[0]['image'].shape==im.shape
assert c.push(im,9,1800,[det()])==[];checks.append('stable target emits once; duplicate push idempotent')
for i in range(10,25):c.push(im,i,i*200,[])
M=np.float32([[1,0,60],[0,1,20]]);shifted=cv2.warpAffine(im,M,(960,540));b2=[360,220,480,340]
for i in range(25,33):assert c.push(shifted,i,i*200,[det(b2)])==[]
assert c.relinks>=1;checks.append('lost track and camera translation reconnect emitted target')
c2=UniqueTargetCore('two');events=[]
for i in range(7):events+=c2.push(im,i,i*200,[det(),det([450,200,570,320])])
assert len(events)==2 and len({e['target_id'] for e in events})==2;checks.append('co-visible neighbors remain distinct')
c3=UniqueTargetCore('gaps');events=[]
for i in range(9):events+=c3.push(im,i,i*200,[det()] if i%2==0 else [])
assert not events;assert c3.flush()==[];assert c3.flush()==[];checks.append('window rejects insufficient evidence; flush does not force output')
c4=UniqueTargetCore('sparse');events=[]
for i in range(6):events+=c4.push(im,i*30,i*600,[det()])
assert not events;checks.append('confirmation uses source time not received-frame count')
try:c2.push(im,0,0,[det()]);raise AssertionError('expected order error')
except ValueError:checks.append('out-of-order frames rejected')
g=Geometry();f=g.extract(im,[b],0);new=g.extract(shifted,[b2],1)
assert g.verify(f,new,b,b2) is not None
assert g.verify(f,new,b,[700,400,820,520]) is None
flat=g.extract(np.zeros_like(im),[b2],2);assert g.verify(f,flat,b,b2) is None;checks.append('geometry accepts translation, rejects wrong target and no texture')
cm=UniqueTargetCore('moving');moving=[]
for i in range(8):moving+=cm.push(im,i,i*200,[det([100+i*35,200,220+i*35,320])])
assert not moving;checks.append('background-relative moving object is not confirmed as stationary house')
ce=UniqueTargetCore('export');export_events=[]
for i in range(6):export_events+=ce.push(im,i,i*200,[det()])
from export import export_event
import tempfile
with tempfile.TemporaryDirectory(dir=ROOT/'unique_target_core') as td:
 metadata=export_event(export_events[0],td)
 assert Path(metadata['files']['original']).exists() and Path(metadata['files']['annotated']).exists()
 json.dumps(metadata)
checks.append('all event fields serialize; original and annotated artifacts exported')
cn=UniqueTargetCore('nested');nested=[]
for i in range(7):nested+=cn.push(im,i,i*200,[det([300,200,550,350],.92),det([295,220,370,360],.96)])
assert len(nested)==1;checks.append('overlapping partial roof box does not create a second business event')
cf=UniqueTargetCore('fragment');fragment=[]
for i in range(6):fragment+=cf.push(im,i,i*200,[det([300,200,550,350])])
for i in range(6,13):fragment+=cf.push(im,i,i*200,[det([300,200,470,350]),det([390,200,550,350])])
assert len(fragment)==1;checks.append('recent whole-object geometry reconnects simultaneous partial boxes')
ca=UniqueTargetCore('stream_a');ca.push(im,0,0,[det()]);cb=UniqueTargetCore('stream_b')
combined=[]
for i in range(1,7):
 combined+=ca.push(im,i,i*200,[det(),det([600,200,720,320])])
 cb.push(im,i,i*200,[det()])
assert len(combined)==2;checks.append('interleaved stream instances do not reset each other track identities')
try:UniqueTargetCore('../outside');raise AssertionError('expected invalid ID')
except ValueError:checks.append('unsafe output IDs rejected')
cborder=UniqueTargetCore('border');border_events=[]
for i in range(8):border_events+=cborder.push(im,i,i*200,[det([0,200,100,320])])
assert not border_events;cborder.flush();assert cborder.review_items
checks.append('border-truncated target retained for review instead of emitting incomplete evidence')
# A long visit overwrites recent backgrounds. Returning to the original entry
# viewpoint after track expiry must reconnect the emitted business target.
cr=UniqueTargetCore('entry_revisit');revisited=[]
for i in range(7):revisited+=cr.push(im,i,i*200,[det()])
changed=cv2.GaussianBlur(rng.integers(0,256,im.shape,dtype=np.uint8),(3,3),0)
for i in range(7,40):revisited+=cr.push(changed,i,i*200,[det()])
for i in range(40,55):cr.push(changed,i,i*200,[])
for i in range(55,63):revisited+=cr.push(im,i,i*200,[det()])
assert len(revisited)==1 and cr.relinks>=1
checks.append('entry viewpoint survives recent-view turnover and reconnects revisit after track expiry')
cdelay=UniqueTargetCore('fragment_revisit');delayed=[]
for i in range(7):delayed+=cdelay.push(im,i,i*200,[det()])
for i in range(7,20):cdelay.push(im,i,i*200,[])
for i in range(20,23):delayed+=cdelay.push(im,i,40000+(i-20)*200,[det([300,200,330,320])])
for i in range(23,31):delayed+=cdelay.push(im,i,40000+(i-20)*200,[det()])
assert len(delayed)==1 and cdelay.relinks>=1
checks.append('partial re-entry track is reconciled before confirming a duplicate business event')
print(json.dumps({'checks':checks,'passed':len(checks)},indent=2))
