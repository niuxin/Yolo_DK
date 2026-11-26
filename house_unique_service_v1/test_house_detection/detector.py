"""Single-frame house detection: full image + optional overlapping tiles, output original-image xyxy."""
from pathlib import Path
import os,time,json
R=Path(__file__).resolve().parent
os.environ.setdefault('YOLO_CONFIG_DIR',str(R/'.cache/yolo'))
os.environ.setdefault('MPLCONFIGDIR',str(R/'.cache/matplotlib'))
import cv2,numpy as np,torch
from torchvision.ops import nms
from ultralytics import YOLO
class HouseDetector:
 def __init__(self,weights=None,conf=None,tiled=None,device='cpu',image_size=None,tile_size=None,iou_threshold=None,max_det=None,nms_iou_threshold=None):
  config=json.loads((R/'inference_config.json').read_text()) if (R/'inference_config.json').exists() else {}
  self.model=YOLO(str(weights or R.parent/'trained_model/house_v3_recommended.pt'))
  self.conf=config.get('conf',.4) if conf is None else conf
  self.tiled=config.get('tiled',True) if tiled is None else tiled
  self.device=device
  self.image_size=config.get('image_size',640) if image_size is None else image_size
  self.tile_size=config.get('tile_size',1280) if tile_size is None else tile_size
  self.iou_threshold=config.get('iou_threshold',.45) if iou_threshold is None else iou_threshold
  self.max_det=config.get('max_det',300) if max_det is None else max_det
  self.nms_iou_threshold=config.get('nms_iou_threshold',.45) if nms_iou_threshold is None else nms_iou_threshold
 def detect(self,image,frame_id=None,conf=None):
  if image is None or image.ndim!=3 or image.shape[2]!=3:raise ValueError('Expected HxWx3 BGR image')
  h,w=image.shape[:2];regions=[(0,0,w,h,True)]
  if self.tiled and max(w,h)>1600:
   size=min(self.tile_size,w,h);step=max(1,int(size*.8))
   xs=sorted(set(list(range(0,w-size+1,step))+[w-size]));ys=sorted(set(list(range(0,h-size+1,step))+[h-size]))
   regions += [(x,y,x+size,y+size,False) for y in ys for x in xs]
  allboxes=[];scores=[]
  for offset in range(0,len(regions),4):
   batch=regions[offset:offset+4]
   results=self.model.predict([image[y:y2,x:x2] for x,y,x2,y2,_ in batch],imgsz=self.image_size,conf=self.conf if conf is None else conf,iou=self.iou_threshold,max_det=self.max_det,device=self.device,quantize=32 if self.device=='cpu' else 16,verbose=False)
   for region,result in zip(batch,results):
    x,y,x2,y2,full=region
    for row in result.boxes.data.cpu().numpy():
     a,b,c,d,score,cls=row
     # Suppress cut-off tile-edge boxes; overlap/full-image pass may provide complete detections.
     if not full and ((a<4 and x>0) or (b<4 and y>0) or (c>x2-x-4 and x2<w) or (d>y2-y-4 and y2<h)):continue
     box=[max(0,float(a+x)),max(0,float(b+y)),min(w,float(c+x)),min(h,float(d+y))]
     if box[2]>box[0] and box[3]>box[1]:allboxes.append(box);scores.append(float(score))
  if not allboxes:return []
  keep=nms(torch.tensor(allboxes),torch.tensor(scores),self.nms_iou_threshold).tolist()[:self.max_det]
  return [{'frame_id':frame_id,'class_id':0,'class_name':'house','bbox_xyxy':allboxes[i],'confidence':scores[i]} for i in keep]
if __name__=='__main__':
 import argparse,json
 p=argparse.ArgumentParser();p.add_argument('image');p.add_argument('--output',required=True);p.add_argument('--weights');p.add_argument('--conf',type=float,default=None);p.add_argument('--full-only',action='store_true');a=p.parse_args()
 torch.set_num_threads(4);im=cv2.imread(a.image)
 results=HouseDetector(a.weights,a.conf,False if a.full_only else None).detect(im)
 Path(a.output).write_text(json.dumps(results,ensure_ascii=False,indent=2));print(len(results),'detections')
