import cv2,numpy as np
from dataclasses import dataclass
@dataclass
class Features:
 frame_id:int
 points:object
 descriptors:object
 scale:float

class Geometry:
 def __init__(self,width=960):
  self.width=width;self.sift=cv2.SIFT_create(nfeatures=1000);self.matches=0;self.accepted=0
 def extract(self,image,boxes,frame_id):
  h,w=image.shape[:2];s=min(1.,self.width/w);gray=cv2.cvtColor(cv2.resize(image,(round(w*s),round(h*s))),cv2.COLOR_BGR2GRAY)
  mask=np.full(gray.shape,255,np.uint8)
  for box in boxes:
   a,b,c,d=np.array(box)*s;cv2.rectangle(mask,(max(0,int(a)-3),max(0,int(b)-3)),(int(c)+3,int(d)+3),0,-1)
  kp,des=self.sift.detectAndCompute(gray,mask)
  return Features(frame_id,np.float32([k.pt for k in kp]).reshape(-1,2),des.astype(np.float16) if des is not None else None,s)
 def verify(self,old,new,old_box,new_box):
  self.matches+=1
  if old.frame_id==new.frame_id or old.descriptors is None or new.descriptors is None:return None
  pairs=cv2.BFMatcher().knnMatch(old.descriptors.astype(np.float32),new.descriptors.astype(np.float32),k=2)
  good=[a for pair in pairs if len(pair)==2 for a,b in [pair] if a.distance<.72*b.distance]
  # Each destination feature may support only one correspondence.
  good=sorted(good,key=lambda a:a.distance);used=set();good=[a for a in good if not(a.trainIdx in used or used.add(a.trainIdx))]
  if len(good)<14:return None
  a=np.float32([old.points[m.queryIdx] for m in good]);b=np.float32([new.points[m.trainIdx] for m in good])
  H,inliers=cv2.findHomography(a,b,cv2.RANSAC,3.)
  if H is None or inliers is None:return None
  valid=inliers.ravel().astype(bool);count=int(valid.sum());ratio=count/len(good)
  if count<14 or ratio<.45:return None
  if min(np.ptp(a[valid],axis=0))<35 or min(np.ptp(b[valid],axis=0))<35:return None
  # Require supporting background features near the target, not only in a distant part of the image.
  ab=np.array(old_box)*old.scale;bb=np.array(new_box)*new.scale
  def local(points,box):
   pad=max(80.,np.linalg.norm(box[2:]-box[:2])*1.2)
   return ((points>=box[:2]-pad)&(points<=box[2:]+pad)).all(1).sum()
  if local(a[valid],ab)<4 or local(b[valid],bb)<4:return None
  x,y,x2,y2=ab;corners=np.float32([[[x,y],[x2,y],[x2,y2],[x,y2]]]);q=cv2.perspectiveTransform(corners,H)[0]/new.scale
  if not np.isfinite(q).all() or not cv2.isContourConvex(q.astype(np.float32)):return None
  proj=np.r_[q.min(0),q.max(0)];box=np.array(new_box);wh=np.maximum(0,np.minimum(proj[2:],box[2:])-np.maximum(proj[:2],box[:2]));intersection=wh.prod();iou=intersection/(np.prod(proj[2:]-proj[:2])+np.prod(box[2:]-box[:2])-intersection+1e-8)
  distance=np.linalg.norm((proj[:2]+proj[2:]-box[:2]-box[2:])/2)/max(np.linalg.norm(box[2:]-box[:2]),1)
  if iou<.4 or distance>.35:return None
  self.accepted+=1
  return {'iou':float(iou),'inliers':count,'inlier_ratio':float(ratio),'center_distance':float(distance)}
