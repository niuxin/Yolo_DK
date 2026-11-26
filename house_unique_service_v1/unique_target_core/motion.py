import cv2,numpy as np
class BackgroundMotion:
 def __init__(self):self.gray=None;self.mask=None;self.scale=None;self.accepted=0;self.failed=0
 def update(self,image,boxes):
  h,w=image.shape[:2];s=min(1.,960/w);gray=cv2.cvtColor(cv2.resize(image,(round(w*s),round(h*s))),cv2.COLOR_BGR2GRAY);mask=np.full(gray.shape,255,np.uint8)
  for b in boxes:
   a,y,c,d=(np.array(b)*s).astype(int);cv2.rectangle(mask,(a,y),(c,d),0,-1)
  H=None
  if self.gray is not None and self.gray.shape==gray.shape:
   pts=cv2.goodFeaturesToTrack(self.gray,maxCorners=350,qualityLevel=.01,minDistance=10,mask=self.mask)
   if pts is not None and len(pts)>=30:
    nxt,ok,_=cv2.calcOpticalFlowPyrLK(self.gray,gray,pts,None,winSize=(21,21),maxLevel=3)
    if nxt is not None:
     back,ok2,_=cv2.calcOpticalFlowPyrLK(gray,self.gray,nxt,None,winSize=(21,21),maxLevel=3)
     if back is not None:
      valid=ok.ravel().astype(bool)&ok2.ravel().astype(bool)&(np.linalg.norm(back-pts,axis=2).ravel()<1.5)
      a=pts[valid].reshape(-1,2);b=nxt[valid].reshape(-1,2)
      if len(a)>=30:
       candidate,inside=cv2.findHomography(a,b,cv2.RANSAC,2.5)
       if candidate is not None and inside is not None and inside.sum()>=25 and inside.mean()>=.5:
        corners=np.float32([[[0,0],[gray.shape[1],0],[gray.shape[1],gray.shape[0]],[0,gray.shape[0]]]])
        q=cv2.perspectiveTransform(corners,candidate)[0]
        area=abs(cv2.contourArea(q))/(gray.shape[0]*gray.shape[1])
        if np.isfinite(q).all() and cv2.isContourConvex(q) and .5<area<2:
         S=np.diag([s,s,1]);H=np.linalg.inv(S)@candidate@S
  self.gray=gray;self.mask=mask;self.scale=s
  if H is None:self.failed+=1
  else:self.accepted+=1
  return H
 def reset(self):self.gray=None;self.mask=None

def project(box,H):
 x,y,x2,y2=box;q=cv2.perspectiveTransform(np.float32([[[x,y],[x2,y],[x2,y2],[x,y2]]]),H)[0]
 if not np.isfinite(q).all() or not cv2.isContourConvex(q):return None
 return np.r_[q.min(0),q.max(0)]

def overlap(a,b):
 a=np.asarray(a);b=np.asarray(b);inter=np.maximum(0,np.minimum(a[2:],b[2:])-np.maximum(a[:2],b[:2])).prod();aa=np.maximum(0,a[2:]-a[:2]).prod();bb=np.maximum(0,b[2:]-b[:2]).prod()
 return float(inter/(aa+bb-inter+1e-8)),float(inter/(min(aa,bb)+1e-8)),float(min(aa,bb)/(max(aa,bb)+1e-8))
