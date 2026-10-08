"""Observed passive-body bounds, separate from detector/pin-crop extents."""
import copy
import math
import cv2
import numpy as np

def body_box(image,box,category,texts=()):
    if category not in ('R','C'): return None
    h,w=image.shape[:2]; a,b,c,d=map(round,box)
    a,b,c,d=max(0,a),max(0,b),min(w,c),min(h,d)
    if min(c-a,d-b)<8: return None
    ink=(cv2.cvtColor(image[b:d,a:c],cv2.COLOR_BGR2GRAY)<180).astype(np.uint8)*255
    for t in texts:
        x1,y1,x2,y2=map(round,t['bbox'])
        ax,ay,bx,by=max(0,x1-a),max(0,y1-b),min(c-a,x2-a),min(d-b,y2-b)
        if ax<bx and ay<by: ink[ay:by,ax:bx]=0
    ch,cw=ink.shape; cx,cy=(cw-1)/2,(ch-1)/2
    if category=='C':
        candidates=[]
        for vertical in (False,True):
            view=ink.T if vertical else ink
            opened=cv2.morphologyEx(view,cv2.MORPH_OPEN,np.ones((1,5),np.uint8))
            _,_,stats,_=cv2.connectedComponentsWithStats(opened)
            bars=[(x,y,bw,bh) for x,y,bw,bh,_ in stats[1:] if bw>=5 and bw>=2*bh and bh<=5]
            for i,(x,y,bw,bh) in enumerate(bars):
                for xx,yy,ww,hh in bars[:i]:
                    spacing=abs(y+bh/2-yy-hh/2)
                    if not 2<=spacing<=view.shape[0]*.45 or not .5<=bw/ww<=2: continue
                    overlap=min(x+bw,xx+ww)-max(x,xx)
                    if overlap<min(bw,ww)*.7: continue
                    bounds=[min(x,xx),min(y,yy),max(x+bw,xx+ww),max(y+bh,yy+hh)]
                    if vertical: bounds=[bounds[1],bounds[0],bounds[3],bounds[2]]
                    bx,by=(bounds[0]+bounds[2])/2,(bounds[1]+bounds[3])/2
                    if abs(bx-cx)>cw*.25 or abs(by-cy)>ch*.25: continue
                    score=overlap-math.hypot(bx-cx,by-cy)
                    candidates.append((score,bounds))
        if not candidates: return None
        _,bounds=max(candidates,key=lambda v:v[0])
        method='observed_parallel_capacitor_plates'
    else:
        contours,_=cv2.findContours(ink,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
        candidates=[]
        for contour in contours:
            x,y,bw,bh=cv2.boundingRect(contour)
            if min(bw,bh)<4 or max(bw,bh)<8: continue
            approx=cv2.approxPolyDP(contour,.025*cv2.arcLength(contour,True),True)
            if len(approx)!=4 or cv2.contourArea(contour)<bw*bh*.65: continue
            if abs(x+bw/2-cx)>cw*.25 or abs(y+bh/2-cy)>ch*.25: continue
            if bw>=cw*.95 and bh>=ch*.95: continue
            candidates.append((bw*bh,[x,y,x+bw,y+bh]))
        if not candidates: return None
        _,bounds=max(candidates,key=lambda v:v[0]); method='observed_closed_resistor_body'
    x1,y1,x2,y2=bounds
    result=[float(a+max(0,x1-2)),float(b+max(0,y1-2)),float(a+min(cw,x2+2)),float(b+min(ch,y2+2))]
    return {'bbox':result,'method':method,'detector_bbox':list(box)}

def propose_body_boxes(image,observations):
    h=image.shape[0]; result={}; evidence=[]
    flip=lambda box:[box[0],h-box[3],box[2],h-box[1]]
    texts=[dict(t,bbox=flip(t['bbox'])) for t in observations.get('texts',[])]
    for key,c in observations['components'].items():
        category=observations.get('component_categories',{}).get(key)
        if category is None: category={'resistor':'R','r':'R','capacitor':'C','c':'C'}.get(c['type'])
        record=body_box(image,flip(c['bbox']),category,texts)
        if record:
            result[key]=flip(record['bbox']); evidence.append(dict(component=key,**record))
    return result,{'module':'ObservedPassiveBodyBounds','accepted':len(result),'evidence':evidence,'policy':'output bounds only; original detector box retained for OCR, pin crops and wire masking'}

def apply_body_boxes(result,observations):
    result=copy.deepcopy(result)
    for key,box in observations.get('component_body_boxes',{}).items():
        if key not in result['components']: raise ValueError('body box references unknown component')
        result['components'][key]['bbox']=list(box)
    return result
