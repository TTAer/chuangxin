"""Topology Perception: crossing states from appearance plus local geometry.

Color is an input feature, never a rule that declares an electrical connection.
"""
from pathlib import Path
import json
import math
import cv2
import numpy as np


def patch_tensor(image,x,y,size=48,radius=16):
    h,w=image.shape[:2]; x=int(round(x)); y=int(round(y))
    padded=cv2.copyMakeBorder(image,radius,radius,radius,radius,cv2.BORDER_CONSTANT,value=(255,255,255))
    crop=padded[max(0,y):min(h+2*radius,y+2*radius),max(0,x):min(w+2*radius,x+2*radius)]
    if not crop.size: raise ValueError('junction patch outside image')
    crop=cv2.resize(crop,(size,size))
    rgb=crop[:,:,::-1].astype(np.float32)/255
    gray=cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY).astype(np.float32)/255
    ink=(gray<.86).astype(np.float32)
    horizontal=cv2.morphologyEx(ink,cv2.MORPH_OPEN,np.ones((1,7),np.uint8))
    vertical=cv2.morphologyEx(ink,cv2.MORPH_OPEN,np.ones((7,1),np.uint8))
    return np.ascontiguousarray(np.concatenate((rgb.transpose(2,0,1),gray[None],horizontal[None],vertical[None])))


def crossing_candidates(segments):
    candidates=[]; seen=set()
    for i,(a,b) in enumerate(segments):
        for j,(c,d) in enumerate(segments[:i]):
            if (a[0]==b[0])==(c[0]==d[0]): continue
            horizontal=(a,b) if a[1]==b[1] else (c,d)
            vertical=(c,d) if a[1]==b[1] else (a,b)
            x,y=vertical[0][0],horizontal[0][1]
            if min(horizontal[0][0],horizontal[1][0])-2<=x<=max(horizontal[0][0],horizontal[1][0])+2 and min(vertical[0][1],vertical[1][1])-2<=y<=max(vertical[0][1],vertical[1][1])+2:
                key=(round(x),round(y))
                if key in seen: continue
                seen.add(key)
                geometry=np.clip([abs(x-horizontal[0][0])/32,abs(x-horizontal[1][0])/32,abs(y-vertical[0][1])/32,abs(y-vertical[1][1])/32],0,1).tolist()
                endpoint=min(math.dist((x,y),p) for p in (a,b,c,d))<=3
                candidates.append({'x':x,'y':y,'wires':[i,j],'geometry':geometry,'endpoint_contact':endpoint})
    return candidates


def dot_evidence(mask,x,y):
    h,w=mask.shape; x=round(x); y=round(y)
    if x<12 or y<12 or x+12>=w or y+12>=h: return False
    ink=mask[y-12:y+13,x-12:x+13]>0
    horizontal=float(np.median(ink[8:17,:8].sum(axis=0)))
    vertical=float(np.median(ink[:8,8:17].sum(axis=1)))
    expected=5*horizontal+5*vertical-horizontal*vertical
    observed=float(ink[10:15,10:15].sum())
    return observed-expected>=max(3,expected*.25)


def three_arm_evidence(mask,x,y):
    h,w=mask.shape; x=round(x); y=round(y)
    if x<12 or y<12 or x+12>=w or y+12>=h: return False
    patch=(mask[y-12:y+13,x-12:x+13]>0).astype(np.uint8)
    _,labels=cv2.connectedComponents(patch,connectivity=8)
    center=set(labels[11:14,11:14].flatten())-{0}
    regions=(labels[10:15,:5],labels[10:15,-5:],labels[:5,10:15],labels[-5:,10:15])
    return any(sum(bool(np.any(region==component)) for region in regions)==3 for component in center)


class TopologyPerception:
    def __init__(self,model=None,threads=2):
        self.session=None; self.metadata={}
        if model:
            import onnxruntime as ort
            path=Path(model); self.metadata=json.loads(path.with_suffix('.json').read_text(encoding='utf-8'))
            options=ort.SessionOptions(); options.intra_op_num_threads=threads; options.inter_op_num_threads=threads
            self.session=ort.InferenceSession(str(path),sess_options=options,providers=['CPUExecutionProvider'])

    def perceive(self,image,component_boxes,texts):
        from .vision import wire_lines
        gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        mask=(gray<220).astype(np.uint8)*255
        for box in component_boxes+[t['bbox'] for t in texts]:
            a,b,c,d=map(int,box); mask[max(0,b):min(mask.shape[0],d),max(0,a):min(mask.shape[1],c)]=0
        segments,_=wire_lines(mask,split_junctions=False)
        candidates=crossing_candidates(segments); junctions=[]
        for item in candidates:
            x,y=item['x'],item['y']
            dot=dot_evidence(mask,x,y); three_arm=item['endpoint_contact'] and three_arm_evidence(mask,x,y)
            if self.session:
                logits=self.session.run(None,{'patch':patch_tensor(image,x,y)[None],'geometry':np.asarray(item['geometry'],np.float32)[None]})[0]
                probability=float(1/(1+np.exp(-np.clip(logits.flatten()[0],-30,30))))
                low=self.metadata.get('disconnect_threshold',.1); high=self.metadata.get('connect_threshold',.9)
                state='connected' if probability>=high else 'disconnected' if probability<=low else 'uncertain'
                if state=='uncertain' and ((three_arm and probability>=.2) or (dot and probability>=.5)):
                    state='connected'
                item.update(probability=probability,evidence='learned_topology_perception')
            else:
                state='connected' if three_arm or dot else 'uncertain'
                item.update(probability=None,evidence='three_arm_and_stroke_normalized_dot')
            item.update(dot=dot,three_arm=three_arm)
            item['state']=state
            if state=='connected': junctions.append((x,y))
        # Split only confirmed junctions. Uncertain interior crossings remain
        # separate wires and go to the review queue.
        split_segments=[]
        for a,b in segments:
            length=(b[0]-a[0])**2+(b[1]-a[1])**2; ts={0.,1.}
            for x,y in junctions:
                if not length: continue
                t=((x-a[0])*(b[0]-a[0])+(y-a[1])*(b[1]-a[1]))/length
                q=(a[0]+t*(b[0]-a[0]),a[1]+t*(b[1]-a[1]))
                if 0<t<1 and math.dist((x,y),q)<=2: ts.add(t)
            ordered=sorted(ts)
            for s,t in zip(ordered,ordered[1:]):
                p=(a[0]+s*(b[0]-a[0]),a[1]+s*(b[1]-a[1])); q=(a[0]+t*(b[0]-a[0]),a[1]+t*(b[1]-a[1]))
                if math.dist(p,q)>=2: split_segments.append((p,q))
        return split_segments,junctions,mask,{'module':'TopologyPerception','version':2,'color_policy':'RGB feature only; no color-to-connection rule','decision_policy':'neural confidence plus connected three-arm/stroke-normalized-dot topology; geometric near-touch alone cannot commit a junction','learned':bool(self.session),'crossings':candidates}
