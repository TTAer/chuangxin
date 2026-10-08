"""Official page-separated IC bands plus independently generated synthetic layouts."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
import cv2
import numpy as np
from pcbparse.vision import read_image
from pcbparse.side_strips import band,SIDES
rng=np.random.default_rng(20261006); arrays=[]; records=[]
def add(gray,box,points,page,split,source):
    for side in SIDES:
        lo,hi=(box[1],box[3]) if side in ('left','right') else (box[0],box[2])
        for start in range(int(lo)-16,int(hi)+1,224):
            target=[]
            for x,y in points:
                distances=[abs(x-box[0]),abs(x-box[2]),abs(y-box[1]),abs(y-box[3])]
                if SIDES[int(np.argmin(distances))]!=side: continue
                axis=(y if side in ('left','right') else x)-start
                normal={'left':x-box[0],'right':box[2]-x,'top':y-box[1],'bottom':box[3]-y}[side]
                if 0<=axis<256 and abs(normal)<30: target.append([axis,normal])
            arrays.append(band(gray,box,side,start)); records.append({'page':page,'split':split,'source':source,'side':side,'start':start,'box':box,'targets':target})

audit=json.loads((ROOT/'output/official-audit/inventory.json').read_text(encoding='utf-8')); seen=set()
for item in audit['details']['train200']['records']:
    if item['pixels_sha256'] in seen: continue
    seen.add(item['pixels_sha256']); image=read_image(ROOT/item['image']); h,w=image.shape[:2]
    raw=json.loads((ROOT/item['annotation']).read_text(encoding='utf-8'))
    for key,c in raw['components'].items():
        if c['type'] not in ('box','ic'): continue
        b=c['bbox']; points=[(p['point']['x'],h-p['point']['y']) for p in raw['pins'].get(key,{}).values()]
        if not points or min(b[2]-b[0],b[3]-b[1])<20: continue
        box=[min(b[0],min(x for x,y in points)),min(h-b[3],min(y for x,y in points)),max(b[2],max(x for x,y in points)),max(h-b[1],max(y for x,y in points))]
        add(cv2.cvtColor(image,cv2.COLOR_BGR2GRAY),box,points,'official:'+item['case'],'train' if item['training_overlap'] else 'validation','official')

destination=ROOT/'data/raw/SidePinBench'; destination.mkdir(parents=True,exist_ok=True); synthetic=[]
for family in range(240):
    image=np.full((448,448),255,np.uint8); box=[90,60,358,388]; points=[]
    cv2.rectangle(image,(90,60),(358,388),0,int(rng.integers(1,3)))
    pitch=int(rng.integers(12,37)); offset=int(rng.integers(6,15)); sides=rng.choice(SIDES,size=int(rng.integers(1,5)),replace=False)
    for side in sides:
        lo,hi=(box[1],box[3]) if side in ('left','right') else (box[0],box[2])
        for axis in range(lo+offset,hi-12,pitch):
            if rng.random()<.18: continue
            length=int(rng.integers(10,27)); number=str(int(rng.integers(1,150)))
            if side=='left': tip=(90-length,axis); base=(90,axis); label=(92,axis-3)
            elif side=='right': tip=(358+length,axis); base=(358,axis); label=(365,axis-3)
            elif side=='top': tip=(axis,60-length); base=(axis,60); label=(axis+3,50)
            else: tip=(axis,388+length); base=(axis,388); label=(axis+3,408)
            cv2.line(image,tip,base,0,int(rng.integers(1,3))); points.append(tip)
            if rng.random()<.7: cv2.putText(image,number,label,cv2.FONT_HERSHEY_SIMPLEX,float(rng.uniform(.25,.45)),0,1,cv2.LINE_AA)
    for _ in range(5):
        x,y=rng.integers(10,420,size=2); cv2.putText(image,'R'+str(int(rng.integers(1,30))),(int(x),int(y)),cv2.FONT_HERSHEY_SIMPLEX,.35,0,1)
    if family%3==0: image=cv2.GaussianBlur(image,(3,3),.5)
    # Source family split is chosen before augmentation/cropping.
    split='validation' if family%5==0 else 'train'; outer=[min([box[0]]+[x for x,y in points]),min([box[1]]+[y for x,y in points]),max([box[2]]+[x for x,y in points]),max([box[3]]+[y for x,y in points])]
    cv2.imencode('.png',image)[1].tofile(str(destination/(str(family)+'.png')))
    synthetic.append({'family':family,'split':split,'box':outer,'pins':points,'sha256':hashlib.sha256((destination/(str(family)+'.png')).read_bytes()).hexdigest()})
    add(image,outer,points,'synthetic:'+str(family),split,'synthetic')
(destination/'manifest.json').write_text(json.dumps({'generator':'self-generated randomized IC layouts; not real PCB designs','license':'CC0-1.0','seed':20261006,'records':synthetic},indent=2),encoding='utf-8')
out=ROOT/'data/processed/side-strips'; out.mkdir(parents=True,exist_ok=True)
np.savez_compressed(out/'bands.npz',gray=np.stack(arrays)); (out/'records.json').write_text(json.dumps(records),encoding='utf-8')
summary={'bands':len(records),'counts':dict(Counter(r['source']+':'+r['split'] for r in records)),'synthetic_families':240,'gt10_training':False,'frozen_amsnet_used':False,'coordinate_step_source_pixels':1,'target_caveat':'oracle union boxes; real predicted-box end-to-end validation still required'}
(out/'preparation.json').write_text(json.dumps(summary,indent=2),encoding='utf-8'); print(json.dumps(summary),flush=True)
