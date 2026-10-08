"""Partial-supervision dense fields: real PCB vectors, CGHD ink, own synthetic.

All tiles from one page/drafter or procedural layout family share a split.
External binary ink masks never become wire or electrical-junction labels.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
import cv2
import numpy as np
from pcbparse.dataset import index_dataset
from pcbparse.vision import read_image

SIZE=192
def heat(target,channel,point):
    x,y=map(round,point); h,w=target.shape[1:]
    if not 0<=x<w or not 0<=y<h: return
    a,b=max(0,x-5),max(0,y-5); c,d=min(w,x+6),min(h,y+6); yy,xx=np.mgrid[b:d,a:c]
    value=np.rint(255*np.exp(-((xx-x)**2+(yy-y)**2)/(2*1.5**2))).astype(np.uint8)
    target[channel,b:d,a:c]=np.maximum(target[channel,b:d,a:c],value)

def samples(image,target,centers,count,rng):
    h,w=image.shape[:2]
    for i in range(count):
        cx,cy=centers[i%len(centers)] if centers and i<count-1 else (rng.uniform(0,w),rng.uniform(0,h))
        x=max(0,min(max(0,w-SIZE),round(cx-SIZE/2+rng.uniform(-35,35))))
        y=max(0,min(max(0,h-SIZE),round(cy-SIZE/2+rng.uniform(-35,35))))
        crop=image[y:y+SIZE,x:x+SIZE]; label=target[:,y:y+SIZE,x:x+SIZE]
        pad=(max(0,SIZE-crop.shape[0]),max(0,SIZE-crop.shape[1]))
        crop=cv2.copyMakeBorder(crop,0,pad[0],0,pad[1],cv2.BORDER_CONSTANT,value=(255,255,255))
        label=np.pad(label,((0,0),(0,pad[0]),(0,pad[1])))
        yield crop,label,[x,y,x+SIZE,y+SIZE]

def prepare(source,out):
    rng=np.random.default_rng(20261005); images=[]; targets=[]; records=[]; seen=set(); skipped=Counter()
    def append(image,target,valid,source,page,split,centers,count=6):
        centers=list(centers); rng.shuffle(centers)
        for crop,label,box in samples(image,target,centers,count,rng):
            images.append(crop.transpose(2,0,1)); targets.append(label)
            records.append({'source':source,'page':page,'split':split,'valid_heads':valid,'box':box})
    if source=='pcbnet':
        for item in index_dataset(ROOT/'data/raw/PCBnet/Dataset300'):
            if item['id']=='0237' or item['sha256'] in seen: continue
            seen.add(item['sha256']); raw=json.loads(Path(item['annotation']).read_text(encoding='utf-8')); image=read_image(item['image']); h,w=image.shape[:2]
            if (raw['canvas']['height'],raw['canvas']['width'])!=(h,w): skipped['canvas_mismatch']+=1; continue
            target=np.zeros((5,h,w),np.uint8); centers=[]
            target[4]=(cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)<220).astype(np.uint8)*255
            for comp in raw['components'].values():
                if comp.get('state') not in ('active',None): continue
                for p in comp.get('pins',{}).values():
                    tip=p.get('tip')
                    if isinstance(tip,list) and len(tip)==2: heat(target,2,tip); centers.append(tip)
            for edge in raw['edges'].values():
                if edge.get('state') not in ('active',None): continue
                for point in edge.get('keyPoints',{}).get('divergePoints',{}).get('wire',[]): heat(target,3,point)
                for path in edge.get('keyPointPaths',{}).get('wire',[]):
                    for a,b in zip(path,path[1:]):
                        if a==b: continue
                        dx,dy=abs(a[0]-b[0]),abs(a[1]-b[1]); channel=0 if dx>=dy else 1
                        cv2.line(target[channel],tuple(map(round,a)),tuple(map(round,b)),255,3)
            append(image,target,[1]*5,source,item['id'],item['split'],centers)
    elif source=='cghd':
        base=ROOT/'data/raw/CGHD-v16-subset'; rows=json.loads((base/'records.json').read_text(encoding='utf-8'))
        for r in rows:
            image=read_image(base/r['image']); mask=read_image(base/r['segmentation']); h,w=image.shape[:2]
            if mask.shape[:2]!=(h,w): skipped['segmentation_shape_mismatch']+=1; continue
            # Reduce camera-resolution images for meaningful local context. This
            # is not used to rescale PCB inference images or evaluation labels.
            scale=min(1,1200/max(h,w)); image=cv2.resize(image,(round(w*scale),round(h*scale)),interpolation=cv2.INTER_AREA)
            ink=cv2.resize(cv2.cvtColor(mask,cv2.COLOR_BGR2GRAY),(image.shape[1],image.shape[0]),interpolation=cv2.INTER_NEAREST)<128
            target=np.zeros((5,*image.shape[:2]),np.uint8); target[4]=ink.astype(np.uint8)*255
            ys,xs=np.where(ink); centers=list(zip(xs[::max(1,len(xs)//100)],ys[::max(1,len(ys)//100)]))
            drafter=int(r['drafter'].split('_')[-1]); split='validation' if drafter%5==0 else 'train'
            append(image,target,[0,0,0,0,1],source,r['sample'],split,centers,count=4)
    else:
        # Original procedural drawings with exact semantic fields. A family has
        # one split; style variants cannot leak into the other split.
        for family in range(128):
            for style in range(2):
                image=np.full((384,384,3),255,np.uint8); target=np.zeros((5,384,384),np.uint8); centers=[]
                color=(30,110,30) if style else (20,20,20); thickness=1+family%3
                def wire(a,b):
                    cv2.line(image,a,b,color,thickness); ch=0 if abs(a[0]-b[0])>=abs(a[1]-b[1]) else 1; cv2.line(target[ch],a,b,255,3)
                for row in range(3):
                    y=60+100*row+int(rng.integers(-10,11)); left=40+int(rng.integers(-8,9)); right=340+int(rng.integers(-8,9)); center=190+int(rng.integers(-20,21))
                    wire((left,y),(center-25,y)); wire((center+25,y),(right,y)); heat(target,2,(center-25,y)); heat(target,2,(center+25,y)); centers.extend([(center-25,y),(center+25,y)])
                    if family%2: cv2.rectangle(image,(center-25,y-8),(center+25,y+8),(30,30,180) if style else (30,30,30),1)
                    else:
                        p=np.asarray([(center-25,y),(center-18,y-9),(center-9,y+9),(center,y-9),(center+9,y+9),(center+18,y-9),(center+25,y)],np.int32); cv2.polylines(image,[p],False,(30,30,180) if style else (30,30,30),1)
                    cv2.putText(image,'R'+str(row+1),(center-15,y-18),cv2.FONT_HERSHEY_SIMPLEX,.45,(110,35,25) if style else (25,25,25),1)
                    wire((left,y),(left,35 if row==0 else y-60))
                    if row:
                        wire((right,y),(right,y-100)); cv2.circle(image,(right,y),3,color,-1); heat(target,3,(right,y))
                # A nonconnected interior crossover with both orientation fields;
                # never mark it as an electrical junction.
                wire((70,335),(330,335)); wire((110,300),(110,365))
                target[4]=(cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)<220).astype(np.uint8)*255
                append(image,target,[1]*5,source,str(family),'validation' if family%5==0 else 'train',centers,count=2)
    out.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(out/(source+'.npz'),images=np.stack(images),targets=np.stack(targets))
    (out/(source+'-records.json')).write_text(json.dumps(records),encoding='utf-8')
    summary={'source':source,'tiles':len(records),'pages':{s:len({r['page'] for r in records if r['split']==s}) for s in ('train','validation')},'split_tiles':dict(Counter(r['split'] for r in records)),'size':SIZE,'skipped':dict(skipped),'supervision':'PCB vectors/annotated pin tips/weak junctions; CGHD ink-only; original synthetic full fields','gt10_used_for_training':False}
    (out/(source+'-preparation.json')).write_text(json.dumps(summary,indent=2),encoding='utf-8'); print(json.dumps(summary),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('source',choices=['pcbnet','cghd','synthetic']); p.add_argument('--output',default='data/processed/fields'); a=p.parse_args(); prepare(a.source,Path(a.output))
