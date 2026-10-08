"""CPU export check and held-out tile diagnostics; no end-to-end claim."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
import cv2
import numpy as np
from pcbparse.field_perception import FieldPerception,field_input

def points(mask,threshold):
    local=(mask>=cv2.dilate(mask,np.ones((3,3),np.uint8))) & (mask>=threshold)
    ys,xs=np.where(local); ordered=sorted(zip(xs,ys),key=lambda p:float(mask[p[1],p[0]]),reverse=True)
    kept=[]; suppressed=np.zeros(mask.shape,bool)
    for p in ordered:
        x,y=p
        if suppressed[y,x]: continue
        kept.append(p)
        a,b=max(0,x-5),max(0,y-5); c,d=min(mask.shape[1],x+6),min(mask.shape[0],y+6)
        yy,xx=np.mgrid[b:d,a:c]; suppressed[b:d,a:c]|=(xx-x)**2+(yy-y)**2<=25
    return kept

def main(a):
    model=FieldPerception(a.model); reference=np.load(Path(a.training)/'export-reference.npz')
    actual=model.session.run(None,{'image':reference['image']})[0]
    error=float(np.abs(actual-reference['fields']).max())
    sigmoid=lambda x:1/(1+np.exp(-np.clip(x,-30,30)))
    probability_error=float(np.abs(sigmoid(actual)-sigmoid(reference['fields'])).max())
    # GroupNorm export has small CPU reduction-order differences. Probability
    # drift is the meaningful quantity for the runtime threshold gate.
    if error>1e-3 or probability_error>1e-4: raise ValueError('ONNX export drift')
    counts={}; keypoints={}
    for source in a.sources.split(','):
        cache=ROOT/'data/processed/fields'; data=np.load(cache/(source+'.npz'))
        records=json.loads((cache/(source+'-records.json')).read_text())
        ids=[i for i,r in enumerate(records) if r['split']=='validation']
        for start in range(0,len(ids),8):
            batch=ids[start:start+8]; inputs=np.stack([field_input(data['images'][i].transpose(1,2,0)) for i in batch])
            logits=model.session.run(None,{'image':inputs})[0]; probs=1/(1+np.exp(-np.clip(logits,-30,30)))
            for i,p,x in zip(batch,probs,inputs):
                truth=data['targets'][i].astype(np.float32)/255
                for ch in (0,1,4):
                    if not records[i]['valid_heads'][ch]: continue
                    methods={'learned':p[ch]>=model.threshold}
                    if ch<2: methods['morphology']=x[ch+1]>=.5
                    for method,pred in methods.items():
                        t=truth[ch]>=.5; key=f'{source}/{ch}/{method}'
                        c=counts.setdefault(key,np.zeros(3,np.int64)); c+=np.asarray([(pred&t).sum(),pred.sum(),t.sum()])
                for ch in (2,3):
                    if not records[i]['valid_heads'][ch]: continue
                    golden=points(truth[ch],.99)
                    for threshold in (.3,.5,.7):
                        predicted=points(p[ch],threshold); edges=sorted(((px-gx)**2+(py-gy)**2,j,k) for j,(px,py) in enumerate(predicted) for k,(gx,gy) in enumerate(golden) if (px-gx)**2+(py-gy)**2<=25)
                        usedp=set(); usedg=set()
                        for _,j,k in edges:
                            if j not in usedp and k not in usedg: usedp.add(j); usedg.add(k)
                        c=keypoints.setdefault(f'{source}/{ch}/{threshold}',np.zeros(3,np.int64)); c+=np.asarray([len(usedp),len(predicted),len(golden)])
        print(json.dumps({'validated_source':source,'tiles':len(ids)}),flush=True)
    def metrics(rows):
        return {k:{'f1':float(2*c[0]/max(1,c[1]+c[2])),'precision':float(c[0]/max(1,c[1])),'recall':float(c[0]/max(1,c[2])),'matched':int(c[0]),'predicted':int(c[1]),'golden':int(c[2])} for k,c in rows.items()}
    result={'model':a.model,'onnx_max_absolute_error':error,'onnx_probability_max_absolute_error':probability_error,'export_tolerances':{'logits':.001,'probabilities':.0001},'pixel_metrics':metrics(counts),'keypoint_5px_metrics':metrics(keypoints),'limitations':'Validation tiles select checkpoints/thresholds; keypoint heads are diagnostic only and do not assign pin numbers; not image-level electrical accuracy.'}
    out=Path(a.output); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({'export_error':error,'output':a.output}),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--model',required=True); p.add_argument('--training',required=True); p.add_argument('--sources',default='pcbnet,synthetic'); p.add_argument('--output',required=True); main(p.parse_args())
