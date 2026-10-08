"""Appearance-color ablation on held-out junction patches with fixed geometry."""
import json
from pathlib import Path
import sys
import cv2
import numpy as np
import onnxruntime as ort
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
with np.load(ROOT/'data/processed/topology/patches.npz') as archive: data={k:archive[k] for k in archive.files}
records=json.loads((ROOT/'data/processed/topology/records.json').read_text())
ids=[i for i,r in enumerate(records) if r['split']=='validation']
options=ort.SessionOptions(); options.intra_op_num_threads=2
session=ort.InferenceSession(str(ROOT/'models/topology/topology-perception.onnx'),sess_options=options,providers=['CPUExecutionProvider'])
report={'samples':len(ids),'scope':'junction crop color ablation; fixed geometric features; validation not independent test','official_equivalence':False,'modes':{}}
outputs={}
for mode in ('rgb','grayscale','geometry_neutral'):
    probabilities=[]
    for i in ids:
        rgb=data['images'][i].transpose(1,2,0).astype(np.float32)/255; gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
        if mode=='grayscale': rgb=np.repeat(gray[:,:,None],3,axis=2)
        ink=(gray<.86).astype(np.float32); h=cv2.morphologyEx(ink,cv2.MORPH_OPEN,np.ones((1,7),np.uint8)); v=cv2.morphologyEx(ink,cv2.MORPH_OPEN,np.ones((7,1),np.uint8))
        x=np.concatenate((rgb.transpose(2,0,1),gray[None],h[None],v[None])).astype(np.float32)
        geometry=data['geometry'][i].copy()
        if mode=='geometry_neutral': geometry[:]=.5
        logit=session.run(None,{'patch':x[None],'geometry':geometry[None]})[0].flatten()[0]
        probabilities.append(float(1/(1+np.exp(-np.clip(logit,-30,30)))))
    probabilities=np.asarray(probabilities); predicted=probabilities>=.5; golden=data['labels'][ids]>0
    tp=int((predicted&golden).sum()); fp=int((predicted&~golden).sum()); fn=int((~predicted&golden).sum())
    report['modes'][mode]={'f1':2*tp/max(1,2*tp+fp+fn),'precision':tp/max(1,tp+fp),'recall':tp/max(1,tp+fn),'tp':tp,'fp':fp,'fn':fn}; outputs[mode]=predicted
report['rgb_vs_grayscale_decision_changes']=int((outputs['rgb']!=outputs['grayscale']).sum())
target=ROOT/'output/ablation/topology-color.json'; target.parent.mkdir(parents=True,exist_ok=True); target.write_text(json.dumps(report,indent=2),encoding='utf-8'); print(json.dumps(report,indent=2))
