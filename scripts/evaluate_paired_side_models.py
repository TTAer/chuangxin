"""Evaluate both models on each identical held-out crop distribution."""
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
import numpy as np
from pcbparse.side_strips import SideStripLocator,tensor
rows=[]
for domain in ('oracle','detector'):
    directory=ROOT/f'data/processed/side-strips-{domain}-paired'
    images=np.load(directory/'bands.npz')['gray']; records=json.loads((directory/'records.json').read_text(encoding='utf-8'))
    ids=[i for i,r in enumerate(records) if r['split']=='validation']
    for model in ('oracle','detector'):
        locator=SideStripLocator(ROOT/f'models/candidates/side-strip-{model}-paired/side-strip.onnx')
        matched=predicted=golden=0
        for i in ids:
            y=locator.session.run(None,{locator.input:tensor(images[i])[None]})[0][0]
            probability=1/(1+np.exp(-np.clip(y[0],-30,30))); positions=[]
            for index in np.argsort(probability)[::-1]:
                if probability[index]<.5: break
                if not any(abs(index-j)<4 for j in positions): positions.append(int(index))
            targets=records[i]['targets']; used_p=set(); used_g=set()
            distances=sorted((float(np.hypot(index-position,np.tanh(y[1,index])*32-offset)),a,b) for a,index in enumerate(positions) for b,(position,offset) in enumerate(targets))
            for distance,a,b in distances:
                if distance<=5 and a not in used_p and b not in used_g: used_p.add(a); used_g.add(b)
            matched+=len(used_p); predicted+=len(positions); golden+=len(targets)
        row={'model':model,'crop_domain':domain,'fixed_threshold':.5,'bands':len(ids),'matched':matched,'predicted':predicted,'golden':golden,
             'precision':matched/max(1,predicted),'recall':matched/max(1,golden),'f1':2*matched/max(1,predicted+golden)}
        rows.append(row); print(json.dumps(row),flush=True)
(ROOT/'output/ablation/side-paired-cross-crops.json').write_text(json.dumps({'rows':rows,'official_equivalence':False,'gt10_used':False},indent=2),encoding='utf-8')
