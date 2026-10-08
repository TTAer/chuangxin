"""Finite isolated fixtures, separate from training/development; do not calibrate from them."""
import sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,cv2
from experiments.stroke_residual import residual_scores

dest=ROOT/'output/stroke-residual-fixtures.json'
if dest.exists():raise ValueError('Preserve fixtures')
rows=[]
for width in (1,3,7):
    for sigma in (0,.5,1,2):
        im=np.zeros((33,33),np.float32);cv2.line(im,(0,16),(32,16),1,width);cv2.line(im,(16,0),(16,32),1,width);dot=im.copy();cv2.circle(dot,(16,16),max(6,width+3),1,-1)
        if sigma:im=cv2.GaussianBlur(im,(7,7),sigma);dot=cv2.GaussianBlur(dot,(7,7),sigma)
        a,b=residual_scores(np.stack([im,dot])[:,None]);rows.append({'width':width,'sigma':sigma,'CROSS':float(a),'JOIN':float(b),'ordered':bool(b>a+1e-3)})
dest.write_text(json.dumps({'scope':'isolated finite wire/dot fixtures; no page-context guarantee','records':rows,'ordered_count':sum(d['ordered'] for d in rows)},indent=2),encoding='utf-8');print(json.dumps({'ordered':sum(d['ordered'] for d in rows),'total':len(rows)}))
