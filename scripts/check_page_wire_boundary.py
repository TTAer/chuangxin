"""Construct a mixed-linewidth counterexample outside shared-profile assumptions."""
import sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,cv2
from PIL import Image
from scripts.ordered_line_rectification_v2 import register_ordered
from experiments.page_wire_residual import page_scores
from experiments.stroke_residual import decisions

out=ROOT/'output/page-wire-boundary';out.mkdir(parents=True,exist_ok=True)
if (out/'report.json').exists():raise ValueError('Preserve boundary')
thresholds=json.loads((ROOT/'output/page-wire-residual-strict-development/report.json').read_text())['thresholds'];rows=[]
for name in ('uniform','mixed'):
    im=np.full((256,256),255,np.uint8)
    for i in range(6):
        w=1 if name=='uniform' or i==0 else 7
        cv2.line(im,(16,36+36*i),(240,36+36*i),0,w);cv2.line(im,(36+36*i,16),(36+36*i,240),0,w)
    cv2.circle(im,(36,36),6,0,-1);Image.fromarray(im).save(out/f'{name}.png')
    aligned,g=register_ordered(im);s,slots=page_scores(im,aligned,g);index=slots.index(2) if 2 in slots else None
    rows.append({'style':name,'geometry_ready':g['ready'],'first_slot_score':float(s[index]) if index is not None else None,
                 'first_slot_rule':int(decisions(s,**thresholds)[index]) if index is not None else None,'expected':9})
report={'scope':'constructed6x6 fixture pair; shared linewidth assumption violated in mixed case, not independent real validation','thresholds':thresholds,'records':rows}
(out/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report))
