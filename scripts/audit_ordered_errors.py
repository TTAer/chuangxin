"""Posthoc failure inventory only: do not alter geometry, weights or thresholds."""
import sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from PIL import Image,ImageDraw
from scripts.prepare_circuit_program import execute


def main():
    out=ROOT/'output/ordered-rectification-holdout'
    if (out/'irregular-error-inventory.json').exists():raise ValueError('Preserve audit')
    with np.load(out/'irregular-inputs.npz') as z:gold=z['program'];raw=z['observed_images']
    with np.load(out/'irregular-predictions.npz') as z:pred=z['ordered']
    details=json.loads((out/'irregular-geometry.json').read_text())['ordered'];rows=[]
    for i,(g,p,d) in enumerate(zip(gold,pred,details)):
        slots=np.flatnonzero(g!=p)
        if not len(slots):continue
        a,b=execute(g),execute(p)
        rows.append({'case_index':i,'ready':d['ready'],'count_prediction_error':bool(np.any(g[:2]!=p[:2])),
                     'crossing_error':bool(np.any(g[2:38]!=p[2:38])), 'interval_error':bool(np.any(g[38:]!=p[38:])),
                     'exact_net':bool(np.array_equal(a[1],b[1]) and np.array_equal(a[0],b[0])),
                     'wrong_slots':slots.tolist(),'gold':g[slots].tolist(),'predicted':p[slots].tolist()})
    report={'scope':'posthoc only; frozen models and algorithms not retuned','program_error_pages':len(rows),
            'ready_error_pages':sum(d['ready'] for d in rows),'rejected_error_pages':sum(not d['ready'] for d in rows),'cases':rows}
    (out/'irregular-error-inventory.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    canvas=Image.new('RGB',(1024,280*((len(rows)+3)//4)),'white');draw=ImageDraw.Draw(canvas)
    for j,d in enumerate(rows):
        x=(j%4)*256;y=(j//4)*280
        draw.text((x+3,y+3),f'case {d["case_index"]} ready={d["ready"]}',fill='black')
        canvas.paste(Image.fromarray(raw[d['case_index'],0]).convert('RGB'),(x,y+24))
    canvas.save(out/'irregular-errors.png')
    demo=ROOT/'output/ordered-rectification-demo';demo.mkdir(exist_ok=True)
    for name in ('irregular','shear','rotation'):
        with np.load(out/f'{name}-inputs.npz') as z:Image.fromarray(z['observed_images'][0,0]).save(demo/f'{name}.png')
    print(json.dumps({k:v for k,v in report.items() if k!='cases'}))


if __name__=='__main__':main()
