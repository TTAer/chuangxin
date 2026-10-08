"""Image-evidence-only output-box change; cached recognition stays fixed."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from pcbparse.__main__ import load,save
from pcbparse.component_boxes import propose_body_boxes,apply_body_boxes
from pcbparse.core import evaluate
from pcbparse.competition import evaluate_qa
from pcbparse.dataset import index_dataset
from pcbparse.vision import read_image

p=argparse.ArgumentParser(); p.add_argument('--dataset',choices=['pcb55','gt10'],required=True); p.add_argument('--prediction-root'); p.add_argument('--output'); a=p.parse_args()
out=Path(a.output) if a.output else ROOT/'output'/('body-boxes-'+a.dataset); rows=[]
if a.dataset=='pcb55':
    base=ROOT/'output/benchmark-graph-topology-v2-55'; samples={s['id']:s for s in index_dataset(ROOT/'data/raw/PCBnet/Dataset300')}
else: base=ROOT/'output/benchmark-official-pin-consensus-live'
for folder in sorted(base.iterdir()):
    if not (folder/'observations.json').exists(): continue
    obs=load(folder/'observations.json')
    if a.dataset=='pcb55':
        prediction=load(ROOT/'output/benchmark-pin-consensus-55'/folder.name/'prediction.json'); truth=load(folder/'golden.json'); image=samples[folder.name]['image']; evaluator=evaluate
    else:
        prediction=load(folder/'prediction.json'); target=ROOT/'data/raw/official/gt-0930/10GT'/folder.name; truth=load(target/(folder.name+'.json')); image=target/(folder.name+'.png'); evaluator=evaluate_qa
    obs['component_body_boxes'],report=propose_body_boxes(read_image(image),obs)
    candidate=load(Path(a.prediction_root)/folder.name/'new_pin_semantic-prediction.json') if a.prediction_root else prediction
    proposed=apply_body_boxes(candidate,obs)
    before,after=evaluator(prediction,truth),evaluator(proposed,truth)
    save(out/folder.name/'prediction.json',proposed); save(out/folder.name/'box-evidence.json',report)
    rows.append({'case':folder.name,'changed_boxes':report['accepted'],'before':before,'after':after,'delta':after['Score_Accuracy_SingleCase_proxy']-before['Score_Accuracy_SingleCase_proxy']})
keys=['ComponentF1','PinF1','NetHypergraphF1','NetLineF1','Score_Accuracy_SingleCase_proxy']
summary={'dataset':a.dataset,'completed':len(rows),'means':{mode:{k:sum(r[mode][k] for r in rows)/len(rows) for k in keys} for mode in ('before','after')},'improved':sum(r['delta']>1e-9 for r in rows),'regressed':sum(r['delta']< -1e-9 for r in rows),'same':sum(abs(r['delta'])<=1e-9 for r in rows),'records':rows,'official_equivalence':False,'evaluation':'cached predictions plus original-image body evidence; truth loaded only by evaluator'}
save(out/'summary.json',summary); print(json.dumps({k:v for k,v in summary.items() if k!='records'}),flush=True)
