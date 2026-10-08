"""Four-way semantic/pin ablation, fixed detector/OCR observations per domain."""
import argparse
import copy
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
import cv2
import numpy as np
from pcbparse.__main__ import load,save
from pcbparse.core import evaluate
from pcbparse.competition import evaluate_qa
from pcbparse.graphschematic import parse_graph
from pcbparse.semantic_library import SemanticLibrary
from pcbparse.dataset import index_dataset
from pcbparse.vision import read_image

def mask_image(image,obs):
    h=image.shape[0]; mask=(cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)<220).astype(np.uint8)*255
    for box in obs.get('masked_symbol_boxes',[c['bbox'] for c in obs['components'].values()])+[t['bbox'] for t in obs.get('texts',[])]:
        x1,y1,x2,y2=box; a,b,c,d=map(int,(x1,h-y2,x2,h-y1)); mask[max(0,b):min(h,d),max(0,a):min(mask.shape[1],c)]=0
    return mask

p=argparse.ArgumentParser(); p.add_argument('--dataset',choices=['pcb55','gt10'],required=True); a=p.parse_args()
official=a.dataset=='gt10'; base=ROOT/'output'/('benchmark-official-pin-consensus-live' if official else 'benchmark-graph-topology-v2-55')
aux=ROOT/'output'/('benchmark-official-new-pins-controlled' if official else 'benchmark-graph-official-pins-55')
out=ROOT/'output'/('semantic-grid-'+a.dataset); lib=SemanticLibrary(); rows=[]
samples={s['id']:s for s in index_dataset(ROOT/'data/raw/PCBnet/Dataset300')}
for folder in sorted(base.iterdir()):
    if not (folder/'observations.json').exists(): continue
    default_obs=load(folder/'observations.json'); aux_obs=load(aux/folder.name/'observations.json')
    default_pred=load(folder/'prediction.json') if official else load(ROOT/'output/benchmark-pin-consensus-55'/folder.name/'prediction.json')
    default_obs['pins']=copy.deepcopy(default_pred['pins'])
    aux_pred=load(aux/folder.name/'prediction.json')
    if official:
        target=ROOT/'data/raw/official/gt-0930/10GT'/folder.name; image=read_image(target/(folder.name+'.png')); truth_path=target/(folder.name+'.json'); evaluator=evaluate_qa
    else: image=read_image(samples[folder.name]['image']); truth_path=folder/'golden.json'; evaluator=evaluate
    predictions={'default':default_pred,'new_pin':aux_pred}; reports={}
    for name,obs,unchanged in [('semantic',default_obs,default_pred),('new_pin_semantic',aux_obs,aux_pred)]:
        refined,report=lib.refine(obs); reports[name]=report
        if refined==obs: prediction=unchanged
        else: prediction,_=parse_graph(refined,line_mask=mask_image(image,obs))
        predictions[name]=prediction
        save(out/folder.name/(name+'-prediction.json'),prediction); save(out/folder.name/(name+'-report.json'),report)
    truth=load(truth_path)
    metrics={name:evaluator(pred,truth) for name,pred in predictions.items()}
    rows.append({'case':folder.name,'metrics':metrics,'semantic_edits':{name:report['edits'] for name,report in reports.items()}})
keys=['ComponentF1','PinF1','NetHypergraphF1','NetLineF1','Score_Accuracy_SingleCase_proxy']; names=['default','semantic','new_pin','new_pin_semantic']
summary={'dataset':a.dataset,'completed':len(rows),'mean_metrics':{name:{k:sum(r['metrics'][name][k] for r in rows)/len(rows) for k in keys} for name in names},'records':rows,'official_equivalence':False,'evaluation':'cached detector/OCR and original-image support; labels used after inference; unchanged semantic observations preserve baseline associations','evaluator':'QA-informed proxy' if official else 'legacy strict proxy; location boxes; not numerically comparable to GT10'}
save(out/'summary.json',summary); print(json.dumps({k:v for k,v in summary.items() if k!='records'}),flush=True)
