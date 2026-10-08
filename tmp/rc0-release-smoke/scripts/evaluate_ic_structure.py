"""Evaluate optional IC structure on fixed current-default observations."""
import argparse
import copy
import json
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from pcbparse.__main__ import load,save
from pcbparse.core import evaluate
from pcbparse.competition import evaluate_qa,export_competition
from pcbparse.component_boxes import apply_body_boxes
from pcbparse.graphschematic import parse_graph
from pcbparse.ic_structure import StructureAwareICParser
from pcbparse.semantic_library import SemanticLibrary
from pcbparse.dataset import index_dataset
from pcbparse.vision import read_image
import cv2
import numpy as np

def mask_image(image,obs):
    h,w=image.shape[:2]; mask=(cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)<220).astype(np.uint8)*255
    for box in obs.get('masked_symbol_boxes',[])+[t['bbox'] for t in obs.get('texts',[])]:
        x1,y1,x2,y2=box; a,b,c,d=map(int,(x1,h-y2,x2,h-y1))
        a=max(0,a); b=max(0,b); c=min(w,c); d=min(h,d)
        if a<c and b<d: mask[b:d,a:c]=0
    return mask

def main():
    p=argparse.ArgumentParser(); p.add_argument('--dataset',choices=['pcb55','gt10'],required=True); p.add_argument('--names-only',action='store_true'); p.add_argument('--geometry',action='store_true'); a=p.parse_args()
    official=a.dataset=='gt10'; baseline=ROOT/'output'/('benchmark-recognition-gt10' if official else 'benchmark-recognition-pcb55')
    out=ROOT/'output'/('ic-structure-'+a.dataset+('-geometry' if a.geometry else '-names' if a.names_only else '')); parser=StructureAwareICParser(SemanticLibrary()); rows=[]; tic=time.perf_counter()
    samples={s['id']:s for s in index_dataset(ROOT/'data/raw/PCBnet/Dataset300')}
    for folder in sorted(baseline.iterdir()):
        if not (folder/'observations.json').exists(): continue
        obs=load(folder/'observations.json'); old=load(folder/'prediction.json'); refined,report=parser.refine(obs,resolve_numbers=not a.names_only)
        if official: target=ROOT/'data/raw/official/gt-0930/10GT'/folder.name; image_path=target/(folder.name+'.png'); truth_path=target/(folder.name+'.json'); evaluator=evaluate_qa
        else: image_path=samples[folder.name]['image']; truth_path=folder/'golden.json'; evaluator=evaluate
        if a.geometry: refined,report=parser.refine_geometry(obs,read_image(image_path))
        if report['edits']:
            prediction,_=parse_graph(refined,line_mask=mask_image(read_image(image_path),obs)); prediction=apply_body_boxes(prediction,refined)
            if official: prediction=export_competition(prediction,obs.get('component_categories'))
        else: prediction=copy.deepcopy(old)
        save(out/folder.name/'prediction.json',prediction); save(out/folder.name/'observations.json',refined); save(out/folder.name/'report.json',report)
        truth=load(truth_path); metrics=evaluator(prediction,truth); before=evaluator(old,truth)
        rows.append({'case':folder.name,'edits':report['edits'],'metrics':metrics,'baseline':before})
    keys=['ComponentF1','PinF1','NetHypergraphF1','NetLineF1','Score_Accuracy_SingleCase_proxy']; score=keys[-1]
    summary={'completed':len(rows),'failures':0,'seconds':time.perf_counter()-tic,'dataset':a.dataset,'names_only':a.names_only,'geometry':a.geometry,
             'mean_metrics':{k:sum(r['metrics'][k] for r in rows)/len(rows) for k in keys},
             'baseline':{k:sum(r['baseline'][k] for r in rows)/len(rows) for k in keys},
             'improved':sum(r['metrics'][score]>r['baseline'][score]+1e-9 for r in rows),
             'regressed':sum(r['metrics'][score]<r['baseline'][score]-1e-9 for r in rows),
             'records':rows,'official_equivalence':False,'training':False,'policy':'cached current detector/OCR/observed endpoints; golden loaded only after candidate inference'}
    save(out/'summary.json',summary); print(json.dumps({k:v for k,v in summary.items() if k!='records'}),flush=True)
if __name__=='__main__': main()
