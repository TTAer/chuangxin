"""Actual side proposal inference against fixed current detector/OCR/geometry."""
import argparse
import json
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from pcbparse.__main__ import load,save
from pcbparse.side_strips import SideStripLocator,refine_observations
from pcbparse.semantic_library import SemanticLibrary
from pcbparse.dataset import index_dataset
from pcbparse.vision import read_image
from pcbparse.core import evaluate,validate
from pcbparse.competition import evaluate_qa,export_competition
from pcbparse.component_boxes import apply_body_boxes
from pcbparse.graphschematic import parse_graph
from evaluate_ic_structure import mask_image

def main():
    p=argparse.ArgumentParser(); p.add_argument('--dataset',choices=['gt10','pcb55'],required=True); p.add_argument('--model',required=True); p.add_argument('--without-structure',action='store_true'); p.add_argument('--threshold',type=float); p.add_argument('--output'); p.add_argument('--name-anchors',action='store_true'); a=p.parse_args()
    official=a.dataset=='gt10'; base=ROOT/'output'/('benchmark-recognition-gt10' if official else 'benchmark-recognition-pcb55'); out=ROOT/'output'/('side-strip-'+Path(a.model).parent.name+'-'+a.dataset+('-raw' if a.without_structure else ''))
    locator=SideStripLocator(a.model); lib=SemanticLibrary(); samples={s['id']:s for s in index_dataset(ROOT/'data/raw/PCBnet/Dataset300')}; rows=[]; started=time.perf_counter()
    if a.output: out=ROOT/a.output
    if a.threshold is not None: locator.meta['threshold']=a.threshold
    funnels=[]
    for folder in sorted(base.iterdir()):
        if not (folder/'observations.json').exists(): continue
        tic=time.perf_counter(); obs=load(folder/'observations.json'); before=load(folder/'prediction.json')
        if official: target=ROOT/'data/raw/official/gt-0930/10GT'/folder.name; image_path=target/(folder.name+'.png'); truth_path=target/(folder.name+'.json'); evaluator=evaluate_qa
        else: image_path=samples[folder.name]['image']; truth_path=folder/'golden.json'; evaluator=evaluate
        image=read_image(image_path); refined,report=refine_observations(obs,image,locator,lib,structure=not a.without_structure,name_anchors=a.name_anchors)
        funnels.extend(dict(r,case=folder.name) for r in report['funnel'])
        if refined!=obs:
            pred,_=parse_graph(refined,line_mask=mask_image(image,obs)); pred=apply_body_boxes(pred,refined)
            if official: pred=export_competition(pred,obs.get('component_categories'))
        else: pred=before
        assert not validate(pred)
        save(out/folder.name/'prediction.json',pred); save(out/folder.name/'observations.json',refined); save(out/folder.name/'report.json',report)
        truth=load(truth_path); metrics=evaluator(pred,truth); baseline=evaluator(before,truth)
        rows.append({'case':folder.name,'seconds':time.perf_counter()-tic,'metrics':metrics,'baseline':baseline})
    keys=['ComponentF1','PinF1','NetHypergraphF1','NetLineF1','Score_Accuracy_SingleCase_proxy']; k=keys[-1]
    summary={'dataset':a.dataset,'model':a.model,'with_structure':not a.without_structure,'completed':len(rows),'failures':0,'seconds':time.perf_counter()-started,
             'mean_metrics':{k:sum(r['metrics'][k] for r in rows)/len(rows) for k in keys},'baseline':{k:sum(r['baseline'][k] for r in rows)/len(rows) for k in keys},
             'improved':sum(r['metrics'][k]>r['baseline'][k]+1e-9 for r in rows),'regressed':sum(r['metrics'][k]<r['baseline'][k]-1e-9 for r in rows),
             'records':rows,'official_equivalence':False,'timing_scope':'cached detector/OCR/topology; actual learned side proposals and graph reconstruction; not full pipeline latency'}
    summary['threshold']=locator.meta['threshold']; summary['name_anchors']=a.name_anchors; summary['funnel']=funnels
    summary['funnel_totals']={k:sum(r[k] for r in funnels) for k in ('proposals','numbered_candidates','existing_pins','accepted_add','accepted_refine')}
    summary['funnel_totals']['ic_components']=len(funnels)
    summary['funnel_totals']['library_identified']=sum(bool(r['library_model']) for r in funnels)
    summary['funnel_totals']['pages_with_accepted_changes']=len({r['case'] for r in funnels if r['accepted_add']+r['accepted_refine']})
    save(out/'summary.json',summary); print(json.dumps({k:v for k,v in summary.items() if k not in ('records','funnel')}),flush=True)
if __name__=='__main__': main()
