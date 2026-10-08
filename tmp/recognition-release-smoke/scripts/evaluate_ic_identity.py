"""Identity recovery ablation on fixed development observations, GT loaded later."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from pcbparse.__main__ import load,save
from pcbparse.ic_identity import ICIdentityResolver
from pcbparse.semantic_library import SemanticLibrary
from pcbparse.side_strips import SideStripLocator,refine_observations
from pcbparse.dataset import index_dataset
from pcbparse.vision import read_image
from pcbparse.competition import evaluate_qa,export_competition
from pcbparse.core import evaluate,validate
from pcbparse.graphschematic import parse_graph
from pcbparse.component_boxes import apply_body_boxes
from evaluate_ic_structure import mask_image

def main():
    p=argparse.ArgumentParser(); p.add_argument('--dataset',choices=['gt10','pcb55'],required=True); p.add_argument('--side',action='store_true'); a=p.parse_args()
    lib=SemanticLibrary(); solver=ICIdentityResolver(lib); locator=SideStripLocator(ROOT/'models/candidates/side-strip-local/side-strip.onnx') if a.side else None
    base=ROOT/('output/benchmark-recognition-'+a.dataset); out=ROOT/('output/ic-identity-'+a.dataset+('-side' if a.side else ''))
    samples={s['id']:s for s in index_dataset(ROOT/'data/raw/PCBnet/Dataset300')}; rows=[]; records=[]
    for folder in sorted(base.iterdir()):
        if not (folder/'observations.json').exists(): continue
        obs=load(folder/'observations.json'); before=load(folder/'prediction.json'); refined,report=solver.refine(obs)
        path=ROOT/'data/raw/official/gt-0930/10GT'/folder.name/(folder.name+'.png') if a.dataset=='gt10' else Path(samples[folder.name]['image'])
        image=read_image(path); side=None
        if locator: refined,side=refine_observations(refined,image,locator,lib,name_anchors=True)
        if refined!=obs:
            pred,_=parse_graph(refined,line_mask=mask_image(image,obs)); pred=apply_body_boxes(pred,refined)
            if a.dataset=='gt10': pred=export_competition(pred,obs.get('component_categories'))
        else: pred=before
        assert not validate(pred)
        save(out/folder.name/'prediction.json',pred); save(out/folder.name/'report.json',report); save(out/folder.name/'observations.json',refined)
        records.extend(dict(r,case=folder.name) for r in report['components'])
        truth=load(path.with_suffix('.json') if a.dataset=='gt10' else folder/'golden.json'); evaluator=evaluate_qa if a.dataset=='gt10' else evaluate
        rows.append({'case':folder.name,'metrics':evaluator(pred,truth),'baseline':evaluator(before,truth),'identity_edits':report['edits']})
    keys=['ComponentF1','PinF1','NetHypergraphF1','NetLineF1','Score_Accuracy_SingleCase_proxy']; score=keys[-1]
    summary={'dataset':a.dataset,'side_enabled':a.side,'completed':len(rows),'failures':0,'mean_metrics':{k:sum(r['metrics'][k] for r in rows)/len(rows) for k in keys},
        'baseline':{k:sum(r['baseline'][k] for r in rows)/len(rows) for k in keys},'improved':sum(r['metrics'][score]>r['baseline'][score]+1e-9 for r in rows),
        'regressed':sum(r['metrics'][score]<r['baseline'][score]-1e-9 for r in rows),'identity_records':records,'records':rows,'official_equivalence':False,'default_promoted':False}
    save(out/'summary.json',summary); print(json.dumps({k:v for k,v in summary.items() if k not in ('records','identity_records')}),flush=True)
if __name__=='__main__': main()
