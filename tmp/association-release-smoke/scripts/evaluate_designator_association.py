"""Actual pixel inference on two fixed development sets; never input GT."""
import argparse
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from pcbparse.__main__ import load,save
from pcbparse.vision import ImageParser
from pcbparse.dataset import index_dataset
from pcbparse.competition import export_competition,evaluate_qa
from pcbparse.core import evaluate,validate

def main():
    p=argparse.ArgumentParser(); p.add_argument('--domain',choices=['gt10','pcb55'],required=True); p.add_argument('--mode',choices=['typed','boxed','guarded'],default='typed'); p.add_argument('--cached-ocr',action='store_true'); a=p.parse_args()
    parser=ImageParser(designator_mode=a.mode,ic_identity_mode='retrieval',side_pin_model='models/candidates/side-strip-local/side-strip.onnx',side_name_anchors=True)
    samples={s['id']:s for s in index_dataset(ROOT/'data/raw/PCBnet/Dataset300')}; rows=[]
    root=ROOT/('output/benchmark-recognition-'+a.domain); out=ROOT/('output/designator-'+a.mode)/a.domain
    for folder in sorted(root.iterdir()):
        if not (folder/'observations.json').exists(): continue
        path=ROOT/'data/raw/official/gt-0930/10GT'/folder.name/(folder.name+'.png') if a.domain=='gt10' else Path(samples[folder.name]['image'])
        if a.cached_ocr:
            cached=load(folder/'observations.json'); h=cached['image']['height']
            ts=[dict(t,bbox=[t['bbox'][0],h-t['bbox'][3],t['bbox'][2],h-t['bbox'][1]]) for t in cached['texts']]
            parser.texts=lambda image:ts
        pred,obs,diag,_=parser.parse(path)
        if a.domain=='gt10': pred=export_competition(pred,obs.get('component_categories'))
        assert not validate(pred)
        save(out/folder.name/'prediction.json',pred); save(out/folder.name/'observations.json',obs); save(out/folder.name/'diagnostics.json',diag)
        truth=load(path.with_suffix('.json') if a.domain=='gt10' else folder/'golden.json'); scorer=evaluate_qa if a.domain=='gt10' else evaluate
        baseline=load(ROOT/('output/ic-identity-'+a.domain+'-side')/folder.name/'prediction.json')
        default=load(folder/'prediction.json')
        rows.append({'case':folder.name,'metrics':scorer(pred,truth),'baseline':scorer(baseline,truth),'default':scorer(default,truth),'seconds':diag['seconds'],'prediction_changed':pred!=baseline})
        print(a.domain,folder.name,round(rows[-1]['metrics']['Score_Accuracy_SingleCase_proxy'],4),flush=True)
    keys=['ComponentF1','PinF1','NetHypergraphF1','NetLineF1','Score_Accuracy_SingleCase_proxy']; score=keys[-1]
    summary={'domain':a.domain,'mode':a.mode,'pages':len(rows),'mean':{k:sum(r['metrics'][k] for r in rows)/len(rows) for k in keys},'baseline':{k:sum(r['baseline'][k] for r in rows)/len(rows) for k in keys},'improved':sum(r['metrics'][score]>r['baseline'][score]+1e-9 for r in rows),'regressed':sum(r['metrics'][score]<r['baseline'][score]-1e-9 for r in rows),'records':rows,'actual_pixel_inference':True,'cached_ocr':a.cached_ocr,'official_equivalence':False,'default_promoted':False}
    save(out/'summary.json',summary); print({k:v for k,v in summary.items() if k!='records'},flush=True)
if __name__=='__main__': main()
