"""Replay cached image observations; golden answers are used only for scoring."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from pcbparse.__main__ import load,save
from pcbparse.pin_consensus import fuse_observations,electrical_signature,replay_electrical_associations
from pcbparse.graphschematic import parse_graph
from pcbparse.core import evaluate
from pcbparse.competition import evaluate_qa

p=argparse.ArgumentParser(); p.add_argument('--official',action='store_true'); a=p.parse_args()
baseline=ROOT/'output'/('benchmark-official-baseline' if a.official else 'benchmark-graph-topology-v2-55')
auxiliary=ROOT/'output'/('benchmark-official-new-pins-controlled' if a.official else 'benchmark-graph-official-pins-55')
out=ROOT/'output'/('benchmark-official-pin-consensus' if a.official else 'benchmark-pin-consensus-55'); rows=[]
for case in sorted(p for p in baseline.iterdir() if p.is_dir() and (p/'observations.json').exists()):
    base,aux=load(case/'observations.json'),load(auxiliary/case.name/'observations.json')
    fused,report=fuse_observations(base,aux)
    # Conservative replay has no new geometric association proposals.
    prediction,graph=parse_graph(fused,optimize=False)
    # Reproduce already-validated baseline associations; omitting them can
    # wrongly roll back a safe fusion merely because this replay has no pixels.
    baseline_graph=case/'graph.json'
    if not baseline_graph.exists(): baseline_graph=case/'dual-graph.json'
    if baseline_graph.exists():
        baseline_report=load(baseline_graph)
        prediction,_=replay_electrical_associations(fused,baseline_report)
        report['baseline_associations_replayed']=len(baseline_report['association']['accepted'])
    original=load(case/'prediction.json')
    if electrical_signature(prediction)!=electrical_signature(original):
        prediction=original; report['final_gate']='rollback_final_net_or_wire_change'; report['accepted']=[]
    else: report['final_gate']='electrical_signature_preserved'
    if a.official:
        golden=load(ROOT/'data/raw/official/gt-0930/10GT'/case.name/(case.name+'.json')); metric=evaluate_qa(prediction,golden)
    else: golden=load(case/'golden.json'); metric=evaluate(prediction,golden)
    save(out/case.name/'prediction.json',prediction); save(out/case.name/'consensus.json',report)
    rows.append({'case':case.name,'accepted':len(report['accepted']),**metric})
keys=['ComponentF1','PinF1','NetHypergraphF1','NetLineF1','Score_Accuracy_SingleCase_proxy']
summary={'completed':len(rows),'official_equivalence':False,'mean_metrics':{k:sum(r[k] for r in rows)/len(rows) for k in keys},'accepted_pin_updates':sum(r['accepted'] for r in rows),'records':rows,'evaluation':'cached image-only observations; labels loaded after fusion; not independent final test'}
save(out/'summary.json',summary); print(json.dumps({k:v for k,v in summary.items() if k!='records'},indent=2))
