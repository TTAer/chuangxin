"""Cached visual-candidate replay with neural scoring; truth loaded last."""
import argparse
import copy
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from pcbparse.__main__ import load,save
from pcbparse.graphschematic import DualGraph,circuit_checks
from pcbparse.learned_association import LearnedAssociation
from pcbparse.pin_consensus import fuse_observations,electrical_signature,replay_electrical_associations
from pcbparse.core import evaluate

p=argparse.ArgumentParser(); p.add_argument('--model',default='models/candidates/association-balanced/association.onnx'); p.add_argument('--output',default='output/benchmark-gea-cached-55'); a=p.parse_args()
model=LearnedAssociation(a.model); base=ROOT/'output/benchmark-graph-topology-v2-55'; aux=ROOT/'output/benchmark-graph-official-pins-55'; out=Path(a.output); rows=[]
for case in sorted(c for c in base.iterdir() if (c/'observations.json').exists()):
    obs=load(case/'observations.json'); old=load(case/'graph.json'); graph=DualGraph(obs)
    # Preserve exact image evidence generated at original inference. Models do
    # not see golden pins, net membership, text or per-case evaluation results.
    candidates=old['geometry_graph']['hypotheses']; selected,learned=model.filter(graph,candidates)
    uf,association=graph.associate(selected); before,electrical=graph.export(uf)
    old['association']=association; old['electrical_hypergraph']=electrical; old['learned_association']=learned
    fused,consensus=fuse_observations(obs,load(aux/case.name/'observations.json'))
    prediction,report=replay_electrical_associations(fused,old)
    if electrical_signature(prediction)!=electrical_signature(before): prediction=before; consensus['final_gate']='rollback_final_net_or_wire_change'
    else: consensus['final_gate']='electrical_signature_preserved'
    report['pin_consensus']=consensus
    metric=evaluate(prediction,load(case/'golden.json'))
    save(out/case.name/'prediction.json',prediction); save(out/case.name/'graph.json',report)
    rows.append({'case':case.name,'candidate_count':len(candidates),'accepted':len(association['accepted']),'learned_vetoes':len(learned['blocked']),**metric})
keys=['ComponentF1','PinF1','NetHypergraphF1','NetLineF1','Score_Accuracy_SingleCase_proxy']
summary={'completed':len(rows),'mean_metrics':{k:sum(r[k] for r in rows)/len(rows) for k in keys},'records':rows,'association_model':a.model,'official_equivalence':False,'evaluation':'saved image-only candidates and observations; 55-page public development split also used for model selection; oracle training validation is separate from this inference replay'}
save(out/'summary.json',summary); print(json.dumps({k:v for k,v in summary.items() if k!='records'},indent=2))
