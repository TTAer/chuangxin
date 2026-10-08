"""Compare real image inference with cached GEA candidate replay."""
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from pcbparse.vision import ImageParser
from pcbparse.__main__ import load,save
from pcbparse.core import evaluate

parser=ImageParser(association_model='models/candidates/association-balanced/association.onnx',pin_ensemble='auto',semantic_library='none',text_constraints=True)
out=ROOT/'output/association-live-check'; rows=[]
for case in ['0101','0159']:
    base=ROOT/'output/benchmark-graph-topology-v2-55'/case
    prediction,obs,diag,_=parser.parse(load(base/'diagnostics.json')['input'])
    cached=load(ROOT/'output/benchmark-gea-cached-55'/case/'prediction.json')
    row={'case':case,'seconds':diag['seconds'],'prediction_matches_cached':prediction==cached,'learned_association_status':parser.last_graph_report['learned_association']['status'],**evaluate(prediction,load(base/'golden.json'))}
    save(out/case/'prediction.json',prediction); save(out/case/'graph.json',parser.last_graph_report); rows.append(row)
    print(json.dumps(row),flush=True)
save(out/'summary.json',{'records':rows,'passed':all(r['prediction_matches_cached'] for r in rows),'official_equivalence':False})
