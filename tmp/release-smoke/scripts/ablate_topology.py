"""Rebuild saved image-only observations; no repeated OCR or annotation input to build."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from pcbparse.core import build,evaluate
from pcbparse.__main__ import save

p=argparse.ArgumentParser(); p.add_argument('run'); p.add_argument('--output',default='output/ablation/topology.json'); a=p.parse_args()
root=Path(a.run); summary=json.loads((root/'summary.json').read_text(encoding='utf-8')); results={}
for tolerance in (3,4,5):
    records=[]
    for item in summary['records']:
        if item['status']!='completed' or item['id']=='0237': continue
        directory=root/item['id']; obs=json.loads((directory/'observations.json').read_text(encoding='utf-8'))
        pred=build(obs,tolerance=tolerance)
        gt=json.loads((directory/'golden.json').read_text(encoding='utf-8'))
        records.append({'id':item['id'],**evaluate(pred,gt)})
    results[str(tolerance)]={'cases':len(records),'metrics':{k:sum(r[k] for r in records)/len(records) for k in ('ComponentF1','PinF1','NetHypergraphF1','NetLineF1','Score_Accuracy_SingleCase_proxy')},'records':records}
report={'official_equivalence':False,'source':a.run,'tolerances':results,'scope':'validation tuning, not independent test performance'}
save(a.output,report); print(json.dumps({k:v['metrics'] for k,v in results.items()},indent=2))
