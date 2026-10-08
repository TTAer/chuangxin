"""Rebuild saved image-only observations; no repeated OCR or annotation input to build."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from pcbparse.core import build,evaluate
from pcbparse.__main__ import save

p=argparse.ArgumentParser(); p.add_argument('run'); p.add_argument('--output',default='output/ablation/topology.json'); p.add_argument('--limit',type=int,default=5); a=p.parse_args()
root=Path(a.run); summary=json.loads((root/'summary.json').read_text(encoding='utf-8')); results={}
for tolerance in (3,4,5):
    records=[]
    selected=[r for r in summary['records'] if r['status']=='completed' and r['id']!='0237']
    if a.limit>0: selected=selected[:a.limit]
    for item in selected:
        if item['status']!='completed' or item['id']=='0237': continue
        directory=root/item['id']; obs=json.loads((directory/'observations.json').read_text(encoding='utf-8'))
        if tolerance==3:
            records.append({k:v for k,v in item.items() if k in ('id','ComponentF1','PinF1','NetHypergraphF1','NetLineF1','Score_Accuracy_SingleCase_proxy')})
            continue
        pred=build(obs,tolerance=tolerance)
        gt=json.loads((directory/'golden.json').read_text(encoding='utf-8'))
        records.append({'id':item['id'],**evaluate(pred,gt)})
        save(Path(a.output).with_suffix('.progress.json'),{'tolerance':tolerance,'records':records})
        print(json.dumps({'tolerance':tolerance,'case':item['id']}),flush=True)
    results[str(tolerance)]={'cases':len(records),'metrics':{k:sum(r[k] for r in records)/len(records) for k in ('ComponentF1','PinF1','NetHypergraphF1','NetLineF1','Score_Accuracy_SingleCase_proxy')},'records':records}
report={'official_equivalence':False,'source':a.run,'tolerances':results,'scope':'validation tuning, not independent test performance'}
save(a.output,report); print(json.dumps({k:v['metrics'] for k,v in results.items()},indent=2))
