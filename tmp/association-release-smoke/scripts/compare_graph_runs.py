"""Paired GraphSchematic development ablation; no official score equivalence."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
runs=['benchmark-multisource-55','benchmark-graph-association-55','benchmark-graph-topology-55','benchmark-graph-topology-v2-55']
summaries={name:json.loads((ROOT/'output'/name/'summary.json').read_text(encoding='utf-8')) for name in runs}
base={r['id']:r for r in summaries[runs[0]]['records'] if r['status']=='completed'}
result={'official_equivalence':False,'excluded_near_duplicate':'0237','evaluation':'same 55 development pages; checkpoint/architecture selection, not independent final test','runs':{}}
for name,s in summaries.items():
    cases={r['id']:r for r in s['records'] if r['status']=='completed'}
    if cases.keys()!=base.keys(): raise ValueError('case sets differ: '+name)
    changes=[cases[k]['Score_Accuracy_SingleCase_proxy']-base[k]['Score_Accuracy_SingleCase_proxy'] for k in base]
    result['runs'][name]={'completed':len(cases),'mean_metrics':s['mean_metrics'],'improved':sum(d>1e-9 for d in changes),'regressed':sum(d< -1e-9 for d in changes),'unchanged':sum(abs(d)<=1e-9 for d in changes),'mean_case_seconds':s['mean_case_seconds'],'by_case_proxy_delta':dict(zip(base,changes))}
path=ROOT/'output/ablation/graph-55.json'
path.write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps({k:{a:b for a,b in v.items() if a!='by_case_proxy_delta'} for k,v in result['runs'].items()},indent=2))
