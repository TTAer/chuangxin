"""Paired official-GT development ablation, explicitly not official scoring."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
names=['benchmark-official-baseline','benchmark-official-semantics-integrated','benchmark-official-new-pins-controlled','benchmark-official-new-pins-semantics']
summaries={n:json.loads((ROOT/'output'/n/'summary.json').read_text(encoding='utf-8')) for n in names}
baseline={r['case']:r for r in summaries[names[0]]['records']}
result={'golden_version':'10GT-0930','official_equivalence':False,'evaluation':'public GT, architecture selection; Q&A-informed local proxy, not private final test','runs':{}}
for name,s in summaries.items():
    rows={r['case']:r for r in s['records']}
    if rows.keys()!=baseline.keys(): raise ValueError('case sets differ')
    delta={k:rows[k]['Score_Accuracy_SingleCase_proxy']-baseline[k]['Score_Accuracy_SingleCase_proxy'] for k in baseline}
    result['runs'][name]={'mean_metrics':s['mean_metrics'],'improved':sum(v>1e-9 for v in delta.values()),'regressed':sum(v< -1e-9 for v in delta.values()),'unchanged':sum(abs(v)<=1e-9 for v in delta.values()),'proxy_delta':delta}
out=ROOT/'output/ablation/official-gt10.json'; out.write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps({k:{a:b for a,b in v.items() if a!='proxy_delta'} for k,v in result['runs'].items()},indent=2))
