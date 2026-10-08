"""Paired development ablations; exclude the visually confirmed near-duplicate page."""
import argparse
import json
from pathlib import Path
from collections import Counter

KEYS=['ComponentF1','PinF1','NetHypergraphF1','NetLineF1','Score_Accuracy_SingleCase_proxy']
p=argparse.ArgumentParser(); p.add_argument('runs',nargs='+'); p.add_argument('--output',default='output/ablation/summary.json'); p.add_argument('--exclude-case',action='append',default=['0237']); a=p.parse_args()
runs={Path(run).name:json.loads((Path(run)/'summary.json').read_text(encoding='utf-8')) for run in a.runs}
common=set.intersection(*(set(r['id'] for r in run['records']) for run in runs.values()))-set(a.exclude_case)
report={'official_equivalence':False,'cases':len(common),'case_ids':sorted(common),'excluded_cases':a.exclude_case,'exclusion_reason':'0237 is the same underlying TAS2764 diagram as training page 0027 (visual audit), despite JPEG/PNG and minor text differences','runs':{}}
baseline=None
for name,run in runs.items():
    records=[r for r in run['records'] if r['id'] in common]
    complete=[r for r in records if r['status']=='completed']
    means={k:sum(r[k] for r in complete)/max(1,len(records)) for k in KEYS}
    if baseline is None: baseline={r['id']:r for r in complete}
    scores=[r['Score_Accuracy_SingleCase_proxy']-baseline[r['id']]['Score_Accuracy_SingleCase_proxy'] for r in complete if r['id'] in baseline]
    report['runs'][name]={'completed':len(complete),'failed':len(records)-len(complete),'mean_metrics':means,'mean_seconds':sum(r['seconds'] for r in complete)/max(1,len(complete)),'score_change_vs_first':{'improved':sum(s>1e-8 for s in scores),'worse':sum(s< -1e-8 for s in scores),'same':sum(abs(s)<=1e-8 for s in scores)},'styles':{style:{k:sum(r[k] for r in complete if r.get('style')==style)/sum(r.get('style')==style for r in complete) for k in KEYS} for style in sorted({r.get('style','unknown') for r in complete})}}
out=Path(a.output); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in report.items() if k!='case_ids'},indent=2))
