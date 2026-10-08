"""Development-only threshold-to-end-score curves, never auto-promote a model."""
import csv
import json
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]
rows=[]
for model in ('base','local'):
    for threshold in (.2,.3,.4,.5,.6,.7,.8,.9):
        for domain in ('gt10','pcb55'):
            output=f'output/side-threshold-sweep/{model}-{threshold:.1f}-{domain}'
            run=subprocess.run([sys.executable,str(ROOT/'scripts/evaluate_side_strips.py'),'--dataset',domain,
                '--model',f'models/candidates/side-strip-{model}/side-strip.onnx','--threshold',str(threshold),'--output',output],cwd=ROOT,capture_output=True,text=True,encoding='utf-8',errors='replace')
            if run.returncode: raise RuntimeError(run.stdout+run.stderr)
            s=json.loads((ROOT/output/'summary.json').read_text(encoding='utf-8'))
            row={'model':model,'threshold':threshold,'domain':domain,**s['mean_metrics'],**s['funnel_totals'],
                 'improved':s['improved'],'regressed':s['regressed']}
            rows.append(row); print(json.dumps(row),flush=True)
out=ROOT/'output/ablation/side-thresholds.json'
out.write_text(json.dumps({'development_only':True,'default_promoted':False,'official_equivalence':False,'rows':rows},indent=2),encoding='utf-8')
with out.with_suffix('.csv').open('w',encoding='utf-8',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
