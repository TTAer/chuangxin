"""Counterfactual single-slot electrical costs from train/dev supervision only."""
import sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scripts.prepare_circuit_program import execute

out=ROOT/'data/processed/topology-cost-v1';out.mkdir(parents=True,exist_ok=True)
if (out/'protocol.json').exists():raise ValueError('Preserve costs')
protocol={'date':'2026-10-08','source':'radial-evidence-v1 tiny/train/validation only; no pressure',
          'definition':'flip one CROSS/JOIN with all other GT instructions fixed, count upper-triangle FP+FN',
          'weight':'1+min(FP+FN,128)/32; normalize by train-set mean, CE baseline weight1',
          'scope':'single-flip sensitivity not additive multi-error loss; GT never used at inference',
          'models':'fresh same SpatialEvidenceNet CE vs topology-weighted CE, same shuffle, tiny400 then8epochs batch512 micro128',
          'selection':'minimum worst clean/fixed-blurred dev mean single-flip error cost, then accuracy; no pressure selection',
          'gate':'new primary coverage>=.90 F1>=.98 exact-net>=.95; risk-vs-CE full pair-F1 gain>=.002, no increased FP, rotation accept<=.05'}
(out/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8');report={}
for name in ('tiny','train','validation'):
    with np.load(ROOT/f'data/processed/radial-evidence-v1/{name}.npz') as z:program=z['program']
    with np.load(ROOT/f'data/processed/radial-evidence-v1/{name}-patches.npz') as z:pages=z['page_ids'];slots=z['slot_ids'];labels=z['labels']
    fp=[];fn=[];cache={}
    for page,slot,label in zip(pages,slots,labels):
        key=int(page),int(slot)
        if key not in cache:
            g=program[page];assert g[slot]==label+8
            _,truth,_,_=execute(g);altered=g.copy();altered[slot]=9 if g[slot]==8 else 8;_,changed,_,_=execute(altered)
            upper=np.triu(np.ones((42,42),bool),1);cache[key]=(int(((changed>.5)&(truth<=.5)&upper).sum()),int(((changed<=.5)&(truth>.5)&upper).sum()))
        a,b=cache[key];fp.append(a);fn.append(b)
    fp=np.array(fp);fn=np.array(fn);np.savez_compressed(out/f'{name}.npz',FP=fp,FN=fn,weight=1+np.minimum(fp+fn,128)/32)
    report[name]={'patches':len(fp),'zero_cost':int(((fp+fn)==0).sum()),'max_FP':int(fp.max()),'max_FN':int(fn.max()),'mean_cost':float((fp+fn).mean())}
(out/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report))
