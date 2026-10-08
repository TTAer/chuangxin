"""Check counterfactual merge cost against independent component size products."""
import sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scripts.prepare_circuit_program import execute

out=ROOT/'output/topology-cost-check.json'
if out.exists():raise ValueError('Preserve check')
with np.load(ROOT/'data/processed/radial-evidence-v1/tiny.npz') as z:program=z['program']
with np.load(ROOT/'data/processed/radial-evidence-v1/tiny-patches.npz') as z:pages=z['page_ids'];slots=z['slot_ids'];labels=z['labels']
with np.load(ROOT/'data/processed/topology-cost-v1/tiny.npz') as z:fp=z['FP'];fn=z['FN']
checked=0
for page,slot,label,a,b in zip(pages,slots,labels,fp,fn):
    if label==0:
        _,_,_,groups=execute(program[page]);i,j=divmod(int(slot)-2,6);node=6+i*6+j
        ga=next(g for g in groups if i in g);gb=next(g for g in groups if node in g)
        expected=0 if ga==gb else len(ga)*len(gb);assert a==expected and b==0;checked+=1
    else:assert a==0
result={'CROSS_merge_component_product_checks':checked,'JOIN_deletion_never_creates_FP':True,'scope':'tiny analytical cost validation, not accuracy evidence'}
out.write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result))
