"""Post-hoc controls; never select checkpoints with these results."""
import json, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scripts.train_probe_flow import metrics

def main():
    out=ROOT/'output/training/probe-flow-sparse'
    with np.load(out/'new-style-holdout.npz') as z:d={k:z[k] for k in z.files}
    with np.load(out/'r2-sparse-audit-responses.npz') as z:logits=z['conductance_logits']
    n=len(d['valid']);controls={}
    for name in ('all-crossings-connected','all-crossings-separated','local-logit-positive-CC'):
        response=np.tile(np.eye(12,dtype=np.float32),(n,1,1))
        for i in range(n):
            for j in np.flatnonzero(d['edge_mask'][i]):
                if name=='all-crossings-connected' or (name=='local-logit-positive-CC' and logits[i,j]>0):
                    a,b=d['edges'][i,j];response[i,a,b]=response[i,b,a]=1
        controls[name]=metrics(response,d['truth'],d['valid'])['mean']
    result={'controls':controls,'scope':'post-hoc synthetic holdout comparison, same observed candidate graph',
            'local_control':'same trained encoder and conductance head; graph decoding only, not separately trained local baseline',
            'causal_ablation':False,'checkpoint_selection_changed':False}
    (out/'controls.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result))

if __name__=='__main__':main()
