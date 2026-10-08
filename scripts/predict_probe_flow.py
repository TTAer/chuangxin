"""Research-only physical groups from supplied observed graph patches; no GT read."""
import argparse, json, sys, hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np, torch
from experiments.probe_flow_sparse import SparseProbeFlowNet

def main():
    parser=argparse.ArgumentParser();parser.add_argument('input',type=Path);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--weights',type=Path,default=ROOT/'output/training/probe-flow-sparse/best.pt');args=parser.parse_args()
    torch.set_num_threads(4)
    with np.load(args.input) as z:raw={k:z[k] for k in ('patches','edges','edge_mask','valid')}
    p,e,m,v=(raw[k] for k in ('patches','edges','edge_mask','valid'))
    b=len(p)
    if p.shape!=(b,36,1,24,24) or p.dtype!=np.uint8 or e.shape!=(b,36,2) or m.shape!=(b,36) or v.shape!=(b,12):raise ValueError('Expected uint8 patches [B,36,1,24,24], edges [B,36,2], mask [B,36], valid [B,12]; no truncation supported')
    if not np.isin(m,[0,1]).all() or not np.isin(v,[0,1]).all() or not np.issubdtype(e.dtype,np.integer) or e.min()<0 or e.max()>=12:raise ValueError('Invalid masks or node IDs')
    for i in range(b):
        n=int(v[i].sum())
        if n<1 or not np.array_equal(v[i],np.arange(12)<n) or (e[i][m[i]>0]>=n).any():raise ValueError('Valid nodes must be a nonempty prefix and observed edges must use valid IDs')
    model=SparseProbeFlowNet().eval();model.load_state_dict(torch.load(args.weights,map_location='cpu',weights_only=True));records=[]
    with torch.no_grad():
        for i in range(b):
            r,_=model(1-torch.tensor(p[i:i+1]).float()/255,torch.tensor(e[i:i+1]).long(),torch.tensor(m[i:i+1]).float())
            n=int(v[i].sum());a=r[0,:n,:n].numpy()>.5;remaining=set(range(n));groups=[]
            while remaining:
                frontier={remaining.pop()};group=set(frontier)
                while frontier:
                    found={int(j) for k in frontier for j in np.flatnonzero(a[k])}&remaining;remaining-=found;group|=found;frontier=found
                groups.append(sorted(group))
            records.append({'physical_wire_run_groups':groups,'root_response':r[0,:n,:n].tolist()})
    result={'scope':'research physical-run oracle; NOT competition pin/net output','checkpoint_sha256':hashlib.sha256(args.weights.read_bytes()).hexdigest(),'input_sha256':hashlib.sha256(args.input.read_bytes()).hexdigest(),'threshold':.5,'pages':records}
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps({'pages':b,'output':str(args.output),'ground_truth_fields_read':False}))

if __name__=='__main__':main()
