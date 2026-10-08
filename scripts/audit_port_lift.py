"""Same-cohort controls and invariance; no checkpoint selection."""
import json,hashlib,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch,numpy as np
from experiments.port_lift import PortLiftNet
from scripts.train_port_lift import scores
from experiments.probe_flow_sparse import SparseProbeFlowNet

def main():
    torch.set_num_threads(4);torch.manual_seed(20261007);out=ROOT/'output/training/port-lift-v1'
    with np.load(out/'stress.npz') as z:d={k:z[k] for k in z.files}
    model=PortLiftNet().eval();model.load_state_dict(torch.load(out/'best.pt',map_location='cpu',weights_only=True));cs=[]
    with torch.no_grad():
        for i in range(0,len(d['patches']),16):
            _,c,_=model(1-torch.tensor(d['patches'][i:i+16]).float()/255,torch.tensor(d['edges'][i:i+16]),torch.tensor(d['edge_mask'][i:i+16]),torch.tensor(d['ports'][i:i+16]));cs.append(c.numpy())
    pred=np.concatenate(cs);mask=np.tile(np.triu(np.ones((10,10)),1),(len(pred),1,1));truth=d['component_truth'];closed=pred>.5
    for k in range(10):closed=closed|(closed[:,:,k,None]&closed[:,None,k,:])
    controls={'direct_component_pair':scores(pred,truth,mask),'incorrect_component_transitive_closure':scores(closed.astype(float),truth,mask),
              'all_component_pairs_positive':scores(np.ones_like(pred),truth,mask),'all_component_pairs_negative':scores(np.zeros_like(pred),truth,mask)}
    samewire=(d['ports'][:,:, :,None,None]==d['ports'][:,None,None,:,:]).any((2,4))
    controls['observed_same_wire_only_no_image']=scores(samewire.astype(float),truth,mask)
    prior=SparseProbeFlowNet().eval();prior.load_state_dict(torch.load(ROOT/'output/training/probe-flow-sparse/best.pt',map_location='cpu',weights_only=True));old=[]
    with torch.no_grad():
        for i in range(0,len(d['patches']),16):
            r,_=prior(1-torch.tensor(d['patches'][i:i+16]).float()/255,torch.tensor(d['edges'][i:i+16]),torch.tensor(d['edge_mask'][i:i+16]))
            ports=torch.nn.functional.one_hot(torch.tensor(d['ports'][i:i+16]),12).float()
            old.append(torch.einsum('bcpi,bij,bdqj->bcpdq',ports,r,ports).amax((2,4)).numpy())
    controls['frozen_ProbeFlow_R2_plus_same_port_pool']=scores(np.concatenate(old),truth,mask)
    patch=1-torch.tensor(d['patches'][:4]).float()/255;edge=torch.tensor(d['edges'][:4]);em=torch.tensor(d['edge_mask'][:4]);port=torch.tensor(d['ports'][:4])
    with torch.no_grad():
        _,original,_=model(patch,edge,em,port)
        _,swapped,_=model(patch,edge,em,port.flip(-1))
        rename=torch.randperm(12);_,renamed,_=model(patch,rename[edge],em,rename[port])
    checks={'port_order_max_delta':float((original-swapped).abs().max()),'wire_id_permutation_max_delta':float((original-renamed).abs().max())}
    assert max(checks.values())<.002
    frozen=[]
    for name,key in [('output/probe-flow-research-manifest.json','sha256'),('data/splits/rc0-evaluation.json','runtime_sha256'),('output/netset-oracle-research-manifest.json','sha256'),('output/topology-agreement-manifest.json','sha256')]:
        manifest=json.loads((ROOT/name).read_text(encoding='utf-8'));bad=[p for p,h in manifest[key].items() if hashlib.sha256((ROOT/p).read_bytes()).hexdigest()!=h];assert not bad;frozen.append({'manifest':name,'files':len(manifest[key]),'mismatches':0})
    result={'controls':controls,'invariance':checks,'frozen_checks':frozen,'scope':'synthetic known-port graph; not real dataset accuracy'}
    (out/'audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8');np.savez_compressed(out/'stress-predictions.npz',component_probability=pred)
    print(json.dumps(result))

if __name__=='__main__':main()
