"""Candidate-recall, selective coverage and soundness on synthetic observations."""
import json,sys,hashlib,argparse
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,torch
from itertools import product
from experiments.port_lift import PortLiftNet
from experiments.port_lift_uncertainty import component_bounds
from scripts.train_port_lift import compile_bank

def main():
    torch.set_num_threads(4);torch.manual_seed(20261010);rng=np.random.default_rng(20261010)
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=ROOT/'output/port-lift-uncertainty');args=parser.parse_args()
    out=args.output;out.mkdir(parents=True,exist_ok=True)
    if (out/'audit.json').exists():raise ValueError('Preserve audit')
    protocol={'threshold':.5,'cases':['complete_two_port','complete_four_port','missing_true_candidate'],
              'unknown_policy':'lower>0.5 positive, upper<=0.5 negative; otherwise abstain','seed':20261010,
              'scope':'synthetic known-wire relation and candidate attachment observations','training_executed':False}
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    d=compile_bank(range(81000,81064),True);model=PortLiftNet().eval();model.load_state_dict(torch.load(ROOT/'output/training/port-lift-v1/best.pt',map_location='cpu',weights_only=True))
    with torch.no_grad():r,_,_=model(1-torch.tensor(d['patches']).float()/255,torch.tensor(d['edges']),torch.tensor(d['edge_mask']),torch.tensor(d['ports']))
    physical=(r>.5).float();results=[];artifacts={}
    for name in protocol['cases']:
        ports=d['ports'].copy()
        if name=='complete_four_port':ports=np.stack([np.stack([rng.choice(int(d['valid'][i].sum()),4,replace=False) for _ in range(10)]) for i in range(64)])
        candidates=np.stack((ports,ports),axis=-1);changed=np.zeros_like(ports,dtype=bool)
        for i in range(64):
            n=int(d['valid'][i].sum())
            for ci in range(10):
                for pi in range(ports.shape[-1]):
                    if rng.random()<.3:
                        candidates[i,ci,pi,1]=(ports[i,ci,pi]+int(rng.integers(1,n)))%n;changed[i,ci,pi]=True
                        if name=='missing_true_candidate':candidates[i,ci,pi,0]=candidates[i,ci,pi,1]
        cm=torch.ones(candidates.shape);pm=torch.ones(ports.shape)
        lower,upper=component_bounds(physical,torch.tensor(candidates),cm,pm,torch.ones_like(pm));low=lower.numpy();high=upper.numpy()
        target=np.stack([d['truth'][i][p[:,:,None,None],p[None,None,:,:]].max((1,3)) for i,p in enumerate(ports)])>.5
        mask=np.tile(np.triu(np.ones((10,10),bool),1),(64,1,1));pos=(low>.5)&mask;neg=(high<=.5)&mask;covered=pos|neg
        incorrect=int((pos&~target).sum()+(neg&target).sum());total=int(mask.sum());known=int(covered.sum())
        result={'case':name,'candidate_recall':float(np.mean((candidates==ports[:,:,:,None]).any(-1))),
                'total_component_pairs':total,'covered':known,'unknown':total-known,'coverage':known/total,
                'covered_errors':incorrect,'covered_accuracy':1-incorrect/max(known,1),'false_positive':int((pos&~target).sum()),'false_negative':int((neg&target).sum())}
        if name!='missing_true_candidate':assert incorrect==0,result
        results.append(result);artifacts[name+'_lower']=low;artifacts[name+'_upper']=high
        if name=='missing_true_candidate':
            # No reliable completeness evidence: default is unknown, not certainty.
            safe_low,safe_high=component_bounds(physical,torch.tensor(candidates),cm,pm)
            safe_known=((safe_low.numpy()>.5)|(safe_high.numpy()<=.5))&mask
            results.append({'case':'default_unverified_completeness','total_component_pairs':total,'covered':int(safe_known.sum()),'unknown':total-int(safe_known.sum()),'coverage':float(safe_known.sum()/total)})
            assert not safe_known.any()
    # Exhaustive small assignment fixture verifies conservative, not exact, lower bound.
    wire=torch.tensor([[[1.,0.,1.],[0.,1.,0.],[1.,0.,1.]]]);cand=torch.tensor([[[[0,1]],[[2,2]]]])
    lo,hi=component_bounds(wire,cand,torch.ones_like(cand),torch.ones((1,2,1)),torch.ones((1,2,1)));assert lo[0,0,1]==0 and hi[0,0,1]==1
    missing=torch.zeros_like(cand);lm,hm=component_bounds(wire,cand,missing,torch.ones((1,2,1)));assert lm[0,0,1]==0 and hm[0,0,1]==1
    order=torch.tensor([1,0]);lp,hp=component_bounds(wire,cand[:,:,:,order],torch.ones_like(cand),torch.ones((1,2,1)),torch.ones((1,2,1)));assert torch.equal(lo,lp) and torch.equal(hi,hp)
    # Exhaust all 64 port-choice assignments in each of 32 random tiny graphs.
    fixture_rng=np.random.default_rng(20261011)
    for _ in range(32):
        labels=fixture_rng.integers(0,3,4);rel=(labels[:,None]==labels[None,:]).astype(float);options=fixture_rng.integers(0,4,(3,2,2))
        lows,highs=component_bounds(torch.tensor(rel[None]),torch.tensor(options[None]),torch.ones((1,3,2,2)),torch.ones((1,3,2)),torch.ones((1,3,2)))
        actual=[]
        for choice in product(range(2),repeat=6):
            selected=np.take_along_axis(options,np.array(choice).reshape(3,2,1),axis=-1)[...,0]
            actual.append(rel[selected[:,:,None,None],selected[None,None,:,:]].max((1,3)))
        actual=np.stack(actual);assert np.all(lows.numpy()[0]<=actual.min(0)) and np.all(highs.numpy()[0]>=actual.max(0))
    frozen=[]
    for path,key in [('output/port-lift-research-manifest.json','sha256'),('output/probe-flow-research-manifest.json','sha256'),('data/splits/rc0-evaluation.json','runtime_sha256')]:
        m=json.loads((ROOT/path).read_text(encoding='utf-8'));bad=[p for p,h in m[key].items() if hashlib.sha256((ROOT/p).read_bytes()).hexdigest()!=h];assert not bad;frozen.append({'manifest':path,'files':len(m[key]),'mismatches':0})
    result={'cases':results,'fixtures_passed':True,'exhaustive_assignment_fixture':{'graphs':32,'assignments_per_graph':64},'frozen_checks':frozen,'warning':'coverage is conditional on true candidate being present and the physical graph being correct; no guarantee for missed candidates',
            'training_executed':False,'real_observation_evaluation':False,'RC0_changed':False}
    np.savez_compressed(out/'relation-bounds.npz',**artifacts);(out/'audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result))

if __name__=='__main__':main()
