"""Post-training solver/invariance audit and non-misleading pair micro counts."""
import json,sys,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,torch
from experiments.probe_flow import ProbeFlowNet
from experiments.probe_flow_sparse import SparseProbeFlowNet
from scripts.train_probe_flow import metrics
from scripts.prepare_review_handoff import check_lock


def main():
    torch.set_num_threads(4);bank=ROOT/'data/processed/probe-flow';out=ROOT/'output/training/probe-flow-sparse'
    if not (out/'summary.json').exists():raise ValueError('Wait for fixed recovery training to complete')
    results=[]
    for name,cls,path,source in [('v1-scratch',ProbeFlowNet,ROOT/'output/training/probe-flow/scratch/best.pt',bank/'stress.npz'),
                                ('v1-contrastive',ProbeFlowNet,ROOT/'output/training/probe-flow/ci2n-contrastive/best.pt',bank/'stress.npz'),
                                ('r2-sparse',SparseProbeFlowNet,out/'best.pt',out/'new-style-holdout.npz')]:
        model=cls().eval();model.load_state_dict(torch.load(path,weights_only=True,map_location='cpu'))
        with np.load(source) as z:d={k:torch.tensor(z[k]) for k in z.files}
        d['patches']=1-d['patches'].float()/255;responses=[];edge_logits=[]
        with torch.no_grad():
            for i in range(0,len(d['patches']),8):
                response,logits=model(d['patches'][i:i+8],d['edges'][i:i+8],d['edge_mask'][i:i+8]);responses.append(response.numpy());edge_logits.append(logits.numpy())
        probability=np.concatenate(responses);valid=d['valid'].numpy();truth=d['truth'].numpy();totals={'TP':0,'FP':0,'FN':0};empty=0
        for r,g,m in zip(probability,truth,valid):
            n=int(m.sum());mask=np.triu(np.ones((n,n),bool),1);p=r[:n,:n]>.5;t=g[:n,:n]>.5
            tp=int((p&t&mask).sum());fp=int((p&~t&mask).sum());fn=int((~p&t&mask).sum())
            for k,v in zip(totals,(tp,fp,fn)):totals[k]+=v
            empty+=not bool((t&mask).any())
        result={'variant':name,'evaluation_file':source.relative_to(ROOT).as_posix(),'summary':metrics(probability,truth,valid)['mean'],
                'raw_pair_micro':{**totals,'F1':2*totals['TP']/max(1,2*totals['TP']+totals['FP']+totals['FN'])},'pages_without_positive_pairs':int(empty)}
        np.savez_compressed(out/(name+'-audit-responses.npz'),response=probability,conductance_logits=np.concatenate(edge_logits));results.append(result)
    model=SparseProbeFlowNet().eval();model.load_state_dict(torch.load(out/'best.pt',weights_only=True,map_location='cpu'))
    patch=d['patches'][:4];edges=d['edges'][:4];mask=d['edge_mask'][:4]
    with torch.no_grad():
        original,_=model(patch,edges,mask);rotated,_=model(torch.rot90(patch,1,(-2,-1)),edges,mask)
        perm=torch.randperm(12);renamed,_=model(patch,perm[edges],mask);back=renamed[:,perm][:,:,perm]
        order=torch.randperm(36);reordered,_=model(patch[:,order],edges[:,order],mask[:,order])
    deltas={'rotation_response_max_delta':float((original-rotated).abs().max()),'node_permutation_response_max_delta':float((original-back).abs().max()),
            'candidate_order_response_max_delta':float((original-reordered).abs().max())};assert max(deltas.values())<.002,deltas
    finite_model=SparseProbeFlowNet();r,logits=finite_model(patch,edges,mask);(r.mean()+logits.square().mean()).backward()
    finite=all(torch.isfinite(p.grad).all() for p in finite_model.parameters() if p.grad is not None);assert finite
    frozen=[]
    for file,key in [('data/splits/rc0-evaluation.json','runtime_sha256'),('output/netset-oracle-research-manifest.json','sha256'),('output/topology-agreement-manifest.json','sha256')]:
        manifest=json.loads((ROOT/file).read_text(encoding='utf-8'));bad=[p for p,h in manifest[key].items() if hashlib.sha256((ROOT/p).read_bytes()).hexdigest()!=h];assert not bad;frozen.append({'manifest':file,'files':len(manifest[key]),'mismatches':bad})
    lock=json.loads((ROOT/'output/review-handoff/convention-pilot-protocol-lock.json').read_text(encoding='utf-8'));check_lock(ROOT,lock)
    result={'results':results,'invariance_checks':deltas,'finite_gradients':finite,'frozen_checks':frozen,'review_protocol_unchanged':True,
            'empty_pair_note':'legacy macro pair F1 records 0 on empty-positive pages, even if exact partition is correct; use counts/micro for interpretation',
            'new_dataset_annotations_used_for_training':False,'independent_verified_designs':0,'runtime_changed':False}
    (out/'audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result))

if __name__=='__main__':main()
