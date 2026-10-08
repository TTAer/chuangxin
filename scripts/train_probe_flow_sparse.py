"""Separate recovery protocol: old stress observed, new holdout scored once."""
import json,sys,time,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,torch
from torch import nn
from experiments.probe_flow import partition_loss
from experiments.probe_flow_sparse import SparseProbeFlowNet
from scripts.prepare_probe_flow import compile_page
from scripts.train_probe_flow import metrics


def main():
    out=ROOT/'output/training/probe-flow-sparse';out.mkdir(parents=True,exist_ok=True)
    if (out/'protocol.json').exists():raise ValueError('Preserve recovery experiment')
    torch.set_num_threads(4);torch.backends.cudnn.enabled=False;torch.cuda.set_per_process_memory_fraction(.4);torch.manual_seed(20261007)
    bank=ROOT/'data/processed/probe-flow';data={}
    for name in ('train','validation','stress'):
        with np.load(bank/(name+'.npz')) as z:data[name]={k:torch.tensor(z[k],device='cuda') for k in z.files}
        data[name]['patches']=1-data[name]['patches'].float()/255
    # Frozen new synthetic styles; never used in fitting or checkpoint selection.
    records=[compile_page(seed) for seed in range(40000,40128)]
    raw={k:np.stack([r[i] for r in records]) for i,k in enumerate(('patches','edges','edge_mask','edge_truth','truth','valid'))}
    import cv2
    for i in range(len(records)):
        for j in np.flatnonzero(raw['edge_mask'][i]):
            rotated=np.rot90(raw['patches'][i,j,0],(i+j)%4).copy();raw['patches'][i,j,0]=cv2.GaussianBlur(rotated,(3,3),.7)
    np.savez_compressed(out/'new-style-holdout.npz',**raw)
    data['new_holdout']={k:torch.tensor(v,device='cuda') for k,v in raw.items()};data['new_holdout']['patches']=1-data['new_holdout']['patches'].float()/255
    protocol={'architecture':'R2 C4 feature orbit averaging + zero-permitting conductance + Green response','epochs':12,'batch':8,'lr':.003,
              'initialization':'CI2N image-only pretrained weights from this turn; no V1 topology-finetuned checkpoint',
              'augmentation':'per-page brightness/noise and Gaussian blur sigma0.7; C4 invariance built into model',
              'old_stress':'already observed; diagnostic development only','new_holdout':'seeds40000:40128 and patch rotations/blur, scored once after checkpoint selection',
              'selection':'development membership F1; ties resolved by validation reachability BCE fixed before run',
              'threshold':.5,'grounding_epsilon':.003,'loss':'balanced global reachability + 0.2 observed edge BCE',
              'new_holdout_sha256':hashlib.sha256((out/'new-style-holdout.npz').read_bytes()).hexdigest(),'real_independent_validation':False,'runtime_changed':False}
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    model=SparseProbeFlowNet().cuda();model.load_state_dict(torch.load(ROOT/'output/training/probe-flow/ci2n-pretrained.pt',weights_only=True))
    optimizer=torch.optim.AdamW(model.parameters(),lr=.003);rng=np.random.default_rng(20261007);started=time.perf_counter();history=[];best=(-1,float('-inf'))
    def evaluate(name):
        model.eval();responses=[];losses=[]
        with torch.no_grad():
            for i in range(0,len(data[name]['patches']),8):
                d={k:v[i:i+8] for k,v in data[name].items()};r,_=model(d['patches'],d['edges'],d['edge_mask']);responses.append(r.cpu().numpy());losses.append(float(partition_loss(r,d['truth'],d['valid'])))
        result=metrics(np.concatenate(responses),data[name]['truth'].cpu().numpy(),data[name]['valid'].cpu().numpy());result['reachability_loss']=float(np.mean(losses));return result
    for epoch in range(12):
        model.train();losses=[]
        for indices in rng.permutation(len(data['train']['patches'])).reshape(-1,8):
            d={k:v[indices] for k,v in data['train'].items()};patches=d['patches']
            if rng.random()<.5:patches=nn.functional.avg_pool2d(nn.functional.pad(patches.flatten(0,1),(1,1,1,1),mode='replicate'),3,1).reshape_as(patches)
            patches=(patches*float(rng.uniform(.7,1.15))+torch.randn_like(patches)*.015).clamp(0,1)
            response,logits=model(patches,d['edges'],d['edge_mask']);edge_loss=nn.functional.binary_cross_entropy_with_logits(logits,d['edge_truth'],reduction='none')
            loss=partition_loss(response,d['truth'],d['valid'])+.2*(edge_loss*d['edge_mask']).sum()/d['edge_mask'].sum()
            optimizer.zero_grad();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5);optimizer.step()
            if not torch.isfinite(loss):raise ValueError('Nonfinite R2 loss')
            losses.append(float(loss.detach()))
        validation=evaluate('validation');key=(validation['mean']['membership_F1'],-validation['reachability_loss']);r={'epoch':epoch+1,'loss':float(np.mean(losses)),**validation['mean'],'validation_loss':validation['reachability_loss']};history.append(r);print(json.dumps(r),flush=True)
        if key>best:
            best=key;torch.save(model.state_dict(),out/'best.pt');(out/'best.json').write_text(json.dumps({'epoch':epoch+1,'validation':validation},indent=2),encoding='utf-8')
    (out/'history.json').write_text(json.dumps(history,indent=2),encoding='utf-8');model.load_state_dict(torch.load(out/'best.pt',weights_only=True));model.eval()
    result={'best':json.loads((out/'best.json').read_text(encoding='utf-8')),'old_stress_diagnostic':evaluate('stress'),'new_style_holdout':evaluate('new_holdout'),
            'train':evaluate('train'),'seconds':time.perf_counter()-started,'CUDA_peak_allocated_bytes':torch.cuda.max_memory_allocated(),'independent_validation':False,'runtime_changed':False}
    (out/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps({'finished':True,'new_holdout':result['new_style_holdout']['mean'],'seconds':result['seconds']}),flush=True)

if __name__=='__main__':main()
