"""Preregistered tiny fit -> image-contrastive -> physical reachability learning."""
import json,sys,time,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,torch
from torch import nn
from experiments.probe_flow import ProbeFlowNet,partition_loss
from pcbparse.core import assignment,f1


def metrics(response,truth,valid):
    records=[]
    for probability,target,mask in zip(response,truth,valid):
        n=int(mask.sum());pred=probability[:n,:n]>.5;gold=target[:n,:n]>.5;upper=np.triu(np.ones((n,n),bool),1)
        tp=int((pred&gold&upper).sum());fp=int((pred&~gold&upper).sum());fn=int((~pred&gold&upper).sum())
        # Connected-component decoding guarantees an equivalence relation, but
        # may amplify a single false bridge; report raw and closed relations.
        def components(adjacency):
            remaining=set(range(n));sets=[]
            while remaining:
                frontier={remaining.pop()};group=set(frontier)
                while frontier:
                    found={int(j) for i in frontier for j in np.flatnonzero(adjacency[i])}&remaining
                    remaining-=found;group|=found;frontier=found
                sets.append(group)
            return sets
        sets=components(pred);real=components(gold);matches=assignment([[f1(len(a&b),len(a),len(b)) for b in real] for a in sets]);hits=sum(len(sets[i]&real[j]) for i,j in matches)
        closed=np.zeros_like(pred)
        for group in sets:
            ids=list(group);closed[np.ix_(ids,ids)]=True
        ctp=int((closed&gold&upper).sum());cfp=int((closed&~gold&upper).sum());cfn=int((~closed&gold&upper).sum())
        records.append({'membership_F1':hits/n,'exact_partition':float({frozenset(s) for s in sets}=={frozenset(s) for s in real}),
                        'raw_pair_F1':2*tp/max(1,2*tp+fp+fn),'closed_pair_F1':2*ctp/max(1,2*ctp+cfp+cfn),
                        'raw_FP':fp,'closed_FP':cfp,'predicted_groups':len(sets),'true_groups':len(real)})
    return {'mean':{k:float(np.mean([r[k] for r in records])) for k in records[0]},'records':records}


def main():
    out=ROOT/'output/training/probe-flow';out.mkdir(parents=True,exist_ok=True)
    if (out/'protocol.json').exists():raise ValueError('Preserve existing experiment')
    torch.set_num_threads(4);torch.backends.cudnn.enabled=False;torch.cuda.set_per_process_memory_fraction(.35)
    seed=20261006;torch.manual_seed(seed);rng=np.random.default_rng(seed)
    bank=ROOT/'data/processed/probe-flow';datasets={}
    for name in ('tiny','train','validation','stress'):
        with np.load(bank/(name+'.npz')) as z:datasets[name]={k:torch.tensor(z[k],device='cuda') for k in z.files}
        datasets[name]['patches']=1-datasets[name]['patches'].float()/255
    protocol={'architecture':'CNN learned conductance + differentiable grounded graph Laplacian inverse','seed':seed,'device':torch.cuda.get_device_name(),
              'tiny_steps':400,'tiny_gate':{'membership_F1_min':.95,'exact_partition_min':.9},'full_epochs':16,'batch_pages':16,'pretrain_epochs':6,
              'variants':['scratch','ci2n-contrastive'],'response_threshold':.5,'grounding_epsilon':.003,'conductance':'sigmoid(logit)^4',
              'loss':'balanced global root reachability BCE + 0.2 physical edge BCE','learning_rate':.003,
              'selection':'highest synthetic development membership F1; evaluate stress once after selection',
              'data_manifest_sha256':hashlib.sha256((bank/'manifest.json').read_bytes()).hexdigest(),
              'scope':'synthetic physical-wire-run oracle, NOT end-to-end PCB pin/net recognition','GT10_used':False,'Freeze_A_used':False,'runtime_changed':False}
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8');started=time.perf_counter()
    def batch(name,indices):return {k:v[indices] for k,v in datasets[name].items()}
    def loss_for(model,d):
        response,logits=model(d['patches'],d['edges'],d['edge_mask'])
        auxiliary=nn.functional.binary_cross_entropy_with_logits(logits,d['edge_truth'],reduction='none')
        return partition_loss(response,d['truth'],d['valid'])+.2*(auxiliary*d['edge_mask']).sum()/d['edge_mask'].sum(),response
    def evaluate(model,name):
        model.eval();outputs=[]
        with torch.no_grad():
            for i in range(0,len(datasets[name]['patches']),16):
                d=batch(name,slice(i,i+16));response,_=model(d['patches'],d['edges'],d['edge_mask']);outputs.append(response.cpu().numpy())
        response=np.concatenate(outputs);d=datasets[name]
        return metrics(response,d['truth'].cpu().numpy(),d['valid'].cpu().numpy())
    initial=ProbeFlowNet();initial_state={k:v.clone() for k,v in initial.state_dict().items()}
    tiny=ProbeFlowNet().cuda();tiny.load_state_dict(initial_state);opt=torch.optim.AdamW(tiny.parameters(),lr=.003)
    tiny_history=[]
    for step in range(400):
        tiny.train();loss,_=loss_for(tiny,datasets['tiny']);opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(tiny.parameters(),5);opt.step()
        if not torch.isfinite(loss):raise ValueError('Nonfinite tiny loss')
        if (step+1)%40==0:
            result=evaluate(tiny,'tiny');r={'step':step+1,'loss':float(loss.detach()),**result['mean']};tiny_history.append(r);print(json.dumps({'phase':'tiny',**r}),flush=True)
    tiny_result=evaluate(tiny,'tiny');passed=tiny_result['mean']['membership_F1']>=.95 and tiny_result['mean']['exact_partition']>=.9
    torch.save(tiny.state_dict(),out/'tiny.pt');(out/'tiny.json').write_text(json.dumps({'gate_passed':passed,'history':tiny_history,'result':tiny_result},indent=2),encoding='utf-8')
    if not passed:
        (out/'summary.json').write_text(json.dumps({'tiny_gate_passed':False,'larger_training_executed':False,'seconds':time.perf_counter()-started,'runtime_changed':False},indent=2),encoding='utf-8');return
    # Image-only contrastive pretraining on actual new train images; no external
    # pixel/node/net labels. Checkpoint starts from the same fresh initialization.
    pretrained=ProbeFlowNet().cuda();pretrained.load_state_dict(initial_state)
    with np.load(bank/'ci2n-contrastive.npz') as z:real=torch.tensor(z['patches'],device='cuda').float()/255
    real=1-real;optimizer=torch.optim.AdamW(pretrained.parameters(),lr=.001);prehistory=[]
    def augment(x):
        if rng.random()<.5:x=x.flip(-1)
        if rng.random()<.5:x=x.flip(-2)
        return (x*float(rng.uniform(.65,1.2))+torch.randn_like(x)*.025).clamp(0,1)
    for epoch in range(6):
        losses=[]
        for indices in rng.permutation(len(real)).reshape(-1,64):
            x=real[indices];views=torch.cat((augment(x),augment(x)))
            z=nn.functional.normalize(pretrained.projection(pretrained.encoder(views)),dim=1)
            similarity=z@z.T/.2;similarity=similarity.masked_fill(torch.eye(len(z),device='cuda',dtype=torch.bool),-1e4)
            target=(torch.arange(len(z),device='cuda')+len(x))%len(z);loss=nn.functional.cross_entropy(similarity,target)
            optimizer.zero_grad();loss.backward();optimizer.step();losses.append(float(loss.detach()))
        r={'epoch':epoch+1,'loss':float(np.mean(losses))};prehistory.append(r);print(json.dumps({'phase':'contrastive',**r}),flush=True)
    torch.save(pretrained.state_dict(),out/'ci2n-pretrained.pt');(out/'pretraining.json').write_text(json.dumps(prehistory,indent=2),encoding='utf-8')
    reports=[]
    for name in ('scratch','ci2n-contrastive'):
        torch.manual_seed(seed+1);model=ProbeFlowNet().cuda();model.load_state_dict(initial_state if name=='scratch' else pretrained.state_dict());optimizer=torch.optim.AdamW(model.parameters(),lr=.003)
        order=np.random.default_rng(seed+1);run=out/name;run.mkdir(exist_ok=True);best=-1;history=[]
        for epoch in range(16):
            model.train();losses=[]
            for indices in order.permutation(len(datasets['train']['patches'])).reshape(-1,16):
                loss,_=loss_for(model,batch('train',indices));optimizer.zero_grad();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5);optimizer.step()
                if not torch.isfinite(loss):raise ValueError('Nonfinite global loss')
                losses.append(float(loss.detach()))
            val=evaluate(model,'validation');r={'epoch':epoch+1,'loss':float(np.mean(losses)),**val['mean']};history.append(r);print(json.dumps({'phase':name,**r}),flush=True)
            if val['mean']['membership_F1']>best:
                best=val['mean']['membership_F1'];torch.save(model.state_dict(),run/'best.pt');(run/'best.json').write_text(json.dumps({'epoch':epoch+1,'result':val},indent=2),encoding='utf-8')
        (run/'history.json').write_text(json.dumps(history,indent=2),encoding='utf-8');model.load_state_dict(torch.load(run/'best.pt',weights_only=True))
        stress=evaluate(model,'stress');train=evaluate(model,'train');(run/'post-training.json').write_text(json.dumps({'train':train,'stress':stress},indent=2),encoding='utf-8')
        reports.append({'variant':name,'best':json.loads((run/'best.json').read_text(encoding='utf-8')),'train':train['mean'],'stress':stress['mean']})
    summary={'tiny_gate_passed':True,'runs':reports,'seconds':time.perf_counter()-started,'CUDA_peak_allocated_bytes':torch.cuda.max_memory_allocated(),
             'parameters':sum(p.numel() for p in initial.parameters()),'runtime_changed':False,'independent_validation':False,'official_evaluator_equivalence':False}
    (out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8');print(json.dumps({'finished':True,'seconds':summary['seconds']}),flush=True)

if __name__=='__main__':main()
