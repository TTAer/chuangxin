"""New preregistered synthetic feasibility, no external annotation admission."""
import json,sys,time,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,torch,cv2
from torch import nn
from scripts.prepare_probe_flow import compile_page
from experiments.port_lift import PortLiftNet,balanced_bce

def compile_bank(seeds,stress=False):
    rows=[]
    for seed in seeds:
        patch,edge,mask,edge_gt,wire,valid,_=compile_page(seed)
        rng=np.random.default_rng(seed+900000);n=int(valid.sum())
        ports=np.stack([rng.choice(n,2,replace=False) for _ in range(10)])
        target=wire[ports[:, :,None,None],ports[None,None,:,:]].max((1,3))
        if stress:
            for j in np.flatnonzero(mask):patch[j,0]=cv2.GaussianBlur(np.rot90(patch[j,0],seed%4).copy(),(3,3),.7)
        rows.append((patch,edge,mask,edge_gt,wire,valid,ports,target))
    return {k:np.stack([r[i] for r in rows]) for i,k in enumerate(('patches','edges','edge_mask','edge_truth','truth','valid','ports','component_truth'))}

def scores(pred,true,mask):
    p=pred>.5;t=true>.5;m=mask.astype(bool);tp=int((p&t&m).sum());fp=int((p&~t&m).sum());fn=int((~p&t&m).sum())
    return {'TP':tp,'FP':fp,'FN':fn,'micro_F1':2*tp/max(1,2*tp+fp+fn),'exact_pages':float(np.mean(np.all((p==t)|~m,axis=(1,2))))}

def main():
    out=ROOT/'output/training/port-lift-v1';out.mkdir(parents=True,exist_ok=True)
    if (out/'protocol.json').exists():raise ValueError('Preserve experiment; no overwrite')
    torch.set_num_threads(4);torch.backends.cudnn.enabled=False;torch.cuda.set_per_process_memory_fraction(.4);torch.manual_seed(20261007)
    protocols={'seed':20261007,'epochs':12,'batch':16,'lr':.003,'tiny_steps':240,'tiny_gate_component_F1':.95,
               'selection':'component-pair development micro F1; ties by lower development loss','threshold':.5,
               'loss':'balanced component incidence + 0.5 balanced wire relation + 0.2 edge BCE',
               'cohorts':{'tiny':[500,504],'train':[50000,51024],'validation':[60000,60128],'stress':[70000,70128]},
               'scope':'procedural known wire graph plus known two-port attachment, not visual pin detection or real supervised training',
               'no_internal_device_edge':True,'external_annotations_used':False,'RC0_changed':False}
    (out/'protocol.json').write_text(json.dumps(protocols,indent=2),encoding='utf-8')
    data={};hashes={}
    for name,(start,stop) in protocols['cohorts'].items():
        raw=compile_bank(range(start,stop),name=='stress');np.savez_compressed(out/(name+'.npz'),**raw)
        hashes[name]=hashlib.sha256((out/(name+'.npz')).read_bytes()).hexdigest()
        data[name]={k:torch.tensor(v,device='cuda') for k,v in raw.items()};data[name]['patches']=1-data[name]['patches'].float()/255
    (out/'data-hashes.json').write_text(json.dumps(hashes,indent=2),encoding='utf-8')
    augmentation_rng=np.random.default_rng(20261008)
    def loss(model,d,augment=False):
        patches=d['patches']
        if augment and augmentation_rng.random()<.5:patches=nn.functional.avg_pool2d(nn.functional.pad(patches.flatten(0,1),(1,1,1,1),mode='replicate'),3,1).reshape_as(patches)
        r,c,l=model(patches,d['edges'],d['edge_mask'],d['ports']);cm=(1-torch.eye(10,device='cuda'))[None]
        wm=d['valid'][:,:,None]*d['valid'][:,None,:]*(1-torch.eye(12,device='cuda'))[None]
        edge=nn.functional.binary_cross_entropy_with_logits(l,d['edge_truth'],reduction='none')
        value=balanced_bce(c,d['component_truth'],cm)+.5*balanced_bce(r,d['truth'],wm)+.2*(edge*d['edge_mask']).sum()/d['edge_mask'].sum()
        return value,r,c
    def evaluate(model,name):
        model.eval();rs=[];cs=[];ls=[]
        with torch.no_grad():
            for i in range(0,len(data[name]['patches']),16):
                d={k:v[i:i+16] for k,v in data[name].items()};l,r,c=loss(model,d);rs.append(r.cpu().numpy());cs.append(c.cpu().numpy());ls.append(float(l))
        valid=data[name]['valid'].cpu().numpy();wm=valid[:,:,None]*valid[:,None,:]*np.triu(np.ones((12,12)),1)[None]
        cm=np.tile(np.triu(np.ones((10,10)),1),(len(valid),1,1))
        return {'component':scores(np.concatenate(cs),data[name]['component_truth'].cpu().numpy(),cm),
                'wire':scores(np.concatenate(rs),data[name]['truth'].cpu().numpy(),wm),'loss':float(np.mean(ls))}
    model=PortLiftNet().cuda();initial={k:v.clone() for k,v in model.state_dict().items()};opt=torch.optim.AdamW(model.parameters(),lr=.003);started=time.perf_counter();tiny=[]
    for step in range(240):
        model.train();v,_,_=loss(model,data['tiny']);opt.zero_grad();v.backward();nn.utils.clip_grad_norm_(model.parameters(),5);opt.step()
        if (step+1)%40==0:
            r=evaluate(model,'tiny');tiny.append({'step':step+1,**r});print(json.dumps({'phase':'tiny',**tiny[-1]}),flush=True)
    passed=tiny[-1]['component']['micro_F1']>=.95 and tiny[-1]['wire']['micro_F1']>=.95
    (out/'tiny.json').write_text(json.dumps({'passed':passed,'history':tiny},indent=2),encoding='utf-8')
    if not passed:(out/'summary.json').write_text(json.dumps({'tiny_passed':False,'full_training_executed':False}),encoding='utf-8');return
    model.load_state_dict(initial);opt=torch.optim.AdamW(model.parameters(),lr=.003);rng=np.random.default_rng(20261007);history=[];best=(-1,float('-inf'))
    for epoch in range(12):
        for indices in rng.permutation(1024).reshape(-1,16):
            d={k:v[indices] for k,v in data['train'].items()};model.train();v,_,_=loss(model,d,True);opt.zero_grad();v.backward();nn.utils.clip_grad_norm_(model.parameters(),5);opt.step()
            if not torch.isfinite(v):raise ValueError('Nonfinite loss')
        r=evaluate(model,'validation');history.append({'epoch':epoch+1,**r});print(json.dumps(history[-1]),flush=True)
        key=(r['component']['micro_F1'],-r['loss'])
        if key>best:best=key;torch.save(model.state_dict(),out/'best.pt');(out/'best.json').write_text(json.dumps(history[-1],indent=2),encoding='utf-8')
    model.load_state_dict(torch.load(out/'best.pt',weights_only=True));model.eval();d={k:v[:4] for k,v in data['stress'].items()}
    with torch.no_grad():
        r,c,_=model(d['patches'],d['edges'],d['edge_mask'],d['ports']);_,rot,_=model(torch.rot90(d['patches'],1,(-2,-1)),d['edges'],d['edge_mask'],d['ports'])
        perm=torch.randperm(10,device='cuda');_,cp,_=model(d['patches'],d['edges'],d['edge_mask'],d['ports'][:,perm]);expected=c[:,perm][:,:,perm]
    checks={'rotation_max_delta':float((rot-c).abs().max()),'component_permutation_max_delta':float((cp-expected).abs().max())};assert max(checks.values())<.002
    result={'tiny_passed':True,'train':evaluate(model,'train'),'validation':evaluate(model,'validation'),'stress':evaluate(model,'stress'),
            'checks':checks,'parameters':sum(p.numel() for p in model.parameters()),'seconds':time.perf_counter()-started,'CUDA_peak_bytes':torch.cuda.max_memory_allocated(),
            'runtime_changed':False,'real_training':False,'independent_validation':False}
    (out/'history.json').write_text(json.dumps(history,indent=2),encoding='utf-8');(out/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result),flush=True)

if __name__=='__main__':main()
