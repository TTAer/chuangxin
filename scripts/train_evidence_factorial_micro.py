"""Four matched-budget training trials, no new or old pressure pixels used for selection."""
import sys,json,time,hashlib,argparse
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch,numpy as np,cv2
from experiments.radial_evidence_net import RadialEvidenceNet
from experiments.spatial_evidence_net import SpatialEvidenceNet


def main():
    p=argparse.ArgumentParser();p.add_argument('--architecture',choices=['radial','spatial'],required=True);p.add_argument('--augmentation',choices=['narrow','broad'],required=True);a=p.parse_args();key=f'{a.architecture}-{a.augmentation}'
    out=ROOT/f'output/training/evidence-factorial-v1-{key}-micro';out.mkdir(parents=True,exist_ok=True)
    if (out/'protocol.json').exists():raise ValueError('Preserve trial')
    protocol={'date':'2026-10-08','trial':key,'seed':20261081,'data':'existing radial-evidence-v1 train/dev raw patches only; no pressure',
              'factorial':'radial/spatial x no added blur/additional per-batch Gaussian sigma uniform0:1.2',
              'memory_adaptation':'logical batch512 accumulated in micro128, same optimizer steps, no batchnorm/dropout; lower GPU allocation cap .20', 'tiny_steps':400,'tiny_gate':'final96/96 crossing accuracy','epochs':8,'batch':512,'lr':.001,
              'selection':'minimum(clean dev accuracy, fixed additionally blurred dev accuracy), then mean; first best',
              'fixed_dev_blur':'sigma per-patch uniform0:1.2, seed20261082, before training',
              'composition':'frozen base prefix/interval, hard crossing replacement only observed active slots',
              'pressure_gate':'coverage>=.90 F1>=.98 exact-net>=.95 net-exact improvement>=.01 and broken exact-program pages<=fixed pages; rotated acceptance<=.05',
              'real_accuracy_verified':False,'RC0_changed':False}
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    torch.manual_seed(protocol['seed']);torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(.20)
    rng=np.random.default_rng(protocol['seed']);data={}
    for n in ('tiny','train','validation'):
        with np.load(ROOT/f'data/processed/radial-evidence-v1/{n}-patches.npz') as z:data[n]=(z['raw'],z['labels'])
    r=np.random.default_rng(20261082);x,g=data['validation'];blur=np.stack([cv2.GaussianBlur(im[0],(7,7),float(r.uniform(0,1.2)))[None] for im in x]);data['blurred_validation']=(blur,g)
    model=(RadialEvidenceNet() if a.architecture=='radial' else SpatialEvidenceNet()).cuda();initial={k:v.clone() for k,v in model.state_dict().items()};start=time.perf_counter()
    def batch(n,ids):return torch.tensor(data[n][0][ids],device='cuda'),torch.tensor(data[n][1][ids],device='cuda')
    def evaluate(n):
        model.eval();pred=[]
        with torch.no_grad():
            for i in range(0,len(data[n][1]),128):
                x,g=batch(n,slice(i,i+128));pred.append(model(x).argmax(-1).cpu().numpy())
        p=np.concatenate(pred);g=data[n][1];return {'patches':len(g),'accuracy':float((p==g).mean()),'FP':int(((p==1)&(g==0)).sum()),'FN':int(((p==0)&(g==1)).sum())}
    def augment(x):
        # Isotropic added blur preserves labels; same deterministic sigma sequence in broad trials.
        sigma=float(rng.uniform(0,1.2))
        if sigma<.05:return x
        axis=torch.arange(-3,4,device=x.device,dtype=x.dtype);w=torch.exp(-axis.square()/(2*sigma*sigma));w=w/w.sum();kernel=(w[:,None]*w[None]).reshape(1,1,7,7)
        return torch.nn.functional.conv2d(torch.nn.functional.pad(x,(3,3,3,3),mode='replicate'),kernel)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001);schedule=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,400,eta_min=.0001);tiny=[];x,g=batch('tiny',slice(None))
    for step in range(400):
        model.train();loss=torch.nn.functional.cross_entropy(model(x),g);optimizer.zero_grad();loss.backward();optimizer.step();schedule.step()
        if (step+1)%50==0:tiny.append({'step':step+1,**evaluate('tiny')})
    passed=tiny[-1]['accuracy']==1
    (out/'tiny.json').write_text(json.dumps({'gate_passed':passed,'history':tiny},indent=2),encoding='utf-8');print(json.dumps({'trial':key,'tiny_passed':passed}),flush=True)
    if not passed:
        (out/'summary.json').write_text(json.dumps({'tiny_gate_passed':False,'full_training_executed':False,'tiny_steps':400}),encoding='utf-8');return
    model.load_state_dict(initial);optimizer=torch.optim.AdamW(model.parameters(),lr=.001);best=(-1,-1);history=[];steps=0
    # Separate shuffle from augmentation RNG so narrow and broad use identical page/patch order.
    shuffle=np.random.default_rng(protocol['seed'])
    for epoch in range(8):
        ids=shuffle.permutation(len(data['train'][1]));losses=[]
        for i in range(0,len(ids),512):
            x,g=batch('train',ids[i:i+512]);model.train()
            if a.augmentation=='broad':x=augment(x)
            optimizer.zero_grad();batch_loss=0.
            for m in range(0,len(g),128):
                fraction=len(g[m:m+128])/len(g)
                loss=torch.nn.functional.cross_entropy(model(x[m:m+128]),g[m:m+128])*fraction
                if not torch.isfinite(loss):raise ValueError('Nonfinite loss')
                loss.backward();batch_loss+=float(loss.detach())
            optimizer.step();steps+=1;losses.append(batch_loss)
        clean=evaluate('validation');blurred=evaluate('blurred_validation');score=(min(clean['accuracy'],blurred['accuracy']),(clean['accuracy']+blurred['accuracy'])/2)
        row={'epoch':epoch+1,'loss':float(np.mean(losses)),'clean':clean,'blurred':blurred,'selection_score':score};history.append(row);print(json.dumps({'trial':key,**row}),flush=True)
        if score>best:
            best=score;torch.save(model.state_dict(),out/'best.pt');(out/'best.json').write_text(json.dumps(row,indent=2),encoding='utf-8')
    report={'tiny_gate_passed':True,'full_training_executed':True,'trial':key,'parameters':sum(p.numel() for p in model.parameters()),'memory_adaptation':'logical batch512 accumulated in micro128, same optimizer steps, no batchnorm/dropout; lower GPU allocation cap .20', 'tiny_steps':400,'main_steps':steps,'seconds':time.perf_counter()-start,'real_accuracy_verified':False}
    (out/'history.json').write_text(json.dumps(history,indent=2),encoding='utf-8');(out/'summary.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report),flush=True)


if __name__=='__main__':main()
