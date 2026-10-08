"""Paired fresh-model radial evidence trials. Pressure is not read by this trainer."""
import sys,json,argparse,time,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch,numpy as np
from experiments.radial_evidence_net import RadialEvidenceNet


def main():
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['raw','canonical'],required=True);a=p.parse_args()
    bank=ROOT/'data/processed/radial-evidence-v1';assert (bank/'manifest.json').exists()
    out=ROOT/f'output/training/radial-evidence-v1-{a.mode}';out.mkdir(parents=True,exist_ok=True)
    if (out/'protocol.json').exists():raise ValueError('Preserve run')
    protocol=json.loads((bank/'protocol.json').read_text());protocol['mode']=a.mode
    protocol['architecture_sha256']=hashlib.sha256((ROOT/'experiments/radial_evidence_net.py').read_bytes()).hexdigest()
    protocol['manifest_sha256']=hashlib.sha256((bank/'manifest.json').read_bytes()).hexdigest()
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    torch.manual_seed(protocol['seed']);torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(.55)
    rng=np.random.default_rng(protocol['seed']);data={}
    for n in ('tiny','train','validation'):
        with np.load(bank/f'{n}-patches.npz') as z:data[n]=(z[a.mode],z['labels'])
    model=RadialEvidenceNet().cuda();initial={k:v.clone() for k,v in model.state_dict().items()};start=time.perf_counter()
    def batch(n,ids):return torch.tensor(data[n][0][ids],device='cuda'),torch.tensor(data[n][1][ids],device='cuda')
    def evaluate(n):
        model.eval();pred=[]
        with torch.no_grad():
            for i in range(0,len(data[n][1]),512):
                x,g=batch(n,slice(i,i+512));pred.append(model(x).argmax(-1).cpu().numpy())
        pred=np.concatenate(pred);g=data[n][1];tp=int(((pred==1)&(g==1)).sum());fp=int(((pred==1)&(g==0)).sum());fn=int(((pred==0)&(g==1)).sum())
        return {'patches':len(g),'accuracy':float((pred==g).mean()),'JOIN_TP':tp,'JOIN_FP':fp,'JOIN_FN':fn,'JOIN_F1':2*tp/max(1,2*tp+fp+fn)}
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001);schedule=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,400,eta_min=.0001)
    x,g=batch('tiny',slice(None));history=[]
    for step in range(400):
        model.train();loss=torch.nn.functional.cross_entropy(model(x),g);optimizer.zero_grad();loss.backward();optimizer.step();schedule.step()
        if (step+1)%50==0:history.append({'step':step+1,**evaluate('tiny')})
    passed=history[-1]['accuracy']==1
    (out/'tiny.json').write_text(json.dumps({'gate_passed':passed,'history':history},indent=2),encoding='utf-8')
    print(json.dumps({'mode':a.mode,'tiny_passed':passed,'last':history[-1]}),flush=True)
    if not passed:
        (out/'summary.json').write_text(json.dumps({'tiny_gate_passed':False,'full_training_executed':False,'tiny_steps':400}),encoding='utf-8');return
    model.load_state_dict(initial);optimizer=torch.optim.AdamW(model.parameters(),lr=.001);best=-1;history=[];steps=0
    for epoch in range(8):
        ids=rng.permutation(len(data['train'][1]));losses=[]
        for i in range(0,len(ids),512):
            x,g=batch('train',ids[i:i+512]);model.train();loss=torch.nn.functional.cross_entropy(model(x),g);optimizer.zero_grad();loss.backward();optimizer.step();steps+=1
            if not torch.isfinite(loss):raise ValueError('Nonfinite loss')
            losses.append(float(loss.detach()))
        r={'epoch':epoch+1,'loss':float(np.mean(losses)),**evaluate('validation')};history.append(r);print(json.dumps({'mode':a.mode,**r}),flush=True)
        if r['accuracy']>best:
            best=r['accuracy'];torch.save(model.state_dict(),out/'best.pt');(out/'best.json').write_text(json.dumps(r,indent=2),encoding='utf-8')
    report={'tiny_gate_passed':True,'full_training_executed':True,'mode':a.mode,'parameters':sum(p.numel() for p in model.parameters()),
            'tiny_steps':400,'main_steps':steps,'seconds':time.perf_counter()-start,'selection':'dev crossing accuracy only','real_accuracy_verified':False,'RC0_changed':False}
    (out/'history.json').write_text(json.dumps(history,indent=2),encoding='utf-8');(out/'summary.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
