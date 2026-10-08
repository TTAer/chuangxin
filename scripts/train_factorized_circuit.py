"""Preregistered independent visual slot reader; stress is scored only at the end."""
import json,sys,time,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,torch
from torch import nn
from experiments.factorized_circuit_reader import FactorizedCircuitReader
from scripts.train_circuit_program import metrics


def main():
    out=ROOT/'output/training/factorized-circuit-v1';out.mkdir(parents=True,exist_ok=True)
    if (out/'protocol.json').exists():raise ValueError('Preserve existing run')
    torch.set_num_threads(4);torch.manual_seed(20261011);rng=np.random.default_rng(20261011)
    torch.backends.cudnn.benchmark=False;torch.cuda.set_per_process_memory_fraction(.55)
    bank=ROOT/'data/processed/factorized-circuit-v1'
    protocol={'model':'FactorizedCircuitReader, fresh random initialization, no transformer or denoising',
              'seed':20261011,'whole_raster_resolution':256,'tiny_steps':400,'tiny_gate':'final exact program and net partition 4/4',
              'tiny_lr':'cosine .001 to .0001','main_epochs':8,'main_batch':16,'main_lr':.001,
              'selection':'dev net-pair F1 then exact program; stress once after selection',
              'stress_success_criterion':'net-pair F1 >=.95 and exact net partition >=.90, research only',
              'representation':'shared raster encoder, fixed slot sampling, typed classification heads, image-only count heads',
              'real_net_supervision':False,'external_image_pretraining':False,'RC0_changed':False,
              'manifest_sha256':hashlib.sha256((bank/'manifest.json').read_bytes()).hexdigest()}
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    data={}
    for name in ('tiny','train','validation','stress'):
        with np.load(bank/f'{name}.npz') as z:data[name]={'images':z['images'],'program':z['program']}
    model=FactorizedCircuitReader().cuda();initial={k:v.clone() for k,v in model.state_dict().items()}
    scaler=torch.amp.GradScaler('cuda');start=time.perf_counter()
    def batch(d,ids):
        return torch.tensor(1-d['images'][ids].astype(np.float32)/255,device='cuda'),torch.tensor(d['program'][ids],device='cuda')
    def loss_for(x,g):
        counts,cross,interval=model(x);active_c=g[:,2:38]!=7;active_i=g[:,38:]!=7
        return nn.functional.cross_entropy(counts.float().reshape(-1,4),(g[:,:2]-3).reshape(-1))+nn.functional.cross_entropy(cross.float()[active_c],(g[:,2:38]-8)[active_c])+nn.functional.cross_entropy(interval.float()[active_i],(g[:,38:]-10)[active_i])
    def evaluate(name,save=False,blank=False):
        model.eval();pred=[]
        with torch.no_grad():
            for i in range(0,len(data[name]['images']),16):
                x,g=batch(data[name],slice(i,i+16))
                if blank:x.zero_()
                with torch.autocast('cuda',dtype=torch.float16):p=model.generate(x)
                pred.append(p.cpu().numpy())
        pred=np.concatenate(pred);result=metrics(pred,data[name]['program'])
        if save:np.savez_compressed(out/f'{name}-predictions.npz',program=pred)
        return result
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001);schedule=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,400,eta_min=.0001)
    tiny=[]
    x,g=batch(data['tiny'],slice(None))
    for step in range(400):
        model.train()
        with torch.autocast('cuda',dtype=torch.float16):loss=loss_for(x,g)
        optimizer.zero_grad();scaler.scale(loss).backward();scaler.unscale_(optimizer);nn.utils.clip_grad_norm_(model.parameters(),5);scaler.step(optimizer);scaler.update();schedule.step()
        if (step+1)%50==0:
            r={'step':step+1,**evaluate('tiny')};tiny.append(r);print(json.dumps({'phase':'tiny',**r}),flush=True)
    passed=tiny[-1]['exact_program']==1 and tiny[-1]['exact_net_partition']==1
    (out/'tiny.json').write_text(json.dumps({'gate_passed':passed,'history':tiny},indent=2),encoding='utf-8')
    if not passed:
        (out/'summary.json').write_text(json.dumps({'tiny_gate_passed':False,'full_training_executed':False,'seconds':time.perf_counter()-start},indent=2),encoding='utf-8');return
    model.load_state_dict(initial);optimizer=torch.optim.AdamW(model.parameters(),lr=.001);history=[];best=(-1,-1);main_steps=0
    for epoch in range(8):
        losses=[]
        for ids in rng.permutation(len(data['train']['images'])).reshape(-1,16):
            x,g=batch(data['train'],ids);model.train()
            with torch.autocast('cuda',dtype=torch.float16):loss=loss_for(x,g)
            optimizer.zero_grad();scaler.scale(loss).backward();scaler.unscale_(optimizer);nn.utils.clip_grad_norm_(model.parameters(),5);scaler.step(optimizer);scaler.update()
            if not torch.isfinite(loss):raise ValueError('Nonfinite loss')
            losses.append(float(loss.detach()));main_steps+=1
        row={'epoch':epoch+1,'loss':float(np.mean(losses)),**evaluate('validation')};history.append(row);print(json.dumps(row),flush=True)
        key=(row['net_pair']['micro_F1'],row['exact_program'])
        if key>best:
            best=key;torch.save(model.state_dict(),out/'best.pt');(out/'best.json').write_text(json.dumps(row,indent=2),encoding='utf-8')
    model.load_state_dict(torch.load(out/'best.pt',weights_only=True));model.eval()
    validation=evaluate('validation',True);stress=evaluate('stress',True)
    report={'tiny_gate_passed':True,'full_training_executed':True,'validation':validation,'stress':stress,
            'blank_validation':evaluate('validation',blank=True),'blank_stress':evaluate('stress',blank=True),
            'stress_gate_passed':stress['net_pair']['micro_F1']>=.95 and stress['exact_net_partition']>=.90,
            'parameters':sum(p.numel() for p in model.parameters()),'main_optimizer_steps':main_steps,'tiny_optimizer_steps':400,
            'seconds':time.perf_counter()-start,'CUDA_peak_allocated_bytes':torch.cuda.max_memory_allocated(),
            'independent_real_accuracy':False,'RC0_changed':False}
    (out/'history.json').write_text(json.dumps(history,indent=2),encoding='utf-8');(out/'summary.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({'finished':True,**report}),flush=True)


if __name__=='__main__':main()
