"""GPU multisource pretraining and scratch-controlled topology fine-tuning."""
import argparse,hashlib,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,torch
from experiments.masked_stroke_transformer import MaskedStrokeTransformer
from experiments.port_partition import RELATIONS
from scripts.train_port_partition import metrics


def main():
    p=argparse.ArgumentParser();p.add_argument('--pretrain-epochs',type=int,default=6);p.add_argument('--finetune-epochs',type=int,default=24)
    p.add_argument('--device',default='cuda');args=p.parse_args()
    torch.set_num_threads(4);torch.manual_seed(20261006);rng=np.random.default_rng(20261006)
    torch.backends.cudnn.enabled=False
    torch.backends.mha.set_fastpath_enabled(False)
    torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False)
    if args.device=='cuda':torch.cuda.set_per_process_memory_fraction(.025)
    out=ROOT/'output/training/masked-stroke-transformer';out.mkdir(parents=True,exist_ok=True)
    if (out/'pretrained.pt').exists():raise ValueError('Preserve completed pretraining')
    with np.load(ROOT/'data/processed/masked-stroke/pretrain.npz') as z:images=z['images'];source=z['source']
    groups=[np.flatnonzero(source==s) for s in sorted(set(source.tolist()))]
    model=MaskedStrokeTransformer().to(args.device);opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.001)
    started=time.perf_counter();history=[]
    for epoch in range(args.pretrain_epochs):
        model.train();losses=[]
        # Balanced fixed 4096 patches per epoch; complete bank available, sampled across epochs.
        chosen=np.array([rng.choice(groups[j%len(groups)]) for j in range(4096)]);rng.shuffle(chosen)
        for ids in np.array_split(chosen,256):
            x=torch.tensor(images[ids],dtype=torch.float32,device=args.device)/255
            if rng.random()<.5:x=x.mean(1,keepdim=True).expand(-1,3,-1,-1)
            mask=torch.rand(len(ids),36,device=args.device)<.6
            target=1-model.patches(x).reshape(-1,36,3,64).mean(2)
            reconstruction=model.reconstruct(x,mask)
            weight=1+4*(target>.1).float()
            loss=(((reconstruction-target)**2)*weight*mask[:,:,None]).sum()/(mask.sum()*64).clamp_min(1)
            opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);opt.step();losses.append(loss.item())
        row={'epoch':epoch+1,'masked_reconstruction_loss':float(np.mean(losses)),'seconds':time.perf_counter()-started};history.append(row)
        (out/'pretraining-history.json').write_text(json.dumps(history,indent=2),encoding='utf-8');print(json.dumps(row),flush=True)
    torch.save(model.state_dict(),out/'pretrained.pt')
    with np.load(ROOT/'data/processed/port-partition-expanded/relations.npz') as z:data={k:z[k] for k in z.files}
    train=np.flatnonzero(data['split']==0);val=np.flatnonzero(data['split']==1)
    compatible=np.all((RELATIONS[None]==data['targets'][:,None])|~data['valid'][:,None].astype(bool),axis=2)
    for pretrained in (False,True):
        variant='multisource-pretrained' if pretrained else 'scratch'
        dest=ROOT/'models/candidates/masked-stroke-transformer'/variant;dest.mkdir(parents=True,exist_ok=True)
        run=out/variant;run.mkdir(parents=True,exist_ok=True)
        torch.manual_seed(20261006);rng=np.random.default_rng(20261006);model=MaskedStrokeTransformer().to(args.device)
        if pretrained:model.load_state_dict(torch.load(out/'pretrained.pt',weights_only=True,map_location=args.device))
        opt=torch.optim.AdamW(model.parameters(),lr=.0005,weight_decay=.001);best=-1;history=[]
        for epoch in range(args.finetune_epochs):
            model.train();losses=[]
            for ids in np.array_split(rng.permutation(train),int(np.ceil(len(train)/16))):
                x=torch.tensor(data['images'][ids],dtype=torch.float32,device=args.device)/255
                if rng.random()<.5:x=x.mean(1,keepdim=True).expand(-1,3,-1,-1)
                logp=model(x).log_softmax(1);valid=torch.tensor(compatible[ids],device=args.device)
                loss=-torch.logsumexp(logp.masked_fill(~valid,-1e9),1).mean()
                opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);opt.step();losses.append(loss.item())
            model.eval()
            with torch.no_grad():logits=np.concatenate([model(torch.tensor(data['images'][ids],dtype=torch.float32,device=args.device)/255).cpu().numpy() for ids in np.array_split(val,9)])
            result=metrics(RELATIONS[logits.argmax(1)],data['targets'][val],data['valid'][val]);score=result['cross_axis']['f1']
            row={'epoch':epoch+1,'loss':float(np.mean(losses)),**result};history.append(row)
            if score>best:
                best=score;torch.save(model.state_dict(),run/'best.pt')
                (run/'best.json').write_text(json.dumps({'architecture':'MaskedStrokeTransformer: 36 patch tokens, 2 transformer blocks, direct 15-way legal partition head',
                    'variant':variant,'parameters':sum(p.numel() for p in model.parameters()),'device':args.device,'best':row,
                    'pretrain_data_sha256':hashlib.sha256((ROOT/'data/processed/masked-stroke/pretrain.npz').read_bytes()).hexdigest(),
                    'fine_tune_data_sha256':hashlib.sha256((ROOT/'data/processed/port-partition-expanded/relations.npz').read_bytes()).hexdigest(),
                    'seed':20261006,'selection':'original133 development cross-axis F1; no independent validation','runtime_changed':False,
                    'weight_scope':'local research only; source licenses and attribution in masked-stroke preparation; no contest promotion'},indent=2),encoding='utf-8')
            (run/'history.json').write_text(json.dumps(history,indent=2),encoding='utf-8');print(json.dumps({'variant':variant,**row}),flush=True)
        model.load_state_dict(torch.load(run/'best.pt',weights_only=True,map_location='cpu'));model.cpu().eval()
        torch.onnx.export(model,torch.zeros(1,3,48,48),str(dest/'model.onnx'),input_names=['image'],output_names=['logits'],opset_version=17,dynamic_axes={'image':{0:'batch'},'logits':{0:'batch'}},dynamo=False)
        (dest/'model.json').write_bytes((run/'best.json').read_bytes())
        x=torch.tensor(data['images'][val[:8]],dtype=torch.float32)/255
        with torch.no_grad():expected=model(x).numpy()
        np.savez_compressed(run/'export-reference.npz',image=x.numpy(),logits=expected)
    (out/'runtime.json').write_text(json.dumps({'device':args.device,'peak_allocated_bytes':torch.cuda.max_memory_allocated() if args.device=='cuda' else None,
        'peak_reserved_bytes':torch.cuda.max_memory_reserved() if args.device=='cuda' else None,'seconds':time.perf_counter()-started}),encoding='utf-8')


if __name__=='__main__':main()
