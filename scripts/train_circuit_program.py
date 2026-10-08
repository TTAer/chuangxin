"""New whole-raster -> executable-program run; no old model checkpoints."""
import json,sys,time,hashlib,argparse
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,torch
from torch import nn
from experiments.circuit_program import CircuitProgramNet,MASK,LENGTH
from experiments.circuit_program_anchored import AnchoredCircuitProgramNet
from experiments.circuit_program_hierarchical import HierarchicalCircuitProgramNet
from scripts.prepare_circuit_program import execute,render_program

def metrics(prediction,gold):
    records=[];tp=fp=fn=ntp=nfp=nfn=0;invalid=0
    for p,g in zip(prediction,gold):
        gv,gt,gc,groups=execute(g)
        try:pv,pt,pc,pgroup=execute(p)
        except ValueError:invalid+=1;pv=np.zeros(42,bool);pt=np.zeros((42,42));pc=[];pgroup=[]
        upper=np.triu(np.ones((42,42),bool),1);mask=(pv|gv)[:,None]&(pv|gv)[None,:]&upper
        tp+=int(((pt>.5)&(gt>.5)&mask).sum());fp+=int(((pt>.5)&(gt<=.5)&mask).sum());fn+=int(((pt<=.5)&(gt>.5)&mask).sum())
        ntp+=int((pv&gv).sum());nfp+=int((pv&~gv).sum());nfn+=int((~pv&gv).sum())
        true_components={(tuple(c['slot']),c['kind']) for c in gc};pred_components={(tuple(c['slot']),c['kind']) for c in pc}
        records.append({'exact_net_partition':float({frozenset(x) for x in pgroup}=={frozenset(x) for x in groups}),
                        'exact_component_set':float(pred_components==true_components),'exact_program':float(np.array_equal(p,g)),
                        'counts_correct':float(np.array_equal(p[:2],g[:2])),'token_accuracy':float(np.mean(p==g)),
                        'component_TP':len(true_components&pred_components),'component_FP':len(pred_components-true_components),'component_FN':len(true_components-pred_components)})
    ct=sum(r['component_TP'] for r in records);cf=sum(r['component_FP'] for r in records);cn=sum(r['component_FN'] for r in records)
    return {'pages':len(records),'invalid_programs':invalid,'net_pair':{'TP':tp,'FP':fp,'FN':fn,'micro_F1':2*tp/max(1,2*tp+fp+fn)},
            'wire_node_F1':2*ntp/max(1,2*ntp+nfp+nfn),'component_type_and_slot_F1':2*ct/max(1,2*ct+cf+cn),
            **{k:float(np.mean([r[k] for r in records])) for k in ('exact_net_partition','exact_component_set','exact_program','counts_correct','token_accuracy')}}

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=ROOT/'output/training/circuit-program-v1');parser.add_argument('--bank',type=Path,default=ROOT/'data/processed/circuit-program-v1')
    parser.add_argument('--warm-image-checkpoint',type=Path);parser.add_argument('--pretrain-epochs',type=int,default=4);parser.add_argument('--tiny-mixed',action='store_true');parser.add_argument('--anchored',action='store_true');parser.add_argument('--hierarchical',action='store_true');parser.add_argument('--tiny-best',action='store_true');parser.add_argument('--tiny-cosine',action='store_true');parser.add_argument('--seed',type=int,default=20261007);args=parser.parse_args()
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    if (out/'protocol.json').exists():raise ValueError('Preserve this run')
    torch.set_num_threads(4);torch.manual_seed(args.seed);torch.backends.cudnn.enabled=True;torch.backends.cudnn.benchmark=False;torch.cuda.set_per_process_memory_fraction(.55)
    bank=args.bank.resolve();manifest=json.loads((bank/'manifest.json').read_text());rng=np.random.default_rng(args.seed)
    protocol={'model':'Hierarchical renderer-anchored CircuitProgramNet' if args.hierarchical else ('Renderer-anchored CircuitProgramNet' if args.anchored else 'CircuitProgramNet whole-raster masked instruction denoiser'),'device':torch.cuda.get_device_name(),'seed':args.seed,
              'initialization':'fresh random weights; multisource image-only encoder pretraining in this run, no prior checkpoint',
              'pretrain_epochs':args.pretrain_epochs,'pretrain_batch':128,'tiny_steps':400,'tiny_corruption':'same all-mask/partial mixture as main' if args.tiny_mixed else 'all MASK only','tiny_gate':'all four exact programs and exact net partitions correct at final step',
              'main_epochs':12,'main_batch':32,'lr':.001,'decode_steps':6,'grammar':'68 typed instructions, counts + crossing + vertical wire/R/C interval',
              'selection':'highest validation net-pair micro F1, then exact-program, then lower all-mask CE; no stress selection',
              'training_corruption':'50% batches all MASK; otherwise independent tokens masked with ratio uniform0.3:1',
              'augmentation':'30% probability 3x3 mean blur; compiler brightness/noise; no full-page rotation because grammar does not encode horizontal components',
              'loss':'masked-token weighted CE; JOIN/R/C weight3, count weight2, ABSENT weight0.25',
              'data_manifest_sha256':hashlib.sha256((bank/'manifest.json').read_bytes()).hexdigest(),'source_patch_counts':manifest['image_patch_counts'],
              'synthetic_program_cohorts':manifest['program_layouts'],'geometry_oracle_supplied':False,'external_net_labels_used':False,'RC0_changed':False}
    if args.tiny_best:protocol['tiny_gate']='at least one preregistered 50-step checkpoint with all four exact programs and partitions correct; tiny diagnostic not an independent score'
    protocol['tiny_optimizer_schedule']='cosine .001 to .0001' if args.tiny_cosine else 'constant .002'
    if args.warm_image_checkpoint:protocol.update(initialization='Fresh program decoder; encoder/projection only from this request image-only pretraining, no supervised checkpoint',warm_image_checkpoint_sha256=hashlib.sha256(args.warm_image_checkpoint.read_bytes()).hexdigest(),warm_keys=['encoder.*','projection.*'])
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8');started=time.perf_counter();model=(HierarchicalCircuitProgramNet() if args.hierarchical else (AnchoredCircuitProgramNet() if args.anchored else CircuitProgramNet())).cuda()
    if args.warm_image_checkpoint:
        warm=torch.load(args.warm_image_checkpoint,weights_only=True,map_location='cuda');model.load_state_dict({k:v for k,v in warm.items() if k.startswith(('encoder.','projection.'))},strict=False)
    with np.load(bank/'multisource-image-only.npz') as z:pretrain=torch.tensor(z['images'],device='cuda').float()/255
    optimizer=torch.optim.AdamW(list(model.encoder.parameters())+list(model.projection.parameters()),lr=.001);scaler=torch.amp.GradScaler('cuda');prehistory=[];pre_steps=0
    def augment(x):
        if rng.random()<.5:x=x.flip(-1)
        if rng.random()<.5:x=x.flip(-2)
        x=torch.rot90(x,int(rng.integers(4)),(-2,-1));return (1-x*float(rng.uniform(.8,1.15))+torch.randn_like(x)*.015).clamp(0,1)
    for epoch in range(args.pretrain_epochs):
        model.train();losses=[]
        for indices in np.array_split(rng.permutation(len(pretrain)),int(np.ceil(len(pretrain)/128))):
            if len(indices)<2:continue
            x=pretrain[indices];views=torch.cat((augment(x),augment(x)));n=len(x)
            with torch.autocast('cuda',dtype=torch.float16):
                feature=model.encoder(views).mean((-2,-1));z=nn.functional.normalize(model.projection(feature).float(),dim=-1);sim=z@z.T/.2;sim.fill_diagonal_(-1e4)
                target=(torch.arange(2*n,device='cuda')+n)%(2*n);loss=nn.functional.cross_entropy(sim,target)
            optimizer.zero_grad();scaler.scale(loss).backward();scaler.step(optimizer);scaler.update();losses.append(float(loss.detach()));pre_steps+=1
        row={'epoch':epoch+1,'loss':float(np.mean(losses))};prehistory.append(row);print(json.dumps({'phase':'image-only-pretrain',**row}),flush=True)
    torch.save(model.state_dict(),out/'image-pretrained.pt');(out/'pretraining.json').write_text(json.dumps({'history':prehistory,'optimizer_steps':pre_steps,'patches':len(pretrain)},indent=2),encoding='utf-8');del pretrain;torch.cuda.empty_cache()
    initial={k:v.clone() for k,v in model.state_dict().items()};data={}
    for name in ('tiny','train','validation','stress'):
        with np.load(bank/(name+'.npz')) as z:data[name]={'images':torch.tensor(z['images'],device='cuda').float()/255,'program':torch.tensor(z['program'],device='cuda')}
        data[name]['images']=1-data[name]['images']
    weights=torch.ones(14,device='cuda');weights[3:7]=2;weights[7]=.25;weights[[9,11,12]]=3
    def token_loss(d,corrupt=True,augment=False):
        target=d['program'];masked=torch.ones_like(target,dtype=torch.bool)
        if corrupt and rng.random()<.5:masked=torch.rand(target.shape,device='cuda')<float(rng.uniform(.3,1))
        tokens=target.masked_fill(masked,MASK);images=d['images']
        if augment and rng.random()<.3:images=nn.functional.avg_pool2d(nn.functional.pad(images,(1,1,1,1),mode='replicate'),3,1)
        with torch.autocast('cuda',dtype=torch.float16):
            logits=model(images,tokens);ce=nn.functional.cross_entropy(logits.transpose(1,2).float(),target,weight=weights,reduction='none');loss=(ce*masked).sum()/(weights[target]*masked).sum().clamp_min(1)
        return loss
    def evaluate(name,save=False):
        model.eval();predictions=[];raw_predictions=[];losses=[]
        with torch.no_grad():
            for i in range(0,len(data[name]['images']),16):
                d={k:v[i:i+16] for k,v in data[name].items()}
                with torch.autocast('cuda',dtype=torch.float16):
                    p=model.generate(d['images'],6);raw=model(d['images'],torch.full_like(d['program'],MASK)).argmax(-1)
                predictions.append(p.cpu().numpy());raw_predictions.append(raw.cpu().numpy());losses.append(float(token_loss(d,False)))
        gold=data[name]['program'].cpu().numpy();pred=np.concatenate(predictions);raw=np.concatenate(raw_predictions);result=metrics(pred,gold);result['all_mask_CE']=float(np.mean(losses));result['raw_onepass']=metrics(raw,gold)
        if save:np.savez_compressed(out/(name+'-predictions.npz'),program=pred,raw_onepass=raw)
        return result
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001 if args.tiny_cosine else .002);tiny=[]
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,400,eta_min=.0001) if args.tiny_cosine else None
    for step in range(400):
        model.train();loss=token_loss(data['tiny'],args.tiny_mixed);optimizer.zero_grad();scaler.scale(loss).backward();scaler.unscale_(optimizer);nn.utils.clip_grad_norm_(model.parameters(),5);scaler.step(optimizer);scaler.update()
        if scheduler:scheduler.step()
        if (step+1)%50==0:
            result=evaluate('tiny');row={'step':step+1,**result};tiny.append(row);print(json.dumps({'phase':'tiny',**row}),flush=True)
    eligible=tiny if args.tiny_best else tiny[-1:];passed=any(r['exact_program']==1 and r['exact_net_partition']==1 for r in eligible)
    (out/'tiny.json').write_text(json.dumps({'gate_passed':passed,'history':tiny},indent=2),encoding='utf-8');torch.save(model.state_dict(),out/'tiny.pt')
    if not passed:
        (out/'summary.json').write_text(json.dumps({'tiny_gate_passed':False,'full_training_executed':False,'pretrain_steps':pre_steps,'seconds':time.perf_counter()-started,'RC0_changed':False},indent=2),encoding='utf-8');return
    model.load_state_dict(initial);optimizer=torch.optim.AdamW(model.parameters(),lr=.001);history=[];best=(-1,-1,float('-inf'));main_steps=0
    for epoch in range(12):
        losses=[]
        for indices in rng.permutation(len(data['train']['images'])).reshape(-1,32):
            d={k:v[indices] for k,v in data['train'].items()};model.train();loss=token_loss(d,True,True)
            optimizer.zero_grad();scaler.scale(loss).backward();scaler.unscale_(optimizer);nn.utils.clip_grad_norm_(model.parameters(),5);scaler.step(optimizer);scaler.update()
            if not torch.isfinite(loss):raise ValueError('Nonfinite program loss')
            losses.append(float(loss.detach()));main_steps+=1
        validation=evaluate('validation');row={'epoch':epoch+1,'training_loss':float(np.mean(losses)),**validation};history.append(row);print(json.dumps(row),flush=True)
        key=(validation['net_pair']['micro_F1'],validation['exact_program'],-validation['all_mask_CE'])
        if key>best:best=key;torch.save(model.state_dict(),out/'best.pt');(out/'best.json').write_text(json.dumps(row,indent=2),encoding='utf-8')
    model.load_state_dict(torch.load(out/'best.pt',weights_only=True));model.eval()
    result={'tiny_gate_passed':True,'full_training_executed':True,'validation':evaluate('validation',True),'stress':evaluate('stress',True),
            'parameters':sum(p.numel() for p in model.parameters()),'pretrain_optimizer_steps':pre_steps,'main_optimizer_steps':main_steps,'tiny_optimizer_steps':400,
            'seconds':time.perf_counter()-started,'CUDA_peak_allocated_bytes':torch.cuda.max_memory_allocated(),'external_image_pretraining':True,'real_net_supervised_training':False,'independent_validation':False,'RC0_changed':False}
    (out/'history.json').write_text(json.dumps(history,indent=2),encoding='utf-8');(out/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps({'finished':True,**result}),flush=True)

if __name__=='__main__':main()
