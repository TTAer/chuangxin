"""Train partial-supervision directional fields with centerline continuity."""
import argparse
import json
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
import cv2
import numpy as np

def train(args):
    import torch
    import torch.nn.functional as F
    from torch.utils.data import Dataset,DataLoader
    from pcbparse.field_model import DirectionalFieldNet,centerline_loss
    from pcbparse.field_perception import field_input
    torch.set_num_threads(4); torch.manual_seed(20261005); rng=np.random.default_rng(20261005)
    images=[]; targets=[]; records=[]; preparation=[]; cache=Path(args.cache)
    for source in args.sources.split(','):
        with np.load(cache/(source+'.npz')) as z: images.extend(z['images']); targets.extend(z['targets'])
        records.extend(json.loads((cache/(source+'-records.json')).read_text(encoding='utf-8')))
        preparation.append(json.loads((cache/(source+'-preparation.json')).read_text(encoding='utf-8')))
    images=np.stack(images); targets=np.stack(targets)
    class Tiles(Dataset):
        def __init__(self,split): self.ids=[i for i,r in enumerate(records) if r['split']==split]; self.split=split
        def __len__(self): return len(self.ids)
        def __getitem__(self,j):
            i=self.ids[j]; image=images[i].transpose(1,2,0).copy(); label=targets[i].astype(np.float32)/255; valid=np.asarray(records[i]['valid_heads'],np.float32)
            if self.split=='train':
                # 90-degree rotation swaps horizontal and vertical fields.
                if rng.random()<.5:
                    image=np.rot90(image).copy(); label=np.rot90(label,axes=(1,2)).copy(); label[[0,1]]=label[[1,0]]
                if rng.random()<.3:
                    image=cv2.cvtColor(cv2.cvtColor(image,cv2.COLOR_BGR2GRAY),cv2.COLOR_GRAY2BGR)
                if rng.random()<.3:
                    image=np.clip(image.astype(np.float32)*rng.uniform(.8,1.1),0,255).astype(np.uint8)
            return field_input(image),label,valid,i
    training=Tiles('train'); validation=Tiles('validation')
    train_loader=DataLoader(training,batch_size=args.batch,shuffle=True,num_workers=0); val_loader=DataLoader(validation,batch_size=args.batch,num_workers=0)
    device='cuda' if torch.cuda.is_available() else 'cpu'; model=DirectionalFieldNet(strip=not args.local_only).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001); scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,args.epochs,eta_min=.00005)
    out=Path(args.output); out.mkdir(parents=True,exist_ok=True); history=[]; best=-1; started=time.perf_counter()
    weights=torch.tensor([8,8,12,12,2],device=device).view(1,5,1,1)
    channel_weights=torch.tensor([1,1,1,1,.25],device=device).view(1,5)
    for epoch in range(args.epochs):
        model.train(); losses=[]
        for x,target,valid,_ in train_loader:
            x=x.to(device); target=target.to(device); valid=valid.to(device)
            logits=model(x); probability=logits.sigmoid()
            bce=F.binary_cross_entropy_with_logits(logits,target,pos_weight=weights,reduction='none').mean((2,3))
            # Dice applies to broad wire/ink masks, not sparse heatmap labels.
            dice=1-(2*(probability*target).sum((2,3))+1)/(probability.sum((2,3))+target.sum((2,3))+1)
            loss=((bce+.3*dice)*valid*channel_weights).sum()/(valid*channel_weights).sum().clamp(min=1)
            if not args.no_continuity:
                active=valid[:,0]*valid[:,1]
                wire_p=probability[:,:2].amax(1,keepdim=True); wire_t=target[:,:2].amax(1,keepdim=True)
                loss+=.1*(centerline_loss(wire_p,wire_t)*active).sum()/active.sum().clamp(min=1)
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),3); opt.step(); losses.append(loss.item())
        scheduler.step(); model.eval(); counts={}; thresholds=[.3,.5,.7]
        with torch.no_grad():
            for x,t,v,ids in val_loader:
                probs=model(x.to(device)).sigmoid().cpu().numpy(); truth=t.numpy(); valid=v.numpy()
                for i,prob,label,available in zip(ids.tolist(),probs,truth,valid):
                    source=records[i]['source']
                    for threshold in thresholds:
                        for channel in (0,1,4):
                            if not available[channel]: continue
                            key=f'{source}/{channel}/{threshold}'; count=counts.setdefault(key,np.zeros(3,np.int64))
                            pp=prob[channel]>=threshold; tt=label[channel]>=.5
                            count+=np.asarray([(pp&tt).sum(),pp.sum(),tt.sum()])
        metrics={k:{'pixel_f1':float(2*c[0]/max(1,c[1]+c[2])),'precision':float(c[0]/max(1,c[1])),'recall':float(c[0]/max(1,c[2])),'matched':int(c[0]),'predicted':int(c[1]),'golden':int(c[2])} for k,c in counts.items()}
        threshold=max(thresholds,key=lambda t:sum(metrics[f'pcbnet/{c}/{t}']['pixel_f1'] for c in (0,1)))
        score=sum(metrics[f'pcbnet/{c}/{threshold}']['pixel_f1'] for c in (0,1))/2
        row={'epoch':epoch+1,'loss':float(np.mean(losses)),'seconds':time.perf_counter()-started,'pcb_directional_pixel_f1':score,'threshold':threshold}; history.append(row)
        if score>best:
            best=score; torch.save(model.state_dict(),out/'best.pt')
            meta={'architecture':'DirectionalFieldNet: gated horizontal/vertical depthwise strip context + multi-scale decoder + partial-supervised dense fields','size':192,'wire_threshold':threshold,'heads':['horizontal_wire','vertical_wire','pin_tip','junction','ink'],'strip_context':not args.local_only,'centerline_loss':not args.no_continuity,'train_tiles':len(training),'validation_tiles':len(validation),'parameters':sum(p.numel() for p in model.parameters()),'metrics':metrics,'best_epoch':row,'data_preparation':preparation,'evaluation':'PCB held-out-page tile pixel metrics; validation selects checkpoint and threshold; not end-to-end accuracy','keypoint_evaluation':'tip/junction heads trained, but pixel metrics do not establish keypoint detection quality','CGHD_supervision':'ink head only; no wire or electrical-junction labels from binary strokes'}
            (out/'best.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
            if 'cghd' in args.sources.split(','):
                meta['external_attribution']={'author':'Johannes Bayer','source':'https://zenodo.org/records/17469897','modification':'resized/cropped CGHD ink-only auxiliary training','metadata_license':'CC BY 4.0','archive_readme_license':'CC BY-SA 3.0','weight_license':'CC BY-SA 3.0','notice':'docs/NEW_DATA_SOURCES.md'}
                (out/'best.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
        (out/'history.json').write_text(json.dumps(history,indent=2),encoding='utf-8'); print(json.dumps(row),flush=True)
    model.load_state_dict(torch.load(out/'best.pt',weights_only=True)); model.cpu().eval(); dest=Path(args.model_output); dest.mkdir(parents=True,exist_ok=True)
    torch.onnx.export(model,torch.zeros(1,3,192,192),str(dest/'field.onnx'),input_names=['image'],output_names=['fields'],opset_version=17,dynamic_axes={'image':{0:'batch'},'fields':{0:'batch'}})
    (dest/'field.json').write_bytes((out/'best.json').read_bytes())
    # Reference used by CPU-only export verification; no labels enter inference.
    x=field_input(images[validation.ids[0]].transpose(1,2,0))[None]
    with torch.no_grad(): expected=model(torch.from_numpy(x)).numpy()
    np.savez_compressed(out/'export-reference.npz',image=x,fields=expected)

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--cache',default='data/processed/fields'); p.add_argument('--sources',default='pcbnet,synthetic,cghd'); p.add_argument('--epochs',type=int,default=8); p.add_argument('--batch',type=int,default=16); p.add_argument('--local-only',action='store_true'); p.add_argument('--no-continuity',action='store_true'); p.add_argument('--output',default='output/training/directional-fields'); p.add_argument('--model-output',default='models/candidates/directional-fields'); a=p.parse_args(); train(a)
