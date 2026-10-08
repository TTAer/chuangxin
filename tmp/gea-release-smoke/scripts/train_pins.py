"""Train a local coordinate-conditioned tip/base network; page splits stay fixed."""
import argparse
import json
import os
from pathlib import Path
import random
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import cv2
import numpy as np
from pcbparse.dataset import index_dataset
from pcbparse.pins import crop_tensor, peaks
from pcbparse.vision import read_image


def prepare(root, destination):
    xs=[]; records=[]
    for item in index_dataset(root):
        raw=json.loads(Path(item['annotation']).read_text(encoding='utf-8'))
        image=read_image(item['image'])
        for cid,c in raw['components'].items():
            pins=list(c.get('pins',{}).values())
            if not pins or c.get('state') not in ('active',None): continue
            b=[c['location'][k] for k in ('x1','y1','x2','y2')]
            if min(b[2]-b[0],b[3]-b[1])<3: continue
            tensor,box=crop_tensor(image,b)
            targets=[]
            for channel,key in enumerate(('tip','base')):
                seen=set()
                for pin in pins:
                    point=pin.get(key)
                    if not isinstance(point,list) or len(point)!=2: continue
                    x=(point[0]-box[0])/(box[2]-box[0])*96-.5
                    y=(point[1]-box[1])/(box[3]-box[1])*96-.5
                    if not (0<=x<96 and 0<=y<96) or (x,y) in seen: continue
                    seen.add((x,y)); targets.append([channel,x,y])
            if not any(p[0]==0 for p in targets): continue
            xs.append(np.rint(tensor[:4]*255).clip(0,255).astype(np.uint8))
            records.append({'page':item['id'],'split':item['split'],'category':c['category'],'box':box,'targets':targets})
    destination.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(destination/'crops.npz',images=np.stack(xs))
    (destination/'records.json').write_text(json.dumps(records,ensure_ascii=False),encoding='utf-8')
    print(json.dumps({'crops':len(records),'train':sum(r['split']=='train' for r in records),'validation':sum(r['split']=='validation' for r in records)}),flush=True)


def train(args):
    import torch
    import torch.nn.functional as F
    from torch.utils.data import Dataset,DataLoader
    from pcbparse.pin_model import PinHeatmapNet
    torch.set_num_threads(4); torch.manual_seed(20261004); np.random.seed(20261004); random.seed(20261004)
    directory=Path(args.cache)
    if not (directory/'crops.npz').exists(): prepare(args.data,directory)
    images=np.load(directory/'crops.npz')['images']
    records=json.loads((directory/'records.json').read_text(encoding='utf-8'))
    yy,xx=np.mgrid[:96,:96].astype(np.float32)
    coords=np.stack((xx/95*2-1,yy/95*2-1))
    class Crops(Dataset):
        def __init__(self,split): self.ids=[i for i,r in enumerate(records) if r['split']==split]; self.split=split
        def __len__(self): return len(self.ids)
        def __getitem__(self,j):
            i=self.ids[j]; x=images[i].astype(np.float32)/255
            if self.split=='train':
                if random.random()<.3:
                    gray=x[:3].mean(axis=0); x[:3]=gray[None]
                x[:3]=np.clip(x[:3]*random.uniform(.8,1.1),0,1)
            targets=np.zeros((2,96,96),np.float32); valid=np.zeros(2,np.float32)
            for channel,a,b in records[i]['targets']:
                a=int(round(a)); b=int(round(b)); a=min(95,a); b=min(95,b)
                heat=np.exp(-((xx-a)**2+(yy-b)**2)/(2*1.5**2))
                targets[channel]=np.maximum(targets[channel],heat); valid[channel]=1
            return np.concatenate((x,coords)),targets,valid,i
    training=Crops('train'); validation=Crops('validation')
    train_loader=DataLoader(training,batch_size=args.batch,shuffle=True,num_workers=0)
    val_loader=DataLoader(validation,batch_size=args.batch,num_workers=0)
    device=torch.device(args.device); model=PinHeatmapNet().to(device)
    if args.warm_start: model.load_state_dict(torch.load(args.warm_start,map_location=device,weights_only=True))
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,args.epochs,eta_min=.00003)
    out=ROOT/args.output; out.mkdir(parents=True,exist_ok=True)
    history=[]; best=-1; started=time.perf_counter()
    for epoch in range(args.epochs):
        model.train(); loss_sum=0
        for x,t,v,_ in train_loader:
            x=x.to(device); t=t.to(device); v=v.to(device)[:,:,None,None]
            logits=model(x); prob=logits.sigmoid()
            positive=(t==1).float(); negative=(t<1).float()
            loss=(-F.logsigmoid(logits)*(1-prob)**2*positive-F.logsigmoid(-logits)*prob**2*(1-t)**4*negative)*v
            loss=loss.sum()/((positive*v).sum().clamp(min=1))
            optimizer.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),5); optimizer.step()
            loss_sum+=loss.item()
        scheduler.step()
        model.eval(); totals={t:[0,0,0] for t in (.15,.25,.35,.5)}
        with torch.no_grad():
            for x,_,_,ids in val_loader:
                maps=model(x.to(device)).sigmoid().cpu().numpy()
                for heat,i in zip(maps,ids.tolist()):
                    r=records[i]; bw=r['box'][2]-r['box'][0]; bh=r['box'][3]-r['box'][1]
                    truth=[(a,b) for c,a,b in r['targets'] if c==0]
                    for threshold,counts in totals.items():
                        predictions=peaks(heat[0],threshold)
                        pairs=sorted(((((a-u)*bw/96)**2+((b-v)*bh/96)**2)**.5,j,k) for j,(a,b,_) in enumerate(predictions) for k,(u,v) in enumerate(truth))
                        used_p=set(); used_g=set()
                        for distance,j,k in pairs:
                            if distance<=5 and j not in used_p and k not in used_g: used_p.add(j); used_g.add(k)
                        counts[0]+=len(used_p); counts[1]+=len(predictions); counts[2]+=len(truth)
        metrics={t:{'precision':a/max(1,b),'recall':a/max(1,c),'f1':2*a/max(1,b+c),'matched':a,'predicted':b,'golden':c} for t,(a,b,c) in totals.items()}
        threshold=max(metrics,key=lambda t:metrics[t]['f1']); metric=metrics[threshold]
        entry={'epoch':epoch+1,'loss':loss_sum/len(train_loader),'seconds':time.perf_counter()-started,'threshold':threshold,**metric}
        history.append(entry); (out/'history.json').write_text(json.dumps(history,indent=2),encoding='utf-8')
        if metric['f1']>best:
            best=metric['f1']; torch.save(model.state_dict(),out/'best.pt')
            (out/'best.json').write_text(json.dumps({'size':96,'margin':.15,'threshold':threshold,'architecture':'coordinate-conditioned dual tip/base heatmaps with skip connections','evaluation':'oracle component boxes on held-out PCBnet pages; 5px location tolerance; not end-to-end pin-number F1','train_crops':len(training),'validation_crops':len(validation),'metrics':entry,'threshold_metrics':metrics},indent=2),encoding='utf-8')
            preparation=directory/'preparation.json'
            if preparation.exists():
                metadata=json.loads((out/'best.json').read_text(encoding='utf-8'))
                metadata['data_preparation']=json.loads(preparation.read_text(encoding='utf-8'))
                metadata['evaluation']='oracle training-cache boxes on held-out pages; 5px location tolerance; validation selects checkpoint/threshold; not end-to-end pin-number F1'
                (out/'best.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
        print(json.dumps(entry),flush=True)
    model.load_state_dict(torch.load(out/'best.pt',weights_only=True)); model.cpu().eval()
    target=ROOT/args.model_output; target.mkdir(parents=True,exist_ok=True)
    torch.onnx.export(model,torch.zeros(1,6,96,96),str(target/'pin-heatmap.onnx'),input_names=['crop'],output_names=['heatmaps'],opset_version=17)
    (target/'pin-heatmap.json').write_bytes((out/'best.json').read_bytes())


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--data',default='data/raw/PCBnet/Dataset300'); p.add_argument('--cache',default='data/processed/pins'); p.add_argument('--epochs',type=int,default=20); p.add_argument('--batch',type=int,default=64); p.add_argument('--device',default='cuda:0'); p.add_argument('--prepare-only',action='store_true')
    p.add_argument('--warm-start'); p.add_argument('--output',default='output/training/pin-heatmap'); p.add_argument('--model-output',default='models/pins')
    args=p.parse_args()
    if args.prepare_only: prepare(args.data,Path(args.cache))
    else: train(args)
