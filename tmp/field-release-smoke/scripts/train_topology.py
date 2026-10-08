"""Real junction supervision from source divergePoints and distinct interior crossings."""
import argparse
import json
import os
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
import cv2
import numpy as np
from pcbparse.dataset import index_dataset
from pcbparse.topology_perception import crossing_candidates,patch_tensor
from pcbparse.vision import read_image,wire_lines


def prepare(data,cache):
    xs=[]; geometry=[]; labels=[]; records=[]; counts={}
    for item in index_dataset(data):
        if item['id']=='0237': continue
        raw=json.loads(Path(item['annotation']).read_text(encoding='utf-8')); image=read_image(item['image'])
        mask=(cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)<220).astype(np.uint8)*255
        for c in raw['components'].values():
            b=c['location']; a,y,z,w=[int(b[k]) for k in ('x1','y1','x2','y2')]; mask[max(0,y):w,max(0,a):z]=0
        for text in raw['texts'].values():
            b=text.get('location',{})
            if b:
                a,y,z,w=[int(b[k]) for k in ('x1','y1','x2','y2')]; mask[max(0,y):w,max(0,a):z]=0
        segments,_=wire_lines(mask,split_junctions=False); visual=crossing_candidates(segments)
        positives=[]; gt_segments=[]
        for eid,edge in raw['edges'].items():
            if edge.get('state') not in ('active',None): continue
            positives.extend(edge.get('keyPoints',{}).get('divergePoints',{}).get('wire',[]))
            for path in edge.get('keyPointPaths',{}).get('wire',[]):
                for a,b in zip(path,path[1:]):
                    if a==b or (abs(a[0]-b[0])>.1 and abs(a[1]-b[1])>.1): continue
                    gt_segments.append((eid,a,b))
        # Confirm a negative only at an interior crossing of distinct annotated
        # wire groups, away from all annotated branch points and endpoints.
        negatives=[]
        horizontal=[x for x in gt_segments if x[1][1]==x[2][1]]
        vertical=[x for x in gt_segments if x[1][0]==x[2][0]]
        for ea,a,b in horizontal:
            for eb,c,d in vertical:
                if ea==eb: continue
                x,y=c[0],a[1]
                if not(min(a[0],b[0])+5<x<max(a[0],b[0])-5 and min(c[1],d[1])+5<y<max(c[1],d[1])-5): continue
                if any(np.hypot(x-p[0],y-p[1])<=5 for p in positives): continue
                negatives.append((x,y))
        seen=set()
        for candidate in visual:
            x,y=candidate['x'],candidate['y']; key=(round(x),round(y))
            if key in seen: continue
            positive=any(np.hypot(x-p[0],y-p[1])<=4 for p in positives)
            negative=not candidate['endpoint_contact'] and any(np.hypot(x-p[0],y-p[1])<=4 for p in negatives)
            if not positive and not negative: continue
            seen.add(key); label=int(positive)
            tensor=patch_tensor(image,x,y)
            xs.append(np.rint(tensor[:3]*255).clip(0,255).astype(np.uint8)); geometry.append(candidate['geometry']); labels.append(label)
            records.append({'page':item['id'],'split':item['split'],'label':label,'point':[x,y],'source':'annotated_diverge_point' if positive else 'distinct_wire_groups_interior_cross'})
            count_key=item['split']+'/'+str(label); counts[count_key]=counts.get(count_key,0)+1
        print(json.dumps({'page':item['id'],'samples':len(records)}),flush=True)
    if set(labels)!={0,1}: raise ValueError('insufficient real positive/negative junction supervision')
    cache.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(cache/'patches.npz',images=np.stack(xs),geometry=np.asarray(geometry,np.float32),labels=np.asarray(labels,np.float32))
    (cache/'records.json').write_text(json.dumps(records),encoding='utf-8')
    (cache/'preparation.json').write_text(json.dumps({'counts':counts,'samples':len(records),'excluded_pages':['0237'],'labels':'weak source-annotation supervision; uncertain/local-contact cases skipped','color_policy':'color is not a class label'},indent=2),encoding='utf-8')
    print(json.dumps(counts),flush=True)


def train(args):
    import torch
    from torch.utils.data import Dataset,DataLoader
    from pcbparse.topology_model import TopologyNet
    torch.set_num_threads(4); torch.manual_seed(20261004); np.random.seed(20261004)
    cache=Path(args.cache)
    if not (cache/'patches.npz').exists(): prepare(args.data,cache)
    with np.load(cache/'patches.npz') as archive:
        raw={key:archive[key] for key in ('images','geometry','labels')}
    records=json.loads((cache/'records.json').read_text(encoding='utf-8'))
    class Patches(Dataset):
        def __init__(self,split): self.ids=[i for i,r in enumerate(records) if r['split']==split]; self.split=split
        def __len__(self): return len(self.ids)
        def __getitem__(self,j):
            i=self.ids[j]; rgb=raw['images'][i].transpose(1,2,0).astype(np.float32)/255
            geometry=raw['geometry'][i].copy()
            if self.split=='train':
                if np.random.random()<.5:
                    rgb=np.rot90(rgb).copy(); geometry=geometry[[2,3,1,0]]
                if np.random.random()<.3: geometry[:]=.5
            gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
            # Frequent grayscale augmentation breaks a color shortcut.
            if self.split=='train' and np.random.random()<.6: rgb=np.repeat(gray[:,:,None],3,axis=2)
            ink=(gray<.86).astype(np.float32)
            h=cv2.morphologyEx(ink,cv2.MORPH_OPEN,np.ones((1,7),np.uint8)); v=cv2.morphologyEx(ink,cv2.MORPH_OPEN,np.ones((7,1),np.uint8))
            tensor=np.concatenate((rgb.transpose(2,0,1),gray[None],h[None],v[None]))
            return tensor,geometry,raw['labels'][i]
    training=Patches('train'); validation=Patches('validation')
    if not len(training) or not len(validation): raise ValueError('both page splits required')
    train_loader=DataLoader(training,batch_size=64,shuffle=True,num_workers=0); val_loader=DataLoader(validation,batch_size=128,num_workers=0)
    model=TopologyNet().cuda(); optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    if args.weights: model.load_state_dict(torch.load(args.weights,weights_only=True))
    ys=raw['labels'][training.ids]; positive_weight=float((ys==0).sum()/max(1,(ys==1).sum()))
    loss_fn=torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([positive_weight],device='cuda'))
    out=ROOT/'output/training/topology-perception'; out.mkdir(parents=True,exist_ok=True); history=[]; best=-1; started=time.perf_counter()
    for epoch in range(args.epochs):
        model.train(); loss_sum=0
        for x,g,y in train_loader:
            loss=loss_fn(model(x.cuda(),g.cuda()).flatten(),y.cuda()); optimizer.zero_grad(); loss.backward(); optimizer.step(); loss_sum+=loss.item()
        model.eval(); probabilities=[]; truth=[]
        with torch.no_grad():
            for x,g,y in val_loader: probabilities.extend(model(x.cuda(),g.cuda()).sigmoid().flatten().cpu().tolist()); truth.extend(y.tolist())
        probabilities=np.asarray(probabilities); truth=np.asarray(truth); pred=probabilities>=.5
        tp=int(((truth==1)&pred).sum()); fp=int(((truth==0)&pred).sum()); fn=int(((truth==1)&~pred).sum()); tn=int(((truth==0)&~pred).sum())
        metrics={'epoch':epoch+1,'loss':loss_sum/len(train_loader),'seconds':time.perf_counter()-started,'f1':2*tp/max(1,2*tp+fp+fn),'precision':tp/max(1,tp+fp),'recall':tp/max(1,tp+fn),'tp':tp,'fp':fp,'fn':fn,'tn':tn}
        history.append(metrics); (out/'history.json').write_text(json.dumps(history,indent=2),encoding='utf-8')
        if metrics['f1']>best:
            best=metrics['f1']; torch.save(model.state_dict(),out/'best.pt')
            connect=1.01; disconnect=-.01
            for threshold in np.arange(.5,.991,.01):
                selected=probabilities>=threshold
                if selected.sum()>=10 and (truth[selected]==1).mean()>=.97: connect=float(threshold); break
            for threshold in np.arange(.5,.009,-.01):
                selected=probabilities<=threshold
                if selected.sum()>=10 and (truth[selected]==0).mean()>=.97: disconnect=float(threshold); break
            (out/'best.json').write_text(json.dumps({'architecture':'appearance encoder + geometric arm MLP; fused local electrical-contact classifier','module':'TopologyPerception','size':48,'radius':16,'connect_threshold':connect,'disconnect_threshold':disconnect,'train_samples':len(training),'validation_samples':len(validation),'metrics':metrics,'supervision':'source divergePoints positives and distinct interior-wire crossings negatives; weak annotations; not official ground truth','color_augmentation':'60% grayscale train patches','evaluation':'page-held-out validation used for checkpoint and threshold selection, not independent test'},indent=2),encoding='utf-8')
        print(json.dumps(metrics),flush=True)
    model.load_state_dict(torch.load(out/'best.pt',weights_only=True)); model.cpu().eval()
    target=ROOT/'models/topology'; target.mkdir(parents=True,exist_ok=True)
    torch.onnx.export(model,(torch.zeros(1,6,48,48),torch.zeros(1,4)),str(target/'topology-perception.onnx'),input_names=['patch','geometry'],output_names=['logit'],opset_version=17)
    (target/'topology-perception.json').write_bytes((out/'best.json').read_bytes())


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--data',default='data/raw/PCBnet/Dataset300'); p.add_argument('--cache',default='data/processed/topology'); p.add_argument('--epochs',type=int,default=25); p.add_argument('--weights'); p.add_argument('--prepare-only',action='store_true'); args=p.parse_args()
    if args.prepare_only: prepare(args.data,Path(args.cache))
    else: train(args)
