"""Train side-axis endpoint responses and normal offsets; fixed page splits."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import sys
import time
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader
from pcbparse.side_strip_model import SideStripNet
from pcbparse.side_strips import tensor

def main():
    p=argparse.ArgumentParser(); p.add_argument('--epochs',type=int,default=6); p.add_argument('--synthetic',action='store_true'); p.add_argument('--local',action='store_true'); p.add_argument('--output',required=True); p.add_argument('--data-dir',default='data/processed/side-strips'); a=p.parse_args()
    seed=20261006; torch.manual_seed(seed); random.seed(seed); np.random.seed(seed); torch.set_num_threads(4)
    directory=ROOT/a.data_dir; images=np.load(directory/'bands.npz')['gray']; records=json.loads((directory/'records.json').read_text(encoding='utf-8'))
    axis=np.arange(256,dtype=np.float32)
    class Bands(Dataset):
        def __init__(self,split): self.ids=[i for i,r in enumerate(records) if r['split']==split and (r['source']=='official' or a.synthetic)]; self.split=split
        def __len__(self): return len(self.ids)
        def __getitem__(self,j):
            i=self.ids[j]; patch=images[i].copy(); r=records[i]
            if self.split=='train':
                patch=np.clip(patch.astype(np.float32)*random.uniform(.9,1.05),0,255).astype(np.uint8)
            target=np.zeros(256,np.float32); normal=np.zeros(256,np.float32); weight=np.zeros(256,np.float32)
            for position,offset in r['targets']:
                heat=np.exp(-((axis-position)**2)/(2*1.5**2)); mask=heat>target
                normal[mask]=offset/32; target=np.maximum(target,heat); weight=np.maximum(weight,heat)
            return tensor(patch),target,normal,weight,i
    train=Bands('train'); validation=Bands('validation'); loader=DataLoader(train,batch_size=48,shuffle=True,num_workers=0); vl=DataLoader(validation,batch_size=48,num_workers=0)
    model=SideStripNet(dilated=not a.local).cuda(); optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    out=ROOT/a.output; out.mkdir(parents=True,exist_ok=True); history=[]; best=-1; started=time.perf_counter()
    for epoch in range(a.epochs):
        model.train(); losses=[]
        for x,target,normal,weight,_ in loader:
            x=x.cuda(); target=target.cuda(); normal=normal.cuda(); weight=weight.cuda(); y=model(x)
            probability=y[:,0].sigmoid(); positive=(target>.95).float(); negative=1-positive
            detection=(-F.logsigmoid(y[:,0])*(1-probability)**2*positive-F.logsigmoid(-y[:,0])*probability**2*(1-target)**4*negative).sum()/positive.sum().clamp(min=1)
            offset=(F.smooth_l1_loss(y[:,1].tanh(),normal,reduction='none')*weight).sum()/weight.sum().clamp(min=1)
            loss=detection+2*offset; optimizer.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),5); optimizer.step(); losses.append(loss.item())
        model.eval(); counts={source:{t:[0,0,0] for t in (.15,.3,.5,.7)} for source in ('official','synthetic')}
        with torch.no_grad():
            for x,_,_,_,ids in vl:
                output=model(x.cuda()).cpu().numpy()
                for y,index in zip(output,ids.tolist()):
                    r=records[index]; probability=1/(1+np.exp(-np.clip(y[0],-30,30)))
                    for threshold,totals in counts[r['source']].items():
                        proposals=[]
                        for index in np.argsort(probability)[::-1]:
                            if probability[index]<threshold: break
                            if not any(abs(index-j)<4 for j in proposals): proposals.append(int(index))
                        pairs=sorted((float(np.hypot(index-position,np.tanh(y[1,index])*32-offset)),i,j) for i,index in enumerate(proposals) for j,(position,offset) in enumerate(r['targets']))
                        used_p=set(); used_g=set()
                        for distance,i,j in pairs:
                            if distance<=5 and i not in used_p and j not in used_g: used_p.add(i); used_g.add(j)
                        totals[0]+=len(used_p); totals[1]+=len(proposals); totals[2]+=len(r['targets'])
        metrics={source:{str(t):{'precision':m/max(1,n),'recall':m/max(1,g),'f1':2*m/max(1,n+g),'matched':m,'predicted':n,'golden':g} for t,(m,n,g) in cs.items()} for source,cs in counts.items()}
        # Only existing-domain held-out pages select the threshold/checkpoint.
        threshold=max(metrics['official'],key=lambda t:metrics['official'][t]['f1']); metric=metrics['official'][threshold]
        entry={'epoch':epoch+1,'loss':sum(losses)/len(losses),'seconds':time.perf_counter()-started,'threshold':float(threshold),'official':metric,'synthetic':metrics['synthetic'][threshold]}; history.append(entry)
        if metric['f1']>best:
            best=metric['f1']; torch.save(model.state_dict(),out/'best.pt')
            metadata={'architecture':'SideStripNet normal-bin compression + '+('local' if a.local else 'dilated')+' axial response/normal-offset heads','parameters':sum(p.numel() for p in model.parameters()),'threshold':float(threshold),'seed':seed,'train_bands':len(train),'validation_bands':len(validation),'synthetic_training':a.synthetic,'selection':'official held-out page oracle IC bands; not end-to-end score','coordinate_step_source_pixels':1,'metrics':entry,'preparation':json.loads((directory/'preparation.json').read_text(encoding='utf-8')),'license':'project code; synthetic corpus CC0; official source supervision retains original competition terms'}
            (out/'side-strip.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
        (out/'history.json').write_text(json.dumps(history,indent=2),encoding='utf-8'); print(json.dumps(entry),flush=True)
    model.load_state_dict(torch.load(out/'best.pt',weights_only=True)); model.cpu().eval()
    torch.onnx.export(model,torch.zeros(1,3,64,256),str(out/'side-strip.onnx'),input_names=['band'],output_names=['axis_heads'],opset_version=17)
    meta=json.loads((out/'side-strip.json').read_text(encoding='utf-8')); meta['onnx_sha256']=hashlib.sha256((out/'side-strip.onnx').read_bytes()).hexdigest(); (out/'side-strip.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
if __name__=='__main__': main()
