"""Train geometry/electrical relation network using real nets + endpoint jitter.

    This is oracle-geometry augmentation, not predicted-detector supervision.
    Page split excludes 0237 and preserves existing PCBnet partitions.
"""
import argparse
import copy
from collections import Counter
import json
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
import cv2
import numpy as np
from pcbparse.dataset import index_dataset,convert_raw
from pcbparse.core import references
from pcbparse.graphschematic import DualGraph
from pcbparse.association_features import candidate_graph
from pcbparse.vision import read_image

def prepare(args):
    rng=np.random.default_rng(20261004); out=Path(args.cache); out.mkdir(parents=True,exist_ok=True)
    records=[]; seen=set(); skipped=Counter()
    for item in index_dataset(args.data):
        if item['id']=='0237' or item['sha256'] in seen: continue
        seen.add(item['sha256'])
        raw=json.loads(Path(item['annotation']).read_text(encoding='utf-8'))
        try: golden,_=convert_raw(raw)
        except ValueError: skipped['invalid_source_graph']+=1; continue
        image=read_image(item['image']); h,w=image.shape[:2]
        mask=(cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)<220).astype(np.uint8)*255
        # Mask symbol bodies, mirroring inference; no net membership in features.
        for c in golden['components'].values():
            a,b,d,e=map(round,c['bbox']); mask[max(0,h-e):min(h,h-b),max(0,a):min(w,d)]=0
        wires=[]; wire_net=[]; pin_net={}; ambiguous=set()
        for key,net in golden['nets'].items():
            for ref in references(net['hyperGraph']):
                if ref in pin_net and pin_net[ref]!=key: ambiguous.add(ref)
                pin_net[ref]=key
            for edge in net['edges'].values():
                if edge[0]==edge[1]: continue
                wires.append(edge); wire_net.append(key)
        if not wires: continue
        base={'components':golden['components'],'pins':golden['pins'],'wires':wires,'image':{'width':w,'height':h,'origin':'bottom-left'}}
        for repeat in range(args.augmentations):
            obs=copy.deepcopy(base)
            for pins in obs['pins'].values():
                for p in pins.values():
                    angle=rng.uniform(0,2*np.pi); radius=rng.uniform(2.5,7)
                    p['point']['x']+=float(np.cos(angle)*radius); p['point']['y']+=float(np.sin(angle)*radius)
            graph=DualGraph(obs); candidates=graph.candidates(mask)
            # Skip unannotated or multiply attributed pins. No implicit negatives.
            candidates=[c for c in candidates if c['pin'] in pin_net and c['pin'] not in ambiguous]
            if not candidates: continue
            # A contracted observed group that crosses annotated nets is noisy:
            # exclude those pairs rather than teach wrong topology as truth.
            groups={}
            for i,label in enumerate(wire_net): groups.setdefault(graph.uf.root(i),set()).add(label)
            candidates=[c for c in candidates if len(groups[graph.uf.root(c['wire'])])==1]
            if not candidates: continue
            try: tensors=candidate_graph(graph,candidates)
            except ValueError: skipped['large_graph']+=1; continue
            labels=np.asarray([int(pin_net[c['pin']]==wire_net[c['wire']]) for c in candidates],np.float32)
            name=f"{item['id']}-{repeat}.npz"; np.savez_compressed(out/name,**tensors,labels=labels)
            records.append({'file':name,'page':item['id'],'split':item['split'],'positive':int(labels.sum()),'negative':int((1-labels).sum())})
        print(json.dumps({'page':item['id'],'graphs':len(records)}),flush=True)
    (out/'records.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
    summary={'graphs':len(records),'pages':{split:len({r['page'] for r in records if r['split']==split}) for split in ('train','validation')},'counts':{split:{k:sum(r[k] for r in records if r['split']==split) for k in ('positive','negative')} for split in ('train','validation')},'excluded':['0237'],'skipped':dict(skipped),'supervision':'source net membership with synthetic 2.5-7 pixel endpoint displacement; oracle symbols and wires; not end-to-end performance','official_gt10_used':False}
    (out/'preparation.json').write_text(json.dumps(summary,indent=2),encoding='utf-8'); print(json.dumps(summary))

def metrics(prob,y,threshold=.5):
    pred=prob>=threshold; tp=int(((y==1)&pred).sum()); fp=int(((y==0)&pred).sum()); fn=int(((y==1)&~pred).sum()); tn=int(((y==0)&~pred).sum())
    return {'f1':2*tp/max(1,2*tp+fp+fn),'precision':tp/max(1,tp+fp),'recall':tp/max(1,tp+fn),'negative_f1':2*tn/max(1,2*tn+fn+fp),'balanced_accuracy':.5*(tp/max(1,tp+fn)+tn/max(1,tn+fp)),'tp':tp,'fp':fp,'fn':fn,'tn':tn}

def train(args):
    import torch
    from pcbparse.association_model import ElectricalAssociationNet
    torch.set_num_threads(4); torch.manual_seed(20261004)
    cache=Path(args.cache)
    if not (cache/'records.json').exists(): prepare(args)
    records=json.loads((cache/'records.json').read_text(encoding='utf-8')); data={'train':[],'validation':[]}
    device='cuda' if torch.cuda.is_available() else 'cpu'
    for r in records:
        with np.load(cache/r['file']) as z: tensors={k:torch.from_numpy(z[k]).to(device) for k in z.files}
        data[r['split']].append(tensors)
    if not all(data.values()): raise ValueError('empty split')
    model=ElectricalAssociationNet(layers=args.layers).to(device); opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    positive=sum(r['positive'] for r in records if r['split']=='train'); negative=sum(r['negative'] for r in records if r['split']=='train')
    lossfn=torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(negative/max(1,positive) if args.balance else 1.,device=device)); out=Path(args.output); out.mkdir(parents=True,exist_ok=True)
    history=[]; best=-1; rng=np.random.default_rng(20261004); started=time.perf_counter()
    def forward(d): return model(*(d[k] for k in ('nodes','component_relation','incidence_relation','pairs','edge_features')))
    for epoch in range(args.epochs):
        model.train(); losses=[]
        for i in rng.permutation(len(data['train'])):
            d=data['train'][i]; loss=lossfn(forward(d),d['labels']); opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),2); opt.step(); losses.append(loss.item())
        model.eval(); probs=[]; ys=[]
        with torch.no_grad():
            for d in data['validation']: probs.extend(forward(d).sigmoid().cpu().tolist()); ys.extend(d['labels'].cpu().tolist())
        prob=np.asarray(probs); y=np.asarray(ys); result=metrics(prob,y); row={'epoch':epoch+1,'loss':float(np.mean(losses)),'seconds':time.perf_counter()-started,**result}; history.append(row)
        selection_metric=result['balanced_accuracy'] if args.balance else result['f1']
        if selection_metric>best:
            best=selection_metric; torch.save(model.state_dict(),out/'best.pt')
            # Veto only if >=30 selected validation pairs and >=97% negative
            # precision; if unsupported, abstain by using threshold -0.01.
            threshold=-.01; selection={}
            for t in np.arange(.01,.501,.01):
                take=prob<=t
                if take.sum()>=30 and (y[take]==0).mean()>=.97:
                    threshold=float(t); selection={'selected':int(take.sum()),'negative_precision':float((y[take]==0).mean())}
            meta={'architecture':'GEA-Net: relation-specific pin/component and pin/observed-wire-group message passing + edge decoder','layers':args.layers,'hidden':48,'negative_threshold':threshold,'negative_selection':selection,'metrics':result,'training':json.loads((cache/'preparation.json').read_text(encoding='utf-8')),'evaluation':'page-held-out oracle/jitter validation used for checkpoint and threshold selection; not independent end-to-end test','policy':'negative veto only; uncertain predictions abstain; all image/topology/circuit gates retained'}
            meta['class_balanced_loss']=args.balance
            if args.layers==0: meta['architecture']='node encoder + pair/geometry MLP decoder; no relational message passing'
            meta['policy']='learned/heuristic soft gain within original hard image/topology/circuit gates; validated negative veto when supported'
            meta['checkpoint_metric']='balanced_accuracy' if args.balance else 'positive_f1'
            (out/'best.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
            np.savez_compressed(out/'validation.npz',probabilities=prob,labels=y)
        (out/'history.json').write_text(json.dumps(history,indent=2),encoding='utf-8'); print(json.dumps(row),flush=True)
    model.load_state_dict(torch.load(out/'best.pt',weights_only=True)); model.cpu().eval()
    example=data['validation'][0]; inputs=tuple(example[k].cpu() for k in ('nodes','component_relation','incidence_relation','pairs','edge_features'))
    target=Path(args.model_output); target.mkdir(parents=True,exist_ok=True)
    torch.onnx.export(model,inputs,str(target/'association.onnx'),input_names=['nodes','component_relation','incidence_relation','pairs','edge_features'],output_names=['logits'],opset_version=17,dynamic_axes={'nodes':{0:'nodes'},'component_relation':{0:'nodes',1:'nodes'},'incidence_relation':{0:'nodes',1:'nodes'},'pairs':{0:'edges'},'edge_features':{0:'edges'},'logits':{0:'edges'}})
    (target/'association.json').write_bytes((out/'best.json').read_bytes())

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--data',default='data/raw/PCBnet/Dataset300'); p.add_argument('--cache',default='data/processed/association'); p.add_argument('--epochs',type=int,default=15); p.add_argument('--augmentations',type=int,default=3); p.add_argument('--layers',type=int,default=2); p.add_argument('--output',default='output/training/association'); p.add_argument('--model-output',default='models/candidates/association'); p.add_argument('--prepare-only',action='store_true'); p.add_argument('--balance',action='store_true'); a=p.parse_args()
    if a.prepare_only: prepare(a)
    else: train(a)
