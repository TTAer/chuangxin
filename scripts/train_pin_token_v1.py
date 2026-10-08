"""Small joint role/affinity experiment, with a no-visual MLP control."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from experiments.pin_token_v1 import K,ROLES,make_model

def metrics(role,association,truth_role,truth_association):
    f1={}
    for i,name in enumerate(ROLES):
        if not np.any(truth_role==i): continue
        tp=int(((role==i)&(truth_role==i)).sum()); fp=int(((role==i)&(truth_role!=i)).sum()); fn=int(((role!=i)&(truth_role==i)).sum())
        f1[name]=2*tp/max(1,2*tp+fp+fn)
    pins=truth_association<K; null=~pins
    return {'role_macro_f1_present_classes':float(np.mean(list(f1.values()))),'role_f1':f1,
        'association_accuracy':float((association==truth_association).mean()),
        'pin_token_association_accuracy':float((association[pins]==truth_association[pins]).mean()) if pins.any() else None,
        'null_accuracy':float((association[null]==K).mean()) if null.any() else None,
        'tokens':len(role),'pin_tokens':int(pins.sum()),'null_tokens':int(null.sum())}

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--epochs',type=int,default=15); parser.add_argument('--no-visual',action='store_true'); args=parser.parse_args()
    torch.set_num_threads(4); torch.manual_seed(20261005); np.random.seed(20261005)
    cache=ROOT/'data/processed/pin-token-v1'; source=cache/'tokens.npz'; data=np.load(source)
    device='cuda' if torch.cuda.is_available() else 'cpu'; model=make_model(not args.no_visual).to(device)
    inputs=[torch.from_numpy(data[k]).to(device) for k in ('patch','text','pairs','valid')]; inputs[0]=inputs[0].float()/255
    yr=torch.from_numpy(data['role']).long().to(device); ya=torch.from_numpy(data['association']).long().to(device)
    train=np.flatnonzero(data['split']==0); validation=np.flatnonzero(data['split']==1)
    assert len(train) and len(validation)
    opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    counts=np.bincount(data['role'][train],minlength=len(ROLES)); weights=np.sqrt(len(train)/np.maximum(1,counts)); weights=weights/weights.mean()
    role_loss=torch.nn.CrossEntropyLoss(weight=torch.tensor(weights,dtype=torch.float32,device=device)); pair_loss=torch.nn.CrossEntropyLoss()
    variant='geometry-text' if args.no_visual else 'visual-joint'
    out=ROOT/'output/training/pin-token-v1'/variant; out.mkdir(parents=True,exist_ok=True)
    best=-1; history=[]; started=time.perf_counter()
    def predict():
        model.eval(); rs=[]; pairs=[]
        with torch.no_grad():
            for start in range(0,len(validation),256):
                indices=validation[start:start+256]; r,a=model(*(x[indices] for x in inputs)); rs.extend(r.argmax(1).cpu().tolist()); pairs.extend(a.argmax(1).cpu().tolist())
        return np.array(rs),np.array(pairs)
    for epoch in range(args.epochs):
        model.train(); losses=[]
        for indices in np.array_split(np.random.permutation(train),max(1,int(np.ceil(len(train)/128)))):
            r,a=model(*(x[indices] for x in inputs)); loss=role_loss(r,yr[indices])+pair_loss(a,ya[indices]); opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),2); opt.step(); losses.append(loss.item())
        r,a=predict(); result=metrics(r,a,data['role'][validation],data['association'][validation]); score=(result['role_macro_f1_present_classes']+result['association_accuracy'])/2
        row={'epoch':epoch+1,'loss':float(np.mean(losses)),'elapsed_seconds':time.perf_counter()-started,**result}; history.append(row)
        if score>best:
            best=score; torch.save(model.state_dict(),out/'best.pt')
            np.savez_compressed(out/'validation.npz',role=r,association=a,indices=validation)
            meta={'variant':variant,'architecture':'shared token image encoder + text syntax embedding + joint role/endpoint affinity heads + null endpoint',
                'parameters':sum(p.numel() for p in model.parameters()),'device':device,'epoch':epoch+1,'seed':20261005,'metrics':result,
                'training_tokens':len(train),'validation_tokens':len(validation),'training_data_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
                'input_ocr':'actual RapidOCR top1; no OCR replacement','candidate_endpoints':K,'selection':'mean of validation role macro F1 and association accuracy',
                'limitations':['weak explicit CVJson reference labels, not manually reviewed token roles','oracle bodies and endpoints; not end-to-end recovery rate or competition PinF1','top8 candidate failures and unmatched OCR excluded; conditional accuracy only','no other-class supervision; cannot claim six-role discrimination'],
                'rc0_runtime_modified':False,'freeze_a_used':False,'freeze_b_used':False}
            (out/'best.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
        (out/'history.json').write_text(json.dumps(history,indent=2),encoding='utf-8'); print(json.dumps(row),flush=True)
    target=data['association'][validation]; nearest=np.zeros(len(validation),int)
    pins=target<K
    baseline={'pin_token_nearest_accuracy':float((nearest[pins]==target[pins]).mean()),'scope':'pin tokens only, oracle endpoint set; no token role or null prediction'}
    (out/'baseline.json').write_text(json.dumps(baseline,indent=2),encoding='utf-8')
    print(json.dumps({'variant':variant,'best':best,'baseline':baseline}),flush=True)
if __name__=='__main__': main()
