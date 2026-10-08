"""Frozen model stress: arbitrary port counts and incorrect observed attachment."""
import json,sys,argparse
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,torch
from scripts.train_port_lift import compile_bank,scores
from experiments.port_lift import PortLiftNet

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=ROOT/'output/port-lift-observation-stress');args=parser.parse_args()
    torch.set_num_threads(4);out=args.output;out.mkdir(parents=True,exist_ok=True)
    if (out/'report.json').exists():raise ValueError('Preserve stress cohort')
    d=compile_bank(range(80000,80064),True);model=PortLiftNet().eval();model.load_state_dict(torch.load(ROOT/'output/training/port-lift-v1/best.pt',map_location='cpu',weights_only=True))
    cases={};rng=np.random.default_rng(20261009)
    clean=d['ports'];corrupt=clean.copy()
    for i in range(64):
        n=int(d['valid'][i].sum())
        for c in range(10):
            if rng.random()<.3:corrupt[i,c,0]=(corrupt[i,c,0]+int(rng.integers(1,n)))%n
    extra=np.stack([np.stack([rng.choice(int(d['valid'][i].sum()),4,replace=False) for _ in range(10)]) for i in range(64)])
    mask=np.tile(np.triu(np.ones((10,10)),1),(64,1,1))
    for name,ports in [('clean_two_port',clean),('corrupted_attachment_30pct_components',corrupt),('clean_three_port',extra[:,:,:3]),('clean_four_port',extra)]:
        truth=d['component_truth'] if name not in ('clean_three_port','clean_four_port') else np.stack([d['truth'][i][p[:,:,None,None],p[None,None,:,:]].max((1,3)) for i,p in enumerate(ports)])
        outputs=[]
        with torch.no_grad():
            for i in range(0,64,16):
                _,c,_=model(1-torch.tensor(d['patches'][i:i+16]).float()/255,torch.tensor(d['edges'][i:i+16]),torch.tensor(d['edge_mask'][i:i+16]),torch.tensor(ports[i:i+16]));outputs.append(c.numpy())
        cases[name]=scores(np.concatenate(outputs),truth,mask)
    result={'scope':'synthetic physical graph oracle; observation corruption diagnostic, not detector output','seeds':[80000,80064],'cases':cases,'model_retrained':False,'RC0_changed':False}
    (out/'report.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result))

if __name__=='__main__':main()
