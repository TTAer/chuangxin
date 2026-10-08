"""Retrospective page-wise topology cost audit, no model or threshold changes."""
import sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from PIL import Image,ImageDraw
from scripts.prepare_circuit_program import execute
from scripts.prepare_separable_circuit import render_separable


def costs(p,g):
    pv,pt,_,pg=execute(p);gv,gt,_,gg=execute(g)
    mask=(pv|gv)[:,None]&(pv|gv)[None,:]&np.triu(np.ones((42,42),bool),1)
    return {'FP':int(((pt>.5)&(gt<=.5)&mask).sum()),'FN':int(((pt<=.5)&(gt>.5)&mask).sum()),
            'exact_net':{frozenset(x) for x in pg}=={frozenset(x) for x in gg},'exact_program':bool(np.array_equal(p,g))}


def main():
    folder=ROOT/'output/topology-cost-holdout';dest=folder/'selected-topology-cost-audit.json'
    if dest.exists():raise ValueError('Preserve audit')
    report=json.loads((folder/'report.json').read_text());trial='risk';result={'scope':'posthoc selected model, no re-selection','trial':trial,'cohorts':{}}
    for name in ('primary','hard'):
        with np.load(folder/f'{name}-inputs.npz') as z:gold=z['program'];raw=z['observed_images']
        with np.load(folder/f'{name}-predictions.npz') as z:base=z['baseline'];pred=z[trial]
        cases=[]
        for i,(a,b,g) in enumerate(zip(base,pred,gold)):
            if np.array_equal(a,b):continue
            before=costs(a,g);after=costs(b,g)
            cases.append({'case_index':i,'before':before,'after':after,'delta_FP':after['FP']-before['FP'],'delta_FN':after['FN']-before['FN'],
                          'changed_slots':np.flatnonzero(a!=b).tolist()})
        result['cohorts'][name]={'changed_pages':len(cases),'sum_delta_FP':sum(d['delta_FP'] for d in cases),'sum_delta_FN':sum(d['delta_FN'] for d in cases),'cases':cases}
        worst=sorted(cases,key=lambda d:d['delta_FP'],reverse=True)[:3];canvas=Image.new('RGB',(768,280*len(worst)),'white');draw=ImageDraw.Draw(canvas)
        for row,d in enumerate(worst):
            i=d['case_index'];draw.text((4,row*280+4),f'{name} case{i} deltaFP={d["delta_FP"]}: raw | baseline program | selected program',fill='black')
            for col,im in enumerate((raw[i,0],render_separable(base[i],2),render_separable(pred[i],2))):canvas.paste(Image.fromarray(im).convert('RGB'),(col*256,row*280+24))
        canvas.save(folder/f'{name}-topology-cost-cases.png')
    dest.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({k:{x:y for x,y in v.items() if x!='cases'} for k,v in result['cohorts'].items()}))


if __name__=='__main__':main()
