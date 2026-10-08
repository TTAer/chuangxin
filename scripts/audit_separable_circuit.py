"""Same-bank fixed model/template/composition controls; no pressure-based selection."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from PIL import Image,ImageDraw
from scripts import audit_factorized_circuit as prior
from scripts.prepare_separable_circuit import render_separable
from scripts.train_circuit_program import metrics

PROTOCOL={'model':'dev-selected fixed single-pass reader',
          'template':'previous deterministic algorithm, only substitute new renderer',
          'composition':'model counts/crossings plus template intervals if counts agree; otherwise model proposal unchanged',
          'agreement':'full program equality; report coverage and errors, never automatically verified',
          'selection':'no threshold search or checkpoint selection using pressure',
          'gate':'pre-existing net F1>=.95 and exact net partition>=.90 for individual model; comparisons descriptive',
          'preview':'first4 pages each cohort'}


def main():
    out=ROOT/'output/separable-circuit-audit';out.mkdir(parents=True,exist_ok=True)
    if (out/'report.json').exists():raise ValueError('Preserve audit')
    registered=json.loads((out/'protocol.json').read_text())
    assert registered==PROTOCOL
    prior.render_native=render_separable
    report={'protocol':PROTOCOL,'cohorts':{}}
    for name in ('validation','stress'):
        with np.load(ROOT/f'data/processed/separable-circuit-v1/{name}.npz') as z:
            image,gold,styles=z['images'],z['program'],z['styles']
        with np.load(ROOT/f'output/training/separable-circuit-v1/{name}-predictions.npz') as z:model=z['program']
        template=prior.template_predict(image);composition=model.copy();samecount=np.all(model[:,:2]==template[:,:2],axis=1)
        composition[samecount,38:]=template[samecount,38:]
        agree=np.all(model==template,axis=1)
        records={k:metrics(p,gold) for k,p in [('model',model),('template',template),('composition',composition)]}
        records['agreement']={'accepted_pages':int(agree.sum()),'coverage':float(agree.mean()),
                              'exact_program_errors':int(np.any(model[agree]!=gold[agree],axis=1).sum()),
                              'accepted_metrics':metrics(model[agree],gold[agree]) if agree.any() else None}
        records['by_width']={str(int(w)):{k:metrics(p[styles[:,0]==w],gold[styles[:,0]==w])
                                       for k,p in [('model',model),('template',template),('composition',composition)]}
                             for w in sorted(set(styles[:,0]))}
        report['cohorts'][name]=records
        np.savez_compressed(out/f'{name}-predictions.npz',template=template,composition=composition,agreement=agree)
        canvas=Image.new('RGB',(768,1120),'white');draw=ImageDraw.Draw(canvas)
        for i in range(4):
            draw.text((4,i*280+3),f'{name} {i} width{int(styles[i,0])}: input | gold | model',fill='black')
            for col,im in enumerate((image[i,0],render_separable(gold[i],2),render_separable(model[i],2))):canvas.paste(Image.fromarray(im).convert('RGB'),(col*256,i*280+24))
        canvas.save(out/f'{name}-first4.png')
    (out/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({name:{k:v if k=='agreement' else v['net_pair']['micro_F1'] for k,v in rec.items() if k!='by_width'} for name,rec in report['cohorts'].items()}))


if __name__=='__main__':
    if '--register' in sys.argv:
        out=ROOT/'output/separable-circuit-audit';out.mkdir(parents=True,exist_ok=True)
        if (out/'protocol.json').exists():raise ValueError('Already registered')
        (out/'protocol.json').write_text(json.dumps(PROTOCOL,indent=2),encoding='utf-8')
    else:main()
