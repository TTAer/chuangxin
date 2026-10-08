"""Preregistered raw-vs-registered frozen readers, then optional development-gated training."""
import sys,json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,torch
from experiments.factorized_circuit_reader import FactorizedCircuitReader
from scripts import audit_factorized_circuit as templates
from scripts.prepare_separable_circuit import render_separable
from scripts.train_circuit_program import metrics
from PIL import Image,ImageDraw

PROTOCOL={'scope':'translation/anisotropic scale only; fixed grid grammar still assumed',
          'model':'frozen separable-circuit-v1 best.pt, no weight changes',
          'comparisons':'raw model/template vs image-only registration model/template on same pages',
          'registration':'left/top line overhangs estimate coordinates; source GT transforms never supplied',
          'training_gate':'train only if development registered-model F1<.95 or exact net partition<.90',
          'pressure':'report once, no tuning registration or weights using these outputs',
          'template':'same deterministic reader, visible-dot renderer only',
          'preview':'first4 of each cohort', 'RC0_changed':False}


def main():
    out=ROOT/'output/registered-circuit-evaluation';out.mkdir(parents=True,exist_ok=True)
    if '--register' in sys.argv:
        if (out/'protocol.json').exists():raise ValueError('Protocol already registered')
        (out/'protocol.json').write_text(json.dumps(PROTOCOL,indent=2),encoding='utf-8');return
    assert json.loads((out/'protocol.json').read_text())==PROTOCOL
    if (out/'report.json').exists():raise ValueError('Preserve report')
    torch.set_num_threads(4);model=FactorizedCircuitReader().cuda().eval()
    checkpoint=ROOT/'output/training/separable-circuit-v1/best.pt';model.load_state_dict(torch.load(checkpoint,weights_only=True,map_location='cuda'))
    templates.render_native=render_separable
    def infer(images):
        rows=[]
        with torch.no_grad():
            for i in range(0,len(images),16):
                x=torch.tensor(1-images[i:i+16].astype(np.float32)/255,device='cuda')
                with torch.autocast('cuda',dtype=torch.float16):rows.append(model.generate(x).cpu().numpy())
        return np.concatenate(rows)
    report={'protocol':PROTOCOL,'checkpoint_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),'cohorts':{}}
    for name in ('validation','stress'):
        if name=='stress' and report.get('new_training_required'):
            report['pressure_deferred_until_training']=True
            break
        with np.load(ROOT/f'data/processed/registered-circuit-v1/{name}.npz') as z:
            raw,aligned,gold,width=z['observed_images'],z['images'],z['program'],z['width']
        predictions={'raw_model':infer(raw),'registered_model':infer(aligned),
                     'raw_template':templates.template_predict(raw),'registered_template':templates.template_predict(aligned)}
        records={k:metrics(v,gold) for k,v in predictions.items()}
        records['by_width']={str(w):{k:metrics(v[width==w],gold[width==w]) for k,v in predictions.items()} for w in sorted(set(width.tolist()))}
        details=json.loads((ROOT/f'data/processed/registered-circuit-v1/{name}-registration.json').read_text())
        records['registration_ready']=sum(d['ready'] for d in details)
        report['cohorts'][name]=records
        np.savez_compressed(out/f'{name}-predictions.npz',**predictions)
        canvas=Image.new('RGB',(768,1120),'white');draw=ImageDraw.Draw(canvas)
        for i in range(4):
            draw.text((4,i*280+3),f'{name} {i}: raw | image-registered | registered-model proposal',fill='black')
            for col,im in enumerate((raw[i,0],aligned[i,0],render_separable(predictions['registered_model'][i],2))):canvas.paste(Image.fromarray(im).convert('RGB'),(col*256,i*280+24))
        canvas.save(out/f'{name}-first4.png')
        if name=='validation':
            d=records['registered_model'];report['new_training_required']=d['net_pair']['micro_F1']<.95 or d['exact_net_partition']<.90
            # The decision is written before pressure is scored.
            (out/'development-decision.json').write_text(json.dumps({'new_training_required':report['new_training_required'],'basis':d},indent=2),encoding='utf-8')
    (out/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({'new_training_required':report['new_training_required'],
                     'results':{c:{k:(m['net_pair']['micro_F1'],m['exact_net_partition']) for k,m in r.items() if isinstance(m,dict) and 'net_pair' in m} for c,r in report['cohorts'].items()}}))


if __name__=='__main__':main()
