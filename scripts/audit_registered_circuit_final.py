"""Final fixed training outcome, registration coverage, and same-image template control."""
import sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,torch
from PIL import Image,ImageDraw
from experiments.factorized_circuit_reader import FactorizedCircuitReader
from scripts.train_circuit_program import metrics
from scripts import audit_factorized_circuit as template
from scripts.prepare_separable_circuit import render_separable


def main():
    out=ROOT/'output/registered-circuit-final';out.mkdir(parents=True,exist_ok=True)
    if (out/'report.json').exists():raise ValueError('Preserve report')
    training=json.loads((ROOT/'output/training/registered-circuit-v1/summary.json').read_text())
    trained=training['full_training_executed']
    checkpoint=ROOT/('output/training/registered-circuit-v1/best.pt' if trained else 'output/training/separable-circuit-v1/best.pt')
    model=FactorizedCircuitReader().cuda().eval();model.load_state_dict(torch.load(checkpoint,weights_only=True,map_location='cuda'));torch.set_num_threads(4)
    template.render_native=render_separable
    report={'new_full_training_executed':trained,'model_status':'new model' if trained else 'frozen model, tiny gate failed',
            'cohorts':{},'pressure_not_used_for_tuning':True,'scope':'uniform grid affine changes only'}
    for name in ('validation','stress'):
        with np.load(ROOT/f'data/processed/registered-circuit-v1/{name}.npz') as z:images,raw,gold,true_transforms=z['images'],z['observed_images'],z['program'],z['transforms']
        predictions=[]
        with torch.no_grad():
            for i in range(0,len(images),16):
                x=torch.tensor(1-images[i:i+16].astype(np.float32)/255,device='cuda')
                with torch.autocast('cuda',dtype=torch.float16):predictions.append(model.generate(x).cpu().numpy())
        pred=np.concatenate(predictions)
        if name=='validation':
            with np.load(ROOT/'output/registered-circuit-evaluation/validation-predictions.npz') as z:baseline=z['registered_template']
        else:baseline=template.template_predict(images)
        details=json.loads((ROOT/f'data/processed/registered-circuit-v1/{name}-registration.json').read_text())
        ready=np.array([r['ready'] for r in details])
        count_errors=0;coordinate_errors=[]
        for i,d in enumerate(details):
            if not d['ready']:continue
            count_errors+=int(len(d['observed_rows'])!=gold[i,0] or len(d['observed_columns'])!=gold[i,1])
            xs=np.arange(int(gold[i,1]))*36+36;ys=np.arange(int(gold[i,0]))*36+36
            tx=true_transforms[i]
            coordinate_errors.extend(np.abs((xs*tx[0,0]+tx[0,2]-d['translation_x'])/d['scale_x']-xs).tolist())
            coordinate_errors.extend(np.abs((ys*tx[1,1]+tx[1,2]-d['translation_y'])/d['scale_y']-ys).tolist())
        report['cohorts'][name]={'all_model':metrics(pred,gold),'all_template':metrics(baseline,gold),
                               'ready_pages':int(ready.sum()),'ready_coverage':float(ready.mean()),
                               'ready_model':metrics(pred[ready],gold[ready]) if ready.any() else None,
                               'ready_template':metrics(baseline[ready],gold[ready]) if ready.any() else None,
                               'not_ready_pages_abstain':int((~ready).sum()),
                               'posthoc_GT_registration_diagnostic':{'used_for_predictions':False,'ready_count_identity_errors':count_errors,
                                                                    'mean_abs_canonical_coordinate_error':float(np.mean(coordinate_errors)) if coordinate_errors else None,
                                                                    'max_abs_canonical_coordinate_error':float(np.max(coordinate_errors)) if coordinate_errors else None}}
        np.savez_compressed(out/f'{name}-predictions.npz',model=pred,template=baseline,ready=ready)
        canvas=Image.new('RGB',(768,1120),'white');draw=ImageDraw.Draw(canvas)
        for i in range(4):
            draw.text((4,i*280+3),f'{name} {i}, ready{bool(ready[i])}: observed | registered | model proposal',fill='black')
            for col,im in enumerate((raw[i,0],images[i,0],render_separable(pred[i],2))):canvas.paste(Image.fromarray(im).convert('RGB'),(col*256,i*280+24))
        canvas.save(out/f'{name}-first4.png')
    (out/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({c:{k:(v['net_pair']['micro_F1'],v['exact_net_partition']) if isinstance(v,dict) and 'net_pair' in v else v for k,v in r.items()} for c,r in report['cohorts'].items()}))


if __name__=='__main__':main()
