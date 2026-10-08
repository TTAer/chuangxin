"""Paired model comparisons and whole-page geometry coverage, without retuning."""
import sys,json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,torch
from collections import Counter
from PIL import Image,ImageDraw
from experiments.contrast_edge_circuit_reader import ContrastOnlyReader,ContrastEdgeReader
from scripts.train_circuit_program import metrics
from scripts import audit_factorized_circuit as template
from scripts.prepare_separable_circuit import render_separable
from scripts.circuit_registration_v2 import register_image_v2

AUDIT_PROTOCOL={'models':'paired contrast-only and contrast+signed-edge, no pressure-based selection',
                'full_model_gate':'full pressure F1>=.95 and exact net partition>=.90',
                'geometry_gate':'pressure coverage>=.90 and shear acceptance<=.05',
                'ready_subset':'report descriptively, never substitute for full scores',
                'template':'same known-renderer control on identical guarded pixels',
                'preview':'first4 pages only','independent_real_accuracy':False}


def main():
    out=ROOT/'output/contrast-edge-audit';out.mkdir(parents=True,exist_ok=True)
    if '--register' in sys.argv:
        if (out/'protocol.json').exists():raise ValueError('Already registered')
        (out/'protocol.json').write_text(json.dumps(AUDIT_PROTOCOL,indent=2),encoding='utf-8');return
    assert json.loads((out/'protocol.json').read_text())==AUDIT_PROTOCOL
    if (out/'report.json').exists():raise ValueError('Preserve audit')
    torch.set_num_threads(4);template.render_native=render_separable
    summaries={m:json.loads((ROOT/f'output/training/contrast-edge-v1-{m}/summary.json').read_text()) for m in ('contrast','edge')}
    eligible={m:r for m,r in summaries.items() if r['full_training_executed']}
    report={'scope':'new controlled layouts, paired budgets, no real validation','training':summaries,'cohorts':{}}
    for name in ('validation','stress','shear'):
        with np.load(ROOT/f'data/processed/contrast-edge-v1/{name}.npz') as z:images,raw,gold=z['images'],z['observed_images'],z['program']
        details=json.loads((ROOT/f'data/processed/contrast-edge-v1/{name}-geometry.json').read_text());ready=np.array([d['ready'] for d in details])
        identity=sum(len(d.get('observed_rows',[]))!=g[0] or len(d.get('observed_columns',[]))!=g[1] for d,g in zip(details,gold) if d['ready'])
        results={'pages':len(gold),'geometry_ready':int(ready.sum()),'coverage':float(ready.mean()),'ready_count_identity_errors':int(identity),'models':{}}
        results['geometry_failure_reasons']=dict(Counter(d.get('reason','ready') for d in details))
        predictions={}
        for model_name in eligible:
            model=(ContrastOnlyReader() if model_name=='contrast' else ContrastEdgeReader()).cuda().eval();model.load_state_dict(torch.load(ROOT/f'output/training/contrast-edge-v1-{model_name}/best.pt',weights_only=True,map_location='cuda'))
            pred=[]
            with torch.no_grad():
                for i in range(0,len(images),16):
                    x=torch.tensor(1-images[i:i+16].astype(np.float32)/255,device='cuda')
                    with torch.autocast('cuda',dtype=torch.float16):pred.append(model.generate(x).cpu().numpy())
            p=np.concatenate(pred);predictions[model_name]=p
            results['models'][model_name]={'all':metrics(p,gold),'ready':metrics(p[ready],gold[ready]) if ready.any() else None}
            del model;torch.cuda.empty_cache()
        # Deterministic template only on supported-domain controls. Shear is scored as rejection quality.
        if name!='shear':
            baseline=template.template_predict(images);predictions['template']=baseline
            results['template']={'all':metrics(baseline,gold),'ready':metrics(baseline[ready],gold[ready]) if ready.any() else None}
        if name in ('stress','shear'):
            legacy_ready=np.array([register_image_v2(im)[1]['ready'] for im in raw[:,0]])
            results['unguarded_v2_ready']=int(legacy_ready.sum())
            results['guard_rejected_unguarded_ready']=int((legacy_ready&~ready).sum())
            if name=='shear':results['unsupported_ready_rate']=float(ready.mean())
        np.savez_compressed(out/f'{name}-predictions.npz',**predictions,ready=ready)
        canvas=Image.new('RGB',(768,1120),'white');draw=ImageDraw.Draw(canvas)
        chosen='edge' if 'edge' in predictions else ('contrast' if 'contrast' in predictions else None)
        for i in range(4):
            draw.text((4,i*280+3),f'{name} {i} ready{bool(ready[i])}: observed | guarded input | model',fill='black')
            third=render_separable(predictions[chosen][i],2) if chosen else images[i,0]
            for col,im in enumerate((raw[i,0],images[i,0],third)):canvas.paste(Image.fromarray(im).convert('RGB'),(col*256,i*280+24))
        canvas.save(out/f'{name}-first4.png')
        report['cohorts'][name]=results
        print(json.dumps({'cohort':name,**results}),flush=True)
    report['geometry_gate_passed']=report['cohorts']['stress']['coverage']>=.90 and report['cohorts']['shear']['unsupported_ready_rate']<=.05
    report['gate']={m:(r['stress_gate_passed'] and report['geometry_gate_passed']) for m,r in eligible.items()}
    report['new_pressure_used_for_selection']=False
    (out/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')


if __name__=='__main__':main()
