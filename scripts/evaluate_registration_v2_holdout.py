"""New preregistered affine holdouts and unsupported-shear diagnostic, frozen weights."""
import sys,json,hashlib,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,cv2,torch
from PIL import Image,ImageDraw
from scripts.prepare_separable_circuit import render_separable
from scripts.prepare_circuit_program import compile_page
from scripts.circuit_registration import register_image
from scripts.circuit_registration_v2 import register_image_v2
from scripts.train_circuit_program import metrics
from experiments.factorized_circuit_reader import FactorizedCircuitReader
from scripts import audit_factorized_circuit as template


def main():
    out=ROOT/'output/registration-v2-holdout';out.mkdir(parents=True,exist_ok=True)
    if (out/'protocol.json').exists():raise ValueError('Preserve holdout')
    dev=json.loads((ROOT/'output/registration-v2-development/report.json').read_text())
    assert dev['gate_passed']
    protocol={'scope':'new affine grammar layouts, same frozen model, no training or tuning',
              'primary':'256 affine pages scale.65:.8, shift20, widths6/7, sigma.7 noise5',
              'hard_diagnostic':'256 affine pages scale.5:.65, shift25, widths1..7, sigma1 noise5',
              'unsupported_diagnostic':'64 shear pages; not part of success claim',
              'gate':'primary coverage>=.95, net F1>=.95, exact net partition>=.90',
              'comparison':'old v1/new v2 registration and same frozen neural reader; template on v2 registered pixels',
              'readiness_is_independent_truth':False,'GT_transform_supplied':False,'no_old_program_reuse':True,
              'algorithm_hash':hashlib.sha256((ROOT/'scripts/circuit_registration_v2.py').read_bytes()).hexdigest(),
              'model_hash':hashlib.sha256((ROOT/'output/training/registered-circuit-v1/best.pt').read_bytes()).hexdigest(),
              'preview':'first4 pages per cohort'}
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    seen=set()
    for bank in ('circuit-program-v2','factorized-circuit-v1','separable-circuit-v1','registered-circuit-v1'):
        for name in ('tiny','train','validation','stress'):
            with np.load(ROOT/f'data/processed/{bank}/{name}.npz') as z:seen.update(hashlib.sha256(g.tobytes()).hexdigest() for g in z['program'])
    torch.set_num_threads(4);model=FactorizedCircuitReader().cuda().eval();model.load_state_dict(torch.load(ROOT/'output/training/registered-circuit-v1/best.pt',weights_only=True,map_location='cuda'))
    template.render_native=render_separable
    def infer(images):
        preds=[]
        with torch.no_grad():
            for i in range(0,len(images),16):
                x=torch.tensor(1-images[i:i+16].astype(np.float32)/255,device='cuda')
                with torch.autocast('cuda',dtype=torch.float16):preds.append(model.generate(x).cpu().numpy())
        return np.concatenate(preds)
    report={'protocol':protocol,'cohorts':{}};started=time.perf_counter()
    for name,n,start in (('primary',256,510000),('hard',256,520000),('shear',64,530000)):
        raw=[];gold=[];transforms=[];seeds=[];seed=start
        while len(gold)<n:
            _,g=compile_page(seed);sig=hashlib.sha256(g.tobytes()).hexdigest();r=np.random.default_rng(seed+11311)
            if sig in seen:seed+=1;continue
            seen.add(sig);hard=name=='hard';shear=name=='shear';lo,hi=(.5,.65) if hard else (.65,.8)
            sx,sy=r.uniform(lo,hi,2);shift=25 if hard else 20
            tx=128*(1-sx)+r.uniform(-shift,shift);ty=128*(1-sy)+r.uniform(-shift,shift)
            skew=float(r.choice([-1,1])*r.uniform(.12,.2)) if shear else 0
            transform=np.array([[sx,skew,tx-128*skew],[0,sy,ty]],np.float32)
            width=int(r.integers(1,8) if hard else r.integers(6,8))
            image=cv2.warpAffine(render_separable(g,width),transform,(256,256),borderValue=255)
            image=cv2.GaussianBlur(image,(3,3),1 if hard else .7)
            image=np.clip(image.astype(float)*r.uniform(.7,1)+r.normal(0,5,image.shape),0,255).astype(np.uint8)
            raw.append(image[None]);gold.append(g);transforms.append(transform);seeds.append(seed);seed+=1
        raw=np.stack(raw);gold=np.stack(gold);np.savez_compressed(out/f'{name}-inputs.npz',observed_images=raw,program=gold,transforms=transforms,seeds=seeds)
        aligned={};details={};prediction={}
        for version,algorithm in (('v1',register_image),('v2',register_image_v2)):
            records=[algorithm(im) for im in raw[:,0]];aligned[version]=np.stack([r[0][None] for r in records]);details[version]=[r[1] for r in records]
            prediction[version]=infer(aligned[version])
        baseline=template.template_predict(aligned['v2']);ready=np.array([d['ready'] for d in details['v2']]);identity=sum(len(d['observed_rows'])!=g[0] or len(d['observed_columns'])!=g[1] for d,g in zip(details['v2'],gold) if d['ready'])
        report['cohorts'][name]={'pages':n,'v1_ready':sum(d['ready'] for d in details['v1']),'v2_ready':int(ready.sum()),
                               'v2_coverage':float(ready.mean()),'v2_ready_line_count_identity_errors':int(identity),
                               'v1_model':metrics(prediction['v1'],gold),'v2_model':metrics(prediction['v2'],gold),
                               'v2_template':metrics(baseline,gold),
                               'ready_model':metrics(prediction['v2'][ready],gold[ready]) if ready.any() else None,
                               'ready_template':metrics(baseline[ready],gold[ready]) if ready.any() else None,
                               'unsupported_ready_rate':float(ready.mean()) if shear else None}
        (out/f'{name}-registration.json').write_text(json.dumps(details,indent=2),encoding='utf-8')
        np.savez_compressed(out/f'{name}-predictions.npz',**prediction,template=baseline,ready=ready)
        canvas=Image.new('RGB',(768,1120),'white');draw=ImageDraw.Draw(canvas)
        for i in range(4):
            draw.text((4,i*280+3),f'{name} {i}: observed | v2 aligned | v2 model',fill='black')
            for col,im in enumerate((raw[i,0],aligned['v2'][i,0],render_separable(prediction['v2'][i],2))):canvas.paste(Image.fromarray(im).convert('RGB'),(col*256,i*280+24))
        canvas.save(out/f'{name}-first4.png')
        print(json.dumps({'cohort':name,**report['cohorts'][name]}),flush=True)
    d=report['cohorts']['primary'];report['primary_gate_passed']=d['v2_coverage']>=.95 and d['v2_model']['net_pair']['micro_F1']>=.95 and d['v2_model']['exact_net_partition']>=.90
    report['seconds']=time.perf_counter()-started
    (out/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')


if __name__=='__main__':main()
