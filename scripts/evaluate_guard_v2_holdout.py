"""New blur/geometry holdouts after development gate, frozen contrast model."""
import sys,json,hashlib,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,cv2,torch
from PIL import Image,ImageDraw
from scripts.prepare_circuit_program import compile_page
from scripts.prepare_separable_circuit import render_separable
from scripts.circuit_geometry_guard import register_guarded
from scripts.circuit_geometry_guard_v2 import register_guarded_v2
from experiments.contrast_edge_circuit_reader import ContrastOnlyReader
from scripts.train_circuit_program import metrics


def main():
    out=ROOT/'output/guard-v2-holdout';out.mkdir(parents=True,exist_ok=True)
    if (out/'protocol.json').exists():raise ValueError('Preserve holdout')
    assert json.loads((ROOT/'output/guard-v2-development/report.json').read_text())['gate_passed']
    protocol={'scope':'finite renderer, new pages, frozen ContrastOnlyReader, no new training',
              'primary':'256 pages scale.55:.85 widths1..7 sigma.9:1.2 noise5 brightness.5:1',
              'shear':'256 unsupported .12:.2 shear pages',
              'rotation':'128 unsupported +/-8:15degree pages',
              'gate':'primary coverage>=.90 F1>=.95 exact-net>=.90, shear/rotation acceptance<=.05',
              'comparison':'same new pages with old/new guards and same model, reject scores reported separately',
              'algorithm_sha256':hashlib.sha256((ROOT/'scripts/circuit_geometry_guard_v2.py').read_bytes()).hexdigest(),
              'model_sha256':hashlib.sha256((ROOT/'output/training/contrast-edge-v1-contrast/best.pt').read_bytes()).hexdigest(),
              'no_pressure_selection':True,'preview':'first4 per cohort'}
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    seen=set()
    for bank in ('circuit-program-v2','factorized-circuit-v1','separable-circuit-v1','registered-circuit-v1','contrast-edge-v1'):
        names=['tiny','train','validation','stress']+(['shear'] if bank=='contrast-edge-v1' else [])
        for n in names:
            with np.load(ROOT/f'data/processed/{bank}/{n}.npz') as z:seen.update(hashlib.sha256(g.tobytes()).hexdigest() for g in z['program'])
    for n in ('primary','hard','shear'):
        with np.load(ROOT/f'output/registration-v2-holdout/{n}-inputs.npz') as z:seen.update(hashlib.sha256(g.tobytes()).hexdigest() for g in z['program'])
    torch.set_num_threads(4);model=ContrastOnlyReader().cuda().eval();model.load_state_dict(torch.load(ROOT/'output/training/contrast-edge-v1-contrast/best.pt',weights_only=True,map_location='cuda'))
    def infer(images):
        rows=[]
        with torch.no_grad():
            for i in range(0,len(images),16):
                x=torch.tensor(1-images[i:i+16].astype(np.float32)/255,device='cuda')
                with torch.autocast('cuda',dtype=torch.float16):rows.append(model.generate(x).cpu().numpy())
        return np.concatenate(rows)
    report={'protocol':protocol,'cohorts':{}};started=time.perf_counter()
    for name,n,start in (('primary',256,710000),('shear',256,720000),('rotation',128,730000)):
        raw=[];gold=[];transforms=[];seeds=[];seed=start
        while len(gold)<n:
            _,g=compile_page(seed);sig=hashlib.sha256(g.tobytes()).hexdigest();r=np.random.default_rng(seed+24001)
            if sig in seen:seed+=1;continue
            seen.add(sig);sx,sy=r.uniform(.55,.85,2);tx=128*(1-sx)+r.uniform(-10,10);ty=128*(1-sy)+r.uniform(-10,10)
            skew=float(r.choice([-1,1])*r.uniform(.12,.2)) if name=='shear' else 0
            affine=np.array([[sx,skew,tx-128*skew],[0,sy,ty]],np.float32)
            im=cv2.warpAffine(render_separable(g,int(r.integers(1,8))),affine,(256,256),borderValue=255)
            if name=='rotation':
                rotation=cv2.getRotationMatrix2D((128,128),float(r.choice([-1,1])*r.uniform(8,15)),1)
                im=cv2.warpAffine(im,rotation,(256,256),borderValue=255)
            sigma=float(r.uniform(.9,1.2) if name=='primary' else r.uniform(.2,.8));im=cv2.GaussianBlur(im,(5,5),sigma)
            im=np.clip(im.astype(float)*r.uniform(.5,1)+r.normal(0,5,im.shape),0,255).astype(np.uint8)
            raw.append(im[None]);gold.append(g);transforms.append(affine);seeds.append(seed);seed+=1
        raw=np.stack(raw);gold=np.stack(gold);np.savez_compressed(out/f'{name}-inputs.npz',observed_images=raw,program=gold,source_affine=transforms,seeds=seeds)
        results={};predictions={};details={};images={}
        for version,algorithm in (('old',register_guarded),('new',register_guarded_v2)):
            records=[algorithm(im) for im in raw[:,0]];images[version]=np.stack([a[None] for a,d in records]);details[version]=[d for a,d in records]
            ready=np.array([d['ready'] for d in details[version]]);p=infer(images[version]);predictions[version]=p
            count_errors=sum(len(d.get('observed_rows',[]))!=g[0] or len(d.get('observed_columns',[]))!=g[1] for d,g in zip(details[version],gold) if d['ready'])
            results[version]={'ready_pages':int(ready.sum()),'coverage':float(ready.mean()),'ready_line_count_identity_errors':int(count_errors),
                              'all_model':metrics(p,gold),'ready_model':metrics(p[ready],gold[ready]) if ready.any() else None,
                              'rejected_pages':int((~ready).sum())}
        report['cohorts'][name]=results;np.savez_compressed(out/f'{name}-predictions.npz',**predictions)
        (out/f'{name}-geometry.json').write_text(json.dumps(details,indent=2),encoding='utf-8')
        canvas=Image.new('RGB',(768,1120),'white');draw=ImageDraw.Draw(canvas)
        for i in range(4):
            draw.text((4,i*280+3),f'{name} {i} ready{details["new"][i]["ready"]}: observed | new aligned | proposal',fill='black')
            for col,im in enumerate((raw[i,0],images['new'][i,0],render_separable(predictions['new'][i],2))):canvas.paste(Image.fromarray(im).convert('RGB'),(col*256,i*280+24))
        canvas.save(out/f'{name}-first4.png');print(json.dumps({'cohort':name,**results}),flush=True)
    p=report['cohorts']['primary']['new'];report['gate_passed']=p['coverage']>=.9 and p['all_model']['net_pair']['micro_F1']>=.95 and p['all_model']['exact_net_partition']>=.9 and all(report['cohorts'][n]['new']['coverage']<=.05 for n in ('shear','rotation'))
    report['seconds']=time.perf_counter()-started;(out/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')


if __name__=='__main__':main()
