"""Preregister richer scale/stroke training and new guarded holdouts before results."""
import sys,json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,cv2
from scripts.prepare_circuit_program import compile_page
from scripts.prepare_separable_circuit import render_separable
from scripts.circuit_geometry_guard import register_guarded


def main():
    out=ROOT/'data/processed/contrast-edge-v1';out.mkdir(parents=True,exist_ok=True)
    if (out/'manifest.json').exists():raise ValueError('Preserve bank')
    protocol={'scope':'finite grid, image-only geometry guard and registration, real accuracy unverified',
              'training':'scale .55:1, widths1..7, blur.2:.8, brightness.5:1, noise2:5',
              'development':'same style, disjoint programs',
              'pressure':'scale .55:.85, widths1..7, unseen blur.9:1.2, noise5',
              'unsupported':'256 new shear .12:.2 pages, not training',
              'models':'fresh contrast-only and contrast+signed Sobel heads; same budget and bank',
              'gate':'pressure full F1>=.95 and exact net partition>=.90; geometry shear acceptance<=.05',
              'training_budget':'each tiny400 final gate then8epochs4096 images batch16',
              'selection':'each model dev F1 then exact-program, no pressure selection',
              'GT_used_for_geometry':False,'RC0_changed':False}
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    seen=set()
    for bank in ('circuit-program-v2','factorized-circuit-v1','separable-circuit-v1','registered-circuit-v1'):
        for n in ('tiny','train','validation','stress'):
            with np.load(ROOT/f'data/processed/{bank}/{n}.npz') as z:seen.update(hashlib.sha256(g.tobytes()).hexdigest() for g in z['program'])
    for n in ('primary','hard','shear'):
        with np.load(ROOT/f'output/registration-v2-holdout/{n}-inputs.npz') as z:seen.update(hashlib.sha256(g.tobytes()).hexdigest() for g in z['program'])
    manifest={'protocol':protocol,'cohorts':{},'past_programs_excluded':True}
    for name,n,start in (('tiny',4,610000),('train',4096,620000),('validation',256,630000),('stress',256,640000),('shear',256,650000)):
        images=[];raw=[];gold=[];details=[];seeds=[];transforms=[];seed=start
        while len(gold)<n:
            _,g=compile_page(seed);signature=hashlib.sha256(g.tobytes()).hexdigest();rng=np.random.default_rng(seed+17001)
            if signature in seen:seed+=1;continue
            seen.add(signature);pressure=name=='stress';unsupported=name=='shear';sx,sy=rng.uniform(.55,.85 if pressure else 1,2)
            tx=128*(1-sx)+rng.uniform(-10,10);ty=128*(1-sy)+rng.uniform(-10,10)
            skew=float(rng.choice([-1,1])*rng.uniform(.12,.2)) if unsupported else 0
            affine=np.array([[sx,skew,tx-128*skew],[0,sy,ty]],np.float32);width=int(rng.integers(1,8))
            im=cv2.warpAffine(render_separable(g,width),affine,(256,256),borderValue=255)
            sigma=float(rng.uniform(.9,1.2) if pressure else rng.uniform(.2,.8));im=cv2.GaussianBlur(im,(5,5),sigma)
            im=np.clip(im.astype(float)*rng.uniform(.5,1)+rng.normal(0,5 if pressure else rng.uniform(2,5),im.shape),0,255).astype(np.uint8)
            aligned,info=register_guarded(im)
            images.append(aligned[None]);raw.append(im[None]);gold.append(g);details.append(info);seeds.append(seed);transforms.append(affine);seed+=1
        np.savez_compressed(out/f'{name}.npz',images=np.stack(images),observed_images=np.stack(raw),program=np.stack(gold),transforms=np.stack(transforms),seeds=seeds)
        (out/f'{name}-geometry.json').write_text(json.dumps(details,indent=2),encoding='utf-8')
        manifest['cohorts'][name]={'pages':n,'ready':sum(d['ready'] for d in details),'sha256':hashlib.sha256((out/f'{name}.npz').read_bytes()).hexdigest()}
        print(json.dumps({'cohort':name,**manifest['cohorts'][name]}),flush=True)
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')


if __name__=='__main__':main()
