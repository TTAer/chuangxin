"""Fix new affine-layout cohorts and store image-only registered inputs."""
import json,hashlib,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,cv2
from scripts.prepare_separable_circuit import render_separable
from scripts.prepare_circuit_program import compile_page
from scripts.circuit_registration import register_image


def main():
    out=ROOT/'data/processed/registered-circuit-v1';out.mkdir(parents=True,exist_ok=True)
    if (out/'manifest.json').exists():raise ValueError('Preserve bank')
    seen=set()
    for bank in ('circuit-program-v2','factorized-circuit-v1','separable-circuit-v1'):
        for name in ('tiny','train','validation','stress'):
            with np.load(ROOT/f'data/processed/{bank}/{name}.npz') as z:seen.update(hashlib.sha256(g.tobytes()).hexdigest() for g in z['program'])
    manifest={'scope':'global anisotropic scaling and translation, not free layout/rotation',
              'registration_uses_GT_transform':False,'cohorts':{},
              'development':'scale x/y .85..1, centered + shift +/-10, widths1..5, blur.3 noise2',
              'pressure':'scale x/y .65.. .8, centered + shift +/-20, widths6/7, blur.7 noise5',
              'prior_programs_excluded':True}
    for name,n,start in (('tiny',4,410000),('train',2048,420000),('validation',256,430000),('stress',256,440000)):
        records=[];observations=[];programs=[];transforms=[];details=[];widths=[];seed=start;seeds=[]
        while len(programs)<n:
            _,g=compile_page(seed);sig=hashlib.sha256(g.tobytes()).hexdigest();r=np.random.default_rng(seed+10103)
            if sig in seen:seed+=1;continue
            seen.add(sig);stress=name=='stress';w=int(r.integers(6,8) if stress else r.integers(1,6));low,high=(.65,.8) if stress else (.85,1)
            sx,sy=r.uniform(low,high,2);limit=20 if stress else 10
            tx=128*(1-sx)+r.uniform(-limit,limit);ty=128*(1-sy)+r.uniform(-limit,limit)
            transform=np.array([[sx,0,tx],[0,sy,ty]],np.float32)
            im=cv2.warpAffine(render_separable(g,w),transform,(256,256),borderValue=255)
            im=cv2.GaussianBlur(im,(3,3),.7 if stress else .3)
            im=np.clip(im.astype(float)*r.uniform(.7,1)+r.normal(0,5 if stress else 2,im.shape),0,255).astype(np.uint8)
            aligned,info=register_image(im) # Only image; no program or transform argument.
            records.append(aligned[None]);observations.append(im[None]);programs.append(g);transforms.append(transform);details.append(info);widths.append(w);seeds.append(seed);seed+=1
        np.savez_compressed(out/f'{name}.npz',images=np.stack(records),observed_images=np.stack(observations),program=np.stack(programs),transforms=np.stack(transforms),width=widths)
        (out/f'{name}-registration.json').write_text(json.dumps(details,indent=2),encoding='utf-8')
        manifest['cohorts'][name]={'pages':n,'seeds':seeds,'registration_ready':sum(d['ready'] for d in details),'sha256':hashlib.sha256((out/f'{name}.npz').read_bytes()).hexdigest()}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps({k:{p:v for p,v in m.items() if p!='seeds'} for k,m in manifest['cohorts'].items()}))


if __name__=='__main__':main()
