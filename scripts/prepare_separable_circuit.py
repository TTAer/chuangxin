"""A separately versioned observable-dot renderer; never rewrite prior pressure sets."""
import sys,json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import cv2,numpy as np
from scripts.prepare_factorized_circuit import render_native
from scripts.prepare_circuit_program import compile_page


def render_separable(program,width):
    im=render_native(program,width)
    # Draw visible dot after symbol clearing; ensure it extends outside both wire strips.
    for i in range(int(program[0])):
        for j in range(int(program[1])):
            if program[2+i*6+j]==9:
                cv2.circle(im,(36+36*j,36+36*i),max(6,width+3),0,-1)
    return im


def main():
    out=ROOT/'data/processed/separable-circuit-v1';out.mkdir(parents=True,exist_ok=True)
    if (out/'manifest.json').exists():raise ValueError('Preserve bank')
    manifest={'renderer':'native256 visible dot last, radius=max(6,width+3)',
              'scope':'finite fixed grid grammar; tested fixtures, not universal identifiability proof',
              'training':'width1..5 blur0..0.5 noise2','pressure':'unseen width6/7 blur0.7 noise5',
              'cohorts':{},'prior_programs_excluded':True}
    seen=set()
    for bank in ('circuit-program-v2','factorized-circuit-v1'):
        for name in ('tiny','train','validation','stress'):
            with np.load(ROOT/f'data/processed/{bank}/{name}.npz') as z:
                seen.update(hashlib.sha256(g.tobytes()).hexdigest() for g in z['program'])
    for name,n,start in (('tiny',4,310000),('train',4096,320000),('validation',256,330000),('stress',256,340000)):
        records=[];styles=[];seeds=[];seed=start
        while len(records)<n:
            _,g=compile_page(seed);sig=hashlib.sha256(g.tobytes()).hexdigest();r=np.random.default_rng(seed+9101)
            if sig in seen:seed+=1;continue
            seen.add(sig);w=int(r.integers(6,8) if name=='stress' else r.integers(1,6));sigma=.7 if name=='stress' else float(r.uniform(0,.5))
            im=render_separable(g,w)
            if sigma>.1:im=cv2.GaussianBlur(im,(3,3),sigma)
            im=np.clip(im.astype(float)*r.uniform(.7,1)+r.normal(0,5 if name=='stress' else 2,im.shape),0,255).astype(np.uint8)
            records.append((im[None],g));styles.append([w,sigma]);seeds.append(seed);seed+=1
        np.savez_compressed(out/f'{name}.npz',images=np.stack([x[0] for x in records]),program=np.stack([x[1] for x in records]),styles=styles)
        manifest['cohorts'][name]={'pages':n,'seeds':seeds,'sha256':hashlib.sha256((out/f'{name}.npz').read_bytes()).hexdigest()}
    # Admission screen uses its own programs, not validation or pressure pixels/labels.
    checks=[]
    for seed in range(350000,350016):
        _,g=compile_page(seed)
        for width in range(1,8):
            source=render_separable(g,width)
            for slot in range(2,68):
                if g[slot]==7:continue
                alternatives=([8,9] if slot<38 else [10,11,12])
                for code in alternatives:
                    if code==g[slot]:continue
                    altered=g.copy();altered[slot]=code;different=int(np.count_nonzero(source!=render_separable(altered,width)))
                    checks.append({'seed':seed,'width':width,'slot':slot,'from':int(g[slot]),'to':code,'different_pixels':different})
    screen={'scope':'single instruction counterfactuals on16 separate fixture pages, widths1..7, exact clean raster only',
            'comparisons':len(checks),'collisions':[r for r in checks if r['different_pixels']==0],
            'min_different_pixels':min(r['different_pixels'] for r in checks)}
    (out/'admission.json').write_text(json.dumps(screen,indent=2),encoding='utf-8')
    manifest['admission']=screen
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps(screen))
    if screen['collisions']:raise ValueError('Known renderer collisions block training')


if __name__=='__main__':main()
