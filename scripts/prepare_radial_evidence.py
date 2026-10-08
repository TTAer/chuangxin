"""Register paired raw/canonical local-view training before generating new program splits."""
import sys,json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,cv2
from scripts.prepare_circuit_program import compile_page
from scripts.evaluate_ordered_holdout import render_irregular
from scripts.ordered_line_rectification_v2 import register_ordered
from experiments.radial_evidence_net import crossing_patches


def main():
    out=ROOT/'data/processed/radial-evidence-v1';out.mkdir(parents=True,exist_ok=True)
    if (out/'protocol.json').exists():raise ValueError('Preserve bank')
    protocol={'scope':'new synthetic programs; no independent real accuracy',
              'model':'RadialEvidenceNet, eight sectors x eight radial bands, shared MLP mean/max angular pooling',
              'paired':'same model/training, raw source-frame vs nonuniform rectified patches',
              'train':'4096 pages half uniform half jitter +/-4, scale.55:1 width1:7 blur.2:.8 noise2:5',
              'dev':'256 disjoint pages same style','tiny':'4 separate pages, final active crossing accuracy100% required',
              'pressure':'256 disjoint jitter +/-4 scale.55:.85 blur.9:1.2 noise5; not selection',
              'rotation':'128 unsupported +/-8:15 degrees; no training',
              'selection':'dev crossing accuracy only, first best epoch, each8 epochs batch512 lr.001',
              'composition':'frozen ContrastOnlyReader prefix+interval, learned crossing overwrite only observed slots already active in neural prefix',
              'gate':'pressure coverage>=.90 exact-net>=.95 F1>=.98; net-exact gain>=.01 over same-page frozen baseline; unsupported acceptance<=.05',
              'tiny_steps':400,'seed':20261021,'real_labels':False,'RC0_changed':False}
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    blocked=set()
    for bank in ('circuit-program-v1','circuit-program-v2','factorized-circuit-v1','separable-circuit-v1','registered-circuit-v1','contrast-edge-v1'):
        for p in (ROOT/f'data/processed/{bank}').glob('*.npz'):
            with np.load(p) as z:
                if 'program' in z:blocked.update(hashlib.sha256(g.tobytes()).hexdigest() for g in z['program'])
    for folder in ('registration-v2-holdout','guard-v2-holdout','ordered-rectification-holdout'):
        for p in (ROOT/f'output/{folder}').glob('*-inputs.npz'):
            with np.load(p) as z:blocked.update(hashlib.sha256(g.tobytes()).hexdigest() for g in z['program'])
    manifest={'protocol':protocol,'cohorts':{}}
    for name,n,start in (('tiny',4,910000),('train',4096,920000),('validation',256,930000),('stress',256,940000),('rotation',128,950000)):
        raw=[];aligned=[];gold=[];details=[];seeds=[];patches={'raw':[],'canonical':[]};labels=[];page_ids=[];slot_ids=[];seed=start
        while len(gold)<n:
            _,g=compile_page(seed);sig=hashlib.sha256(g.tobytes()).hexdigest()
            if sig in blocked:seed+=1;continue
            blocked.add(sig);r=np.random.default_rng(seed+45001);base=36+36*np.arange(6);jitter=4 if name in ('stress','rotation') or len(gold)%2 else 0
            xs=base+r.integers(-jitter,jitter+1,6);ys=base+r.integers(-jitter,jitter+1,6)
            pressure=name=='stress';sx,sy=r.uniform(.55,.85 if pressure else 1,2);tx=128*(1-sx)+r.uniform(-10,10);ty=128*(1-sy)+r.uniform(-10,10)
            im=cv2.warpAffine(render_irregular(g,int(r.integers(1,8)),xs,ys),np.array([[sx,0,tx],[0,sy,ty]],np.float32),(256,256),borderValue=255)
            if name=='rotation':im=cv2.warpAffine(im,cv2.getRotationMatrix2D((128,128),float(r.choice([-1,1])*r.uniform(8,15)),1),(256,256),borderValue=255)
            im=cv2.GaussianBlur(im,(5,5),float(r.uniform(.9,1.2) if pressure else r.uniform(.2,.8)))
            im=np.clip(im.astype(float)*r.uniform(.5,1)+r.normal(0,5 if pressure else r.uniform(2,5),im.shape),0,255).astype(np.uint8)
            a,d=register_ordered(im);idx=len(gold);raw.append(im[None]);aligned.append(a[None]);gold.append(g);details.append(d);seeds.append(seed);seed+=1
            if name in ('tiny','train','validation'):
                p,slots=crossing_patches(im,a,d,'raw');q,_=crossing_patches(im,a,d,'canonical')
                # Supervised train/dev only: invisible or misidentified slot mappings are not corrected with GT.
                valid=[k for k,s in enumerate(slots) if g[s]!=7]
                if valid:
                    patches['raw'].append(p[valid]);patches['canonical'].append(q[valid]);labels.extend(int(g[slots[k]]-8) for k in valid);page_ids.extend([idx]*len(valid));slot_ids.extend(slots[k] for k in valid)
        np.savez_compressed(out/f'{name}.npz',observed_images=np.stack(raw),images=np.stack(aligned),program=np.stack(gold),seeds=seeds)
        if name in ('tiny','train','validation'):
            np.savez_compressed(out/f'{name}-patches.npz',raw=np.concatenate(patches['raw']),canonical=np.concatenate(patches['canonical']),labels=np.array(labels),page_ids=page_ids,slot_ids=slot_ids)
        (out/f'{name}-geometry.json').write_text(json.dumps(details,indent=2),encoding='utf-8')
        manifest['cohorts'][name]={'pages':n,'ready':sum(d['ready'] for d in details),'labelled_patches':len(labels),'sha256':hashlib.sha256((out/f'{name}.npz').read_bytes()).hexdigest()}
        print(json.dumps({'cohort':name,**manifest['cohorts'][name]}),flush=True)
    assert manifest['cohorts']['tiny']['ready']==4,'Tiny observations incomplete; stop'
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')


if __name__=='__main__':main()
