"""Compile counterfactual physical wiring, preserving independent graph seeds."""
import json,sys,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import cv2,numpy as np
from pcbparse.vision import read_image
from scripts.prepare_masked_stroke_data import pixel_hash


def compile_page(seed):
    rng=np.random.default_rng(seed);h=int(rng.integers(3,7));v=int(rng.integers(3,7));n=h+v
    ys=np.arange(h)*44+40;xs=np.arange(v)*44+40;im=np.full((int(ys[-1]+41),int(xs[-1]+41)),255,np.uint8)
    width=int(rng.integers(1,3));parents=list(range(n))
    def root(i):
        while parents[i]!=i:i=parents[i]
        return i
    for y in ys:cv2.line(im,(12,int(y)),(im.shape[1]-13,int(y)),0,width)
    for x in xs:cv2.line(im,(int(x),12),(int(x),im.shape[0]-13),0,width)
    patches=np.zeros((36,1,24,24),np.uint8);edges=np.zeros((36,2),np.int64);mask=np.zeros(36,np.float32);edge_truth=np.zeros(36,np.float32)
    probability=float(rng.uniform(.04,.22));decisions=[]
    for i,y in enumerate(ys):
        for j,x in enumerate(xs):
            connected=bool(rng.random()<probability);decisions.append((i,j,connected))
            if connected:
                cv2.circle(im,(int(x),int(y)),int(rng.integers(3,5)),0,-1);parents[root(i)]=root(h+j)
            elif rng.random()<.5:
                # Visible bridge: vertical stroke jumps over horizontal conductor.
                cv2.line(im,(int(x),int(y-5)),(int(x),int(y+5)),255,width+2)
                cv2.line(im,(int(x-9),int(y)),(int(x+9),int(y)),0,width)
                cv2.ellipse(im,(int(x),int(y)),(5,5),0,-90,90,0,width)
    for k,(i,j,c) in enumerate(decisions):
        x,y=int(xs[j]),int(ys[i]);patch=im[y-12:y+12,x-12:x+12].copy()
        # Style perturbations preserve the compiler's graph; not pseudo labels.
        patch=np.clip(patch.astype(float)*rng.uniform(.65,1)+rng.normal(0,rng.uniform(0,7),patch.shape),0,255).astype(np.uint8)
        patches[k,0]=patch;edges[k]=[i,h+j];mask[k]=1;edge_truth[k]=c
    labels=np.array([root(i) for i in range(n)]);truth=np.zeros((12,12),np.float32);truth[:n,:n]=labels[:,None]==labels[None,:];valid=np.zeros(12,np.float32);valid[:n]=1
    return patches,edges,mask,edge_truth,truth,valid,im


def main():
    out=ROOT/'data/processed/probe-flow';out.mkdir(exist_ok=True)
    if (out/'manifest.json').exists():raise ValueError('Preserve procedural experiment data')
    cohorts={'tiny':range(100,104),'train':range(10000,11024),'validation':range(20000,20128),'stress':range(30000,30128)}
    for name,seeds in cohorts.items():
        records=[compile_page(seed) for seed in seeds]
        if name=='stress':
            for r in records:
                for k in np.flatnonzero(r[2]):
                    rotated=np.rot90(r[0][k,0]).copy();r[0][k,0]=cv2.GaussianBlur(rotated,(3,3),.7)
        np.savez_compressed(out/(name+'.npz'),**{k:np.stack([r[i] for r in records]) for i,k in enumerate(('patches','edges','edge_mask','edge_truth','truth','valid'))})
        if name=='tiny':
            for i,r in enumerate(records):cv2.imwrite(str(out/('tiny-'+str(i)+'.png')),r[-1])
    intake=json.loads((ROOT/'data/raw/CI2N-probeflow-sample/intake.json').read_text(encoding='utf-8'));blocked=set()
    for p in (ROOT/'data/raw/official/gt-0930/10GT').glob('*/*.png'):blocked.add(pixel_hash(read_image(p)))
    # Block known PCB development images as well, irrespective of model scope.
    from pcbparse.dataset import index_dataset
    for r in index_dataset(ROOT/'data/raw/PCBnet/Dataset300'):
        if r['split']=='validation' or r['id']=='0237':blocked.add(pixel_hash(read_image(r['image'])))
    real=[];rows=[];excluded=[];seen=set();rng=np.random.default_rng(20261006)
    for r in intake['records']:
        if r.get('status')!='acquired' or not r['path'].endswith('.png'):continue
        image=read_image(ROOT/r['local']);digest=pixel_hash(image)
        if digest in blocked or digest in seen:excluded.append({'path':r['local'],'reason':'pixel_duplicate_or_heldout'});continue
        seen.add(digest);gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY);yy,xx=np.nonzero(gray<200)
        if not len(xx):continue
        for _ in range(16):
            j=int(rng.integers(len(xx)));x,y=int(xx[j]),int(yy[j]);crop=gray[max(0,y-24):y+24,max(0,x-24):x+24]
            real.append(cv2.resize(crop,(24,24),interpolation=cv2.INTER_AREA)[None])
        rows.append({'path':r['local'],'pixel_sha256':digest,'patches':16})
    if real:np.savez_compressed(out/'ci2n-contrastive.npz',patches=np.stack(real))
    result={'compiled_layouts':{k:len(v) for k,v in cohorts.items()},'real_CI2N_train_images':len(rows),'real_image_patches':len(real),'real_records':rows,'excluded':excluded,
            'GT10_used':False,'Freeze_A_used':False,'targets':'reachable physical wire-run partitions from own compiler; NOT real PCB pin/net GT',
            'split_policy':'disjoint procedural graph seeds; graph-isomorphism family independence not established',
            'external_use':'image-only contrastive pretraining, no GT net ownership generated from image geometry or OBB labels',
            'stress_protocol':'unseen procedural graph seeds, 90-degree patch rotation and 3x3 Gaussian blur sigma0.7',
            'limitations':'grid wire-run oracle, explicit-dot/bridge conventions, no symbol detection, OCR or logical-label net recovery'}
    (out/'manifest.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps({k:v for k,v in result.items() if k not in ('real_records','excluded')}))

if __name__=='__main__':main()
