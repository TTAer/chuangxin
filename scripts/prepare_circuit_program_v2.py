"""Add LogicBench official train images, preserve all old program cohorts."""
import json,hashlib,shutil,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import cv2,numpy as np
from scripts.prepare_circuit_program import compile_page
from scripts.prepare_masked_stroke_data import pixel_hash
from pcbparse.vision import read_image

def main():
    old=ROOT/'data/processed/circuit-program-v1';out=ROOT/'data/processed/circuit-program-v2';out.mkdir(parents=True,exist_ok=True)
    if (out/'manifest.json').exists():raise ValueError('Preserve bank')
    manifest=json.loads((old/'manifest.json').read_text());program_seen=set()
    for name in ('train','validation','stress'):
        shutil.copyfile(old/(name+'.npz'),out/(name+'.npz'))
        with np.load(old/(name+'.npz')) as z:program_seen.update(hashlib.sha256(x.tobytes()).hexdigest() for x in z['program'])
    rows=[];seeds=[];seed=1900
    while len(rows)<4:
        row=compile_page(seed);sig=hashlib.sha256(row[1].tobytes()).hexdigest()
        if sig not in program_seen:program_seen.add(sig);rows.append(row);seeds.append(seed)
        seed+=1
    np.savez_compressed(out/'tiny.npz',images=np.stack([r[0] for r in rows]),program=np.stack([r[1] for r in rows]));manifest['program_seeds']['tiny']=seeds
    with np.load(old/'multisource-image-only.npz') as z:images=list(z['images']);codes=list(z['source'])
    names=manifest['image_pretrain_sources'];names.append('logicbench-train');rng=np.random.default_rng(20261008)
    prior=json.loads((ROOT/'data/processed/masked-stroke/preparation.json').read_text());seen={r['pixel_sha256'] for r in prior['pages']}|{r['pixel_sha256'] for r in manifest['CI2N_records']};records=[];excluded=[]
    intake=json.loads((ROOT/'data/raw/LogicBench-train/intake.json').read_text());allowed=set(intake['author_train_ids'])
    for row in intake['records']:
        if row['status']!='acquired' or not row['path'].startswith('images/'):continue
        file=ROOT/row['local'];assert file.stem in allowed;im=read_image(file);digest=pixel_hash(im)
        if digest in seen:excluded.append(row['local']);continue
        seen.add(digest);gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY);ys,xs=np.nonzero(gray<220)
        if len(xs)<20:continue
        for _ in range(16):
            k=int(rng.integers(len(xs)));side=min(96,min(gray.shape));x=max(0,min(gray.shape[1]-side,int(xs[k])-side//2));y=max(0,min(gray.shape[0]-side,int(ys[k])-side//2))
            images.append(cv2.resize(gray[y:y+side,x:x+side],(48,48),interpolation=cv2.INTER_AREA)[None]);codes.append(len(names)-1)
        records.append({'path':row['local'],'pixel_sha256':digest,'patches':16})
    np.savez_compressed(out/'multisource-image-only.npz',images=np.stack(images),source=np.array(codes,np.int64))
    from collections import Counter
    manifest.update(image_pretrain_sources=names,image_patch_counts=dict(Counter(names[c] for c in codes)),new_LogicBench_unique_images=len(records),LogicBench_records=records,LogicBench_excluded_duplicates=excluded,LogicBench_val_test_used=False)
    manifest['sha256']={name:hashlib.sha256((out/(name+'.npz')).read_bytes()).hexdigest() for name in ('tiny','train','validation','stress')}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8');print(json.dumps({k:manifest[k] for k in ('program_layouts','image_patch_counts','new_LogicBench_unique_images')}))

if __name__=='__main__':main()
