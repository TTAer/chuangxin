"""Own executable schematic grammar and multisource image-only warm-up bank."""
import json,hashlib,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import cv2,numpy as np
from scripts.prepare_masked_stroke_data import pixel_hash
from pcbparse.vision import read_image

def execute(tokens):
    tokens=np.asarray(tokens)
    if tokens.shape!=(68,) or not np.issubdtype(tokens.dtype,np.integer):raise ValueError('Expected 68 integer program instructions')
    h,v=map(int,tokens[:2]);parents=list(range(42));valid=np.zeros(42,bool)
    if not 3<=h<=6 or not 3<=v<=6:raise ValueError('Invalid wire counts')
    for i in range(6):
        for j in range(6):
            if (i>=h or j>=v) and tokens[2+i*6+j]!=7:raise ValueError('Instruction at absent crossing')
    for i in range(5):
        for j in range(6):
            if (i>=h-1 or j>=v) and tokens[38+i*6+j]!=7:raise ValueError('Instruction at absent interval')
    def root(i):
        while parents[i]!=i:i=parents[i]
        return i
    def union(a,b):parents[root(a)]=root(b)
    valid[:h]=True
    for i in range(h):
        for j in range(v):
            vid=6+i*6+j;valid[vid]=True;code=int(tokens[2+i*6+j])
            if code not in (8,9):raise ValueError('Invalid active crossing')
            if code==9:union(i,vid)
    components=[]
    for i in range(h-1):
        for j in range(v):
            code=int(tokens[38+i*6+j]);a=6+i*6+j;b=a+6
            if code==10:union(a,b)
            elif code in (11,12):components.append({'kind':'R' if code==11 else 'C','slot':[i,j],'pin_wire_nodes':[a,b]})
            else:raise ValueError('Invalid active vertical interval')
    groups={}
    for i in np.flatnonzero(valid):groups.setdefault(root(int(i)),[]).append(int(i))
    truth=np.zeros((42,42),np.float32)
    for ids in groups.values():truth[np.ix_(ids,ids)]=1
    return valid,truth,components,list(groups.values())

def render_program(tokens,width=1):
    h,v=map(int,tokens[:2]);execute(tokens);im=np.full((128,128),255,np.uint8);xy=np.arange(6)*18+18
    for i in range(h):cv2.line(im,(8,int(xy[i])),(120,int(xy[i])),0,width)
    for j in range(v):cv2.line(im,(int(xy[j]),8),(int(xy[j]),120),0,width)
    for i in range(h):
        for j in range(v):
            if tokens[2+i*6+j]==9:cv2.circle(im,(int(xy[j]),int(xy[i])),3 if width<3 else 4,0,-1)
    for i in range(h-1):
        for j in range(v):
            code=int(tokens[38+i*6+j])
            if code==10:continue
            x=int(xy[j]);y=int((xy[i]+xy[i+1])/2);cv2.rectangle(im,(x-5,y-6),(x+5,y+6),255,-1)
            cv2.line(im,(x,y-9),(x,y-5),0,width);cv2.line(im,(x,y+5),(x,y+9),0,width)
            if code==11:
                pts=np.array([[x,y-5],[x-3,y-3],[x+3,y-1],[x-3,y+1],[x+3,y+3],[x,y+5]],np.int32);cv2.polylines(im,[pts],False,0,1 if width<3 else 2)
            else:
                cv2.line(im,(x,y-5),(x,y-2),0,width);cv2.line(im,(x,y+2),(x,y+5),0,width)
                cv2.line(im,(x-4,y-2),(x+4,y-2),0,width);cv2.line(im,(x-4,y+2),(x+4,y+2),0,width)
    return im

def compile_page(seed,stress=False):
    rng=np.random.default_rng(seed);h=int(rng.integers(3,7));v=int(rng.integers(3,7));tokens=np.full(68,7,np.int64);tokens[:2]=[h,v]
    for i in range(h):
        for j in range(v):tokens[2+i*6+j]=9 if rng.random()<.23 else 8
    for i in range(h-1):
        for j in range(v):tokens[38+i*6+j]=int(rng.choice([10,11,12],p=[.65,.18,.17]))
    im=render_program(tokens,3 if stress else int(rng.integers(1,3)))
    if stress:im=cv2.GaussianBlur(im,(3,3),.7)
    im=np.clip(im.astype(float)*rng.uniform(.7,1)+rng.normal(0,2 if not stress else 5,im.shape),0,255).astype(np.uint8)
    return im[None],tokens

def main():
    out=ROOT/'data/processed/circuit-program-v1';out.mkdir(parents=True,exist_ok=True)
    if (out/'manifest.json').exists():raise ValueError('Preserve prepared data')
    cohorts={'tiny':(900,4),'train':(100000,8192),'validation':(110000,256),'stress':(120000,256)};hashes={};seen_programs=set();used_seeds={}
    for name,(start,count) in cohorts.items():
        records=[];seeds=[];seed=start
        while len(records)<count:
            r=compile_page(seed,name=='stress');signature=hashlib.sha256(r[1].tobytes()).hexdigest()
            if signature not in seen_programs:seen_programs.add(signature);records.append(r);seeds.append(seed)
            seed+=1
        used_seeds[name]=seeds;np.savez_compressed(out/(name+'.npz'),images=np.stack([r[0] for r in records]),program=np.stack([r[1] for r in records]));hashes[name]=hashlib.sha256((out/(name+'.npz')).read_bytes()).hexdigest()
        if name=='tiny':
            for i,r in enumerate(records):cv2.imwrite(str(out/f'tiny-{i}.png'),r[0][0])
    prior=json.loads((ROOT/'data/processed/masked-stroke/preparation.json').read_text());bank=np.load(ROOT/'data/processed/masked-stroke/pretrain.npz');images=[];codes=[];names=list(prior['sources'])
    for im,code in zip(bank['images'],bank['source']):images.append(cv2.cvtColor(im.transpose(1,2,0),cv2.COLOR_RGB2GRAY)[None]);codes.append(int(code))
    blocked=set();seen=set();excluded=[];external=[]
    for r in prior['pages']:seen.add(r['pixel_sha256'])
    from pcbparse.dataset import index_dataset
    for r in index_dataset(ROOT/'data/raw/PCBnet/Dataset300'):
        if r['split']=='validation' or r['id']=='0237':blocked.add(pixel_hash(read_image(r['image'])))
    for p in (ROOT/'data/raw/official/gt-0930/10GT').glob('*/*.png'):blocked.add(pixel_hash(read_image(p)))
    for p in (ROOT/'data/processed/aitee/images/val').glob('*'):
        if p.suffix.lower() in ('.png','.jpg','.jpeg'):blocked.add(pixel_hash(read_image(p)))
    cg_val={r['page'] for r in json.loads((ROOT/'data/processed/fields/cghd-records.json').read_text()) if r['split']=='validation'}
    for p in (ROOT/'data/raw/CGHD-v16-subset').rglob('*'):
        if p.parent.name=='images' and p.stem in cg_val and p.suffix.lower() in ('.png','.jpg','.jpeg'):blocked.add(pixel_hash(read_image(p)))
    intake=json.loads((ROOT/'data/raw/CircuitProgram-expanded/intake.json').read_text());names.append('ci2n-train');rng=np.random.default_rng(20261007)
    for row in intake['records']:
        if row['status']!='acquired' or not row['path'].lower().endswith(('.png','.jpg','.jpeg')):continue
        im=read_image(ROOT/row['local']);digest=pixel_hash(im)
        if digest in seen or digest in blocked:excluded.append(row['local']);continue
        seen.add(digest);gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY);ys,xs=np.nonzero(gray<220)
        if len(xs)<20:continue
        for _ in range(16):
            k=int(rng.integers(len(xs)));side=min(96,min(gray.shape));x=max(0,min(gray.shape[1]-side,int(xs[k])-side//2));y=max(0,min(gray.shape[0]-side,int(ys[k])-side//2))
            images.append(cv2.resize(gray[y:y+side,x:x+side],(48,48),interpolation=cv2.INTER_AREA)[None]);codes.append(len(names)-1)
        external.append({'path':row['local'],'pixel_sha256':digest,'patches':16})
    np.savez_compressed(out/'multisource-image-only.npz',images=np.stack(images),source=np.array(codes,np.int64))
    from collections import Counter
    result={'program_layouts':{k:v[1] for k,v in cohorts.items()},'program_seeds':used_seeds,'exact_program_duplicates_across_splits':0,'program_supervision':'own exact rendering/execution compiler only','sha256':hashes,
            'image_pretrain_sources':names,'image_patch_counts':dict(Counter(names[c] for c in codes)),
            'previous_unique_pages':prior['real_unique_page_counts'],'new_CI2N_unique_images':len(external),'CI2N_records':external,'excluded_new_pixel_duplicates':excluded,
            'external_net_labels_used':False,'Freeze_A_used':False,'GT10_used':False,
            'limitations':'grid-based grammar; only vertical R/C intervals; random programs not independent design families; image patches are not electrical supervision',
            'prior_bank_manifest_sha256':hashlib.sha256((ROOT/'data/processed/masked-stroke/preparation.json').read_bytes()).hexdigest()}
    (out/'manifest.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps({k:v for k,v in result.items() if k not in ('CI2N_records','excluded_new_pixel_duplicates','program_seeds')}))

if __name__=='__main__':main()
