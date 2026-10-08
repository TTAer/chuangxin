"""Native nonuniform geometry, fixed algorithms and weights, new excluded-program holdouts."""
import sys,json,hashlib,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,cv2,torch
from PIL import Image,ImageDraw
from scripts.prepare_circuit_program import compile_page,execute
from scripts.circuit_geometry_guard_v2 import register_guarded_v2
from scripts.ordered_line_rectification_v2 import register_ordered
from experiments.contrast_edge_circuit_reader import ContrastOnlyReader
from scripts.train_circuit_program import metrics


def render_irregular(g,width,xs,ys):
    execute(g);im=np.full((256,256),255,np.uint8);h,v=map(int,g[:2])
    xs=np.rint(xs).astype(int);ys=np.rint(ys).astype(int)
    for y in ys[:h]:cv2.line(im,(16,int(y)),(240,int(y)),0,width)
    for x in xs[:v]:cv2.line(im,(int(x),16),(int(x),240),0,width)
    for i in range(h-1):
        for j in range(v):
            code=int(g[38+i*6+j]);x=int(xs[j]);y=int(round((ys[i]+ys[i+1])/2))
            if code==10:continue
            cv2.rectangle(im,(x-12,y-14),(x+12,y+14),255,-1)
            cv2.line(im,(x,int(ys[i])),(x,y-10),0,width)
            cv2.line(im,(x,y+10),(x,int(ys[i+1])),0,width)
            if code==11:
                points=np.array([(x,y-10),(x-6,y-6),(x+6,y-2),(x-6,y+2),(x+6,y+6),(x,y+10)])
                cv2.polylines(im,[points],False,0,max(1,int(round(width*.6))))
            else:
                cv2.line(im,(x,y-10),(x,y-4),0,width);cv2.line(im,(x,y+4),(x,y+10),0,width)
                cv2.line(im,(x-8,y-4),(x+8,y-4),0,width);cv2.line(im,(x-8,y+4),(x+8,y+4),0,width)
    for i in range(h):
        for j in range(v):
            if g[2+i*6+j]==9:cv2.circle(im,(int(xs[j]),int(ys[i])),max(6,width+3),0,-1)
    return im


def main():
    out=ROOT/'output/ordered-rectification-holdout';out.mkdir(parents=True,exist_ok=True)
    if (out/'protocol.json').exists():raise ValueError('Preserve holdout')
    assert json.loads((ROOT/'output/ordered-rectification-v2-development/report.json').read_text())['gate_passed']
    protocol={'scope':'finite native256 orthogonal renderer, not real schematic validation',
              'new_cohorts':'uniform128; irregular256 each axis jitter integer -4:4; shear128; rotation128',
              'style':'global scale.55:.85, width1..7, blur.9:1.2, noise5, brightness.5:1',
              'native_symbols':'fixed R/C shape drawn at actual interval midpoints, not canonical image warp',
              'excluded':'all previous program banks and registration/guard holdouts',
              'comparison':'frozen GuardV2 vs OrderedV2; same frozen ContrastOnlyReader, no GT geometry inputs',
              'gate':'irregular coverage>=.90 full F1>=.95 exact-net>=.90 and ready identity errors=0; unsupported acceptance<=.05',
              'no_training':True,'no_holdout_selection':True,
              'algorithm_hash':hashlib.sha256((ROOT/'scripts/ordered_line_rectification_v2.py').read_bytes()).hexdigest(),
              'model_hash':hashlib.sha256((ROOT/'output/training/contrast-edge-v1-contrast/best.pt').read_bytes()).hexdigest()}
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    # Separate finite clean-raster counterfactual admission before any holdout images.
    comparisons=[]
    for seed in range(810000,810016):
        _,g=compile_page(seed);rng=np.random.default_rng(seed);xs=36+36*np.arange(6)+rng.integers(-4,5,6);ys=36+36*np.arange(6)+rng.integers(-4,5,6)
        for width in range(1,8):
            im=render_irregular(g,width,xs,ys)
            for slot in range(2,68):
                if g[slot]==7:continue
                for code in ([8,9] if slot<38 else [10,11,12]):
                    if code==g[slot]:continue
                    altered=g.copy();altered[slot]=code
                    comparisons.append({'seed':seed,'width':width,'slot':slot,'different_pixels':int(np.count_nonzero(im!=render_irregular(altered,width,xs,ys)))})
    admission={'scope':'16 separate native fixture pages, widths1..7, clean exact pixels only; finite screen',
               'comparisons':len(comparisons),'collisions':[d for d in comparisons if d['different_pixels']==0],
               'min_different_pixels':min(d['different_pixels'] for d in comparisons)}
    (out/'admission.json').write_text(json.dumps(admission,indent=2),encoding='utf-8')
    if admission['collisions']:raise ValueError('Known native raster collisions')
    seen=set()
    for bank in ('circuit-program-v1','circuit-program-v2','factorized-circuit-v1','separable-circuit-v1','registered-circuit-v1','contrast-edge-v1'):
        for p in (ROOT/f'data/processed/{bank}').glob('*.npz'):
            with np.load(p) as z:
                if 'program' in z:seen.update(hashlib.sha256(g.tobytes()).hexdigest() for g in z['program'])
    for folder in ('registration-v2-holdout','guard-v2-holdout'):
        for p in (ROOT/f'output/{folder}').glob('*-inputs.npz'):
            with np.load(p) as z:seen.update(hashlib.sha256(g.tobytes()).hexdigest() for g in z['program'])
    model=ContrastOnlyReader().cuda().eval();torch.set_num_threads(4)
    model.load_state_dict(torch.load(ROOT/'output/training/contrast-edge-v1-contrast/best.pt',weights_only=True,map_location='cuda'))
    def infer(images):
        rows=[]
        with torch.no_grad():
            for i in range(0,len(images),16):
                x=torch.tensor(1-images[i:i+16].astype(np.float32)/255,device='cuda')
                with torch.autocast('cuda',dtype=torch.float16):rows.append(model.generate(x).cpu().numpy())
        return np.concatenate(rows)
    report={'protocol':protocol,'admission':admission,'cohorts':{}};started=time.perf_counter()
    for name,n,start in (('uniform',128,820000),('irregular',256,830000),('shear',128,840000),('rotation',128,850000)):
        raw=[];gold=[];seeds=[];layouts=[];seed=start
        while len(gold)<n:
            _,g=compile_page(seed);sig=hashlib.sha256(g.tobytes()).hexdigest();r=np.random.default_rng(seed+33001)
            if sig in seen:seed+=1;continue
            seen.add(sig);base=36+36*np.arange(6)
            xs=base+(r.integers(-4,5,6) if name!='uniform' else 0);ys=base+(r.integers(-4,5,6) if name!='uniform' else 0)
            sx,sy=r.uniform(.55,.85,2);tx=128*(1-sx)+r.uniform(-10,10);ty=128*(1-sy)+r.uniform(-10,10)
            skew=float(r.choice([-1,1])*r.uniform(.12,.2)) if name=='shear' else 0
            affine=np.array([[sx,skew,tx-128*skew],[0,sy,ty]],np.float32)
            im=cv2.warpAffine(render_irregular(g,int(r.integers(1,8)),xs,ys),affine,(256,256),borderValue=255)
            angle=float(r.choice([-1,1])*r.uniform(8,15)) if name=='rotation' else 0
            if angle:im=cv2.warpAffine(im,cv2.getRotationMatrix2D((128,128),angle,1),(256,256),borderValue=255)
            im=cv2.GaussianBlur(im,(5,5),float(r.uniform(.9,1.2)))
            im=np.clip(im.astype(float)*r.uniform(.5,1)+r.normal(0,5,im.shape),0,255).astype(np.uint8)
            raw.append(im[None]);gold.append(g);seeds.append(seed);layouts.append({'xs':xs.tolist(),'ys':ys.tolist(),'affine':affine.tolist(),'angle':angle});seed+=1
        raw=np.stack(raw);gold=np.stack(gold)
        np.savez_compressed(out/f'{name}-inputs.npz',observed_images=raw,program=gold,seeds=seeds)
        (out/f'{name}-quarantined-layouts.json').write_text(json.dumps(layouts),encoding='utf-8')
        results={};predictions={};details={};images={}
        for version,algorithm in (('uniform_guard',register_guarded_v2),('ordered',register_ordered)):
            records=[algorithm(im) for im in raw[:,0]];images[version]=np.stack([a[None] for a,d in records]);details[version]=[d for a,d in records]
            ready=np.array([d['ready'] for d in details[version]]);p=infer(images[version]);predictions[version]=p
            errors=[i for i,(d,g) in enumerate(zip(details[version],gold)) if d['ready'] and (len(d['observed_rows'])!=g[0] or len(d['observed_columns'])!=g[1])]
            results[version]={'ready_pages':int(ready.sum()),'coverage':float(ready.mean()),'ready_identity_error_cases':errors,
                              'all_model':metrics(p,gold),'ready_model':metrics(p[ready],gold[ready]) if ready.any() else None}
        report['cohorts'][name]=results;np.savez_compressed(out/f'{name}-predictions.npz',**predictions)
        (out/f'{name}-geometry.json').write_text(json.dumps(details,indent=2),encoding='utf-8')
        canvas=Image.new('RGB',(768,1120),'white');draw=ImageDraw.Draw(canvas)
        for i in range(4):
            draw.text((4,i*280+3),f'{name} {i} raw | uniform guard | ordered warp',fill='black')
            for col,im in enumerate((raw[i,0],images['uniform_guard'][i,0],images['ordered'][i,0])):canvas.paste(Image.fromarray(im).convert('RGB'),(col*256,i*280+24))
        canvas.save(out/f'{name}-first4.png');print(json.dumps({'cohort':name,**results}),flush=True)
    p=report['cohorts']['irregular']['ordered']
    report['gate_passed']=p['coverage']>=.9 and p['all_model']['net_pair']['micro_F1']>=.95 and p['all_model']['exact_net_partition']>=.9 and not p['ready_identity_error_cases'] and all(report['cohorts'][n]['ordered']['coverage']<=.05 for n in ('shear','rotation'))
    report['seconds']=time.perf_counter()-started
    (out/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')


if __name__=='__main__':main()
