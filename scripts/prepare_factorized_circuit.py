"""New native-resolution renderer; preserve all prior banks."""
import json, hashlib, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import cv2,numpy as np
from scripts.prepare_circuit_program import compile_page, execute


def render_native(tokens,width=2,size=256):
    execute(tokens);scale=size/128;h,v=map(int,tokens[:2]);im=np.full((size,size),255,np.uint8)
    def p(x,y):return (int(round(x*scale)),int(round(y*scale)))
    xy=np.arange(6)*18+18
    for i in range(h):cv2.line(im,p(8,xy[i]),p(120,xy[i]),0,width)
    for j in range(v):cv2.line(im,p(xy[j],8),p(xy[j],120),0,width)
    for i in range(h):
        for j in range(v):
            if tokens[2+i*6+j]==9:cv2.circle(im,p(xy[j],xy[i]),max(int(3*scale),int(width/2)+2),0,-1)
    for i in range(h-1):
        for j in range(v):
            code=tokens[38+i*6+j]
            if code==10:continue
            x=xy[j];y=(xy[i]+xy[i+1])/2
            cv2.rectangle(im,p(x-6,y-7),p(x+6,y+7),255,-1)
            cv2.line(im,p(x,y-9),p(x,y-5),0,width);cv2.line(im,p(x,y+5),p(x,y+9),0,width)
            if code==11:
                points=np.array([p(x,y-5),p(x-3,y-3),p(x+3,y-1),p(x-3,y+1),p(x+3,y+3),p(x,y+5)])
                cv2.polylines(im,[points],False,0,max(1,int(round(width*.6))))
            else:
                cv2.line(im,p(x,y-5),p(x,y-2),0,width);cv2.line(im,p(x,y+2),p(x,y+5),0,width)
                cv2.line(im,p(x-4,y-2),p(x+4,y-2),0,width);cv2.line(im,p(x-4,y+2),p(x+4,y+2),0,width)
    return im


def main():
    out=ROOT/'data/processed/factorized-circuit-v1';out.mkdir(parents=True,exist_ok=True)
    if (out/'manifest.json').exists():raise ValueError('Preserve completed bank')
    blocked=set()
    for cohort in ('tiny','train','validation','stress'):
        with np.load(ROOT/f'data/processed/circuit-program-v2/{cohort}.npz') as z:
            blocked.update(hashlib.sha256(g.tobytes()).hexdigest() for g in z['program'])
    manifest={'native_resolution':256,'no_old_program_reuse':True,'cohorts':{},
              'train_style':'width1..5, sigma0..0.5, noise2, brightness0.7..1',
              'stress_style':'unseen widths6/7, sigma0.7, noise5; no selection',
              'geometry_prior':'fixed renderer slots, no per-image GT boxes supplied'}
    for name,count,start in (('tiny',4,210000),('train',4096,220000),('validation',256,230000),('stress',256,240000)):
        images=[];program=[];styles=[];seeds=[];seed=start
        while len(program)<count:
            _,g=compile_page(seed);sig=hashlib.sha256(g.tobytes()).hexdigest();r=np.random.default_rng(seed+7919)
            if sig in blocked:seed+=1;continue
            blocked.add(sig);width=int(r.integers(6,8) if name=='stress' else r.integers(1,6))
            sigma=.7 if name=='stress' else float(r.uniform(0,.5));im=render_native(g,width)
            if sigma>.1:im=cv2.GaussianBlur(im,(3,3),sigma)
            im=np.clip(im.astype(float)*r.uniform(.7,1)+r.normal(0,5 if name=='stress' else 2,im.shape),0,255).astype(np.uint8)
            images.append(im[None]);program.append(g);styles.append([width,sigma]);seeds.append(seed);seed+=1
        np.savez_compressed(out/f'{name}.npz',images=np.stack(images),program=np.stack(program),styles=styles)
        manifest['cohorts'][name]={'pages':count,'seeds':seeds,'sha256':hashlib.sha256((out/f'{name}.npz').read_bytes()).hexdigest()}
    # Controlled pairwise identifiability of isolated crossing and interval fixtures.
    diagnostic=[];base=np.full(68,7,np.int64);base[:2]=6;base[2:38]=8;base[38:]=10
    for size in (128,256):
        for width in range(1,8):
            for family,slot,codes,center in [('crossing',16,[8,9],(54,54)),('interval',52,[10,11,12],(54,63))]:
                patches=[]
                for code in codes:
                    g=base.copy();g[slot]=code;im=render_native(g,width,size);x,y=(int(c*size/128) for c in center);radius=int(7*size/128)
                    patches.append(im[y-radius:y+radius+1,x-radius:x+radius+1])
                pairs=[float(np.mean(patches[i]!=patches[j])) for i in range(len(codes)) for j in range(i+1,len(codes))]
                diagnostic.append({'size':size,'width':width,'kind':family,'pairwise_different_pixel_fractions':pairs,'exact_collision':any(x==0 for x in pairs)})
    (out/'observability.json').write_text(json.dumps({'scope':'isolated fixture exact-raster differences; not human observability or general identifiability','records':diagnostic},indent=2),encoding='utf-8')
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps({'prepared':True,'program_pages':4612,'exact_collision_records':sum(r['exact_collision'] for r in diagnostic)}))


if __name__=='__main__':main()
