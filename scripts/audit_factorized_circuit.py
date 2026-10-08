"""Split observed scores by render identifiability; evaluate a deterministic template control."""
import sys,json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from PIL import Image,ImageDraw
from scripts.prepare_factorized_circuit import render_native
from scripts.train_circuit_program import metrics


def template_predict(images):
    predictions=[]
    for image in images[:,0]:
        binary=image<float(np.median(image))*.5
        h=sum(bool(binary[36+36*i-6:36+36*i+7,20:27].any()) for i in range(6))
        v=sum(bool(binary[20:27,36+36*i-6:36+36*i+7].any()) for i in range(6))
        h=int(np.clip(h,3,6));v=int(np.clip(v,3,6))
        # Estimate stroke thickness from a horizontal line isolated at left border.
        thickness=int(binary[24:49,24].sum())
        widths=[w for w in range(1,8) if (w if w==1 else (w+1 if w%2==0 else w+2))==thickness]
        if not widths:widths=list(range(1,8))
        g=np.full(68,7,np.int64);g[:2]=[h,v]
        g[2:38]=8;g[38:]=10
        for slot in range(2,68):
            crossing=slot<38;k=slot-2 if crossing else slot-38;i,j=divmod(k,6)
            if i>=h-(not crossing) or j>=v:g[slot]=7;continue
            x=36+36*j;y=36+36*i+(0 if crossing else 18)
            rx,ry=(8,8) if crossing else (12,10)
            patch=binary[y-ry:y+ry+1,x-rx:x+rx+1]
            options=[]
            # Template all-wire 6x6 context, independent of the target label.
            for code in ([8,9] if crossing else [10,11,12]):
                for w in widths:
                    t=np.full(68,7,np.int64);t[:2]=6;t[2:38]=8;t[38:]=10;t[slot]=code
                    im=render_native(t,w);ref=im[y-ry:y+ry+1,x-rx:x+rx+1]<128
                    options.append((float(np.mean(patch!=ref)),code))
            g[slot]=min(options)[1]
        # Canonicalize absent slots even though inactive prefixes affect templates.
        for i in range(6):
            for j in range(6):
                if i>=h or j>=v:g[2+i*6+j]=7
        for i in range(5):
            for j in range(6):
                if i>=h-1 or j>=v:g[38+i*6+j]=7
        predictions.append(g)
    return np.stack(predictions)


def main():
    out=ROOT/'output/factorized-circuit-audit';out.mkdir(parents=True,exist_ok=True)
    if (out/'report.json').exists():raise ValueError('Preserve audit')
    summary=json.loads((ROOT/'output/training/factorized-circuit-v1/summary.json').read_text())
    if not summary['full_training_executed']:raise ValueError('Training gate did not pass')
    report={'scope':'fixed native256 grammar only; no official evaluator','cohorts':{}}
    base=np.full(68,7,np.int64);base[:2]=6;base[2:38]=8;base[38:]=10
    changed=base.copy();changed[16]=9
    report['full_raster_join_collision_width7']=bool(np.array_equal(render_native(base,7),render_native(changed,7)))
    for name in ('validation','stress'):
        with np.load(ROOT/f'data/processed/factorized-circuit-v1/{name}.npz') as z:
            image,gold,styles=z['images'],z['program'],z['styles']
        with np.load(ROOT/f'output/training/factorized-circuit-v1/{name}-predictions.npz') as z:pred=z['program']
        template=template_predict(image)
        groups={}
        for width in sorted(set(styles[:,0])):
            mask=styles[:,0]==width
            groups[str(int(width))]={'model':metrics(pred[mask],gold[mask]),'template':metrics(template[mask],gold[mask])}
        report['cohorts'][name]={'model':metrics(pred,gold),'template':metrics(template,gold),'by_width':groups}
        np.savez_compressed(out/f'{name}-template-predictions.npz',program=template)
        canvas=Image.new('RGB',(768,4*280),'white');draw=ImageDraw.Draw(canvas)
        for i in range(4):
            draw.text((4,i*280+3),f'{name} {i}, width {int(styles[i,0])}: input | gold canonical | model canonical',fill='black')
            for col,im in enumerate((image[i,0],render_native(gold[i],2),render_native(pred[i],2))):canvas.paste(Image.fromarray(im).convert('RGB'),(col*256,i*280+24))
        canvas.save(out/f'{name}-first4.png')
    old=json.loads((ROOT/'output/circuit-program-research-manifest.json').read_text())
    bad=[p for p,h in old['sha256'].items() if hashlib.sha256((ROOT/p).read_bytes()).hexdigest()!=h]
    report['previous_research_preserved']={'checked':len(old['sha256']),'mismatches':bad}
    assert not bad
    (out/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({'full_raster_join_collision_width7':report['full_raster_join_collision_width7'],
                      'template_validation':report['cohorts']['validation']['template']['net_pair'],
                      'template_stress':report['cohorts']['stress']['template']['net_pair'],
                      'by_stress_width':report['cohorts']['stress']['by_width']}))


if __name__=='__main__':main()
