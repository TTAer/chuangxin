"""Multi-source image-only pretraining bank, respecting source-held-out images."""
import hashlib,json,sys
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import cv2,numpy as np
from pcbparse.dataset import index_dataset
from pcbparse.vision import read_image


def pixel_hash(image):
    return hashlib.sha256(str(image.shape).encode()+image.tobytes()).hexdigest()


def main():
    out=ROOT/'data/processed/masked-stroke';out.mkdir(parents=True,exist_ok=True)
    if (out/'pretrain.npz').exists():raise ValueError('Preserve existing pretraining bank')
    rng=np.random.default_rng(20261006);blocked=set();seen=set();sources=[];exclusions=[]
    pcb=index_dataset(ROOT/'data/raw/PCBnet/Dataset300')
    for r in pcb:
        if r['split']=='validation' or r['id']=='0237':blocked.add(pixel_hash(read_image(r['image'])))
    for p in (ROOT/'data/raw/official/gt-0930/10GT').glob('*/*.png'):blocked.add(pixel_hash(read_image(p)))
    for split in ('val',):
        for p in (ROOT/'data/processed/aitee/images'/split).glob('*'):
            if p.suffix.lower() in ('.png','.jpg','.jpeg'):blocked.add(pixel_hash(read_image(p)))
    cg_records=json.loads((ROOT/'data/processed/fields/cghd-records.json').read_text(encoding='utf-8'))
    cg_train={r['page'] for r in cg_records if r['split']=='train'}
    cg_val={r['page'] for r in cg_records if r['split']=='validation'}
    cg_images=[p for p in (ROOT/'data/raw/CGHD-v16-subset').rglob('*') if p.parent.name=='images' and p.suffix.lower() in ('.png','.jpg','.jpeg')]
    for p in cg_images:
        if p.stem in cg_val:blocked.add(pixel_hash(read_image(p)))
    for r in pcb:
        if r['split']=='train' and r['id']!='0237':sources.append(('pcbnet',Path(r['image']),r['id']))
    for p in (ROOT/'data/processed/aitee/images/train').glob('*'):
        if p.suffix.lower() in ('.png','.jpg','.jpeg'):sources.append(('aitee',p,p.stem))
    for p in cg_images:
        if p.stem in cg_train:sources.append(('cghd',p,p.stem))
    official=json.loads((ROOT/'output/official-audit/inventory.json').read_text(encoding='utf-8'))
    for r in official['details']['train200']['records']:
        if r['training_overlap']:sources.append(('official-train200',ROOT/r['image'],r['case']))
    intake=ROOT/'data/raw/OpenSchematics-pretrain-sample/intake.json'
    if intake.exists():
        for r in json.loads(intake.read_text(encoding='utf-8')):
            if r.get('image'):sources.append(('open-schematics',ROOT/r['image'],r['repository']))
    patches=[];records=[];pages=[]
    for source,path,family in sources:
        try:image=read_image(path)
        except Exception as e:exclusions.append({'image':str(path.relative_to(ROOT)),'reason':type(e).__name__});continue
        digest=pixel_hash(image)
        if digest in blocked or digest in seen:
            exclusions.append({'image':str(path.relative_to(ROOT)),'reason':'held_out_pixel_duplicate' if digest in blocked else 'cross_source_pixel_duplicate'});continue
        seen.add(digest);gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        yy,xx=np.nonzero(gray<220)
        if len(xx)<30:exclusions.append({'image':str(path.relative_to(ROOT)),'reason':'insufficient_stroke_pixels'});continue
        h,w=gray.shape;accepted=0
        for _ in range(24):
            j=int(rng.integers(len(xx)));side=int(rng.choice([64,96,144,192]));cx,cy=xx[j],yy[j]
            a=max(0,min(max(0,w-side),int(cx-side/2)));b=max(0,min(max(0,h-side),int(cy-side/2)))
            crop=image[b:b+side,a:a+side]
            small=cv2.resize(crop,(48,48),interpolation=cv2.INTER_AREA)
            ink=(small.mean(2)<220).mean()
            if ink<.005 or ink>.75:continue
            patches.append(small[:,:,::-1].transpose(2,0,1).copy())
            records.append({'source':source,'image':str(path.relative_to(ROOT)),'image_pixel_sha256':digest,
                            'family_or_page':family,'box':[a,b,a+crop.shape[1],b+crop.shape[0]]})
            accepted+=1
            if accepted==16:break
        if accepted:pages.append({'source':source,'image':str(path.relative_to(ROOT)),'pixel_sha256':digest,'family_or_page':family,'patches':accepted})
    # Own procedural layout families only; use existing source training split.
    synthetic_records=json.loads((ROOT/'data/processed/fields/synthetic-records.json').read_text(encoding='utf-8'))
    with np.load(ROOT/'data/processed/fields/synthetic.npz') as ar:synthetic=ar['images']
    for i,r in enumerate(synthetic_records):
        if r['split']!='train':continue
        im=synthetic[i].transpose(1,2,0)
        for y,x in [(0,0),(48,48),(96,96),(144,144)]:
            patches.append(im[y:y+48,x:x+48,::-1].transpose(2,0,1).copy())
            records.append({'source':'own-synthetic','family_or_page':r['page'],'source_tile':i,'box':[x,y,x+48,y+48]})
    source_names=sorted({r['source'] for r in records});codes=np.array([source_names.index(r['source']) for r in records],np.int64)
    np.savez_compressed(out/'pretrain.npz',images=np.stack(patches),source=codes)
    (out/'records.json').write_text(json.dumps(records),encoding='utf-8')
    summary={'sources':source_names,'patch_counts':dict(Counter(r['source'] for r in records)),
             'real_unique_page_counts':dict(Counter(p['source'] for p in pages)),
             'procedural_families':len({r['family_or_page'] for r in records if r['source']=='own-synthetic'}),
             'pages':pages,'exclusions':exclusions,'holdout_pixel_hashes':len(blocked),
             'holdout_overlap':0,'supervision':'image-only masked stroke reconstruction; no net labels from external images',
             'sampling':'16 image-guided patches per real page; source-balanced training sampling',
             'Freeze_A_used':False,'GT10_pretrained':False,'external_sources_not_used':['AMSNet rights unresolved','CircuitOCR image scope unresolved','OmniSch sample provenance unresolved','SchGen text/code not rendered'],
             'weight_scope':'research-only local artifact; CGHD archive CC BY-SA 3.0 attribution retained, OpenSchematics original repository licenses retained; no raw images redistributed',
             'limitations':'source/page splits and pixel duplicate exclusion do not prove historical design-family independence'}
    (out/'preparation.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in summary.items() if k not in ('pages','exclusions')}),flush=True)


if __name__=='__main__':main()
