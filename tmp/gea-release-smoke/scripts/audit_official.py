"""Inventory user-provided official data; identify overlap before evaluation."""
import hashlib
import json
from collections import Counter
from pathlib import Path
import sys
import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from pcbparse.core import validate

def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def pixel_digest(p):
    a=cv2.imdecode(np.fromfile(p,dtype=np.uint8),cv2.IMREAD_COLOR)
    return hashlib.sha256(str(a.shape).encode()+a.tobytes()).hexdigest()

train={}
for p in (ROOT/'data/processed/multisource/images/train').glob('*'):
    train.setdefault(pixel_digest(p),[]).append(p.name)
sets={}; details={}
for name,folder in [('train200','public'),('public_gt10','public'),('original_gt10','gt-original'),('updated_gt10','gt-0930')]:
    base=ROOT/'data/raw/official'/folder
    images=sorted(base.rglob('*.png'))
    if name=='train200': images=[p for p in images if '200_train_cases' in p.parts]
    if name=='public_gt10': images=[p for p in images if '10_GTcase' in p.parts]
    records=[]; types=Counter(); errors=[]
    for image in images:
        annotation=next(image.parent.glob('*.json'))
        data=json.loads(annotation.read_text(encoding='utf-8-sig')); types.update(c['type'] for c in data['components'].values())
        issues=validate(data)
        if issues: errors.append({'case':image.stem,'errors':issues})
        pix=pixel_digest(image)
        records.append({'case':image.stem,'image':str(image.relative_to(ROOT)),'annotation':str(annotation.relative_to(ROOT)),'image_sha256':digest(image),'pixels_sha256':pix,'training_overlap':train.get(pix,[])})
    sets[name]={'images':len(images),'types':dict(types),'overlap_with_current_detector_training':sum(bool(r['training_overlap']) for r in records),'schema_invalid_cases':len(errors)}
    details[name]={'records':records,'schema_errors':errors}
old=details['original_gt10']['records']; new=details['updated_gt10']['records']
diff=[]
for a,b in zip(old,new):
    x=json.loads((ROOT/a['annotation']).read_text(encoding='utf-8')); y=json.loads((ROOT/b['annotation']).read_text(encoding='utf-8'))
    diff.append({'case':b['case'],'same_image':a['pixels_sha256']==b['pixels_sha256'],'changed_components':sum(x['components'].get(k)!=y['components'].get(k) for k in x['components'].keys()|y['components'].keys()),'changed_pins':sum(x['pins'].get(k)!=y['pins'].get(k) for k in x['pins'].keys()|y['pins'].keys()),'nets_changed':x['nets']!=y['nets']})
sources={p.name:{'bytes':p.stat().st_size,'sha256':digest(p)} for p in (ROOT/'官方资料').iterdir() if p.suffix!='.tar'}
result={'sources':sources,'sets':sets,'updated_gt_diff':diff,'details':details,'official_evaluator_available':False}
out=ROOT/'output/official-audit/inventory.json'; out.parent.mkdir(exist_ok=True,parents=True); out.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'sets':sets,'updated_gt_diff':diff},ensure_ascii=False,indent=2))
