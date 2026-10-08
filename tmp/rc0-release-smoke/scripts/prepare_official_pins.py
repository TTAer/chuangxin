"""Official pin-coordinate supervision, preserving existing page split."""
from collections import Counter
import json
import math
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from pcbparse.pins import crop_tensor
from pcbparse.vision import read_image

audit=json.loads((ROOT/'output/official-audit/inventory.json').read_text(encoding='utf-8'))
records=[]; arrays=[]; skipped=Counter(); pixels_seen=set()
for item in audit['details']['train200']['records']:
    if item['pixels_sha256'] in pixels_seen: continue
    pixels_seen.add(item['pixels_sha256'])
    image=read_image(ROOT/item['image']); h,w=image.shape[:2]
    raw=json.loads((ROOT/item['annotation']).read_text(encoding='utf-8'))
    split='train' if item['training_overlap'] else 'validation'
    for cid,c in raw['components'].items():
        b=c.get('bbox'); pp=list(raw['pins'].get(cid,{}).values())
        if not b or len(b)!=4 or not all(math.isfinite(v) for v in b) or b[2]<=b[0] or b[3]<=b[1]: skipped['invalid_bbox']+=1; continue
        points=[(p['point']['x'],h-p['point']['y']) for p in pp if isinstance(p.get('point'),dict) and all(isinstance(p['point'].get(k),(float,int)) and math.isfinite(p['point'][k]) for k in ('x','y'))]
        points=[(x,y) for x,y in points if 0<=x<w and 0<=y<h]
        if not points: skipped['no_valid_points']+=1; continue
        # Oracle body + actual annotated endpoints is used only for training crops.
        # It is not an inference lookup. Predicted boxes remain mandatory at runtime.
        box=[min(b[0],min(x for x,y in points)),min(h-b[3],min(y for x,y in points)),max(b[2],max(x for x,y in points)),max(h-b[1],max(y for x,y in points))]
        try: tensor,crop=crop_tensor(image,box)
        except ValueError: skipped['empty_crop']+=1; continue
        targets=[[0,(x-crop[0])/(crop[2]-crop[0])*96-.5,(y-crop[1])/(crop[3]-crop[1])*96-.5] for x,y in set(points)]
        arrays.append(np.rint(tensor[:4]*255).clip(0,255).astype(np.uint8))
        records.append({'page':item['case'],'component':cid,'split':split,'category':c['type'],'box':crop,'targets':targets,'supervision':'official 200 training pin coordinates; raw graph schema not required; weak/noisy annotations'})
out=ROOT/'data/processed/official-pins'; out.mkdir(parents=True,exist_ok=True)
np.savez_compressed(out/'crops.npz',images=np.stack(arrays)); (out/'records.json').write_text(json.dumps(records,ensure_ascii=False),encoding='utf-8')
summary={'crops':len(records),'split_counts':dict(Counter(r['split'] for r in records)),'pages':dict(Counter('train' if r['training_overlap'] else 'validation' for r in audit['details']['train200']['records'])),'skipped':dict(skipped),'updated_gt_used_for_training':False,'evaluation_caveat':'oracle union body/endpoint boxes; not end-to-end pin F1; page split retained from existing detector'}
(out/'preparation.json').write_text(json.dumps(summary,indent=2),encoding='utf-8'); print(json.dumps(summary))
