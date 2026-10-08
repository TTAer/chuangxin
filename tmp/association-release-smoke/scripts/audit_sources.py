"""Inventory staged data without confusing render variants with independent designs."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from pcbparse.vision import read_image

records=[]; thumbnail=[]
for source,directory in (('PCBnet',ROOT/'data/processed/yolo'),('AITEE',ROOT/'data/processed/aitee')):
    for split in ('train','val'):
        for image in sorted((directory/'images'/split).glob('*')):
            if image.suffix.lower() not in ('.png','.jpg','.jpeg'): continue
            array=read_image(image); gray=cv2.cvtColor(array,cv2.COLOR_BGR2GRAY)
            small=cv2.resize(gray,(32,32),interpolation=cv2.INTER_AREA).astype(np.float32)/255
            dct=cv2.dct(small)[:8,:8]; bits=dct>np.median(dct.flatten()[1:])
            records.append({'source':source,'split':split,'image':str(image.relative_to(ROOT)),'sha256':hashlib.sha256(image.read_bytes()).hexdigest(),'decoded_sha256':hashlib.sha256(array.tobytes()).hexdigest(),'aspect_ratio':array.shape[1]/array.shape[0],'phash':sum(int(v)<<i for i,v in enumerate(bits.flatten()))})
            thumbnail.append(small)
flags=[]
for i,a in enumerate(records):
    for j in range(i):
        b=records[j]
        if a['split']==b['split']: continue
        exact=a['decoded_sha256']==b['decoded_sha256']
        distance=(a['phash']^b['phash']).bit_count()
        if exact or (distance<=4 and abs(a['aspect_ratio']/b['aspect_ratio']-1)<.02 and float(np.mean((thumbnail[i]-thumbnail[j])**2))<.002):
            flags.append({'a':a['image'],'b':b['image'],'decoded_exact':exact,'phash_distance':distance,'requires_visual_review':not exact})
omni=ROOT/'data/raw/OmniSch/sample_data'
summary={'counts':dict(Counter(r['source']+'/'+r['split'] for r in records)),'unique_byte_hashes':len({r['sha256'] for r in records}),'cross_split_duplicate_candidates':flags,'omnisch_designs':len(list(omni.glob('*'))),'omnisch_render_images':len(list(omni.rglob('*.png'))),'omnisch_training':'excluded; downloaded sample archive does not provide verified training split/license terms','records':records}
out=ROOT/'output/data-audit/sources.json'; out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(summary,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in summary.items() if k!='records'},indent=2))
