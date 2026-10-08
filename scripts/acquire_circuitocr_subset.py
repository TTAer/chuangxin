"""Acquire 64 training/16 validation image-text pairs; inspect, do not train."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import re
import urllib.request
import cv2
import numpy as np
ROOT=Path(__file__).resolve().parents[1]; out=ROOT/'data/raw/CircuitOCR'; records=[]
def sha(data): return hashlib.sha256(data).hexdigest()
known=set()
audit=json.loads((ROOT/'output/official-audit/inventory.json').read_text(encoding='utf-8'))
for group in audit.get('details',{}).values():
    if isinstance(group,dict): known.update(r.get('pixels_sha256') for r in group.get('records',[]) if r.get('pixels_sha256'))
def get(item):
    split,index,row=item; relative=row['images'][0]; path=Path(relative)
    if path.is_absolute() or '..' in path.parts: return {'split':split,'index':index,'status':'unsafe_path'}
    url='https://raw.githubusercontent.com/ZhangJ83/circuit_ocr_dataset_final/master/dataset_a/'+relative
    destination=out/'subset'/split/path.name
    try:
        if not destination.exists():
            with urllib.request.urlopen(url,timeout=30) as response: data=response.read(5*1024**2+1)
            if len(data)>5*1024**2: raise ValueError('image exceeds download cap')
            destination.parent.mkdir(parents=True,exist_ok=True); destination.write_bytes(data)
        data=destination.read_bytes(); image=cv2.imdecode(np.frombuffer(data,np.uint8),cv2.IMREAD_COLOR)
        if image is None: raise ValueError('invalid image')
        text='\n'.join(m['content'] for m in row['messages'] if m['role']=='assistant')
        design_hash=sha(re.sub(r'\s+','',text).upper().encode('utf-8'))
        pixels=sha(str(image.shape).encode()+image.tobytes())
        return {'split':split,'index':index,'url':url,'local_path':str(destination.relative_to(ROOT)).replace('\\','/'),'status':'downloaded',
                'sha256':sha(data),'pixels_sha256':pixels,'dimensions':[image.shape[1],image.shape[0]],'transcript_sha256':design_hash,
                'transcript':text,'exact_pixel_overlap_official_inventory':pixels in known,'spatial_pin_annotations':False}
    except Exception as error: return {'split':split,'index':index,'url':url,'status':'unavailable','error':str(error)}
items=[]
for split,count in (('train',64),('val',16)):
    rows=[json.loads(s) for s in (out/(split+'.jsonl')).read_text(encoding='utf-8').splitlines()]
    items.extend((split,i,row) for i,row in enumerate(rows[:count]))
with ThreadPoolExecutor(max_workers=4) as pool:
    for record in pool.map(get,items): records.append(record)
downloaded=[r for r in records if r['status']=='downloaded']; train={r['transcript_sha256'] for r in downloaded if r['split']=='train'}; val={r['transcript_sha256'] for r in downloaded if r['split']=='val'}
summary={'requested':len(items),'downloaded':len(downloaded),'unique_pixels':len({r['pixels_sha256'] for r in downloaded}),
         'transcript_groups':len({r['transcript_sha256'] for r in downloaded}),'train_val_transcript_overlap_groups':len(train&val),
         'official_exact_pixel_overlaps':sum(r['exact_pixel_overlap_official_inventory'] for r in downloaded),
         'declared_license':'MIT per README; LICENSE file missing at checked revision','usage':'inspection and OCR-development candidate pool only; not used in SideStripNet training',
         'source':'https://github.com/ZhangJ83/circuit_ocr_dataset_final','records':records}
(out/'subset-manifest.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in summary.items() if k!='records'}),flush=True)
