"""Reserve unseen AMSNet source groups for future adapter validation.

This manifest alone is not a ready PCB test set. No model inference or scores
are generated, and no assertion of graph-isomorphism independence is made.
"""
import hashlib
import json
from pathlib import Path
import re
ROOT=Path(__file__).resolve().parents[1]; base=ROOT/'data/raw/AMSNet-1.0/amsnet_1.0'
output=ROOT/'data/splits/amsnet-frozen.json'
if output.exists():
    print(json.dumps({'status':'already_frozen','manifest':str(output)})); raise SystemExit(0)
groups={}
for image in base.rglob('*.jpg'):
    if image.stem!=image.parent.name: continue
    cir=image.with_suffix('.cir').read_text(encoding='utf-8',errors='replace')
    normalized='\n'.join(re.sub(r'\s+',' ',line.strip()).upper() for line in cir.splitlines() if line.strip() and not line.lstrip().startswith('*'))
    signature=hashlib.sha256(normalized.encode()).hexdigest()
    groups.setdefault(signature,[]).append(image)
selected=[]
for signature,images in sorted(groups.items()):
    if any(p.stem in ('0','127') for p in images): continue  # Previously inspected.
    selected.append({'spice_text_group_sha256':signature,'records':[{'case':p.stem,'image':p.relative_to(ROOT).as_posix(),'image_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'spice':p.with_suffix('.cir').relative_to(ROOT).as_posix()} for p in images]})
    if len(selected)==25: break
manifest={'source':'AMSNet 1.0','status':'frozen source pool; evaluation adapter and provenance checks pending','groups':selected,'group_count':len(selected),'cases':sum(len(g['records']) for g in selected),'selection':'deterministic hash order, exclude previously inspected cases; whole identical normalized SPICE text groups reserved','independence_limit':'SPICE text hashing is not topology isomorphism; distinct IDs/parameters can conceal related designs','training_exclusion_required':True,'scores_generated':False,'pcb_domain_test_ready':False}
output.parent.mkdir(parents=True,exist_ok=True); output.write_text(json.dumps(manifest,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in manifest.items() if k!='groups'}),flush=True)
