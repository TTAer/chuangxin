"""Compile official symbol semantics only; omit case/project provenance."""
import hashlib
import json
from pathlib import Path
from collections import defaultdict
ROOT=Path(__file__).resolve().parents[1]
source=next((ROOT/'data/raw/official/public').rglob('cpntLibrary'))
raw=json.loads(source.read_text(encoding='utf-8')); entries={}
for model,c in raw.items():
    pins=defaultdict(lambda:{'names':set(),'types':set()})
    for p in c['pins']:
        number=str(p['number']).strip()
        if not number or any(ch in number for ch in ' ,().'): continue
        name=str(p['name']).strip()
        if name: pins[number]['names'].add(name)
        if p['pin_type']: pins[number]['types'].add(p['pin_type'])
    entries[model]={'pins':{n:{k:sorted(v) for k,v in p.items()} for n,p in pins.items()}}
result={'version':1,'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'source':'user-provided official public cpntLibrary; not GT inference','ambiguity_policy':'retain all number/name alternatives; abstain on unresolved alternatives','entries':entries}
out=ROOT/'models/semantics/component-library.json'; out.parent.mkdir(exist_ok=True,parents=True); out.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'entries':len(entries),'pin_numbers':sum(len(c['pins']) for c in entries.values()),'ambiguous_number_names':sum(len(p['names'])>1 for c in entries.values() for p in c['pins'].values())}))
