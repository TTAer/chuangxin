"""Source-balanced mixed training; PCB validation and AITEE validation are separate."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil
import yaml

ROOT=Path(__file__).resolve().parents[1]
pcb=ROOT/'data/processed/yolo'; extra=ROOT/'data/processed/aitee'; out=ROOT/'data/processed/multisource'
out.mkdir(parents=True,exist_ok=True)
names=list(yaml.safe_load((pcb/'dataset.yaml').read_text(encoding='utf-8'))['names'].values())
extra_names=['aitee_edge','Gnd','current_source','aitee_junction','R','voltage_source','ident-i','ident-i-dir','ident-res','ident-v','ident-v-dir']
for name in extra_names:
    if name not in names: names.append(name)
mapping={i:names.index(name) for i,name in enumerate(extra_names)}
lists={'train':[],'val':[],'extra-val':[]}; hashes={}; counts=Counter()
for source,base in (('pcbnet',pcb),('aitee',extra)):
    for split in ('train','val'):
        for image in sorted((base/'images'/split).glob('*')):
            if image.suffix.lower() not in ('.png','.jpg','.jpeg'): continue
            if source=='pcbnet' and split=='val' and image.stem=='0237': continue
            digest=hashlib.sha256(image.read_bytes()).hexdigest()
            if digest in hashes:
                if hashes[digest]!=split: raise ValueError('cross-source exact duplicate crosses split')
                continue
            hashes[digest]=split
            name=source+'-'+image.name
            dest=out/'images'/split/name; label=out/'labels'/split/(Path(name).stem+'.txt')
            dest.parent.mkdir(parents=True,exist_ok=True); label.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(image,dest)
            text=(base/'labels'/split/(image.stem+'.txt')).read_text(encoding='utf-8')
            if source=='aitee':
                text='\n'.join(str(mapping[int(line.split()[0])])+' '+' '.join(line.split()[1:]) for line in text.splitlines() if line.strip())+'\n'
            label.write_text(text,encoding='utf-8')
            key='extra-val' if source=='aitee' and split=='val' else split
            lists[key].extend([dest.as_posix()]*(3 if source=='pcbnet' and split=='train' else 1))
            counts[source+'/'+split]+=1
for name,files in lists.items(): (out/(name+'.txt')).write_text('\n'.join(files)+'\n',encoding='utf-8')
for name,val in (('dataset','val'),('external-validation','extra-val')):
    (out/(name+'.yaml')).write_text(yaml.safe_dump({'path':out.as_posix(),'train':'train.txt','val':val+'.txt','names':dict(enumerate(names))},sort_keys=False),encoding='utf-8')
report={'counts':dict(counts),'unique_images':sum(counts.values()),'train_sampling':'PCBnet repeated 3 times, AITEE once','checkpoint_selection':'PCBnet 55 validation pages (0237 excluded after visual duplicate audit); AITEE 166 pages evaluated separately','class_count':len(names),'aitee_mapping':mapping,'near_duplicate_check':'byte hash plus visual confirmation of 0027/0237; full source-design grouping remains unavailable'}
(out/'preparation.json').write_text(json.dumps(report,indent=2),encoding='utf-8'); print(json.dumps(report,indent=2))
