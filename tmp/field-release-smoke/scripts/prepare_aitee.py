"""Deduplicate the author's split copies and retain all eleven class semantics."""
import hashlib
import json
from pathlib import Path
import shutil
import argparse
from collections import Counter
import yaml
from PIL import Image


def prepare(root, output):
    root=Path(root); out=Path(output).resolve(); out.mkdir(parents=True,exist_ok=True)
    names=['E','GND','I','P','R','voltage_source','ident-i','ident-i-dir','ident-res','ident-v','ident-v-dir']
    official={}
    for split in ('train','val'):
        for p in (root/'split/images'/split).glob('*'):
            if p.is_file(): official[p.name]=split
    seen={}; records=[]; counts=Counter(); objects=Counter(); issues=[]
    for p in sorted((root/'images').glob('*')):
        if p.suffix.lower() not in ('.png','.jpg','.jpeg'): continue
        label=root/'labels'/(p.stem+'.txt')
        if not label.exists(): issues.append({'image':p.name,'error':'missing_label'}); continue
        digest=hashlib.sha256(p.read_bytes()).hexdigest()
        split=official.get(p.name,'val' if int(digest[:8],16)%5==0 else 'train')
        if digest in seen:
            if seen[digest]!=split: raise ValueError('author split leaks exact duplicate image')
            continue
        seen[digest]=split
        with Image.open(p) as im: w,h=im.size
        lines=[]
        for line in label.read_text(encoding='utf-8-sig').splitlines():
            values=line.split()
            if len(values)!=5: raise ValueError('invalid YOLO annotation '+str(label))
            category=int(values[0]); x,y,bw,bh=map(float,values[1:])
            if category not in range(11) or not all(0<=v<=1 for v in (x,y,bw,bh)) or min(bw,bh)<=0: raise ValueError('invalid label bounds')
            lines.append(line); objects[names[category]]+=1
        dest=out/'images'/split; labels=out/'labels'/split
        dest.mkdir(parents=True,exist_ok=True); labels.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(p,dest/p.name); (labels/(p.stem+'.txt')).write_text('\n'.join(lines)+'\n',encoding='utf-8')
        counts[split]+=1; records.append({'image':p.name,'sha256':digest,'split':split,'width':w,'height':h,'objects':len(lines)})
    (out/'dataset.yaml').write_text(yaml.safe_dump({'path':out.as_posix(),'train':'images/train','val':'images/val','names':dict(enumerate(names))},sort_keys=False),encoding='utf-8')
    report={'source':'https://github.com/CKnievel/aitee-dataset','license':'Apache-2.0','unique_images':len(records),'counts':dict(counts),'objects':dict(objects),'issues':issues,'split':'author supplied train/val, exact duplicates checked; topology near duplicates not independently verified','semantics':'voltage_source is deliberately distinct from PCBnet V/vcc','records':records}
    (out/'preparation.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='records'},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--data',default='data/raw/AITEE/dataset'); p.add_argument('--output',default='data/processed/aitee'); a=p.parse_args(); prepare(a.data,a.output)
