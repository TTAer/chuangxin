"""Create image-hash-grouped train/validation data for detector fine-tuning."""
import argparse
import json
from pathlib import Path
import shutil
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from pcbparse.dataset import index_dataset


def prepare(root,output):
    items=index_dataset(root); out=Path(output).resolve(); out.mkdir(parents=True,exist_ok=True)
    categories=sorted({c['category'] for i in items for c in json.loads(Path(i['annotation']).read_text(encoding='utf-8'))['components'].values()})
    counts={'train':0,'validation':0}; problems=[]
    for item in items:
        split='val' if item['split']=='validation' else 'train'
        dest=out/'images'/split; labels=out/'labels'/split
        dest.mkdir(parents=True,exist_ok=True); labels.mkdir(parents=True,exist_ok=True)
        p=Path(item['image']); name=item['id']+p.suffix.lower()
        shutil.copyfile(p,dest/name)
        raw=json.loads(Path(item['annotation']).read_text(encoding='utf-8'))
        w,h=raw['canvas']['width'],raw['canvas']['height']; lines=[]
        for cid,c in raw['components'].items():
            b=c['location']; x1=max(0,min(w,b['x1'])); y1=max(0,min(h,b['y1'])); x2=max(0,min(w,b['x2'])); y2=max(0,min(h,b['y2']))
            if x2<=x1 or y2<=y1: problems.append({'case':item['id'],'component':cid,'issue':'invalid_bbox'}); continue
            lines.append(f'{categories.index(c["category"])} {(x1+x2)/2/w:.8f} {(y1+y2)/2/h:.8f} {(x2-x1)/w:.8f} {(y2-y1)/h:.8f}')
        (labels/(item['id']+'.txt')).write_text('\n'.join(lines)+'\n',encoding='utf-8')
        counts[item['split']]+=1
    dataset={'path':out.as_posix(),'train':'images/train','val':'images/val','names':dict(enumerate(categories))}
    import yaml
    (out/'dataset.yaml').write_text(yaml.safe_dump(dataset,sort_keys=False,allow_unicode=True),encoding='utf-8')
    (out/'index.json').write_text(json.dumps(items,ensure_ascii=False,indent=2),encoding='utf-8')
    (out/'preparation.json').write_text(json.dumps({'counts':counts,'categories':categories,'issues':problems,'split_method':'sha256 modulo 5; exact duplicates grouped, near duplicates not yet grouped'},ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'counts':counts,'classes':len(categories),'issues':len(problems),'yaml':str(out/'dataset.yaml')},ensure_ascii=False,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--data',default='data/raw/PCBnet/Dataset300'); p.add_argument('--output',default='data/processed/yolo')
    args=p.parse_args(); prepare(args.data,args.output)
