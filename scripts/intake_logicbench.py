"""Pinned author train-only images and directed graph annotations; CC BY 4.0."""
import json,hashlib,urllib.request,concurrent.futures
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
COMMIT='b5c169fab260487870072b7a6ab0c63f43bc9898'

def main():
    out=ROOT/'data/raw/LogicBench-train';out.mkdir(parents=True,exist_ok=True)
    if (out/'intake.json').exists():raise ValueError('Keep provenance')
    def fetch(path):
        row={'path':path,'url':f'https://raw.githubusercontent.com/tkcl-research/LogicBench1k/{COMMIT}/{path}'}
        try:
            with urllib.request.urlopen(urllib.request.Request(row['url'],headers={'User-Agent':'GraphSchematic-research'}),timeout=25) as response:body=response.read(4000001)
            if len(body)>4000000:raise ValueError('Unexpected file size')
            file=out/path;file.parent.mkdir(parents=True,exist_ok=True);file.write_bytes(body);row.update(status='acquired',bytes=len(body),sha256=hashlib.sha256(body).hexdigest(),local=file.relative_to(ROOT).as_posix())
        except Exception as e:row.update(status='unavailable',error=str(e))
        return row
    records=[fetch(path) for path in ('LICENSE','README.md','annotations/splits/train.txt','annotations/splits/val.txt','annotations/splits/test.txt')]
    split=out/'annotations/splits/train.txt'
    if not split.exists():raise ValueError('Author train split required')
    train=split.read_text().splitlines();val=set((out/'annotations/splits/val.txt').read_text().splitlines());test=set((out/'annotations/splits/test.txt').read_text().splitlines());assert not set(train)&(val|test)
    jobs=[p for key in train for p in (f'images/{key}.jpg',f'annotations/{key}.json')]
    with concurrent.futures.ThreadPoolExecutor(max_workers=12) as pool:records+=list(pool.map(fetch,jobs))
    result={'repository':'tkcl-research/LogicBench1k','commit':COMMIT,'author_train_ids':train,
            'train_images_acquired':sum(r['status']=='acquired' and r['path'].startswith('images/') for r in records),
            'train_annotations_acquired':sum(r['status']=='acquired' and r['path'].startswith('annotations/lb') for r in records),
            'records':records,'license':'CC BY 4.0; cite authors and dataset','use':'train images only for encoder pretraining; directed gate graph annotations quarantined, not electrical pin net labels',
            'val_test_images_read':False,'independent_verified_designs_added':0}
    (out/'intake.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps({k:v for k,v in result.items() if k not in ('records','author_train_ids')}),flush=True)

if __name__=='__main__':main()
