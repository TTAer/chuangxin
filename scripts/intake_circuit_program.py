"""Train-only CI2N expansion and official CircuitSense metadata intake."""
import json,hashlib,urllib.request,urllib.parse,concurrent.futures,shutil,argparse
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=ROOT/'data/raw/CircuitProgram-intake');args=parser.parse_args()
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    if (out/'intake.json').exists():raise ValueError('Keep provenance')
    tree=json.loads((ROOT/'output/probeflow-data-intake/ci2n-tree.json').read_text());commit=tree['commit']
    paths=sorted(p for p in tree['paths'] if p.startswith('jumper_identification/images/train/') and p.lower().endswith(('.png','.jpg','.jpeg')))
    def fetch(path):
        short=hashlib.sha256(path.encode()).hexdigest()[:24]+Path(path).suffix if path not in ('LICENSE','README.en.md') else path
        dest=out/'CI2N'/short;dest.parent.mkdir(parents=True,exist_ok=True);url=f'https://raw.githubusercontent.com/LAD021/ci2n_datasets/{commit}/'+urllib.parse.quote(path,safe='/')
        row={'source':'CI2N','path':path,'url':url}
        try:
            old=ROOT/'data/raw/CI2N-probeflow-sample'/path
            expanded=ROOT/'data/raw/CircuitProgram-intake/CI2N'/path
            if expanded.exists():old=expanded
            downloaded=ROOT/'data/raw/CircuitProgram-intake/full-train/CI2N'/path
            if downloaded.exists():old=downloaded
            if old.exists():shutil.copyfile(old,dest);body=dest.read_bytes();row['reused_previous_download']=True
            else:
                with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'GraphSchematic-research'}),timeout=25) as response:body=response.read(12000001)
                if len(body)>12000000:raise ValueError('Individual image exceeds 12MB intake limit')
                dest.write_bytes(body)
            row.update(status='acquired',bytes=len(body),sha256=hashlib.sha256(body).hexdigest(),local=dest.relative_to(ROOT).as_posix())
        except Exception as e:row.update(status='unavailable',error=str(e))
        return row
    with concurrent.futures.ThreadPoolExecutor(max_workers=12) as pool:records=list(pool.map(fetch,['LICENSE','README.en.md']+paths))
    metadata=[]
    for name,url in [('CircuitSense-api','https://huggingface.co/api/datasets/happynewyue/CircuitSense'),('CircuitSense-card','https://huggingface.co/datasets/happynewyue/CircuitSense/raw/main/README.md')]:
        row={'name':name,'url':url}
        try:
            with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'GraphSchematic-research'}),timeout=20) as response:body=response.read(3000000)
            path=out/(name+'.json' if name.endswith('api') else name+'.md');path.write_bytes(body);row.update(status='acquired',local=path.relative_to(ROOT).as_posix(),sha256=hashlib.sha256(body).hexdigest())
        except Exception as e:row.update(status='unavailable',error=str(e))
        metadata.append(row)
    result={'commit':commit,'CI2N_requested_train_images':len(paths),'CI2N_acquired_train_images':sum(r['status']=='acquired' and r['path'].lower().endswith(('.png','.jpg','.jpeg')) for r in records),'records':records,'metadata':metadata,
            'use':'image-only encoder pretraining; no external net labels','author_validation_used':False,'family_independence_verified':False,
            'license':'CI2N repository Apache-2.0; original image rights require provenance; no redistribution',
            'gated_source_not_downloaded':'hanky2397/schematic_images requires personal contact sharing; no acceptance or bypass'}
    (out/'intake.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps({k:v for k,v in result.items() if k not in ('records','metadata')}),flush=True)

if __name__=='__main__':main()
