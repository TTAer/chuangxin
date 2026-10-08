"""Pinned CI2N train-only image intake; annotations are not electrical truth."""
import concurrent.futures,hashlib,json,urllib.request,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))

def main():
    tree=json.loads((ROOT/'output/probeflow-data-intake/ci2n-tree.json').read_text(encoding='utf-8'));commit=tree['commit']
    dest=ROOT/'data/raw/CI2N-probeflow-sample';dest.mkdir(exist_ok=True)
    paths=sorted(p for p in tree['paths'] if p.startswith('jumper_identification/images/train/') and p.endswith('.png'))[:64]
    jobs=['LICENSE','README.en.md']+paths
    def fetch(path):
        url='https://raw.githubusercontent.com/LAD021/ci2n_datasets/'+commit+'/'+path
        row={'path':path,'url':url}
        try:
            req=urllib.request.Request(url,headers={'User-Agent':'GraphSchematic-research'})
            with urllib.request.urlopen(req,timeout=20) as response:body=response.read(8000000)
            local=dest/path;local.parent.mkdir(parents=True,exist_ok=True);local.write_bytes(body)
            row.update(status='acquired',bytes=len(body),sha256=hashlib.sha256(body).hexdigest(),local=local.relative_to(ROOT).as_posix())
        except Exception as e:row.update(status='unavailable',error=str(e))
        return row
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:records=list(pool.map(fetch,jobs))
    result={'source':'LAD021/ci2n_datasets','commit':commit,'requested_train_images':len(paths),'acquired_train_images':sum(r['status']=='acquired' and r['path'].endswith('.png') for r in records),
            'records':records,'use':'image-only contrastive encoder pretraining, no electrical labels inferred from jumper annotations','author_validation_not_used':True,
            'license':'repository Apache-2.0; source images may retain original source rights; research-only, no raw redistribution','historical_family_independence_verified':False}
    (dest/'intake.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps({k:v for k,v in result.items() if k!='records'}),flush=True)

if __name__=='__main__':main()
