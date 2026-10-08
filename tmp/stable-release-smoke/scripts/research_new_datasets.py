"""Record live author-host metadata; advertised counts are not local counts."""
import json
from pathlib import Path
import urllib.request
import urllib.error

ROOT=Path(__file__).resolve().parents[1]
targets={
    'circuithub':'https://api.github.com/repos/360AILAB-NLP/CircuitPile-9k',
    'cghd_v1':'https://zenodo.org/api/records/6385814',
    'cghd_v16':'https://zenodo.org/api/records/17469897',
    'cghd_mirror':'https://huggingface.co/api/datasets/lowercaseonly/cghd/tree/main?recursive=true&limit=1000',
    'netlistify_readme':'https://raw.githubusercontent.com/NYCU-AI-EDA/Netlistify/main/README.md',
    'amsnet_readme':'https://raw.githubusercontent.com/AMS-Net/ams-net.github.io/main/README.md',
}
out=ROOT/'output/new-dataset-research'; out.mkdir(parents=True,exist_ok=True)
records={}
for name,url in targets.items():
    try:
        req=urllib.request.Request(url,headers={'User-Agent':'GraphSchematic-Research/1.0'})
        with urllib.request.urlopen(req,timeout=30) as r: body=r.read(8*1024**2); status=r.status
        suffix='.txt' if name.endswith('readme') else '.json'
        (out/(name+suffix)).write_bytes(body)
        records[name]={'url':url,'status':status,'metadata_file':str((out/(name+suffix)).relative_to(ROOT))}
        if not name.endswith('readme'):
            data=json.loads(body)
            if isinstance(data,dict): records[name]['files']=data.get('files',[]); records[name]['license']=data.get('metadata',{}).get('license')
            else: records[name]['entries']=len(data)
    except (urllib.error.URLError,ValueError) as e: records[name]={'url':url,'status':'unavailable','error':str(e)}
    print(json.dumps({name:records[name]},ensure_ascii=False),flush=True)
(out/'inventory.json').write_text(json.dumps(records,indent=2,ensure_ascii=False),encoding='utf-8')
