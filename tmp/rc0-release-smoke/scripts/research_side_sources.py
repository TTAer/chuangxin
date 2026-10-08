"""Acquire bounded public-source metadata and a new OCR archive for inspection."""
import hashlib
import json
from pathlib import Path
import urllib.request
import zipfile
ROOT=Path(__file__).resolve().parents[1]; out=ROOT/'data/raw/CircuitOCR'; out.mkdir(parents=True,exist_ok=True)
targets={
 'circuitocr_readme':'https://raw.githubusercontent.com/ZhangJ83/circuit_ocr_dataset_final/master/README.md',
 'circuitocr_license':'https://raw.githubusercontent.com/ZhangJ83/circuit_ocr_dataset_final/master/LICENSE',
 'schgen_tree':'https://huggingface.co/api/datasets/microsoft/SchGen_dataset/tree/main?recursive=true&limit=1000',
 'schgen_readme':'https://huggingface.co/datasets/microsoft/SchGen_dataset/raw/main/README.md',
 'open_schematics_tree':'https://huggingface.co/api/datasets/bshada/open-schematics/tree/main?recursive=false&limit=1000',
 'open_schematics_readme':'https://huggingface.co/datasets/bshada/open-schematics/raw/main/README.md',
 'circuitocr_archive':'https://codeload.github.com/ZhangJ83/circuit_ocr_dataset_final/zip/refs/heads/master'}
records={}
for name,url in targets.items():
    target=out/(name+('.zip' if name.endswith('archive') else '.txt'))
    try:
        if not target.exists():
            request=urllib.request.Request(url,headers={'User-Agent':'GraphSchematic-data-audit'})
            with urllib.request.urlopen(request,timeout=30) as response:
                data=response.read(96*1024**2+1)
            if len(data)>96*1024**2: raise ValueError('archive exceeds bounded 96 MiB download')
            target.write_bytes(data)
        data=target.read_bytes(); record={'url':url,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest(),'status':'downloaded'}
        if name.endswith('archive'):
            with zipfile.ZipFile(target) as archive:
                infos=archive.infolist(); record['members']=len(infos)
                record['image_files']=sum(Path(i.filename).suffix.lower() in ('.png','.jpg','.jpeg') for i in infos)
                record['jsonl_files']=[i.filename for i in infos if i.filename.endswith('.jsonl')]
                record['uncompressed_bytes']=sum(i.file_size for i in infos)
                if record['uncompressed_bytes']>512*1024**2: raise ValueError('archive unpack bound exceeded')
                for info in infos:
                    rel=Path(*Path(info.filename).parts[1:]); destination=(out/'repository'/rel).resolve()
                    if not destination.is_relative_to((out/'repository').resolve()): raise ValueError('unsafe archive member')
                    if info.is_dir(): continue
                    if destination.suffix.lower() not in ('.jsonl','.json','.png','.jpg','.jpeg','.md'): continue
                    destination.parent.mkdir(parents=True,exist_ok=True); destination.write_bytes(archive.read(info))
        records[name]=record
    except Exception as error: records[name]={'url':url,'status':'unavailable','error':str(error)}
    print(json.dumps({name:records[name]}),flush=True)
(out/'source-manifest.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
