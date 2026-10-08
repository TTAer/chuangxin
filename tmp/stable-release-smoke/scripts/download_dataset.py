"""Attempt public PCBnet archive; save errors instead of mistaking HTML for data."""
import hashlib
import json
from pathlib import Path
import zipfile
import requests

ROOT=Path(__file__).resolve().parents[1]
out=ROOT/'data/raw'; out.mkdir(parents=True,exist_ok=True)
url='https://drive.usercontent.google.com/download?id=1q_rwzchFRyd0hJ3nsOWFcPZn-zny_ccW&export=download&confirm=t'
report={'url':url}
try:
    with requests.get(url,timeout=(15,45),stream=True) as r:
        r.raise_for_status()
        kind=r.headers.get('Content-Type','')
        if 'text/html' in kind: raise ValueError('Google Drive returned HTML, not a public archive')
        limit=350*1024*1024
        target=out/'Dataset300.zip'; size=0; digest=hashlib.sha256()
        with target.open('wb') as f:
            for chunk in r.iter_content(1024*1024):
                size+=len(chunk)
                if size>limit: raise ValueError('archive exceeds 350 MB automatic download limit')
                f.write(chunk); digest.update(chunk)
        if not zipfile.is_zipfile(target): raise ValueError('response is not ZIP')
        with zipfile.ZipFile(target) as z:
            names=z.namelist()
            report.update(bytes=size,sha256=digest.hexdigest(),files=names[:30],file_count=len(names))
            # Guard against traversal and expansion bombs before extraction.
            base=(out/'PCBnet').resolve()
            if sum(x.file_size for x in z.infolist())>2*1024**3: raise ValueError('expanded archive exceeds 2 GB limit')
            for n in names:
                if not (base/n).resolve().is_relative_to(base): raise ValueError('unsafe ZIP path')
            z.extractall(base)
        report['status']='downloaded'
except (requests.RequestException,ValueError,zipfile.BadZipFile) as e:
    report.update(status='unavailable',error=str(e))
(ROOT/'data/research/dataset-download.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(report,ensure_ascii=False,indent=2))
