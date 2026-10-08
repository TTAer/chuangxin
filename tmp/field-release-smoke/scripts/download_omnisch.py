"""Stage the official OmniSch archive for license and schema inspection."""
import hashlib
import json
from pathlib import Path
import requests
import zipfile

root=Path(__file__).resolve().parents[1]/'data/raw/OmniSch'
root.mkdir(parents=True,exist_ok=True)
url='https://drive.usercontent.google.com/download?id=1GRPNowJNRixrFIWvS7YSkhp_k7erUV6y&export=download&confirm=t'
target=root/'OmniSch.zip'
report={'project':'https://omnisch.com/','url':url,'training_allowed':'unverified; keep separate from training until terms/splits inspected'}
try:
    with requests.get(url,stream=True,timeout=(20,60)) as r:
        r.raise_for_status()
        if 'text/html' in r.headers.get('Content-Type',''): raise ValueError('Drive returned HTML rather than archive')
        digest=hashlib.sha256(); size=0
        with target.open('wb') as f:
            for chunk in r.iter_content(1024*1024):
                size+=len(chunk)
                if size>4*1024**3: raise ValueError('download exceeds 4 GB limit')
                f.write(chunk); digest.update(chunk)
    if not zipfile.is_zipfile(target): raise ValueError('download is not ZIP')
    with zipfile.ZipFile(target) as z:
        report['files']=z.namelist()[:80]; report['file_count']=len(z.namelist())
        if sum(i.file_size for i in z.infolist())>8*1024**3: raise ValueError('expanded archive exceeds 8 GB')
        for name in z.namelist():
            if not (root/name).resolve().is_relative_to(root.resolve()): raise ValueError('unsafe ZIP path')
        z.extractall(root)
    report.update(status='downloaded',bytes=size,sha256=digest.hexdigest())
except (requests.RequestException,ValueError,zipfile.BadZipFile) as e:
    report.update(status='unavailable',error=str(e))
(root/'download.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report))
