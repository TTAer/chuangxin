from pathlib import Path
import hashlib
import json
import requests

out=Path(__file__).resolve().parents[1]/'models/pretrained'
out.mkdir(parents=True,exist_ok=True)
url='https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt'
r=requests.get(url,timeout=60); r.raise_for_status()
p=out/'yolo11n.pt'; p.write_bytes(r.content)
report={'source':url,'bytes':len(r.content),'sha256':hashlib.sha256(r.content).hexdigest(),'license':'AGPL-3.0 (Ultralytics)'}
(out/'provenance.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(report)
