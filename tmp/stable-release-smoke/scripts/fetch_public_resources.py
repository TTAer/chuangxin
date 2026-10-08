"""Download known public contest resources; no login or credential handling."""
import hashlib
import json
from pathlib import Path
import requests

ROOT=Path(__file__).resolve().parents[1]
out=ROOT/'data/research'
out.mkdir(parents=True,exist_ok=True)
urls={
    'question-six.json':'https://cpipc.acge.org.cn/cw/getContestWebsiteQuestionById?questionId=2c9080179fb07f4f019fb8529dc21ff1',
    'invitation.html':'https://cpipc.acge.org.cn/cw/contestNews/detail/2c9080158ee9c272018f229208b610a6/2c9080159fb07bd8019fcb75f6fd4446?page=0',
    'briefing.html':'https://cpipc.acge.org.cn/cw/contestNews/detail/2c9080158ee9c272018f229208b610a6/2c908017a05a3fdc01a06aefa9d2032a?page=0',
}
manifest=[]
for name,url in urls.items():
    try:
        r=requests.get(url,timeout=30); r.raise_for_status()
        (out/name).write_bytes(r.content)
        manifest.append({'file':name,'url':url,'bytes':len(r.content),'sha256':hashlib.sha256(r.content).hexdigest()})
        print(name,r.status_code,len(r.content))
    except requests.RequestException as e:
        manifest.append({'file':name,'url':url,'error':str(e)})
        print(name,type(e).__name__)
(out/'source-manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
