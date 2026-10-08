"""Optional vision-LLM HTTP adapter. Explicit endpoint/model credentials required.

Supports chat-completion-compatible services; never chooses a paid provider.
Without configuration, writes a not-configured result and performs no network.
"""
import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import sys
import urllib.request
import urllib.error
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from pcbparse.vision import read_image
import cv2

p=argparse.ArgumentParser(); p.add_argument('queue'); p.add_argument('image'); p.add_argument('--output',default='output/graphschematic/llm-response.json'); p.add_argument('--max-calls',type=int,default=6); a=p.parse_args()
queue=json.loads(Path(a.queue).read_text(encoding='utf-8')); destination=Path(a.output); destination.parent.mkdir(parents=True,exist_ok=True)
endpoint=os.getenv('GRAPHSCH_LLM_ENDPOINT'); model=os.getenv('GRAPHSCH_LLM_MODEL'); key=os.getenv('GRAPHSCH_LLM_API_KEY')
if not endpoint or not model or not key:
    destination.write_text(json.dumps({'status':'not_configured','required_environment':['GRAPHSCH_LLM_ENDPOINT','GRAPHSCH_LLM_MODEL','GRAPHSCH_LLM_API_KEY'],'network_calls':0},indent=2),encoding='utf-8'); print('LLM not configured; no network calls'); raise SystemExit(0)
if not endpoint.startswith('https://'): raise ValueError('LLM endpoint must use HTTPS')
digest=hashlib.sha256(Path(a.image).read_bytes()).hexdigest()
if queue.get('image_sha256') and queue['image_sha256']!=digest: raise ValueError('queue/image hash mismatch')
image=read_image(a.image); h,w=image.shape[:2]; corrections=[]; usage={'input_tokens':0,'output_tokens':0}; errors=[]
destination.write_text(json.dumps({'observation_sha256':queue['observation_sha256'],'corrections':[],'usage':usage},indent=2),encoding='utf-8')
for task in queue['tasks'][:max(0,min(a.max_calls,8))]:
    x=round(task['pin_point']['x']); y=round(h-task['pin_point']['y'])
    if task['kind']=='component_text_choice':
        a0,b0,c0,d0=task['bbox']; crop=image[max(0,int(h-d0)-45):min(h,int(h-b0)+46),max(0,int(a0)-45):min(w,int(c0)+46)]
    else: crop=image[max(0,y-64):min(h,y+65),max(0,x-64):min(w,x+65)]
    if not crop.size: continue
    image_data=base64.b64encode(cv2.imencode('.png',crop)[1]).decode('ascii')
    instruction='Inspect this local schematic and existing review task. Topology evidence overrides geometric proximity; color alone is not connection evidence. Do not invent wires, pins, values or labels. Return only JSON with candidate_id, decision from allowed_decisions, a short reason, and text_option_id only for choose_text. Choose only the listed OCR options. Preserve physical wires and same-label flying connections. '
    content=[{'type':'text','text':instruction+json.dumps(task,ensure_ascii=False)},{'type':'image_url','image_url':{'url':'data:image/png;base64,'+image_data}}]
    request=urllib.request.Request(endpoint,data=json.dumps({'model':model,'messages':[{'role':'user','content':content}],'temperature':0,'max_tokens':300}).encode('utf-8'),headers={'Authorization':'Bearer '+key,'Content-Type':'application/json'},method='POST')
    try:
        with urllib.request.urlopen(request,timeout=20) as response:
            body=response.read(1024*1024)
            result=json.loads(body)
        text=result['choices'][0]['message']['content'].strip()
        proposal=json.loads(text)
        if not isinstance(proposal,dict) or proposal.get('candidate_id')!=task['id']: raise ValueError('wrong candidate response')
        corrections.append(proposal)
        u=result.get('usage',{}); usage['input_tokens']+=int(u.get('prompt_tokens',0)); usage['output_tokens']+=int(u.get('completion_tokens',0))
    except (urllib.error.URLError,ValueError,KeyError,TypeError) as exc:
        errors.append({'candidate_id':task['id'],'error_type':type(exc).__name__})
    destination.write_text(json.dumps({'observation_sha256':queue['observation_sha256'],'corrections':corrections,'usage':usage},ensure_ascii=False,indent=2),encoding='utf-8')
destination.with_suffix('.status.json').write_text(json.dumps({'status':'completed_with_errors' if errors else 'completed','calls':len(corrections)+len(errors),'errors':errors,'provider_model':model,'image_sha256':digest},indent=2),encoding='utf-8')
print(json.dumps({'proposals':len(corrections),'errors':len(errors),'usage':usage}))
