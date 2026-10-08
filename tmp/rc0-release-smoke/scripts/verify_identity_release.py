"""Replay packaged identity enhancement from image-only inputs on Windows."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from pcbparse.dataset import index_dataset
from pcbparse.competition import export_competition
inputs=ROOT/'tmp/identity-release-input'; inputs.mkdir(parents=True,exist_ok=True)
samples={s['id']:s for s in index_dataset(ROOT/'data/raw/PCBnet/Dataset300')}
shutil.copyfile(ROOT/'data/raw/official/gt-0930/10GT/1/1.png',inputs/'1.png')
for key in ('0006','0184'): shutil.copyfile(samples[key]['image'],inputs/(key+'.png'))
assert len(list(inputs.iterdir()))==3
out=ROOT/'output/identity-packaged-smoke'
run=subprocess.run([sys.executable,'agent_system','-input',str(inputs),'-output',str(out),'--ic-identity-mode','retrieval',
                    '--side-pin-model','models/candidates/side-strip-local/side-strip.onnx','--side-name-anchors'],
                   cwd=ROOT/'tmp/recognition-release-smoke',capture_output=True,text=True,encoding='utf-8',errors='replace')
assert run.returncode==0,(run.stdout,run.stderr)
def load(p): return json.loads(p.read_text(encoding='utf-8'))
assert load(out/'decision_log.json')['status']=='completed'
for key,domain in [('1','gt10'),('0006','pcb55'),('0184','pcb55')]:
    base=ROOT/'output/ic-identity-live'/domain/key
    expected=load(base/'prediction.json')
    if domain=='pcb55': expected=export_competition(expected,load(base/'observations.json').get('component_categories'))
    assert load(out/'final_results'/(key+'_Pred.json'))==expected,key
record={'archive_sha256':hashlib.sha256((ROOT/'submission/pcbparse-development.zip').read_bytes()).hexdigest(),
        'packaged_identity_side_exact_windows_match':True,'cases':3,'official_evaluator_verified':False}
(ROOT/'output/identity-release-verification.json').write_text(json.dumps(record,indent=2),encoding='utf-8'); print(json.dumps(record),flush=True)
