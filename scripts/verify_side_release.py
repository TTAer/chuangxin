"""Replay optional learned IC candidate from the extracted development archive."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
target=ROOT/'tmp/recognition-release-smoke'
archive=ROOT/'submission/pcbparse-development.zip'
out=ROOT/'output/side-packaged-smoke'
result=subprocess.run([sys.executable,'agent_system','-input',str(ROOT/'tmp/side-current-container-input'),
                       '-output',str(out),'--side-pin-model','models/candidates/side-strip-local/side-strip.onnx','--side-name-anchors'],
                      cwd=target,capture_output=True,text=True,encoding='utf-8',errors='replace')
assert result.returncode==0,(result.stdout,result.stderr)
def load(p): return json.loads(p.read_text(encoding='utf-8'))
assert load(out/'decision_log.json')['status']=='completed'
assert load(out/'final_results/1_Pred.json')==load(ROOT/'output/side-incremental-live/gt10/1/prediction.json')
sys.path.insert(0,str(ROOT))
from pcbparse.competition import export_competition
expected=export_competition(load(ROOT/'output/side-incremental-live/pcb55/0184/prediction.json'),load(ROOT/'output/benchmark-recognition-pcb55/0184/observations.json').get('component_categories'))
assert load(out/'final_results/0184_Pred.json')==expected
report={'archive_sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),
        'packaged_side_strip_incremental_exact_windows_match':True,'cases':2,'official_evaluator_verified':False}
(ROOT/'output/side-strip-verification/release.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report),flush=True)
