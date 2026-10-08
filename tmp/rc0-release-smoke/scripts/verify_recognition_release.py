import hashlib
import json
from pathlib import Path
import subprocess
import sys
import zipfile
ROOT=Path(__file__).resolve().parents[1]; archive=ROOT/'submission/pcbparse-development.zip'; target=ROOT/'tmp/recognition-release-smoke'
def load(p): return json.loads(p.read_text(encoding='utf-8'))
with zipfile.ZipFile(archive) as z:
    manifest=json.loads(z.read('BUILD_MANIFEST.json'))
    for name,record in manifest['files'].items():
        data=z.read(name); assert len(data)==record['bytes'] and hashlib.sha256(data).hexdigest()==record['sha256'],name
        assert (target/name).resolve().is_relative_to(target.resolve())
    z.extractall(target)
def run(workspace,out,extra=()):
    result=subprocess.run([sys.executable,'agent_system','-input',str(ROOT/'data/examples'),'-output',str(ROOT/out),*extra],cwd=workspace,capture_output=True,text=True,encoding='utf-8',errors='replace')
    assert result.returncode==0,(result.stdout,result.stderr)
    assert load(ROOT/out/'decision_log.json')['status']=='completed'
    return load(ROOT/out/'final_results/guide-schematic_Pred.json')
live=run(ROOT,'output/recognition-workspace-smoke'); packaged=run(target,'output/recognition-packaged-smoke'); assert live==packaged
old=('--pin-model','models/pins/pin-heatmap.onnx','--pin-ensemble','models/candidates/official-pins/pin-heatmap.onnx','--semantic-library','none','--component-box-mode','detector')
previous=run(target,'output/recognition-previous-default-smoke',old)
assert previous==load(ROOT/'output/consensus-default-agent-smoke/final_results/guide-schematic_Pred.json')
summary={'archive_sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),'verified_files':len(manifest['files']),'new_default_matches_workspace':True,'previous_default_reproducible':True,'official_evaluator_verified':False,'docker_build_verified':manifest['docker_build_verified']}
(ROOT/'output/recognition-release-verification.json').write_text(json.dumps(summary,indent=2),encoding='utf-8'); print(json.dumps(summary),flush=True)
