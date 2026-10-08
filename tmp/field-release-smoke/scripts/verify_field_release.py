"""Verify every archive hash and compare packaged optional/default inference."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import zipfile
ROOT=Path(__file__).resolve().parents[1]
archive=ROOT/'submission/pcbparse-development.zip'; destination=ROOT/'tmp/field-release-smoke'
def load(path): return json.loads(path.read_text(encoding='utf-8'))
with zipfile.ZipFile(archive) as z:
    manifest=json.loads(z.read('BUILD_MANIFEST.json'))
    for name,record in manifest['files'].items():
        data=z.read(name)
        assert len(data)==record['bytes'] and hashlib.sha256(data).hexdigest()==record['sha256'],name
        assert (destination/name).resolve().is_relative_to(destination.resolve())
    z.extractall(destination)
print(json.dumps({'verified_file_hashes':len(manifest['files'])}),flush=True)
def run(workspace,out,extra=()):
    result=subprocess.run([sys.executable,'agent_system','-input',str(ROOT/'data/examples'),'-output',str(ROOT/out),*extra],cwd=workspace,capture_output=True,text=True,encoding='utf-8',errors='replace')
    assert result.returncode==0,(result.stdout,result.stderr)
    assert load(ROOT/out/'decision_log.json')['status']=='completed'
    return load(ROOT/out/'final_results/guide-schematic_Pred.json')
default=run(destination,'output/fields-packaged-default-smoke')
assert default==load(ROOT/'output/consensus-default-agent-smoke/final_results/guide-schematic_Pred.json')
selector=('--field-model','models/candidates/directional-fields/field.onnx')
workspace=run(ROOT,'output/fields-workspace-smoke',selector)
packaged=run(destination,'output/fields-packaged-smoke',selector)
assert workspace==packaged
graph=load(ROOT/'output/fields-packaged-smoke/runs/local/_diagnostics/guide-schematic-graph.json')
assert graph['directional_fields']['module']=='DirectionalFieldNet'
summary={'archive_sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),'verified_files':len(manifest['files']),'default_matches_prior_workspace':True,'optional_fields_match_workspace':True,'official_evaluator_verified':False,'docker_build_verified':False}
(ROOT/'output/field-release-verification.json').write_text(json.dumps(summary,indent=2),encoding='utf-8'); print(json.dumps(summary),flush=True)
