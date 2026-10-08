import hashlib
import json
from pathlib import Path
import subprocess
import sys
import zipfile

root=Path(__file__).resolve().parents[1]; archive=root/'submission/pcbparse-development.zip'; destination=root/'tmp/gea-release-smoke'
load=lambda p:json.loads(p.read_text(encoding='utf-8'))
with zipfile.ZipFile(archive) as z:
    manifest=json.loads(z.read('BUILD_MANIFEST.json'))
    for name,record in manifest['files'].items():
        data=z.read(name); assert len(data)==record['bytes'] and hashlib.sha256(data).hexdigest()==record['sha256'],name
        assert (destination/name).resolve().is_relative_to(destination.resolve())
    z.extractall(destination)
print(f"Verified {len(manifest['files'])} file hashes",flush=True)
def run(workspace,out,extra=()):
    process=subprocess.run([sys.executable,'agent_system','-input',str(root/'data/examples'),'-output',str(root/out),*extra],cwd=workspace,capture_output=True,text=True,encoding='utf-8',errors='replace')
    assert process.returncode==0,(process.stdout,process.stderr)
    assert load(root/out/'decision_log.json')['status']=='completed'
    return load(root/out/'final_results/guide-schematic_Pred.json')
default=run(destination,'output/gea-packaged-default-smoke')
assert default==load(root/'output/consensus-default-agent-smoke/final_results/guide-schematic_Pred.json')
selector=('--association-model','models/candidates/association-balanced/association.onnx')
live=run(root,'output/gea-agent-smoke',selector)
packaged=run(destination,'output/gea-packaged-agent-smoke',selector)
assert live==packaged
graph=load(root/'output/gea-packaged-agent-smoke/runs/local/_diagnostics/guide-schematic-graph.json')
assert graph['learned_association']['module']=='ElectricalAssociationNet'
assert graph['learned_association']['status'] in ('evaluated','no_candidates')
summary={'archive_sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),'verified_files':len(manifest['files']),'default_matches_prior_workspace':True,'optional_gea_matches_workspace':True,'official_evaluator_verified':False,'docker_build_verified':False}
(root/'output/gea-release-verification.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
print(json.dumps(summary),flush=True)
