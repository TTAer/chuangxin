"""Verify packaged runtime sources and selected weights against immutable image."""
import hashlib
import argparse
import json
from pathlib import Path
import subprocess
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser(); p.add_argument('--record',default='output/ablation/side-current-container.json'); p.add_argument('--output',default='output/ablation/side-current-image-sources.json'); args=p.parse_args()
record=json.loads((ROOT/args.record).read_text(encoding='utf-8'))
paths=list((ROOT/'pcbparse').glob('*.py'))+[ROOT/'agent_system',ROOT/'models/pipeline.json']
for folder in ['models/detector','models/pins','models/topology','models/semantics','models/symbols',
               'models/candidates/official-pins','models/candidates/side-strip-base','models/candidates/side-strip-mixed',
               'models/candidates/side-strip-local','models/candidates/side-strip-oracle-paired','models/candidates/side-strip-detector-paired']:
    paths.extend(p for p in (ROOT/folder).iterdir() if p.suffix in ('.onnx','.json','.npz') and p.name not in ('history.json','dataset-index.json'))
names=[p.relative_to(ROOT).as_posix() for p in paths]
code="import hashlib,json\nfrom pathlib import Path\nprint(json.dumps({name:hashlib.sha256((Path('/submission')/name).read_bytes()).hexdigest() for name in "+repr(names)+"}))"
result=subprocess.run(['wsl','-d','Ubuntu-22.04','-u','root','--exec','/usr/bin/docker','run','--rm','-i','--network','none',
        '--entrypoint','/opt/graphschematic/bin/python',record['image_id'],'-'],input=code,capture_output=True,encoding='utf-8',errors='replace')
assert result.returncode==0,(result.stdout,result.stderr)
actual=json.loads(result.stdout.strip()); expected={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
assert actual==expected,[name for name in expected if actual.get(name)!=expected[name]]
out={'image_id':record['image_id'],'verified_files':len(paths),'source_and_selected_weights_match_workspace':True,'sha256':actual}
(ROOT/args.output).write_text(json.dumps(out,indent=2),encoding='utf-8'); print(json.dumps({k:v for k,v in out.items() if k!='sha256'}),flush=True)
