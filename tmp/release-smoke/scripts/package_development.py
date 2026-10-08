"""Package image inference code and selected weights, excluding raw training data."""
import hashlib
import json
from pathlib import Path
import zipfile

ROOT=Path(__file__).resolve().parents[1]
files=[ROOT/'run.py',ROOT/'requirements-inference.txt',ROOT/'README.md',ROOT/'Dockerfile',ROOT/'pyproject.toml']
files+=list((ROOT/'pcbparse').glob('*.py'))
files+=list((ROOT/'scripts').glob('*.py'))
files+=[ROOT/'requirements-training.txt']
files+=[ROOT/'models/pipeline.json',ROOT/'models/detector/pcb-yolo11n.onnx',ROOT/'models/detector/pcb-yolo11n.json',ROOT/'models/pins/pin-heatmap.onnx',ROOT/'models/pins/pin-heatmap.json',ROOT/'models/symbols/templates.json',ROOT/'models/symbols/templates.npz']
files+=[ROOT/'docs/DATASETS_AND_MODELS.md',ROOT/'docs/COMPETITION_RESEARCH.md']
out=ROOT/'submission/pcbparse-development.zip'; out.parent.mkdir(parents=True,exist_ok=True)
manifest={'status':'development candidate; not final competition submission','official_evaluator_verified':False,'docker_build_verified':False,'limitations':['end-to-end accuracy remains low','official evaluator and runner environment are unavailable','technical report intentionally deferred'],'files':{}}
for p in files:
    if not p.is_file(): raise FileNotFoundError(str(p))
    manifest['files'][p.relative_to(ROOT).as_posix()]={'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED) as z:
    for p in files: z.write(p,p.relative_to(ROOT).as_posix())
    z.writestr('BUILD_MANIFEST.json',json.dumps(manifest,indent=2))
(out.with_suffix('.manifest.json')).write_text(json.dumps(manifest,indent=2),encoding='utf-8')
print(json.dumps({'archive':str(out),'bytes':out.stat().st_size,'sha256':hashlib.sha256(out.read_bytes()).hexdigest(),'files':len(files)},ensure_ascii=False))
