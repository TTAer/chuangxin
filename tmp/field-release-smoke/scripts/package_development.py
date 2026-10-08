"""Package image inference code and selected weights, excluding raw training data."""
import hashlib
import json
from pathlib import Path
import zipfile

ROOT=Path(__file__).resolve().parents[1]
files=[ROOT/'run.py',ROOT/'requirements-inference.txt',ROOT/'README.md',ROOT/'Dockerfile',ROOT/'pyproject.toml']
files+=[ROOT/'agent_system',ROOT/'Dockerfile.competition',ROOT/'models/semantics/component-library.json',ROOT/'docs/OFFICIAL_MATERIALS.md']
files+=[ROOT/'models/candidates/official-pins/pin-heatmap.onnx',ROOT/'models/candidates/official-pins/pin-heatmap.json',ROOT/'output/ablation/official-gt10.json',ROOT/'output/ablation/official-pins-55.json']
files+=[ROOT/'docs/PROJECT_PROGRESS_2026-10-04.md',ROOT/'docs/PROJECT_STATUS.md',ROOT/'output/ablation/pin-consensus.json']
files+=[ROOT/'docs/ARCHITECTURE_GEA_EXPERIMENT.md',ROOT/'output/ablation/gea-architecture.json',ROOT/'output/association-model-verification/summary.json']
for candidate in ('association-balanced','association-mlp-balanced'):
    files+=[ROOT/'models/candidates'/candidate/'association.onnx',ROOT/'models/candidates'/candidate/'association.json']
for candidate in ('directional-fields-base','directional-fields','directional-fields-local'):
    files+=[ROOT/'models/candidates'/candidate/'field.onnx',ROOT/'models/candidates'/candidate/'field.json']
files+=[ROOT/'docs/ARCHITECTURE_DIRECTIONAL_FIELDS.md',ROOT/'docs/NEW_DATA_SOURCES.md',ROOT/'output/ablation/directional-fields.json',ROOT/'output/new-dataset-research/local-audit.json']
files+=[ROOT/'output/field-verification'/name for name in ('base.json','multisource.json','local.json')]
files+=list((ROOT/'pcbparse').glob('*.py'))
files+=list((ROOT/'scripts').glob('*.py'))
files+=[ROOT/'requirements-training.txt']
files+=[ROOT/'models/pipeline.json',ROOT/'models/detector/pcb-yolo11n.onnx',ROOT/'models/detector/pcb-yolo11n.json',ROOT/'models/pins/pin-heatmap.onnx',ROOT/'models/pins/pin-heatmap.json',ROOT/'models/symbols/templates.json',ROOT/'models/symbols/templates.npz']
files+=[ROOT/'models/topology/topology-perception.onnx',ROOT/'models/topology/topology-perception.json']
files+=list((ROOT/'tests').glob('test_*.py'))
files+=[ROOT/'docs/DATASETS_AND_MODELS.md',ROOT/'docs/COMPETITION_RESEARCH.md',ROOT/'docs/GRAPHSCHEMATIC.md']
files=list(dict.fromkeys(files))
out=ROOT/'submission/pcbparse-development.zip'; out.parent.mkdir(parents=True,exist_ok=True)
manifest={'status':'development candidate; not final competition submission','official_evaluator_verified':False,'docker_build_verified':False,'limitations':['end-to-end accuracy remains low','official evaluator and runner environment are unavailable','technical report intentionally deferred'],'files':{}}
manifest['limitations']=['end-to-end accuracy remains low','official evaluator unavailable; base image supplied but Docker build unverified','decision_log detailed official schema unavailable','online LLM correction not evaluated','technical report intentionally deferred']
manifest['default_model_policy']='retain original primary pin model; official-adapted auxiliary model enabled through topology-preserving consensus; semantic association remains optional'
manifest['optional_association_policy']='GEA-Net and MLP soft association scoring are experimental; no stable end-to-end gain established; default disabled'
manifest['optional_field_policy']='DirectionalFieldNet is experimental; new CGHD data improve local pixel metric but regress on GT10 proxy; default disabled; attribution and weight terms in docs/NEW_DATA_SOURCES.md'
for p in files:
    if not p.is_file(): raise FileNotFoundError(str(p))
    manifest['files'][p.relative_to(ROOT).as_posix()]={'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED) as z:
    for p in files: z.write(p,p.relative_to(ROOT).as_posix())
    z.writestr('BUILD_MANIFEST.json',json.dumps(manifest,indent=2))
(out.with_suffix('.manifest.json')).write_text(json.dumps(manifest,indent=2),encoding='utf-8')
print(json.dumps({'archive':str(out),'bytes':out.stat().st_size,'sha256':hashlib.sha256(out.read_bytes()).hexdigest(),'files':len(files)},ensure_ascii=False))
