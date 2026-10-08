"""Package image inference code and selected weights, excluding raw training data."""
import hashlib
import json
from pathlib import Path
import zipfile

ROOT=Path(__file__).resolve().parents[1]
files=[ROOT/'run.py',ROOT/'requirements-inference.txt',ROOT/'README.md',ROOT/'Dockerfile',ROOT/'pyproject.toml']
files+=[ROOT/'agent_system',ROOT/'Dockerfile.competition',ROOT/'models/semantics/component-library.json',ROOT/'docs/OFFICIAL_MATERIALS.md']
files+=[ROOT/'.dockerignore',ROOT/'docs/RECOGNITION_IMPROVEMENT.md',ROOT/'data/splits/amsnet-frozen.json',ROOT/'output/ablation/recognition-pipeline.json',ROOT/'output/recognition-live-verification.json']
files+=[ROOT/'output/semantic-grid-pcb55/summary.json',ROOT/'output/semantic-grid-gt10/summary.json',ROOT/'output/recognition-audit/pcb55.json',ROOT/'output/recognition-audit/gt10.json']
files+=[ROOT/'output/container-validation.json']
files+=[ROOT/'vendor/linux-wheels/README.md']
files+=[ROOT/'docs/IC_STRUCTURE_AND_RUNTIME.md',ROOT/'output/ablation/ic-structure.json',ROOT/'output/ic-structure-live-verification.json',ROOT/'output/container-batch-acceptance.json']
files+=[ROOT/'output/net-observation-audit'/name for name in ('pcb55.json','gt10.json')]
files+=[ROOT/'output'/folder/'summary.json' for folder in ('ic-structure-gt10','ic-structure-pcb55','ic-structure-gt10-geometry','ic-structure-pcb55-geometry')]
files+=[ROOT/'models/candidates/official-pins/pin-heatmap.onnx',ROOT/'models/candidates/official-pins/pin-heatmap.json',ROOT/'output/ablation/official-gt10.json',ROOT/'output/ablation/official-pins-55.json']
files+=[ROOT/'docs/PROJECT_PROGRESS_2026-10-04.md',ROOT/'docs/PROJECT_STATUS.md',ROOT/'output/ablation/pin-consensus.json']
files+=[ROOT/'docs/ARCHITECTURE_GEA_EXPERIMENT.md',ROOT/'output/ablation/gea-architecture.json',ROOT/'output/association-model-verification/summary.json']
for candidate in ('association-balanced','association-mlp-balanced'):
    files+=[ROOT/'models/candidates'/candidate/'association.onnx',ROOT/'models/candidates'/candidate/'association.json']
for candidate in ('directional-fields-base','directional-fields','directional-fields-local'):
    files+=[ROOT/'models/candidates'/candidate/'field.onnx',ROOT/'models/candidates'/candidate/'field.json']
files+=[ROOT/'docs/ARCHITECTURE_DIRECTIONAL_FIELDS.md',ROOT/'docs/NEW_DATA_SOURCES.md',ROOT/'output/ablation/directional-fields.json',ROOT/'output/new-dataset-research/local-audit.json']
files+=[ROOT/'docs/ARCHITECTURE_SIDE_STRIP.md',ROOT/'output/ablation/side-strip.json',ROOT/'output/ablation/proxy-sensitivity.json',ROOT/'output/side-source-audit.json',ROOT/'data/processed/side-strips/preparation.json']
files+=[ROOT/'output/side-strip-verification'/name for name in ('exports.json','live.json','container.json')]
for candidate in ('side-strip-base','side-strip-mixed','side-strip-local'):
    files+=[ROOT/'models/candidates'/candidate/name for name in ('side-strip.onnx','side-strip.json','history.json')]
files+=[ROOT/'docs/SIDE_REFINEMENT_2026-10-05.md',ROOT/'output/container-side-20261005-batch-acceptance.json']
files+=[ROOT/'output/ablation'/name for name in ('side-refinement.json','side-thresholds.json','side-thresholds.csv','side-paired-cross-crops.json','side-paired-exports.json','side-paired-live.json','side-incremental-live.json','side-current-container.json','side-current-image-sources.json')]
for variant in ('oracle','detector'):
    files+=[ROOT/f'data/processed/side-strips-{variant}-paired/preparation.json']
    files+=[ROOT/f'models/candidates/side-strip-{variant}-paired'/name for name in ('side-strip.onnx','side-strip.json','history.json')]
files+=[ROOT/'docs/IC_IDENTITY_RECOVERY.md',ROOT/'output/container-identity-20261005-batch-acceptance.json']
files+=[ROOT/'output/ablation'/name for name in ('ic-identity.json','ic-identity-audit.json','ic-identity-live.json','ic-identity-container.json','ic-identity-image-sources.json')]
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
manifest['default_model_policy']='official-adapted primary pin model plus library-conditioned semantics and observed passive-body output boxes; no pin ensemble; actual-image 55-page and GT10 development comparisons both improve'
manifest['optional_association_policy']='GEA-Net and MLP soft association scoring are experimental; no stable end-to-end gain established; default disabled'
manifest['optional_field_policy']='DirectionalFieldNet is experimental; new CGHD data improve local pixel metric but regress on GT10 proxy; default disabled; attribution and weight terms in docs/NEW_DATA_SOURCES.md'
manifest['optional_ic_structure_policy']='StructureAwareICParser is optional via --ic-pin-mode structure; GT10 gain concentrated in one page and PCB55 mean gain below 0.2 points; default none'
manifest['optional_side_strip_policy']='SideStripNet is optional via --side-pin-model; local convolution control outperforms dilated mixed candidate; synthetic data do not improve both domains; default none; source and limitations in docs/ARCHITECTURE_SIDE_STRIP.md'
container=json.loads((ROOT/'output/container-validation.json').read_text(encoding='utf-8'))
manifest['docker_build_verified']=container.get('build_verified',False)
manifest['container_inference_verified']=container.get('inference_verified',False)
if manifest['docker_build_verified']:
    manifest['limitations'][1]='official evaluator unavailable; supplied Ubuntu24 base image built locally; full official runner acceptance remains unverified'
current=json.loads((ROOT/'output/side-strip-verification/container.json').read_text(encoding='utf-8'))
manifest['current_container_image_id']=current['image_id']
manifest['docker_build_verified']=current['build_verified']
manifest['container_inference_verified']=current['default_exact_windows_match'] and current['candidate_identity_names_nets_identical']
manifest['current_default_container_exact_match']=current['default_exact_windows_match']
manifest['current_optional_container_max_coordinate_difference_pixels']=current['candidate_max_coordinate_difference_pixels']
manifest['current_image_batch_acceptance_verified']=False
manifest['historical_batch_acceptance']='output/container-batch-acceptance.json belongs to prior image; current image verified on two single cases'
latest=json.loads((ROOT/'output/ablation/ic-identity-container.json').read_text(encoding='utf-8'))
batch=json.loads((ROOT/'output/container-identity-20261005-batch-acceptance.json').read_text(encoding='utf-8'))
sources=json.loads((ROOT/'output/ablation/ic-identity-image-sources.json').read_text(encoding='utf-8'))
assert latest['image_id']==batch['image_id']==sources['image_id']
assert {t['cases'] for t in batch['tests']}=={50} and all(t['status']=='passed' and t['profile']=='identity-side' and t['image_id']==latest['image_id'] for t in batch['tests'])
for name,digest in sources['sha256'].items():
    assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
manifest.update(current_container_image_id=latest['image_id'],docker_build_verified=latest['build_verified'],
                container_inference_verified=True,current_default_container_exact_match=True,current_image_batch_acceptance_verified=True,
                current_optional_container_max_coordinate_difference_pixels=max(t['max_geometry_difference_pixels'] for t in batch['tests']),
                current_optional_container_cases=50,current_optional_batch_acceptance_verified=True,current_default_batch_acceptance_verified=False,
                current_default_container_cases=1,runtime_sources_and_selected_weights_match_container=True,function_tests_passed=83)
manifest['optional_name_anchor_policy']='--side-name-anchors plus local SideStripNet; unique observed catalog name and real normal stroke; add missing number only; development PCB55 score16.7617, GT10 unchanged; default disabled'
manifest['optional_ic_identity_policy']='--ic-identity-mode retrieval; constrained OCR prefix split, top5 candidates, visible pin evidence and second-best margin; PCB55 combination16.8075 with5 improves0 regressions; GT10 unchanged; default disabled'
manifest['current_batch_profile']='identity retrieval + local SideStripNet + incremental name anchors; 8GT10 and2PCB55 images repeated5 times'
manifest['historical_batch_acceptance']='default10/50 proofs belong to prior images and remain separately included; latest image default single case and full optional50-case proof included'
for p in files:
    if not p.is_file(): raise FileNotFoundError(str(p))
    manifest['files'][p.relative_to(ROOT).as_posix()]={'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED) as z:
    for p in files: z.write(p,p.relative_to(ROOT).as_posix())
    z.writestr('BUILD_MANIFEST.json',json.dumps(manifest,indent=2))
(out.with_suffix('.manifest.json')).write_text(json.dumps(manifest,indent=2),encoding='utf-8')
print(json.dumps({'archive':str(out),'bytes':out.stat().st_size,'sha256':hashlib.sha256(out.read_bytes()).hexdigest(),'files':len(files)},ensure_ascii=False))
