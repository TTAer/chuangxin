"""Package frozen runtime plus current assessment, preserving older archives."""
import hashlib
import json
from pathlib import Path
import zipfile

ROOT=Path(__file__).resolve().parents[1]
def load(path): return json.loads(path.read_text(encoding='utf-8'))
def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()

previous=ROOT/'submission/graphschematic-stable-development.zip'
with zipfile.ZipFile(previous) as archive:
    old= json.loads(archive.read('BUILD_MANIFEST.json'))
    names=set(old['files'])
card=load(ROOT/'output/rc0-scorecard.json')
protocol=load(ROOT/'data/splits/rc0-evaluation.json')
assert card['runtime_unchanged'] and card['image_id']==protocol['image_id']
assert card['promotion_decision']=='NO_GO_pending_Freeze_A'
for name,sha in protocol['runtime_sha256'].items():
    assert digest(ROOT/name)==sha,name
for proof in card['deployment'].values():
    assert proof['image_id']==protocol['image_id']
    assert proof['status']=='passed' and proof['cases']==50 and proof['failed']==0
    assert proof['identity_names_nets_verified'] and not proof['oom_killed']
names.update(p.relative_to(ROOT).as_posix() for pattern in ('scripts/*.py','tests/*.py') for p in ROOT.glob(pattern))
names.update([
    'README.md','docs/PROJECT_STATUS.md','docs/PROJECT_PROGRESS_2026-10-04.md',
    'docs/RC0_EVALUATION.md','docs/PIN_TOKEN_ARCHITECTURE_PLAN.md',
    'data/splits/rc0-evaluation.json','data/splits/freeze-a-intake-template.json',
    'output/rc0-scorecard.json','output/rc0-default-batch-acceptance.json',
    'output/rc0-candidate-batch-acceptance.json',
    'output/freeze-b-amsnet-rc0/summary-adapter-v3.json',
    'output/ablation/rc0-freeze-b-detection.json',
    'output/ablation/rc0-membership-audit.json',
    'output/ablation/rc0-0197-scoring-path.json',
    'output/ablation/rc0-ocr-misses.json'])
manifest={
    'project':'GraphSchematic','artifact':'RC0 engineering evaluation; not final competition submission',
    'current_container_image_id':protocol['image_id'],'runtime_sources_and_selected_weights_match_container':True,
    'current_default_container_cases':50,'current_optional_container_cases':50,
    'current_image_batch_acceptance_verified':True,
    'previous_archive_sha256':digest(previous),'function_tests_passed':103,
    'official_evaluator_verified':False,'default_promoted':False,
    'promotion_decision':card['promotion_decision'],
    'new_architecture_training_started':False,'freeze_a_ready':False,
    'freeze_b_metric_adapter_version':3,'files':{},
    'limitations':card['limitations']}
for name in sorted(names):
    p=ROOT/name; assert p.is_file(),name
    manifest['files'][name]={'bytes':p.stat().st_size,'sha256':digest(p)}
out=ROOT/'submission/graphschematic-rc0-evaluation.zip'
with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED) as archive:
    for name in sorted(names): archive.write(ROOT/name,name)
    archive.writestr('BUILD_MANIFEST.json',json.dumps(manifest,indent=2))
out.with_suffix('.manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
print(json.dumps({'archive':str(out),'sha256':digest(out),'files':len(names),'bytes':out.stat().st_size},ensure_ascii=False))
