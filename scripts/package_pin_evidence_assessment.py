"""Package assessment source/records, excluding all third-party imagery."""
import hashlib
import json
from pathlib import Path
import zipfile
ROOT=Path(__file__).resolve().parents[1]
def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()
verified={}
for name,key in [('data/splits/rc0-evaluation.json','runtime_sha256'),('data/splits/pin-token-v1-frozen.json','sha256'),('data/splits/pin-token-residual-frozen.json','sha256')]:
    m=json.loads((ROOT/name).read_text(encoding='utf-8'))
    for path,sha in m[key].items(): assert digest(ROOT/path)==sha,path
    verified[name]=len(m[key])
names={'docs/PIN_EVIDENCE_IMPROVEMENT_PLAN.md','docs/HARD56_REVIEW_DRAFT.md','docs/FREEZE_A1_PROGRESS.md',
    'experiments/local_pin_evidence.py','tests/test_local_pin_evidence.py','scripts/evaluate_local_pin_evidence.py',
    'scripts/review_hard56_draft.py','scripts/finalize_hard56_agent_review.py','scripts/prepare_freeze_a1_candidates.py',
    'scripts/package_pin_evidence_assessment.py','scripts/render_freeze_a_source_sample.py',
    'output/ablation/hard56-review-draft.json','output/ablation/hard56-review.html',
    'data/processed/FreezeA-A1-candidates/audit.json',
    'data/splits/rc0-evaluation.json','data/splits/pin-token-v1-frozen.json','data/splits/pin-token-residual-frozen.json'}
names.update(p.relative_to(ROOT).as_posix() for p in ROOT.glob('output/local-pin-evidence/**/*.json'))
manifest={'artifact':'evidence-gated local OCR assessment, not competition runtime; requires existing project data',
    'frozen_verified':verified,'function_tests_passed':118,'new_training':False,
    'local_ocr_commit_runtime_enabled':False,'residual_runtime_enabled':False,'default_promoted':False,
    'freeze_a_ready':False,'reviewed_correctable_HRR':None,'third_party_images_included':False,'files':{}}
for name in sorted(names):
    p=ROOT/name; assert p.is_file(),name
    manifest['files'][name]={'sha256':digest(p),'bytes':p.stat().st_size}
out=ROOT/'submission/pin-evidence-assessment.zip'
with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED) as z:
    for name in sorted(names): z.write(ROOT/name,name)
    z.writestr('BUILD_MANIFEST.json',json.dumps(manifest,indent=2))
with zipfile.ZipFile(out) as z:
    for name,record in manifest['files'].items(): assert hashlib.sha256(z.read(name)).hexdigest()==record['sha256']
out.with_suffix('.manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
print(json.dumps({'archive':str(out),'sha256':digest(out),'verified_files':len(names),'bytes':out.stat().st_size},ensure_ascii=False))
