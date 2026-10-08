"""Experiment evidence bundle, distinct from the immutable RC0 runtime ZIP."""
import hashlib
import json
from pathlib import Path
import zipfile
ROOT=Path(__file__).resolve().parents[1]
def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()
verified={}
for filename,key in [('data/splits/rc0-evaluation.json','runtime_sha256'),('data/splits/pin-token-v1-frozen.json','sha256')]:
    record=json.loads((ROOT/filename).read_text(encoding='utf-8'))
    for name,sha in record[key].items(): assert digest(ROOT/name)==sha,name
    verified[filename]=len(record[key])
names={
    'docs/PIN_TOKEN_RESIDUAL_RESULTS.md','docs/LOCAL_PIN_OCR_RESULTS.md','docs/FREEZE_A_PIXEL_PROGRESS.md',
    'scripts/prepare_pin_token_v1.py','scripts/prepare_pin_token_residual.py','scripts/mine_pin_token_hard_cases.py',
    'scripts/train_pin_token_residual.py','scripts/audit_pin_token_residual.py','scripts/audit_local_pin_ocr.py',
    'scripts/render_freeze_a_source_sample.py','scripts/package_pin_token_residual.py',
    'experiments/pin_token_v1.py','experiments/pin_token_residual.py',
    'tests/test_pin_token_residual.py','tests/test_local_pin_ocr.py',
    'data/splits/pin-token-v1-frozen.json','data/splits/rc0-evaluation.json','data/splits/freeze-a-pixel-audit.json',
    'data/processed/pin-token-residual/preparation.json',
    'output/ablation/pin-token-hard-mining.json','output/ablation/pin-token-residual.json','output/ablation/pin-token-residual-audit.json'}
names.update(p.relative_to(ROOT).as_posix() for p in ROOT.glob('output/ablation/local-pin-ocr*.json'))
names.update(p.relative_to(ROOT).as_posix() for p in ROOT.glob('output/training/pin-token-residual/*') if p.is_file())
manifest={'artifact':'residual and local OCR development evidence; requires existing project/data; not competition runtime',
    'verified_frozen_records':verified,'function_tests_passed':115,'rc0_runtime_modified':False,'v1_modified':False,
    'default_promoted':False,'residual_runtime_enabled':False,'freeze_a_ready':False,
    'third_party_images_CAD_and_derivative_images_included':False,'files':{}}
for name in sorted(names):
    path=ROOT/name; assert path.is_file(),name
    manifest['files'][name]={'sha256':digest(path),'bytes':path.stat().st_size}
out=ROOT/'submission/pin-token-residual-development.zip'
with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED) as archive:
    for name in sorted(names): archive.write(ROOT/name,name)
    archive.writestr('BUILD_MANIFEST.json',json.dumps(manifest,indent=2))
with zipfile.ZipFile(out) as archive:
    for name,record in manifest['files'].items(): assert hashlib.sha256(archive.read(name)).hexdigest()==record['sha256']
out.with_suffix('.manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
print(json.dumps({'archive':str(out),'files':len(names),'sha256':digest(out),'bytes':out.stat().st_size},ensure_ascii=False))
