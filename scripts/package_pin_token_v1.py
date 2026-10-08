"""Separate experiment bundle; never overwrite the frozen RC0 delivery."""
import hashlib
import json
from pathlib import Path
import zipfile
ROOT=Path(__file__).resolve().parents[1]
def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()
protocol=json.loads((ROOT/'data/splits/rc0-evaluation.json').read_text(encoding='utf-8'))
for name,sha in protocol['runtime_sha256'].items(): assert digest(ROOT/name)==sha,name
names={'experiments/pin_token_v1.py','tests/test_pin_token_v1.py','tests/test_box_error_taxonomy.py',
    'docs/PIN_TOKEN_V1_RESULTS.md','docs/RC0_BOX_ERROR_DIAGNOSIS.md','docs/FREEZE_A_PROGRESS.md',
    'scripts/audit_rc0_box_errors.py','scripts/acquire_freeze_a_sources.py',
    'scripts/acquire_freeze_a_small_assets.py','scripts/audit_freeze_a_sources.py',
    'data/processed/pin-token-v1/preparation.json','data/splits/freeze-a-source-intake.json',
    'data/splits/freeze-a-cad-evidence.json','data/splits/freeze-a-readiness-audit.json',
    'output/ablation/pin-token-v1.json','output/ablation/pin-token-v1-export.json',
    'output/ablation/rc0-box-error-taxonomy.json'}
names.update(p.relative_to(ROOT).as_posix() for p in ROOT.glob('scripts/*pin_token_v1.py'))
names.update(p.relative_to(ROOT).as_posix() for p in ROOT.glob('models/candidates/pin-token-v1/**/*') if p.is_file())
names.update(p.relative_to(ROOT).as_posix() for p in ROOT.glob('output/training/pin-token-v1/**/*') if p.is_file())
manifest={'artifact':'PinTokenNet V1 experiment; not runtime release','rc0_image_id':protocol['image_id'],
    'rc0_runtime_verified_files':len(protocol['runtime_sha256']),'rc0_runtime_unchanged':True,
    'function_tests_passed':110,'default_promoted':False,'pin_token_v1_runtime_enabled':False,
    'freeze_a_complete':False,'no_third_party_source_images_or_CAD_redistributed':True,'files':{}}
for name in sorted(names):
    path=ROOT/name; assert path.is_file(),name
    manifest['files'][name]={'sha256':digest(path),'bytes':path.stat().st_size}
out=ROOT/'submission/pin-token-v1-experiment.zip'
with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED) as archive:
    for name in sorted(names): archive.write(ROOT/name,name)
    archive.writestr('BUILD_MANIFEST.json',json.dumps(manifest,indent=2))
with zipfile.ZipFile(out) as archive:
    for name,record in manifest['files'].items(): assert hashlib.sha256(archive.read(name)).hexdigest()==record['sha256']
out.with_suffix('.manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
print(json.dumps({'archive':str(out),'sha256':digest(out),'verified_files':len(names),'bytes':out.stat().st_size},ensure_ascii=False))
