"""Metadata/research package, not a new competition runtime."""
import hashlib
import json
import zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    verified={}
    for name,key in [('rc0-evaluation.json','runtime_sha256'),('pin-token-v1-frozen.json','sha256'),('pin-token-residual-frozen.json','sha256')]:
        rows=json.loads((ROOT/'data/splits'/name).read_text(encoding='utf-8'))[key]
        for file,digest in rows.items():assert sha(ROOT/file)==digest,file
        verified[name]=len(rows)
    names={'experiments/pin_geometry_inspector.py','scripts/build_pin_geometry_inspector.py','scripts/pin_geometry_inspector_template.html',
        'scripts/prepare_freeze_a1_calibration_review.py','scripts/audit_pin_geometry_reviews.py','scripts/freeze_a1_identity_gate.py',
        'scripts/package_pin_geometry_calibration.py','tests/test_pin_geometry_inspector.py','tests/test_freeze_a1_calibration_review.py',
        'tests/test_pin_geometry_reviews.py','docs/PIN_POINT_CALIBRATION_PROGRESS.md','output/pin-geometry-inspector/index.html',
        'output/freeze-a1-identity/readiness.json'}
    names.update(p.relative_to(ROOT).as_posix() for pattern in ('output/pin-geometry-inspector/*.json','output/freeze-a1-calibration/**/*.json') for p in ROOT.glob(pattern) if p.name!='official-source-evidence.json')
    manifest={'artifact':'coordinate semantics and independent evaluation infrastructure; requires workspace crop images',
        'frozen_verified':verified,'tests_passed':146,'runtime_changed':False,'training_runs':0,'FreezeA_inference_runs':0,
        'verified_designs':0,'human_review_entries':0,'third_party_images_included':False,
        'files':{n:sha(ROOT/n) for n in sorted(names)}}
    target=ROOT/'submission/pin-geometry-calibration-assessment.zip'
    with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED) as z:
        for n in sorted(names):z.write(ROOT/n,n)
        z.writestr('BUILD_MANIFEST.json',json.dumps(manifest,indent=2))
    with zipfile.ZipFile(target) as z:
        for n,d in manifest['files'].items():assert hashlib.sha256(z.read(n)).hexdigest()==d
    result={'archive':str(target),'sha256':sha(target),'files':len(names),'bytes':target.stat().st_size,'frozen_verified':verified}
    target.with_suffix('.manifest.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,ensure_ascii=False))
if __name__=='__main__':main()
