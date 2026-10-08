"""Archive new assessment metadata without third-party pixels or runtime edits."""
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
    files={'experiments/semantic_endpoint_audit.py','scripts/prepare_pin_geometry_review_v2.py','scripts/pin_geometry_review_v2_template.html',
        'scripts/audit_freeze_a1_revision_candidates.py','scripts/prepare_freeze_a1_two_layer_truth.py','scripts/freeze_a1_identity_gate.py',
        'scripts/package_semantic_endpoint_review_v2.py','tests/test_semantic_endpoint_audit.py','tests/test_freeze_a1_revision_candidates.py',
        'tests/test_freeze_a1_calibration_review.py','docs/SEMANTIC_ENDPOINT_REVIEW_V2_PROGRESS.md','output/pin-geometry-review-v2/index.html',
        'output/freeze-a1-identity/readiness.json'}
    for pattern in ('output/pin-geometry-review-v2/*.json','output/freeze-a1-revision-audit/*.json','output/freeze-a1-two-layer/*.json'):
        files.update(p.relative_to(ROOT).as_posix() for p in ROOT.glob(pattern))
    manifest={'artifact':'semantic endpoint and whole-page revision evidence; requires existing workspace images',
        'tests_passed':160,'frozen_verified':verified,'training_runs':0,'FreezeA_model_runs':0,'runtime_changed':False,
        'verified_designs':0,'human_reviews':0,'semantic_eligible_P5':None,'third_party_pixels_included':False,
        'files':{f:sha(ROOT/f) for f in sorted(files)}}
    target=ROOT/'submission/semantic-endpoint-review-v2-assessment.zip'
    with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED) as z:
        for file in sorted(files):z.write(ROOT/file,file)
        z.writestr('BUILD_MANIFEST.json',json.dumps(manifest,indent=2))
    with zipfile.ZipFile(target) as z:
        for f,h in manifest['files'].items():assert hashlib.sha256(z.read(f)).hexdigest()==h
    result={'archive':str(target),'files':len(files),'bytes':target.stat().st_size,'sha256':sha(target),'frozen_verified':verified}
    target.with_suffix('.manifest.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,ensure_ascii=False))
if __name__=='__main__':main()
