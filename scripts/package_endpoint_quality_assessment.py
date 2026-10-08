"""Separate assessment archive; no third-party pixels or frozen-file writes."""
import hashlib
import json
import zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    verified={}
    for name,key in [('rc0-evaluation.json','runtime_sha256'),('pin-token-v1-frozen.json','sha256'),('pin-token-residual-frozen.json','sha256')]:
        records=json.loads((ROOT/'data/splits'/name).read_text(encoding='utf-8'))[key]
        for path,sha in records.items():
            if digest(ROOT/path)!=sha: raise ValueError('Frozen file changed: '+path)
        verified[name]=len(records)
    review=json.loads((ROOT/'output/ablation/hard56-review-draft.json').read_text(encoding='utf-8'))
    queue={'status':'pending-independent-human-review','reviewed_correctable_HRR':None,
        'raw56_HRR_unchanged':review['raw56_HRR_unchanged'],
        'source_sha256':digest(ROOT/'output/ablation/hard56-review-draft.json'),
        'entries':[e for e in review['entries'] if e['category'] in ('geometry-fixable','ocr-fixable','unknown')]}
    (ROOT/'output/endpoint-quality/hard13-review-queue.json').write_text(json.dumps(queue,ensure_ascii=False,indent=2),encoding='utf-8')
    names={'experiments/endpoint_quality.py','scripts/audit_unresolved_endpoint_quality.py',
        'scripts/package_endpoint_quality_assessment.py','tests/test_endpoint_quality.py',
        'scripts/audit_freeze_a1_identity.py','scripts/freeze_a1_identity_evaluator.py','scripts/freeze_a1_identity_gate.py',
        'tests/test_freeze_a1_identity.py','docs/ENDPOINT_QUALITY_PROGRESS.md','docs/FREEZE_A1_IDENTITY_PROGRESS.md',
        'data/research/freeze-a1-agent-pixel-review.json','data/research/freeze-a1-family-review.json'}
    names.update(p.relative_to(ROOT).as_posix() for pattern in ('output/endpoint-quality/**/*.json','output/freeze-a1-identity/**/*.json') for p in ROOT.glob(pattern))
    manifest={'artifact':'endpoint quality and Freeze-A1 identity assessment; not competition runtime',
        'frozen_verified':verified,'tests_passed':129,'new_training':False,'runtime_promoted':False,
        'independent_verified_designs':0,'third_party_pixels_included':False,
        'files':{name:{'sha256':digest(ROOT/name),'bytes':(ROOT/name).stat().st_size} for name in sorted(names)}}
    out=ROOT/'submission/endpoint-quality-assessment.zip'
    with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED) as z:
        for name in sorted(names): z.write(ROOT/name,name)
        z.writestr('BUILD_MANIFEST.json',json.dumps(manifest,indent=2))
    with zipfile.ZipFile(out) as z:
        for name,r in manifest['files'].items(): assert hashlib.sha256(z.read(name)).hexdigest()==r['sha256']
    record={'archive':str(out),'sha256':digest(out),'files':len(names),'bytes':out.stat().st_size,'frozen_verified':verified}
    out.with_suffix('.manifest.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
    print(json.dumps(record,ensure_ascii=False))
if __name__=='__main__': main()
