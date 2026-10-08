"""Fail-closed first-inference gate; --freeze only writes when >=10 approved designs."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
MINIMUM_DESIGNS=10

def calibration_assess(audit):
    """Coordinate semantics and full-sheet audit precede any first inference."""
    import sys
    sys.path.insert(0,str(ROOT))
    from scripts.prepare_freeze_a1_calibration_review import review_gate
    records={};hashes={}
    for r in audit['records']:
        imagehash=r['image_sha256']
        if imagehash in records:continue
        path=ROOT/'output/freeze-a1-calibration'/imagehash[:12]/'review-template.json'
        if not path.exists():
            records[imagehash]={'semantic_review_complete':False,'reasons':['calibration_review_form_missing']};continue
        review=json.loads(path.read_text(encoding='utf-8'))
        result=review_gate(review,r)
        hashes[str(path.relative_to(ROOT))]=sha(path)
        evidence=list(review.get('point_convention',{}).get('evidence_records',[]))+[p.get('evidence_record') for p in review.get('review_passes',[]) if p.get('evidence_record')]
        for evidence_name in evidence:
            evidence_path=(ROOT/evidence_name).resolve()
            if not evidence_path.is_relative_to(ROOT.resolve()) or not evidence_path.is_file():
                result['reasons'].append('calibration_evidence_file_missing_or_outside_workspace')
                result['semantic_review_complete']=False
            else:hashes[str(evidence_path.relative_to(ROOT.resolve()))]=sha(evidence_path)
        records[imagehash]=result
    hashes['scripts/prepare_freeze_a1_calibration_review.py']=sha(ROOT/'scripts/prepare_freeze_a1_calibration_review.py')
    return records,hashes

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def assess(audit, approvals):
    ready=[];blockers=[]
    by_repository={r['repository']:r for r in approvals.get('families',[])}
    for row in audit['records']:
        family=by_repository.get(row['repository'],{})
        reasons=[]
        if row['observed_review']['status']!='agent_pixel_matched':reasons.append('pixel_identity_incomplete_or_rejected')
        if not family.get('license_scope_approved'):reasons.append('license_scope_pending')
        if not family.get('cad_pdf_page_revision_verified'):reasons.append('whole_page_revision_pending')
        if not family.get('historical_design_family_exclusion_verified'):reasons.append('design_family_exclusion_pending')
        if not family.get('independent_review_complete'):reasons.append('independent_annotation_review_pending')
        if not family.get('verified_design_id'):reasons.append('verified_design_id_missing')
        if family.get('historical_design_family_exclusion_verified') and not family.get('historical_corpus_manifest'):
            reasons.append('historical_corpus_exclusion_evidence_missing')
        for flag, record in [('license_scope_approved','license_scope_review_record'),
                            ('cad_pdf_page_revision_verified','page_revision_review_record'),
                            ('historical_design_family_exclusion_verified','family_exclusion_review_record'),
                            ('independent_review_complete','independent_annotation_review_record')]:
            if family.get(flag) and not family.get(record):reasons.append(record+'_missing')
        if reasons:blockers.append({'ic_id':row['id'],'repository':row['repository'],'reasons':reasons})
        else:ready.append(row)
    designs={by_repository[r['repository']]['verified_design_id'] for r in ready}
    return {'eligible':len(designs)>=MINIMUM_DESIGNS,'minimum_designs':MINIMUM_DESIGNS,
            'verified_designs':len(designs),'eligible_ic_instances':len(ready),
            'eligible_pages':len({r['image_sha256'] for r in ready}),'blockers':blockers,
            'selected_ids':[r['id'] for r in ready]}

def require_frozen_eligibility(manifest_path=None):
    """Future inference runners must call this before loading any image/model."""
    manifest_path=Path(manifest_path) if manifest_path else ROOT/'data/splits/freeze-a1-identity-frozen.json'
    if not manifest_path.exists():raise ValueError('No eligible frozen identity manifest; inference forbidden')
    manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
    if not manifest.get('eligible') or manifest.get('verified_designs',0)<MINIMUM_DESIGNS:
        raise ValueError('Manifest is not eligible')
    if manifest.get('pin_point_calibration_complete') is not True:
        raise ValueError('Pin point definition and independent whole-sheet review not frozen')
    if manifest.get('CoordinateConventionFrozen') is not True:
        raise ValueError('CoordinateConventionFrozen required before first inference')
    convention=manifest.get('frozen_point_convention')
    if not isinstance(convention,list) or len(convention)!=2 or convention[1]!='image_top_left_integer_pixel':
        raise ValueError('No unified frozen coordinate convention')
    if not manifest.get('source_image_annotation_family_evaluator_sha256'):
        raise ValueError('Frozen dependencies missing')
    for name,digest in manifest['source_image_annotation_family_evaluator_sha256'].items():
        if sha(ROOT/name)!=digest:raise ValueError('Frozen dependency changed: '+name)
    return manifest

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--freeze',action='store_true');args=parser.parse_args()
    audit_path=ROOT/'output/freeze-a1-identity/identity-audit.json'
    approval_path=ROOT/'data/research/freeze-a1-family-review.json'
    audit=json.loads(audit_path.read_text(encoding='utf-8'));approvals=json.loads(approval_path.read_text(encoding='utf-8'))
    result=assess(audit,approvals)
    calibration,calibration_hashes=calibration_assess(audit)
    selected=[]
    for row in audit['records']:
        if row['id'] not in result['selected_ids']:continue
        check=calibration[row['image_sha256']]
        if check['semantic_review_complete']:selected.append(row)
        else:result['blockers'].append({'ic_id':row['id'],'repository':row['repository'],'reasons':check['reasons']})
    by_repository={r['repository']:r for r in approvals.get('families',[])}
    designs={by_repository[r['repository']]['verified_design_id'] for r in selected}
    conventions={(calibration[r['image_sha256']].get('point_definition'),calibration[r['image_sha256']].get('coordinate_system')) for r in selected}
    unified=bool(selected) and len(conventions)==1
    if selected and not unified:
        result['blockers'].append({'ic_id':None,'repository':None,'reasons':['pin_point_conventions_not_unified_across_selected_pages']})
    result.update(selected_ids=[r['id'] for r in selected],verified_designs=len(designs),
        eligible=len(designs)>=MINIMUM_DESIGNS and unified,eligible_ic_instances=len(selected),
        eligible_pages=len({r['image_sha256'] for r in selected}),
        pin_point_calibration_complete=unified and all(calibration[r['image_sha256']]['semantic_review_complete'] for r in selected),
        frozen_point_convention=list(next(iter(conventions))) if unified else None,
        calibration_review=calibration)
    result['CoordinateConventionFrozen']=result['eligible'] and result['pin_point_calibration_complete']
    hashes={}
    hashes.update(calibration_hashes)
    def capture(p):
        p=Path(p);hashes[str(p.relative_to(ROOT))]=sha(p)
    for p in [audit_path,approval_path,ROOT/'data/research/freeze-a1-agent-pixel-review.json',
              ROOT/'data/splits/freeze-a-source-intake.json',Path(__file__).resolve(),
              ROOT/'scripts/freeze_a1_identity_evaluator.py',ROOT/'pcbparse/core.py']:
        capture(p)
    for r in audit['records']:
        for pathkey,hashkey in [('annotation','annotation_sha256'),('image','image_sha256'),('source_cad','source_cad_sha256'),('source_pdf','source_pdf_sha256')]:
            p=ROOT/r[pathkey];capture(p)
            if sha(p)!=r[hashkey]:raise ValueError('source/annotation hash changed: '+str(p))
        for l in r['license_records']:capture(ROOT/l['path'])
    for f in approvals.get('families',[]):
        for p in f.get('evidence_files',[]):capture(ROOT/p)
        if f.get('historical_corpus_manifest'):capture(ROOT/f['historical_corpus_manifest'])
        for key in ('license_scope_review_record','page_revision_review_record','family_exclusion_review_record','independent_annotation_review_record'):
            if f.get(key):capture(ROOT/f[key])
    result.update({'status':'ELIGIBLE' if result['eligible'] else 'NO_GO',
                   'scope':'partial individually reviewed IC identity only; full page and Net not scored',
                   'source_image_annotation_family_evaluator_sha256':hashes,'inference_runs':0})
    (ROOT/'output/freeze-a1-identity/readiness.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    manifest=ROOT/'data/splits/freeze-a1-identity-frozen.json'
    if manifest.exists():
        previous=json.loads(manifest.read_text(encoding='utf-8'))
        if previous!=result:raise ValueError('Frozen evidence or eligibility changed; never overwrite after freeze')
    if args.freeze:
        if not result['eligible']:raise SystemExit('NO_GO: cannot freeze or start first inference without 10 approved designs')
        if not manifest.exists():manifest.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ('blockers','source_image_annotation_family_evaluator_sha256')},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
