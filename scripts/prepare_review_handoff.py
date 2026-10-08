"""Lock the existing V2 cohort and prepare a four-source human handoff.

This does not certify reviewers, freeze truth, or authorize inference.
Existing locks are checked, never regenerated after a change.
"""
import hashlib
import json
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'output/review-handoff'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_lock(root, lock):
    changed = [p for p, digest in lock['sha256'].items()
               if not (root / p).is_file() or sha(root / p) != digest]
    if changed:
        raise ValueError('Locked review evidence changed: ' + ', '.join(changed))


def revision_disposition(matched, expected):
    if expected <= 0 or not 0 <= matched <= expected:
        return 'invalid_evidence'
    # Integer comparison preserves the preregistered 90% / 98% boundaries.
    if matched * 100 >= expected * 98:
        return 'human_priority'
    if matched * 100 >= expected * 90:
        return 'revision_review_required'
    return 'paused_revision_risk'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reviews', nargs=2, metavar=('REVIEW_A', 'REVIEW_B'))
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    cohort = ROOT / 'output/pin-geometry-review-v2'
    cases = json.loads((cohort / 'cases.json').read_text(encoding='utf-8'))
    paths = [cohort / 'cases.json', cohort / 'protocol.json', cohort / 'index.html',
             ROOT / 'scripts/pin_geometry_review_v2_template.html',
             ROOT / 'scripts/prepare_pin_geometry_review_v2.py',
             ROOT / 'experiments/semantic_endpoint_audit.py']
    for c in cases:
        paths += [ROOT / c['image'], (cohort / c['crop']).resolve()]
    lock_path = OUT / 'convention-pilot-protocol-lock.json'
    if lock_path.exists():
        lock = json.loads(lock_path.read_text(encoding='utf-8'))
        check_lock(ROOT, lock)
    else:
        lock = {
            'schema': 'convention-pilot-protocol-lock-v1',
            'purpose': 'lock protocol and reviewed pixels, not coordinate convention or truth',
            'case_ids': [c['id'] for c in cases],
            'case_count': len(cases),
            'sha256': {p.relative_to(ROOT).as_posix(): sha(p) for p in paths},
            'CoordinateConventionFrozen': False,
            'human_certification': False,
            'inference_authorized': False,
        }
        lock_path.write_text(json.dumps(lock, ensure_ascii=False, indent=2), encoding='utf-8')
    source = ROOT / 'output/freeze-a1-revision-audit/summary.json'
    summary = json.loads(source.read_text(encoding='utf-8'))
    rows = []
    priority_ids = {r['image_sha256'] for r in summary['shortlist'][:4]}
    for r in summary['pages_detail']:
        row = {k: r[k] for k in ('image_sha256', 'repository', 'source_cad',
                                'source_cad_sha256', 'source_pdf', 'source_pdf_sha256',
                                'PDF_page', 'visible_nonzero_CAD_pin_segments',
                                'PDF_pin_segments_matching')}
        for path_key, sha_key in [('source_cad', 'source_cad_sha256'),
                                  ('source_pdf', 'source_pdf_sha256')]:
            if sha(ROOT / row[path_key]) != row[sha_key]:
                raise ValueError('Revision audit source changed: ' + row[path_key])
        row['revision_risk_disposition'] = revision_disposition(
            row['PDF_pin_segments_matching'], row['visible_nonzero_CAD_pin_segments'])
        row['active_human_review_scope'] = row['image_sha256'] in priority_ids
        prefix = row['image_sha256'][:12]
        row['whole_sheet_review_form'] = f'output/freeze-a1-calibration/{prefix}/review-template.json'
        row['two_layer_truth'] = f'output/freeze-a1-two-layer/{prefix}.json'
        row['revision_evidence'] = f'output/freeze-a1-revision-audit/{prefix}.json'
        for key in ('whole_sheet_review_form', 'two_layer_truth', 'revision_evidence'):
            if not (ROOT / row[key]).is_file():
                raise ValueError('Missing handoff evidence: ' + row[key])
        row['verified_design'] = False
        rows.append(row)
    result = {
        'schema': 'human-review-handoff-v1',
        'revision_audit_sha256': sha(source),
        'protocol_lock_sha256': sha(lock_path),
        'scope': 'visible CAD pin segment consistency is source risk screening, not model accuracy or revision certification',
        'active_source_count': sum(r['active_human_review_scope'] for r in rows),
        'sources': rows,
        'pilot_minimum_verified_designs': 4,
        'official_minimum_verified_designs': 10,
        'pilot_use': 'risk go/no-go only; no tuning; tuned pilot becomes development',
        'manifest_created': False,
        'inference_authorized': False,
        'CoordinateConventionFrozen': False,
    }
    (OUT / 'source-risk-handoff.json').write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'locked_cases': len(cases), 'active_sources': result['active_source_count'],
                      'inference_authorized': False}))
    if args.reviews:
        # The locked importer verifies GT and raw identity provenance as well.
        subprocess.run([sys.executable, str(ROOT / 'scripts/prepare_pin_geometry_review_v2.py'),
                        '--reviews', *args.reviews], cwd=ROOT, check=True)


if __name__ == '__main__':
    main()
