"""Hash research evidence and verify frozen mainline without repeating inference."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    destination = ROOT/'output/circuit-program-research-manifest.json'
    if destination.exists():
        raise ValueError('Preserve existing research freeze')
    preserved = {}
    for name in ('data/splits/rc0-evaluation.json', 'output/port-lift-research-manifest.json',
                 'output/port-lift-uncertainty-manifest.json', 'output/probe-flow-research-manifest.json',
                 'output/netset-oracle-research-manifest.json', 'output/topology-agreement-manifest.json',
                 'output/review-handoff/convention-pilot-protocol-lock.json'):
        m = json.loads((ROOT/name).read_text(encoding='utf-8-sig'))
        hashes = m.get('runtime_sha256', m.get('sha256', {}))
        failures = [p for p, h in hashes.items() if not (ROOT/p).exists() or digest(ROOT/p) != h]
        preserved[name] = {'checked': len(hashes), 'mismatches': failures}
        if not hashes or failures:
            raise ValueError(f'Frozen evidence failure: {name}: {failures}')
    package = ROOT/'submission/graphschematic-rc0-evaluation.zip'
    expected = '04ef04248830666d6a4020c4ee77a73b82d130b4632d9a8e5946a90f996586b4'
    assert digest(package) == expected
    preserved['submission/graphschematic-rc0-evaluation.zip'] = {'sha256': expected, 'unchanged': True}
    # Repair the RC0 field mapping of the original diagnostic record, no inference rerun.
    audit_path = ROOT/'output/circuit-program-audit/report.json'
    audit = json.loads(audit_path.read_text())
    audit['frozen_checks'] = preserved
    audit_path.write_text(json.dumps(audit, indent=2), encoding='utf-8')
    files = set()
    for pattern in ('experiments/circuit_program*.py', 'scripts/*circuit_program*.py',
                    'scripts/intake_logicbench.py', 'docs/CIRCUIT_PROGRAM*.md',
                    'output/circuit-program*.log'):
        files.update(p for p in ROOT.glob(pattern) if p.is_file())
    for folder in ('data/processed/circuit-program-v1', 'data/processed/circuit-program-v2',
                   'data/raw/CircuitProgram-expanded', 'data/raw/LogicBench-train',
                   'output/circuit-program-audit', 'output/circuit-program-demo',
                   *[f'output/training/circuit-program-v{i}' for i in range(1, 5)]):
        files.update(p for p in (ROOT/folder).rglob('*') if p.is_file())
    hashes = {p.relative_to(ROOT).as_posix(): digest(p) for p in sorted(files)}
    result = {'scope': 'CircuitProgramNet-H V1-V4, downloaded train inputs, actual image pretraining and full synthetic program training',
              'status': 'FROZEN_RESEARCH_ONLY_STYLE_STRESS_NO_GO',
              'external_pixel_admitted_images': 2282, 'image_pretraining_patches': 37918,
              'synthetic_main_training_pages': 8192, 'independent_verified_designs': 0,
              'official_score_improvement_claimed': False, 'preserved': preserved, 'sha256': hashes}
    destination.write_text(json.dumps(result, indent=2), encoding='utf-8')
    assert all(digest(ROOT/p) == h for p, h in hashes.items())
    print(json.dumps({'frozen_files': len(hashes), 'preserved': preserved}))


if __name__ == '__main__':
    main()
