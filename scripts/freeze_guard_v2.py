"""Freeze a controlled-passing geometry candidate without certifying real accuracy."""
import json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def main():
    dest=ROOT/'output/guard-v2-research-manifest.json'
    if dest.exists():raise ValueError('Preserve freeze')
    preserved={}
    for name in ('output/circuit-program-research-manifest.json','output/factorized-circuit-research-manifest.json','output/separable-circuit-research-manifest.json','output/registered-circuit-research-manifest.json','output/registration-v2-research-manifest.json','output/contrast-edge-research-manifest.json','data/splits/rc0-evaluation.json','output/review-handoff/convention-pilot-protocol-lock.json'):
        m=json.loads((ROOT/name).read_text(encoding='utf-8-sig'));hashes=m.get('runtime_sha256',m.get('sha256',{}))
        assert hashes and all(sha(ROOT/p)==h for p,h in hashes.items())
        preserved[name]={'checked':len(hashes),'unchanged':True}
    assert sha(ROOT/'submission/graphschematic-rc0-evaluation.zip')=='04ef04248830666d6a4020c4ee77a73b82d130b4632d9a8e5946a90f996586b4'
    for n in ('shear','rotation'):
        d=json.loads((ROOT/f'output/guard-v2-demo/{n}/prediction.json').read_text())
        assert d['program_proposal'] is None and not d['geometry']['ready'] and d['topology_status']=='unknown'
    files=set()
    for pattern in ('scripts/*guard_v2*.py','scripts/circuit_geometry_guard_v2.py','docs/GUARD_V2_RESULTS.md','output/guard-v2*.log'):
        files.update(p for p in ROOT.glob(pattern) if p.is_file())
    for folder in ('output/guard-v2-development','output/guard-v2-holdout','output/guard-v2-demo'):
        files.update(p for p in (ROOT/folder).rglob('*') if p.is_file())
    for relative in ('scripts/circuit_geometry_guard.py','scripts/circuit_registration_v2.py','scripts/train_circuit_program.py','scripts/predict_contrast_edge.py','scripts/prepare_separable_circuit.py','experiments/contrast_edge_circuit_reader.py','experiments/factorized_circuit_reader.py','output/training/contrast-edge-v1-contrast/best.pt'):
        files.add(ROOT/relative)
    hashes={p.relative_to(ROOT).as_posix():sha(p) for p in sorted(files)}
    result={'scope':'multicontrast geometry, fixed ContrastOnlyReader, new640-page controlled holdout',
            'status':'CONTROLLED_RESEARCH_GATE_PASSED_PENDING_REAL_VALIDATION','new_optimizer_steps':0,
            'independent_real_accuracy':False,'verified_designs':0,'RC0_package_unchanged':True,'preserved':preserved,
            'evaluator_hash':sha(ROOT/'scripts/train_circuit_program.py'),
            'scoring_config_hash':sha(ROOT/'output/guard-v2-holdout/protocol.json'),'sha256':hashes}
    dest.write_text(json.dumps(result,indent=2),encoding='utf-8')
    assert all(sha(ROOT/p)==h for p,h in hashes.items())
    print(json.dumps({'frozen_files':len(hashes),'preserved':preserved,'unsupported_abstention_checked':True}))


if __name__=='__main__':main()
