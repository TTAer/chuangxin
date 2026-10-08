"""Freeze multiband registration evidence and retain the unsupported-input counterexample."""
import json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def main():
    dest=ROOT/'output/registration-v2-research-manifest.json'
    if dest.exists():raise ValueError('Preserve freeze')
    preserved={}
    for name in ('output/circuit-program-research-manifest.json','output/factorized-circuit-research-manifest.json','output/separable-circuit-research-manifest.json','output/registered-circuit-research-manifest.json','data/splits/rc0-evaluation.json','output/review-handoff/convention-pilot-protocol-lock.json'):
        m=json.loads((ROOT/name).read_text(encoding='utf-8-sig'));hashes=m.get('runtime_sha256',m.get('sha256',{}))
        assert hashes and all(sha(ROOT/p)==h for p,h in hashes.items())
        preserved[name]={'checked':len(hashes),'unchanged':True}
    assert sha(ROOT/'submission/graphschematic-rc0-evaluation.zip')=='04ef04248830666d6a4020c4ee77a73b82d130b4632d9a8e5946a90f996586b4'
    for cohort in ('primary','shear'):
        demo=json.loads((ROOT/f'output/registration-v2-demo/{cohort}/prediction.json').read_text())
        assert demo['topology_status']=='unknown' and demo['verified_net_groups'] is None
    files=set()
    for pattern in ('scripts/circuit_registration_v2.py','scripts/*registration_v2*.py','docs/REGISTRATION_V2_RESULTS.md','output/registration-v2*.log'):
        files.update(p for p in ROOT.glob(pattern) if p.is_file())
    for folder in ('output/registration-v2-development','output/registration-v2-holdout','output/registration-v2-demo'):
        files.update(p for p in (ROOT/folder).rglob('*') if p.is_file())
    for relative in ('scripts/circuit_registration.py','scripts/predict_registered_circuit.py','scripts/train_circuit_program.py','scripts/audit_factorized_circuit.py','scripts/prepare_separable_circuit.py','experiments/factorized_circuit_reader.py','output/training/registered-circuit-v1/best.pt'):
        files.add(ROOT/relative)
    hashes={p.relative_to(ROOT).as_posix():sha(p) for p in sorted(files)}
    result={'scope':'image-only multiband registration, frozen model, new576-page controlled holdouts',
            'status':'FROZEN_RESEARCH_ONLY_MODEL_GATE_FAIL_UNSUPPORTED_READY_COUNTEREXAMPLES',
            'new_optimizer_steps':0,'verified_designs':0,'RC0_package_unchanged':True,'preserved':preserved,
            'evaluator_hash':sha(ROOT/'scripts/train_circuit_program.py'),
            'scoring_config_hash':sha(ROOT/'output/registration-v2-holdout/protocol.json'),'sha256':hashes}
    dest.write_text(json.dumps(result,indent=2),encoding='utf-8')
    assert all(sha(ROOT/p)==h for p,h in hashes.items())
    print(json.dumps({'frozen_files':len(hashes),'preserved':preserved,'unknown_boundary_checked':True}))


if __name__=='__main__':main()
