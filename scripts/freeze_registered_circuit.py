"""Freeze registration research, data and gate decisions while preserving baselines."""
import json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def main():
    dest=ROOT/'output/registered-circuit-research-manifest.json'
    if dest.exists():raise ValueError('Preserve freeze')
    preserved={}
    for name in ('output/circuit-program-research-manifest.json','output/factorized-circuit-research-manifest.json','output/separable-circuit-research-manifest.json','data/splits/rc0-evaluation.json','output/review-handoff/convention-pilot-protocol-lock.json'):
        m=json.loads((ROOT/name).read_text(encoding='utf-8-sig'));hashes=m.get('runtime_sha256',m.get('sha256',{}))
        assert hashes and all(sha(ROOT/p)==h for p,h in hashes.items())
        preserved[name]={'checked':len(hashes),'unchanged':True}
    assert sha(ROOT/'submission/graphschematic-rc0-evaluation.zip')=='04ef04248830666d6a4020c4ee77a73b82d130b4632d9a8e5946a90f996586b4'
    abstain=json.loads((ROOT/'output/registered-circuit-abstain-demo/prediction.json').read_text())
    assert abstain['program_proposal'] is None and abstain['topology_status']=='unknown'
    files=set()
    for pattern in ('scripts/*registered_circuit*.py','scripts/circuit_registration.py','docs/REGISTERED_CIRCUIT_RESULTS.md','output/registered-circuit*.log'):
        files.update(p for p in ROOT.glob(pattern) if p.is_file())
    for folder in ('data/processed/registered-circuit-v1','output/training/registered-circuit-v1','output/registered-circuit-evaluation','output/registered-circuit-final','output/registered-circuit-demo','output/registered-circuit-abstain-demo'):
        files.update(p for p in (ROOT/folder).rglob('*') if p.is_file())
    for relative in ('scripts/train_factorized_circuit.py','experiments/factorized_circuit_reader.py','scripts/audit_factorized_circuit.py','scripts/prepare_separable_circuit.py','scripts/train_circuit_program.py'):
        files.add(ROOT/relative)
    hashes={p.relative_to(ROOT).as_posix():sha(p) for p in sorted(files)}
    result={'scope':'image-only registration, affine grammar layouts, fixed developmental training gate',
            'status':'FROZEN_RESEARCH_ONLY_PRESSURE_NO_GO','independent_real_accuracy':False,'verified_designs':0,
            'evaluator_hash':sha(ROOT/'scripts/train_circuit_program.py'),
            'scoring_config_hash':sha(ROOT/'output/registered-circuit-evaluation/protocol.json'),
            'preserved':preserved,'RC0_package_unchanged':True,'sha256':hashes}
    dest.write_text(json.dumps(result,indent=2),encoding='utf-8')
    assert all(sha(ROOT/p)==h for p,h in hashes.items())
    print(json.dumps({'frozen_files':len(hashes),'preserved':preserved,'abstention_verified':True}))


if __name__=='__main__':main()
