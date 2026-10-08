"""Freeze the visible-dot experiment and preserve all earlier baselines."""
import json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def main():
    destination=ROOT/'output/separable-circuit-research-manifest.json'
    if destination.exists():raise ValueError('Preserve completed freeze')
    preserved={}
    for name in ('output/circuit-program-research-manifest.json','output/factorized-circuit-research-manifest.json','data/splits/rc0-evaluation.json','output/review-handoff/convention-pilot-protocol-lock.json'):
        m=json.loads((ROOT/name).read_text(encoding='utf-8-sig'));hashes=m.get('runtime_sha256',m.get('sha256',{}))
        assert hashes and all(sha(ROOT/p)==h for p,h in hashes.items())
        preserved[name]={'checked':len(hashes),'unchanged':True}
    package=ROOT/'submission/graphschematic-rc0-evaluation.zip'
    assert sha(package)=='04ef04248830666d6a4020c4ee77a73b82d130b4632d9a8e5946a90f996586b4'
    files=set()
    for pattern in ('scripts/*separable_circuit*.py','docs/SEPARABLE_CIRCUIT_RESULTS.md','output/separable-circuit*.log'):
        files.update(p for p in ROOT.glob(pattern) if p.is_file())
    for folder in ('data/processed/separable-circuit-v1','output/training/separable-circuit-v1','output/separable-circuit-audit','output/separable-circuit-demo'):
        files.update(p for p in (ROOT/folder).rglob('*') if p.is_file())
    for relative in ('experiments/factorized_circuit_reader.py','scripts/train_factorized_circuit.py','scripts/audit_factorized_circuit.py','scripts/prepare_factorized_circuit.py','scripts/train_circuit_program.py','scripts/prepare_circuit_program.py'):
        files.add(ROOT/relative)
    hashes={p.relative_to(ROOT).as_posix():sha(p) for p in sorted(files)}
    result={'scope':'new finite visible-dot renderer and preregistered model/template/composition comparisons',
            'status':'FROZEN_RESEARCH_ONLY_MODEL_STRESS_NO_GO_TEMPLATE_SYNTHETIC_PASS',
            'real_verified_designs':0,'official_accuracy_claimed':False,'preserved':preserved,
            'evaluator_hash':sha(ROOT/'scripts/train_circuit_program.py'),
            'scoring_config_hash':sha(ROOT/'output/separable-circuit-audit/protocol.json'),
            'RC0_package_unchanged':True,'sha256':hashes}
    destination.write_text(json.dumps(result,indent=2),encoding='utf-8')
    assert all(sha(ROOT/p)==h for p,h in hashes.items())
    print(json.dumps({'frozen_files':len(hashes),'preserved':preserved,'RC0_package_unchanged':True}))


if __name__=='__main__':main()
