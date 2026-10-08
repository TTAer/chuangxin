"""Freeze paired architecture trials, geometry rejection evidence and baseline integrity."""
import json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for data in iter(lambda:f.read(1024*1024),b''):h.update(data)
    return h.hexdigest()


def main():
    dest=ROOT/'output/contrast-edge-research-manifest.json'
    if dest.exists():raise ValueError('Preserve freeze')
    preserved={}
    for name in ('output/circuit-program-research-manifest.json','output/factorized-circuit-research-manifest.json','output/separable-circuit-research-manifest.json','output/registered-circuit-research-manifest.json','output/registration-v2-research-manifest.json','data/splits/rc0-evaluation.json','output/review-handoff/convention-pilot-protocol-lock.json'):
        m=json.loads((ROOT/name).read_text(encoding='utf-8-sig'));hashes=m.get('runtime_sha256',m.get('sha256',{}))
        assert hashes and all(sha(ROOT/p)==h for p,h in hashes.items())
        preserved[name]={'checked':len(hashes),'unchanged':True}
    assert sha(ROOT/'submission/graphschematic-rc0-evaluation.zip')=='04ef04248830666d6a4020c4ee77a73b82d130b4632d9a8e5946a90f996586b4'
    unsupported=json.loads((ROOT/'output/contrast-edge-demo/shear/prediction.json').read_text())
    assert unsupported['program_proposal'] is None and not unsupported['geometry']['ready'] and unsupported['topology_status']=='unknown'
    files=set()
    for pattern in ('scripts/*contrast_edge*.py','scripts/circuit_geometry_guard.py','experiments/contrast_edge_circuit_reader.py','docs/CONTRAST_EDGE_RESULTS.md','output/contrast-edge*.log'):
        files.update(p for p in ROOT.glob(pattern) if p.is_file())
    for folder in ('data/processed/contrast-edge-v1','output/training/contrast-edge-v1-contrast','output/training/contrast-edge-v1-edge','output/contrast-edge-audit','output/contrast-edge-transfer','output/contrast-edge-demo'):
        files.update(p for p in (ROOT/folder).rglob('*') if p.is_file())
    for p in ('scripts/circuit_registration_v2.py','scripts/train_factorized_circuit.py','scripts/train_circuit_program.py','scripts/audit_factorized_circuit.py','scripts/prepare_separable_circuit.py','experiments/factorized_circuit_reader.py'):
        files.add(ROOT/p)
    hashes={p.relative_to(ROOT).as_posix():sha(p) for p in sorted(files)}
    result={'scope':'whole-page orientation, paired same-budget contrast/edge models, controlled blur and shear holdouts',
            'status':'FROZEN_RESEARCH_ONLY_COVERAGE_AND_FULL_MODEL_GATE_FAIL_READY_SUBSET_SIGNAL',
            'new_optimizer_steps':4896,'independent_real_accuracy':False,'verified_designs':0,
            'preserved':preserved,'RC0_package_unchanged':True,
            'evaluator_hash':sha(ROOT/'scripts/train_circuit_program.py'),
            'scoring_config_hash':sha(ROOT/'output/contrast-edge-audit/protocol.json'),'sha256':hashes}
    dest.write_text(json.dumps(result,indent=2),encoding='utf-8')
    assert all(sha(ROOT/p)==h for p,h in hashes.items())
    print(json.dumps({'frozen_files':len(hashes),'preserved':preserved,'unsupported_abstention_checked':True}))


if __name__=='__main__':main()
