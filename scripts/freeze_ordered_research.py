"""Freeze failed/passed developmental trials and new geometry evidence without promoting RC0."""
import json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def main():
    dest=ROOT/'output/ordered-rectification-research-manifest.json'
    if dest.exists():raise ValueError('Preserve freeze')
    preserved={}
    names=['circuit-program','factorized-circuit','separable-circuit','registered-circuit','registration-v2','contrast-edge','guard-v2']
    manifests=[f'output/{n}-research-manifest.json' for n in names]+['data/splits/rc0-evaluation.json','output/review-handoff/convention-pilot-protocol-lock.json']
    for name in manifests:
        m=json.loads((ROOT/name).read_text(encoding='utf-8-sig'));hashes=m.get('runtime_sha256',m.get('sha256',{}))
        assert hashes and all(sha(ROOT/p)==h for p,h in hashes.items()),name
        preserved[name]={'checked':len(hashes),'unchanged':True}
    assert sha(ROOT/'submission/graphschematic-rc0-evaluation.zip')=='04ef04248830666d6a4020c4ee77a73b82d130b4632d9a8e5946a90f996586b4'
    assert not json.loads((ROOT/'output/ordered-rectification-development/report.json').read_text())['gate_passed']
    assert json.loads((ROOT/'output/ordered-rectification-v2-development/report.json').read_text())['gate_passed']
    assert json.loads((ROOT/'output/ordered-rectification-holdout/report.json').read_text())['gate_passed']
    for name in ('irregular','shear','rotation'):
        d=json.loads((ROOT/f'output/ordered-rectification-demo/{name}/prediction.json').read_text())
        assert d['topology_status']=='unknown' and d['requires_review'] and d['verified_net_groups'] is None
        assert (d['program_proposal'] is not None)==(name=='irregular')
    files=set()
    for pattern in ('scripts/*ordered*.py','docs/ORDERED_RECTIFICATION_RESULTS.md','output/ordered-rectification*.log'):
        files.update(p for p in ROOT.glob(pattern) if p.is_file())
    for folder in ('ordered-rectification-development','ordered-rectification-v2-development','ordered-rectification-holdout','ordered-rectification-demo'):
        files.update(p for p in (ROOT/f'output/{folder}').rglob('*') if p.is_file())
    for relative in ('scripts/circuit_geometry_guard_v2.py','scripts/circuit_geometry_guard.py','scripts/circuit_registration_v2.py','scripts/train_circuit_program.py','scripts/prepare_circuit_program.py','scripts/predict_contrast_edge.py','experiments/contrast_edge_circuit_reader.py','experiments/factorized_circuit_reader.py','output/training/contrast-edge-v1-contrast/best.pt'):
        files.add(ROOT/relative)
    hashes={p.relative_to(ROOT).as_posix():sha(p) for p in sorted(files)}
    result={'status':'CONTROLLED_NONUNIFORM_GEOMETRY_GATE_PASSED_PENDING_REAL_VALIDATION',
            'new_optimizer_steps':0,'new_holdout_pages':640,'independent_real_accuracy':False,'verified_designs':0,
            'RC0_changed':False,'preserved':preserved,'evaluator_hash':sha(ROOT/'scripts/train_circuit_program.py'),
            'scoring_config_hash':sha(ROOT/'output/ordered-rectification-holdout/protocol.json'),'sha256':hashes}
    dest.write_text(json.dumps(result,indent=2),encoding='utf-8')
    assert all(sha(ROOT/p)==h for p,h in hashes.items())
    print(json.dumps({'frozen_files':len(hashes),'preserved':preserved,'CPU_unknown_and_abstention_checked':True}))


if __name__=='__main__':main()
