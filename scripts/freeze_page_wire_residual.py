"""Freeze all three development attempts, new tests and violated-assumption witness."""
import json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def main():
    dest=ROOT/'output/page-wire-residual-research-manifest.json'
    if dest.exists():raise ValueError('Preserve freeze')
    names=['circuit-program','factorized-circuit','separable-circuit','registered-circuit','registration-v2','contrast-edge','guard-v2','ordered-rectification','radial-evidence','evidence-factorial','topology-cost'];preserved={}
    for name in [f'output/{n}-research-manifest.json' for n in names]+['data/splits/rc0-evaluation.json','output/review-handoff/convention-pilot-protocol-lock.json']:
        m=json.loads((ROOT/name).read_text(encoding='utf-8-sig'));hashes=m.get('runtime_sha256',m.get('sha256',{}));assert hashes and all(sha(ROOT/p)==h for p,h in hashes.items()),name
        preserved[name]={'checked':len(hashes),'unchanged':True}
    assert sha(ROOT/'submission/graphschematic-rc0-evaluation.zip')=='04ef04248830666d6a4020c4ee77a73b82d130b4632d9a8e5946a90f996586b4'
    assert json.loads((ROOT/'output/page-wire-residual-strict-development/report.json').read_text())['gate_passed']
    assert not json.loads((ROOT/'output/page-wire-residual-development/report.json').read_text())['gate_passed']
    assert not json.loads((ROOT/'output/page-wire-residual-holdout/report.json').read_text())['gate_passed']
    witness=json.loads((ROOT/'output/page-wire-boundary/report.json').read_text());assert witness['records'][1]['geometry_ready'] and witness['records'][1]['first_slot_rule']!=witness['records'][1]['expected']
    for name in ('primary','rotation','mixed'):
        d=json.loads((ROOT/f'output/page-wire-residual-demo/{name}/prediction.json').read_text());assert d['topology_status']=='unknown' and d['requires_review'] and d['verified_net_groups'] is None and not d['automatic_release_enabled'] and not d['shared_linewidth_verified']
        assert (d['program_proposal'] is not None)==(name!='rotation')
    files=set()
    for pattern in ('scripts/*stroke_residual*.py','scripts/*page_wire*.py','experiments/*residual.py','docs/PAGE_WIRE_RESIDUAL_RESULTS.md','output/*wire-residual*.log','output/stroke-residual*.log','output/stroke-residual-fixtures.json'):
        files.update(p for p in ROOT.glob(pattern) if p.is_file())
    for folder in ('output/stroke-residual-development','output/page-wire-residual-development','output/page-wire-residual-strict-development','output/page-wire-residual-holdout','output/page-wire-residual-demo','output/page-wire-boundary'):
        files.update(p for p in (ROOT/folder).rglob('*') if p.is_file())
    for relative in ('experiments/radial_evidence_net.py','scripts/train_circuit_program.py','scripts/evaluate_ordered_holdout.py','scripts/ordered_line_rectification_v2.py','output/training/contrast-edge-v1-contrast/best.pt','output/training/topology-cost-v1-risk/best.pt'):
        files.add(ROOT/relative)
    hashes={p.relative_to(ROOT).as_posix():sha(p) for p in sorted(files)}
    result={'date':'2026-10-08','status':'FAILED_PREREGISTERED_GATE_RESEARCH_ONLY','automatic_release_enabled':False,
            'new_optimizer_steps':0,'new_synthetic_holdout_pages':640,'development_attempts':3,'public_development_reused':True,
            'mixed_linewidth_counterexample':True,'independent_real_accuracy':False,'verified_designs':0,'RC0_changed':False,'preserved':preserved,
            'evaluator_hash':sha(ROOT/'scripts/train_circuit_program.py'),'scoring_config_hash':sha(ROOT/'output/page-wire-residual-holdout/protocol.json'),'sha256':hashes}
    dest.write_text(json.dumps(result,indent=2),encoding='utf-8');assert all(sha(ROOT/p)==h for p,h in hashes.items())
    print(json.dumps({'frozen_files':len(hashes),'preserved':preserved,'counterexample_and_CPU_rejection_checked':True}))


if __name__=='__main__':main()
