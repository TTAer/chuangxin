"""Freeze four completed trials and interrupted attempt without promoting a failed candidate."""
import json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def main():
    dest=ROOT/'output/evidence-factorial-research-manifest.json'
    if dest.exists():raise ValueError('Preserve freeze')
    names=['circuit-program','factorized-circuit','separable-circuit','registered-circuit','registration-v2','contrast-edge','guard-v2','ordered-rectification','radial-evidence'];preserved={}
    for name in [f'output/{n}-research-manifest.json' for n in names]+['data/splits/rc0-evaluation.json','output/review-handoff/convention-pilot-protocol-lock.json']:
        m=json.loads((ROOT/name).read_text(encoding='utf-8-sig'));hashes=m.get('runtime_sha256',m.get('sha256',{}));assert hashes and all(sha(ROOT/p)==h for p,h in hashes.items()),name
        preserved[name]={'checked':len(hashes),'unchanged':True}
    assert sha(ROOT/'submission/graphschematic-rc0-evaluation.zip')=='04ef04248830666d6a4020c4ee77a73b82d130b4632d9a8e5946a90f996586b4'
    report=json.loads((ROOT/'output/evidence-factorial-holdout/report.json').read_text());assert not report['selected_gate_passed']
    assert all(not report['cohorts']['primary'][trial]['gate_passed'] for trial in report['protocol']['four_trials'])
    for name in ('primary','rotation'):
        d=json.loads((ROOT/f'output/evidence-factorial-demo/{name}/prediction.json').read_text());assert d['topology_status']=='unknown' and d['requires_review'] and d['verified_net_groups'] is None and not d['automatic_release_enabled']
        assert (d['program_proposal'] is not None)==(name=='primary')
    files=set()
    for pattern in ('scripts/*evidence_factorial*.py','scripts/check_evidence*.py','experiments/spatial_evidence_net.py','docs/EVIDENCE_FACTORIAL_RESULTS.md','output/evidence*.log','output/evidence*-check.json'):
        files.update(p for p in ROOT.glob(pattern) if p.is_file())
    for folder in list(ROOT.glob('output/training/evidence-factorial-v1-*'))+[ROOT/'output/evidence-factorial-holdout',ROOT/'output/evidence-factorial-demo']:
        files.update(p for p in folder.rglob('*') if p.is_file())
    for relative in ('experiments/radial_evidence_net.py','scripts/train_circuit_program.py','scripts/evaluate_ordered_holdout.py','scripts/ordered_line_rectification_v2.py','experiments/contrast_edge_circuit_reader.py','experiments/factorized_circuit_reader.py','output/training/contrast-edge-v1-contrast/best.pt','data/processed/radial-evidence-v1/train-patches.npz','data/processed/radial-evidence-v1/validation-patches.npz','data/processed/radial-evidence-v1/tiny-patches.npz'):
        files.add(ROOT/relative)
    hashes={p.relative_to(ROOT).as_posix():sha(p) for p in sorted(files)}
    result={'date':'2026-10-08','status':'FAILED_PREREGISTERED_GATE_RESEARCH_ONLY','automatic_release_enabled':False,
            'completed_optimizer_steps':6816,'interrupted_attempt_steps':'unrecorded, excluded from completed total; not asserted zero',
            'new_synthetic_holdout_pages':640,'new_external_real_data':False,'independent_real_accuracy':False,'verified_designs':0,'RC0_changed':False,
            'preserved':preserved,'evaluator_hash':sha(ROOT/'scripts/train_circuit_program.py'),
            'scoring_config_hash':sha(ROOT/'output/evidence-factorial-holdout/protocol.json'),'sha256':hashes}
    dest.write_text(json.dumps(result,indent=2),encoding='utf-8');assert all(sha(ROOT/p)==h for p,h in hashes.items())
    print(json.dumps({'frozen_files':len(hashes),'preserved':preserved,'failed_gates_and_CPU_rejection_checked':True}))


if __name__=='__main__':main()
