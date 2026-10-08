"""Freeze negative neural experiments, all data and audit trails, preserve prior baselines."""
import json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def main():
    dest=ROOT/'output/radial-evidence-research-manifest.json'
    if dest.exists():raise ValueError('Preserve freeze')
    preserved={}
    names=['circuit-program','factorized-circuit','separable-circuit','registered-circuit','registration-v2','contrast-edge','guard-v2','ordered-rectification']
    manifests=[f'output/{n}-research-manifest.json' for n in names]+['data/splits/rc0-evaluation.json','output/review-handoff/convention-pilot-protocol-lock.json']
    for name in manifests:
        m=json.loads((ROOT/name).read_text(encoding='utf-8-sig'));hashes=m.get('runtime_sha256',m.get('sha256',{}))
        assert hashes and all(sha(ROOT/p)==h for p,h in hashes.items()),name
        preserved[name]={'checked':len(hashes),'unchanged':True}
    assert sha(ROOT/'submission/graphschematic-rc0-evaluation.zip')=='04ef04248830666d6a4020c4ee77a73b82d130b4632d9a8e5946a90f996586b4'
    report=json.loads((ROOT/'output/radial-evidence-evaluation/report.json').read_text())
    assert all(not report['cohorts']['stress'][mode]['gate_passed'] for mode in ('raw','canonical'))
    assert not json.loads((ROOT/'output/radial-consensus-holdout/report.json').read_text())['gate_passed']
    for name in ('raw','canonical','consensus','rotation'):
        d=json.loads((ROOT/f'output/radial-evidence-demo/{name}/prediction.json').read_text())
        assert d['topology_status']=='unknown' and d['requires_review'] and d['verified_net_groups'] is None
        assert (d['program_proposal'] is not None)==(name!='rotation')
    files=set()
    for pattern in ('scripts/*radial*.py','experiments/radial_evidence_net.py','docs/RADIAL_EVIDENCE_RESULTS.md','output/radial*.log','output/radial-evidence-representation-check.json'):
        files.update(p for p in ROOT.glob(pattern) if p.is_file())
    for folder in ('data/processed/radial-evidence-v1','output/training/radial-evidence-v1-raw','output/training/radial-evidence-v1-canonical','output/radial-evidence-evaluation','output/radial-consensus-holdout','output/radial-evidence-demo'):
        files.update(p for p in (ROOT/folder).rglob('*') if p.is_file())
    for relative in ('scripts/train_circuit_program.py','scripts/evaluate_ordered_holdout.py','scripts/ordered_line_rectification_v2.py','experiments/contrast_edge_circuit_reader.py','experiments/factorized_circuit_reader.py','output/training/contrast-edge-v1-contrast/best.pt'):
        files.add(ROOT/relative)
    hashes={p.relative_to(ROOT).as_posix():sha(p) for p in sorted(files)}
    result={'status':'FAILED_DIRECT_REPLACEMENT_AND_CONSENSUS_GATES_RESEARCH_ONLY','automatic_release_enabled':False,
            'new_optimizer_steps':3408,'new_synthetic_program_pages':5124,'external_real_dataset_added':False,
            'independent_real_accuracy':False,'verified_designs':0,'RC0_changed':False,'preserved':preserved,
            'evaluator_hash':sha(ROOT/'scripts/train_circuit_program.py'),
            'scoring_config_hash':sha(ROOT/'output/radial-evidence-evaluation/protocol.json'),
            'consensus_scoring_config_hash':sha(ROOT/'output/radial-consensus-holdout/protocol.json'),'sha256':hashes}
    dest.write_text(json.dumps(result,indent=2),encoding='utf-8')
    assert all(sha(ROOT/p)==h for p,h in hashes.items())
    print(json.dumps({'frozen_files':len(hashes),'preserved':preserved,'failed_gates_and_CPU_abstention_checked':True}))


if __name__=='__main__':main()
