"""Freeze only this isolated research branch, excluding Git objects and RC0."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def main():
    files=set()
    for path in ('experiments/probe_flow.py','experiments/probe_flow_sparse.py','docs/PROBE_FLOW_RESULTS.md',
                 'scripts/prepare_probe_flow.py','scripts/train_probe_flow.py','scripts/train_probe_flow_sparse.py',
                 'scripts/intake_probeflow_sources.py','scripts/audit_probe_flow.py','scripts/control_probe_flow.py',
                 'scripts/predict_probe_flow.py','scripts/audit_connectivity_intake.py','scripts/freeze_probe_flow.py'):
        files.add(ROOT/path)
    for directory in ('data/processed/probe-flow','data/raw/CI2N-probeflow-sample','data/raw/CircuitDigitization-connectivity',
                      'output/training/probe-flow','output/training/probe-flow-sparse','output/probeflow-data-intake'):
        files.update(p for p in (ROOT/directory).rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    hashes={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)}
    result={'status':'research only; synthetic feasibility passed; real independent accuracy not established',
            'frozen_on':'2026-10-07','runtime_changed':False,'independent_verified_designs_added':0,
            'sha256':hashes,'files':len(hashes),'raw_redistribution_authorized':False}
    output=ROOT/'output/probe-flow-research-manifest.json'
    if output.exists():raise ValueError('Research already frozen; create a separate experiment for changes')
    output.write_text(json.dumps(result,indent=2),encoding='utf-8')
    assert all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in hashes.items())
    print(json.dumps({'files':len(hashes),'mismatches':0,'manifest':str(output)}))

if __name__=='__main__':main()
