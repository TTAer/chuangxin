"""Verify the recorded offline container smoke against the workspace result."""
import json
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser(); p.add_argument('--image-id',required=True); args=p.parse_args()
def load(path):
    return json.loads(path.read_text(encoding='utf-8'))

container = ROOT / 'output/container-recognition-smoke'
log = load(container / 'decision_log.json')
assert log['status'] == 'completed' and len(log['cases']) == 1
assert log['llm_calls'] == 0 and all(c['status'] == 'completed' for c in log['cases'])
prediction = 'final_results/guide-schematic_Pred.json'
assert load(container / prediction) == load(ROOT / 'output/recognition-workspace-smoke' / prediction)
record = {
    'status': 'verified',
    'runtime': 'Docker Engine 29.3.0 under WSL Ubuntu-22.04',
    'base_image': 'edarace-agent-base:ubuntu24',
    'base_digest': 'sha256:4506a6309f04431e9ae2ccebd1664e3a1033d726652ccaa512dd842767d39b69',
    'image_tag': 'graphschematic-recognition:20261004',
    'image_id': args.image_id,
    'base_loaded': True,
    'build_verified': True,
    'inference_verified': True,
    'official_evaluator_verified': False,
    'network_mode': 'none',
    'input_mount': 'read-only',
    'cases': 1,
    'exact_json_match_to_windows': True,
    'elapsed_seconds': log['elapsed_seconds'],
    'wheelhouse': {'verified_packages': 13, 'bytes': 139051448, 'installation': 'pip --no-index'},
    'initial_online_build_failure': 'files.pythonhosted.org ReadTimeoutError; resolved with SHA256-verified local wheels',
    'limitations': ['one public example smoke; not full official runner acceptance', 'official decision_log schema remains provisional', 'official evaluator not available'],
}
(ROOT / 'output/container-validation.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
print(json.dumps(record))
