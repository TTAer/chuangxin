import hashlib
import json
from pathlib import Path
import subprocess
import sys
import zipfile

root = Path(__file__).resolve().parents[1]
archive = root / 'submission/pcbparse-development.zip'
destination = root / 'tmp/consensus-release-smoke'
with zipfile.ZipFile(archive) as z:
    manifest = json.loads(z.read('BUILD_MANIFEST.json'))
    for name, record in manifest['files'].items():
        data = z.read(name)
        assert len(data) == record['bytes']
        assert hashlib.sha256(data).hexdigest() == record['sha256'], name
        target = (destination / name).resolve()
        assert target.is_relative_to(destination.resolve()), name
    z.extractall(destination)
print(f"Verified {len(manifest['files'])} file hashes", flush=True)
result = subprocess.run([sys.executable, 'agent_system', '-input', str(root / 'data/examples'), '-output', str(root / 'output/consensus-packaged-agent-smoke')], cwd=destination, capture_output=True, text=True, encoding='utf-8', errors='replace')
print(result.stdout)
if result.returncode:
    print(result.stderr)
    raise SystemExit(result.returncode)
case = 'final_results/guide-schematic_Pred.json'
load = lambda p: json.loads(p.read_text(encoding='utf-8'))
assert load(root / 'output/consensus-default-agent-smoke' / case) == load(root / 'output/consensus-packaged-agent-smoke' / case)
log = load(root / 'output/consensus-packaged-agent-smoke/decision_log.json')
assert log['status'] == 'completed' and len(log['cases']) == 1
diagnostics = load(root / 'output/consensus-packaged-agent-smoke/runs/local/_diagnostics/guide-schematic.json')
assert diagnostics['pin_consensus']['module'] == 'TopologyPreservingPinConsensus'
print('Packaged default inference matches workspace output exactly; consensus enabled.')
summary = {'archive_sha256': hashlib.sha256(archive.read_bytes()).hexdigest(), 'verified_files': len(manifest['files']), 'packaged_prediction_matches_workspace': True, 'default_consensus_enabled': True, 'official_evaluator_verified': False, 'docker_build_verified': False}
(root / 'output/consensus-release-verification.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
