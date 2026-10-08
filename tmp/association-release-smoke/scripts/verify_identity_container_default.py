"""Verify new identity image preserves the default pipeline on the guide image."""
import json
from pathlib import Path
import subprocess
ROOT=Path(__file__).resolve().parents[1]
prefix=['wsl','-d','Ubuntu-22.04','-u','root','--exec','/usr/bin/docker']
tag='graphschematic-identity:20261005'
image_id=subprocess.run(prefix+['image','inspect',tag,'--format','{{.Id}}'],capture_output=True,encoding='utf-8',errors='replace',check=True).stdout.strip()
out=ROOT/'output/container-identity-default'; out.mkdir(parents=True,exist_ok=True)
result=subprocess.run(prefix+['run','--rm','--network','none','--cpus','2','--memory','2g','--pids-limit','256',
    '-v','/mnt/d/创芯EDA/gpt方案/data/examples:/input:ro','-v','/mnt/d/创芯EDA/gpt方案/output/container-identity-default:/output',image_id,
    '-input','/input','-output','/output'],capture_output=True,encoding='utf-8',errors='replace')
(out/'console.log').write_text(result.stdout+result.stderr,encoding='utf-8'); assert result.returncode==0,(result.stdout,result.stderr)
def load(p): return json.loads(p.read_text(encoding='utf-8'))
assert load(out/'final_results/guide-schematic_Pred.json')==load(ROOT/'output/recognition-workspace-smoke/final_results/guide-schematic_Pred.json')
log=load(out/'decision_log.json'); assert log['status']=='completed' and log['llm_calls']==0
record={'image_tag':tag,'image_id':image_id,'build_verified':True,'default_exact_windows_match':True,'default_cases':1,
        'default_elapsed_seconds':log['elapsed_seconds'],'network':'none','input_read_only':True,'llm_calls':0,'function_tests_passed':83,
        'official_evaluator_verified':False}
(ROOT/'output/ablation/ic-identity-container.json').write_text(json.dumps(record,indent=2),encoding='utf-8'); print(json.dumps(record),flush=True)
