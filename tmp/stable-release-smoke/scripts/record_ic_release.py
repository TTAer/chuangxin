"""Record tested candidate policy and current Docker acceptance provenance."""
import hashlib
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def load(path): return json.loads((ROOT/path).read_text(encoding='utf-8'))
def save(path,data):
    p=ROOT/path; p.parent.mkdir(parents=True,exist_ok=True); p.write_text(json.dumps(data,indent=2),encoding='utf-8')
image_id=load('output/container-validation.json')['image_id']
runtime=load('output/container-batch-acceptance.json')
for test in runtime['tests']:
    if test.get('image_id')!=image_id:
        raise ValueError('Historical batch image differs from single-case provenance; do not relabel recorded runs')
    console=(ROOT/('output/container-batch-'+str(test['cases']))/'docker-console.log').read_text(encoding='utf-8',errors='replace')
    test['swap_limit_supported']=False if 'does not support swap limit' in console else None
if runtime.get('image_id')!=image_id:
    raise ValueError('Historical batch image provenance must be preserved')
save('output/container-batch-acceptance.json',runtime)
candidate=load('output/container-ic-structure/final_results/1_Pred.json')
assert candidate==load('output/benchmark-ic-structure-gt10/1/prediction.json')
log=load('output/container-ic-structure/decision_log.json')
assert log['status']=='completed' and log['llm_calls']==0
domains={d:load('output/ic-structure-'+d+'-geometry/summary.json') for d in ('gt10','pcb55')}
record={'default_ic_pin_mode':'none','candidate_mode':'structure','candidate_status':'optional; not promoted',
        'training_performed':False,'official_equivalence':False,'independent_test':False,'function_tests_passed':72,
        'selection_reason':'GT10 gain concentrated in one case; PCB55 mean gain below suggested 0.2-point gate with two regressions',
        'domains':{d:{k:s[k] for k in ('completed','failures','mean_metrics','baseline','improved','regressed')} for d,s in domains.items()},
        'live_validation':load('output/ic-structure-live-verification.json'),
        'candidate_container':{'image_id':image_id,'cases':1,'status':'completed','exact_windows_match':True,'network':'none','llm_calls':0,'seconds':log['elapsed_seconds']},
        'default_container_batch_tests':[{k:v for k,v in t.items() if k!='memory_samples'} for t in runtime['tests']],
        'source_sha256':{str(p.relative_to(ROOT)).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'pcbparse/ic_structure.py',ROOT/'pcbparse/vision.py',ROOT/'models/pipeline.json',ROOT/'Dockerfile.competition')}}
save('output/ablation/ic-structure.json',record)
print(json.dumps({'candidate_status':record['candidate_status'],'candidate_container':record['candidate_container'],'batch_cases':[t['cases'] for t in runtime['tests']]}),flush=True)
