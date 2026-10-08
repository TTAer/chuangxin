"""Unified scorecard: development, stress, deployment and blind-test gates."""
import hashlib
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from pcbparse.__main__ import load,save
protocol=load(ROOT/'data/splits/rc0-evaluation.json')
for name,sha in protocol['runtime_sha256'].items(): assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==sha,name
development={}
for domain in ('gt10','pcb55'):
    s=load(ROOT/'output/designator-stable-catalog'/domain/'summary.json'); keys=list(s['mean'])
    default={k:sum(r['default'][k] for r in s['records'])/len(s['records']) for k in keys}
    score='Score_Accuracy_SingleCase_proxy'
    development[domain]={'legacy_default':default,'previous_combo':s['baseline'],'rc0':s['mean'],'pages':s['pages'],
        'rc0_improved_vs_default':sum(r['metrics'][score]>r['default'][score]+1e-9 for r in s['records']),
        'rc0_regressed_vs_default':sum(r['metrics'][score]<r['default'][score]-1e-9 for r in s['records']),
        'rc0_improved_vs_previous_combo':s['improved'],'rc0_regressed_vs_previous_combo':s['regressed'],'independent':False}
runtime={}
for name,prefix in [('legacy_default','rc0-default'),('rc0','rc0-candidate')]:
    record=load(ROOT/'output'/(prefix+'-batch-acceptance.json')); assert record['image_id']==protocol['image_id']
    assert all(r['status']=='passed' and r['cases']==50 for r in record['tests'])
    runtime[name]=record['tests'][0]
    runtime[name]['latency_environment']='two profiles ran concurrently, also overlap with Windows cross-domain inference; not isolated comparative latency'
stress=load(ROOT/'output/freeze-b-amsnet-rc0/summary-adapter-v3.json')
membership=load(ROOT/'output/ablation/rc0-membership-audit.json')
miss=load(ROOT/'output/ablation/rc0-ocr-misses.json')
card={'version':'RC0-stable-catalog','image_id':protocol['image_id'],'runtime_unchanged':True,'function_tests_passed':103,
 'development':development,'freeze_a':{'status':'not_ready','pages':0,'score':None,'reason':'no independently proven PCB designs and complete component/key/name labels acquired'},
 'freeze_b':{'status':'first_inference_completed','groups':stress['groups'],'metric_adapter_version':3,'profiles':stress['profiles'],'scope':'shared-class component bbox; not PCB/Pin/Net benchmark','design_independence_proven':False},
 'deployment':runtime,'official_evaluator':{'verified':False,'score':None},
 'membership':{'audited_pages':len(membership['rows']),'transactions':sum(len(r['nets']) for r in membership['rows']),'merged_previous_nets':sum(r['merged_previous_nets'] for r in membership['rows']),'diagnostic_only':True},
 'ocr_missing_designators':{'cases':miss['samples'],'local_exact_recovery':sum(r['local_ocr_exact_key'] for r in miss['rows']),'unresolved':sum(not r['local_ocr_exact_key'] for r in miss['rows']),'runtime_changed':False},
 'promotion_decision':'NO_GO_pending_Freeze_A','default_promoted':False,'new_architecture_training_started':False,
 'limitations':['v1/v2 stress scores invalid due schema adaptation, preserved only for audit','memory is sampled docker working set, not exact peak RSS','50-case repeats test runtime, not 50 independent designs','cross-domain equality does not establish same-domain blind generalization']}
save(ROOT/'output/rc0-scorecard.json',card)
print({'image':card['image_id'],'promotion':card['promotion_decision'],'default_passed':50,'rc0_passed':50,'stress_groups':25},flush=True)
