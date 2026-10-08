"""Record funnel, threshold, paired-box and incremental-name development evidence."""
import hashlib
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def load(name): return json.loads((ROOT/name).read_text(encoding='utf-8'))
def compact(name):
    s=load(name)
    return {k:s[k] for k in ('mean_metrics','baseline','completed','failures','improved','regressed','threshold','funnel_totals','records')}
record={'date':'2026-10-05','default_promoted':False,'official_equivalence':False,'independent_test':False,'function_tests_passed':79,
        'threshold_sweep':load('output/ablation/side-thresholds.json'),'paired_cross_crops':load('output/ablation/side-paired-cross-crops.json'),
        'paired_models':{},'incremental_name':{},'funnel_baseline':{},'model_exports':load('output/ablation/side-paired-exports.json'),
        'incremental_actual_image_replay':load('output/ablation/side-incremental-live.json'),
        'paired_actual_image_replay':load('output/ablation/side-paired-live.json'),
        'current_container_optional':load('output/ablation/side-current-container.json'),
        'limitations':['one training seed; unequal crop counts/steps','matched detector subset, not additional independent designs',
                      'normal-band targets beyond 30 pixels omitted and counted','endpoint counts are not endpoint recall',
                      'threshold and gates inspected on public development sets; no independence claim',
                      'structure parser may edit existing identities even at zero learned proposals']}
for model in ('oracle','detector'):
    root=f'models/candidates/side-strip-{model}-paired'
    record['paired_models'][model]={'metadata':load(root+'/side-strip.json'),
                                  'domains':{d:compact(f'output/side-paired-{model}-{d}/summary.json') for d in ('gt10','pcb55')}}
for d in ('gt10','pcb55'):
    s=load(f'output/side-threshold-sweep/local-0.5-{d}/summary.json')
    record['funnel_baseline'][d]={'totals':s['funnel_totals'],'components':s['funnel']}
    record['incremental_name'][d]=compact(f'output/side-name-incremental-{d}/summary.json')
record['source_sha256']={str(p.relative_to(ROOT)).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in (ROOT/'pcbparse/side_strips.py',ROOT/'pcbparse/vision.py',ROOT/'pcbparse/batch.py',ROOT/'pcbparse/agent.py',ROOT/'pcbparse/__main__.py',ROOT/'Dockerfile.competition')}
(ROOT/'output/ablation/side-refinement.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
print(json.dumps({'default_promoted':False,'tests':79,'incremental_scores':{d:record['incremental_name'][d]['mean_metrics']['Score_Accuracy_SingleCase_proxy'] for d in record['incremental_name']}}),flush=True)
