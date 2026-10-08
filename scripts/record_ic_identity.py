"""Record tested optional identity retrieval, coverage and container evidence."""
import hashlib
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def load(name): return json.loads((ROOT/name).read_text(encoding='utf-8'))
record={'date':'2026-10-05','default_promoted':False,'training_performed':False,'function_tests_passed':83,
        'official_equivalence':False,'independent_test':False,'domains':{},
        'identity_audit':load('output/ablation/ic-identity-audit.json'),'actual_image_replay':load('output/ablation/ic-identity-live.json'),
        'default_container':load('output/ablation/ic-identity-container.json'),
        'optional_container':load('output/container-identity-20261005-batch-acceptance.json'),
        'image_sources':load('output/ablation/ic-identity-image-sources.json'),
        'limitations':['gain in identity recovery concentrated in PCB0006','three recovered identities, emitted PCB identity coverage only27.3%',
                      'catalog has no verified image-side sequence; numerical order cannot supply it',
                      'many missing real models cannot be resolved by fuzzy retrieval','GT10 Pin/Net targets unmet','proxy evaluator not official']}
for domain in ('gt10','pcb55'):
    record['domains'][domain]={}
    for mode in ('identity','identity_side'):
        s=load('output/ic-identity-'+domain+('-side' if mode=='identity_side' else '')+'/summary.json')
        record['domains'][domain][mode]={k:s[k] for k in ('mean_metrics','baseline','completed','failures','improved','regressed','records','identity_records')}
record['source_sha256']={str(p.relative_to(ROOT)).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'pcbparse/ic_identity.py',ROOT/'pcbparse/vision.py',ROOT/'pcbparse/batch.py',ROOT/'pcbparse/agent.py',ROOT/'pcbparse/__main__.py')}
(ROOT/'output/ablation/ic-identity.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
print(json.dumps({'default_promoted':False,'function_tests':83,'pcb55_score':record['domains']['pcb55']['identity_side']['mean_metrics']['Score_Accuracy_SingleCase_proxy']}),flush=True)
