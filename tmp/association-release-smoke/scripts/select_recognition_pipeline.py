"""Select the reviewed recognition candidate only after actual-image checks."""
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from pcbparse.competition import export_competition
def load(path): return json.loads((ROOT/path).read_text(encoding='utf-8'))
verified={}; summaries={}
for domain in ('pcb55','gt10'):
    actual=load(f'output/benchmark-recognition-{domain}/summary.json')
    cached=load(f'output/recognition-candidate-{domain}/summary.json')
    assert actual['completed']==(55 if domain=='pcb55' else 10)
    for record in cached['records']:
        case=record['case']; expected=load(f'output/recognition-candidate-{domain}/{case}/prediction.json')
        if domain=='gt10': expected=export_competition(expected)
        assert load(f'output/benchmark-recognition-{domain}/{case}/prediction.json')==expected,(domain,case)
    assert actual['mean_metrics']['Score_Accuracy_SingleCase_proxy']>cached['means']['before']['Score_Accuracy_SingleCase_proxy']
    verified[domain]=actual['completed']; summaries[domain]={'before':cached['means']['before'],'after':actual['mean_metrics'],'improved':cached['improved'],'regressed':cached['regressed'],'same':cached['same'],'wall_seconds':actual['wall_seconds']}
config=load('models/pipeline.json'); old=dict(config)
if config.get('version')==4 and config.get('selected_run')=='output/benchmark-recognition-pcb55':
    old=load('output/ablation/recognition-pipeline.json')['previous_default']
config.update(version=4,pin_model='models/candidates/official-pins/pin-heatmap.onnx',pin_ensemble='none',semantic_library='models/semantics/component-library.json',component_box_mode='body',selected_run='output/benchmark-recognition-pcb55',comparison='output/ablation/recognition-pipeline.json',selection_notes='Actual-image 55-page and updated GT10 QA proxy both improve; observed passive-body output bounds preserve detector crop/mask extent; semantic no-op preserves cached associations; public development selection, not independent final test.')
config['scores']['benchmark-recognition-pcb55']=summaries['pcb55']['after']['Score_Accuracy_SingleCase_proxy']
result={'selected_default':config,'previous_default':old,'actual_predictions_matching_cache':verified,'domains':summaries,'official_equivalence':False,'independent_final_test':False,'limits':['PCB55 uses legacy strict/location-box evaluator; GT10 uses QA-informed proxy; scores not cross-domain comparable','PCB55 component F1 slightly regresses while pin/net/line improve','GT10 pin/net metrics remain low','public development sets used for model selection']}
(ROOT/'output/ablation/recognition-pipeline.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
(ROOT/'models/pipeline.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
print(json.dumps({'verified':verified,'domains':summaries}),flush=True)
