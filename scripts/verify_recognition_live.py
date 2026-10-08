"""Check auto selectors and rerun only outputs affected by a code correction."""
import json
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from pcbparse.__main__ import load,save
from pcbparse.competition import export_competition,evaluate_qa
from pcbparse.core import evaluate
from pcbparse.dataset import index_dataset
from pcbparse.vision import ImageParser
samples={s['id']:s for s in index_dataset(ROOT/'data/raw/PCBnet/Dataset300')}
parser=ImageParser(); reruns=[]; checked={}
for domain in ('pcb55','gt10'):
    base=ROOT/f'output/benchmark-recognition-{domain}'; cached=load(ROOT/f'output/recognition-candidate-{domain}/summary.json'); summary=load(base/'summary.json')
    for row in cached['records']:
        case=row['case']; expected=load(ROOT/f'output/recognition-candidate-{domain}/{case}/prediction.json')
        if domain=='gt10': expected=export_competition(expected)
        current=load(base/case/'prediction.json')
        if current!=expected or domain=='pcb55' and case in ('0005','0076','0101'):
            image=ROOT/f'data/raw/official/gt-0930/10GT/{case}/{case}.png' if domain=='gt10' else samples[case]['image']
            prediction,obs,diag,texts=parser.parse(image)
            if domain=='gt10': prediction=export_competition(prediction,obs.get('component_categories'))
            assert prediction==expected,(domain,case)
            save(base/case/'prediction.json',prediction); save(base/case/'observations.json',obs); save(base/case/'diagnostics.json',diag)
            save(base/case/('dual-graph.json' if domain=='gt10' else 'graph.json'),parser.last_graph_report)
            truth=load(ROOT/f'data/raw/official/gt-0930/10GT/{case}/{case}.json') if domain=='gt10' else load(base/case/'golden.json')
            metrics=evaluate_qa(prediction,truth) if domain=='gt10' else evaluate(prediction,truth)
            entry=next(r for r in summary['records'] if r.get('case',r.get('id'))==case); entry.update(metrics,seconds=diag['seconds'])
            reruns.append({'domain':domain,'case':case,'seconds':diag['seconds']}); print(json.dumps(reruns[-1]),flush=True)
    keys=['ComponentF1','PinF1','NetHypergraphF1','NetLineF1','Score_Accuracy_SingleCase_proxy']
    summary['mean_metrics']={k:sum(r[k] for r in summary['records'])/len(summary['records']) for k in keys}
    summary['component_box_mode']='body'; summary['pin_model']='models/candidates/official-pins/pin-heatmap.onnx'; summary['pin_ensemble']='none'; summary['semantic_library']='models/semantics/component-library.json'
    save(base/'summary.json',summary); checked[domain]=len(cached['records'])
save(ROOT/'output/recognition-live-verification.json',{'actual_predictions_matching_cache':checked,'auto_selector_reruns':reruns,'official_equivalence':False}); print(json.dumps({'checked':checked,'auto_selector_reruns':len(reruns)}),flush=True)
