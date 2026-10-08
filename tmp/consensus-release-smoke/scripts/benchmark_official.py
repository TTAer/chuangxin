"""Image-only benchmark of updated GT; no truth read before inference."""
import argparse
import json
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from pcbparse.vision import ImageParser
from pcbparse.competition import export_competition,evaluate_qa
from pcbparse.__main__ import save

p=argparse.ArgumentParser(); p.add_argument('--data',default='data/raw/official/gt-0930/10GT'); p.add_argument('--output',default='output/benchmark-official-baseline'); p.add_argument('--semantic-library',default='none'); p.add_argument('--pin-model',default='auto'); p.add_argument('--pin-ensemble',default='none'); a=p.parse_args()
parser=ImageParser(semantic_library=a.semantic_library,pin_model=a.pin_model,text_constraints=True,pin_ensemble=a.pin_ensemble); records=[]; out=Path(a.output); started=time.perf_counter()
for image in sorted(Path(a.data).rglob('*.png'),key=lambda x:int(x.stem)):
    tic=time.perf_counter(); case=out/image.stem
    result,obs,diag,texts=parser.parse(image)
    save(case/'observations.json',obs); save(case/'diagnostics.json',diag)
    if parser.last_graph_report:
        save(case/'dual-graph.json',parser.last_graph_report)
    result=export_competition(result,obs.get('component_categories')); save(case/'prediction.json',result)
    # Golden answers are loaded only after image inference finishes.
    truth=json.loads(image.with_suffix('.json').read_text(encoding='utf-8'))
    metrics=evaluate_qa(result,truth); save(case/'metrics.json',metrics)
    records.append(dict(case=image.stem,seconds=time.perf_counter()-tic,**metrics))
    print(json.dumps(records[-1]),flush=True)
keys=['ComponentF1','PinF1','NetHypergraphF1','NetLineF1','Score_Accuracy_SingleCase_proxy']
summary={'completed':len(records),'mean_metrics':{k:sum(r[k] for r in records)/len(records) for k in keys},'wall_seconds':time.perf_counter()-started,'records':records,'official_equivalence':False,'golden_version':'10GT-0930','data_overlap_audit':'output/official-audit/inventory.json','test_status':'public competition GT, not private final test','pin_model':a.pin_model,'semantic_library':a.semantic_library,'pin_ensemble':a.pin_ensemble}
save(out/'summary.json',summary); print(json.dumps(summary['mean_metrics']))
