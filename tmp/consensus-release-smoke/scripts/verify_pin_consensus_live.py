"""Cross-style image inference compared with cached fusion; no labels in parser."""
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from pcbparse.vision import ImageParser
from pcbparse.__main__ import load,save
from pcbparse.core import evaluate

parser=ImageParser(text_constraints=True,graph_mode='full',pin_ensemble='models/candidates/official-pins/pin-heatmap.onnx',semantic_library='none')
records=[]; out=ROOT/'output/pin-consensus-live-check'
for case in ['0005','0076','0101']:
    base=ROOT/'output/benchmark-graph-topology-v2-55'/case
    info=load(base/'diagnostics.json')
    prediction,observations,diagnostics,texts=parser.parse(info['input'])
    cached=load(ROOT/'output/benchmark-pin-consensus-55'/case/'prediction.json')
    metrics=evaluate(prediction,load(base/'golden.json'))
    record={'case':case,'colored':diagnostics['colored'],'seconds':diagnostics['seconds'],'prediction_matches_cached_fusion':prediction==cached,'metrics':metrics}
    save(out/case/'prediction.json',prediction); save(out/case/'diagnostics.json',diagnostics); records.append(record)
    print(json.dumps(record),flush=True)
save(out/'summary.json',{'records':records,'passed':all(r['prediction_matches_cached_fusion'] for r in records),'official_equivalence':False})
