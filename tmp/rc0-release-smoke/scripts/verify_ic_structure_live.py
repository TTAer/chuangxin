"""Run affected PCB pages plus controls; compare actual output to cached ablation."""
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from pcbparse.vision import ImageParser
from pcbparse.__main__ import load,save
from pcbparse.dataset import index_dataset
parser=ImageParser(ic_pin_mode='structure'); samples={s['id']:s for s in index_dataset(ROOT/'data/raw/PCBnet/Dataset300')}
out=ROOT/'output/benchmark-ic-structure-pcb55'; records=[]
for folder in sorted((ROOT/'output/ic-structure-pcb55-geometry').iterdir()):
    if not (folder/'report.json').exists(): continue
    if not load(folder/'report.json')['edits'] and folder.name not in ('0005','0076','0101'): continue
    pred,obs,diag,_=parser.parse(samples[folder.name]['image'])
    assert pred==load(folder/'prediction.json'),folder.name
    save(out/folder.name/'prediction.json',pred); save(out/folder.name/'observations.json',obs); save(out/folder.name/'diagnostics.json',diag)
    records.append({'case':folder.name,'seconds':diag['seconds'],'exact_cache_match':True})
official=ROOT/'output/benchmark-ic-structure-gt10'; matched=0
for folder in sorted((ROOT/'output/ic-structure-gt10-geometry').iterdir()):
    if not (folder/'prediction.json').exists(): continue
    assert load(folder/'prediction.json')==load(official/folder.name/'prediction.json'),folder.name
    matched+=1
summary={'gt10_actual_inference_matched':matched,'pcb55_actual_reruns':len(records),'records':records,'unchanged_pages_use_prior_actual_inference':True,'official_equivalence':False}
save(ROOT/'output/ic-structure-live-verification.json',summary); print(json.dumps(summary),flush=True)
