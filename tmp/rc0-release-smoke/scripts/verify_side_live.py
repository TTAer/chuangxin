"""Full image replay of affected PCB pages and controls, plus GT10 comparison."""
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from pcbparse.vision import ImageParser
from pcbparse.dataset import index_dataset
from pcbparse.__main__ import load,save
samples={s['id']:s for s in index_dataset(ROOT/'data/raw/PCBnet/Dataset300')}
parser=ImageParser(side_pin_model='models/candidates/side-strip-local/side-strip.onnx'); records=[]
cache=ROOT/'output/side-strip-side-strip-local-pcb55'; base=ROOT/'output/benchmark-recognition-pcb55'
for folder in sorted(cache.iterdir()):
    if not (folder/'prediction.json').exists(): continue
    expected=load(folder/'prediction.json')
    if expected==load(base/folder.name/'prediction.json') and folder.name not in ('0005','0076','0101'): continue
    prediction,obs,diag,_=parser.parse(samples[folder.name]['image'])
    assert prediction==expected,folder.name
    target=ROOT/'output/benchmark-side-strip-local-pcb55'/folder.name
    save(target/'prediction.json',prediction); save(target/'observations.json',obs); save(target/'diagnostics.json',diag)
    records.append({'case':folder.name,'seconds':diag['seconds'],'exact_match':True})
for number in range(1,11):
    assert load(ROOT/'output/side-strip-side-strip-local-gt10'/str(number)/'prediction.json')==load(ROOT/'output/benchmark-side-strip-local-gt10'/str(number)/'prediction.json')
report={'candidate':'side-strip-local','actual_gt10_pages':10,'actual_pcb55_reruns':len(records),'records':records,
        'unchanged_pages_reuse_prior_actual_results':True,'official_equivalence':False}
save(ROOT/'output/side-strip-verification/live.json',report); print(json.dumps(report),flush=True)
