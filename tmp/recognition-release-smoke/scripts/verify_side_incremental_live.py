"""Verify optional incremental name recovery against cached proposals."""
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from pcbparse.vision import ImageParser
from pcbparse.dataset import index_dataset
from pcbparse.competition import export_competition
from pcbparse.__main__ import load,save
parser=ImageParser(side_pin_model='models/candidates/side-strip-local/side-strip.onnx',side_name_anchors=True)
samples={s['id']:s for s in index_dataset(ROOT/'data/raw/PCBnet/Dataset300')}; rows=[]
for domain in ('gt10','pcb55'):
    cached=ROOT/('output/side-name-incremental-'+domain)
    for folder in sorted(cached.iterdir()):
        if not (folder/'prediction.json').exists(): continue
        expected=load(folder/'prediction.json')
        if domain=='pcb55' and expected==load(ROOT/'output/benchmark-recognition-pcb55'/folder.name/'prediction.json') and folder.name not in ('0005','0076','0101'): continue
        image=ROOT/'data/raw/official/gt-0930/10GT'/folder.name/(folder.name+'.png') if domain=='gt10' else Path(samples[folder.name]['image'])
        actual,obs,diag,_=parser.parse(image)
        if domain=='gt10': actual=export_competition(actual,obs.get('component_categories'))
        assert actual==expected,(domain,folder.name)
        target=ROOT/'output/side-incremental-live'/domain/folder.name
        save(target/'prediction.json',actual); save(target/'diagnostics.json',diag)
        rows.append({'domain':domain,'case':folder.name,'exact_match':True,'seconds':diag['seconds']})
save(ROOT/'output/ablation/side-incremental-live.json',{'rows':rows,'official_equivalence':False}); print(json.dumps({'pages':len(rows),'exact_matches':len(rows)}),flush=True)
