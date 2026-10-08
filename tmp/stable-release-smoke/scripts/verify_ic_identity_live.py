"""Replay changed identity/side predictions from pixels plus unchanged controls."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from pcbparse.__main__ import load,save
from pcbparse.vision import ImageParser
from pcbparse.dataset import index_dataset
from pcbparse.competition import export_competition
parser=ImageParser(ic_identity_mode='retrieval',side_pin_model='models/candidates/side-strip-local/side-strip.onnx',side_name_anchors=True)
samples={s['id']:s for s in index_dataset(ROOT/'data/raw/PCBnet/Dataset300')}; rows=[]
for domain in ('gt10','pcb55'):
    root=ROOT/('output/ic-identity-'+domain+'-side')
    for folder in sorted(root.iterdir()):
        if not (folder/'prediction.json').exists(): continue
        expected=load(folder/'prediction.json')
        if domain=='pcb55' and expected==load(ROOT/'output/benchmark-recognition-pcb55'/folder.name/'prediction.json') and folder.name not in ('0005','0101'): continue
        image=ROOT/'data/raw/official/gt-0930/10GT'/folder.name/(folder.name+'.png') if domain=='gt10' else Path(samples[folder.name]['image'])
        prediction,obs,diag,_=parser.parse(image)
        if domain=='gt10': prediction=export_competition(prediction,obs.get('component_categories'))
        assert prediction==expected,(domain,folder.name)
        out=ROOT/'output/ic-identity-live'/domain/folder.name
        save(out/'prediction.json',prediction); save(out/'observations.json',obs)
        rows.append({'domain':domain,'case':folder.name,'seconds':diag['seconds'],'exact_match':True})
save(ROOT/'output/ablation/ic-identity-live.json',{'rows':rows,'actual_image_pages':len(rows),'official_equivalence':False})
print({'actual_pages':len(rows),'exact_match':True},flush=True)
