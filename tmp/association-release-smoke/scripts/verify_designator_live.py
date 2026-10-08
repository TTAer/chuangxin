"""Fresh OCR replay of all changed guarded predictions and fixed controls."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from pcbparse.__main__ import load,save
from pcbparse.vision import ImageParser
from pcbparse.dataset import index_dataset
from pcbparse.competition import export_competition

def main():
    parser=ImageParser(designator_mode='guarded',ic_identity_mode='retrieval',side_pin_model='models/candidates/side-strip-local/side-strip.onnx',side_name_anchors=True)
    samples={s['id']:s for s in index_dataset(ROOT/'data/raw/PCBnet/Dataset300')}; rows=[]
    for domain in ('gt10','pcb55'):
        summary=load(ROOT/'output/designator-guarded'/domain/'summary.json')
        for record in summary['records']:
            case=record['case']
            if not record['prediction_changed'] and not (domain=='gt10' and case=='1') and not (domain=='pcb55' and case=='0005'): continue
            path=ROOT/'data/raw/official/gt-0930/10GT'/case/(case+'.png') if domain=='gt10' else Path(samples[case]['image'])
            pred,obs,diag,_=parser.parse(path)
            if domain=='gt10': pred=export_competition(pred,obs.get('component_categories'))
            assert pred==load(ROOT/'output/designator-guarded'/domain/case/'prediction.json'),(domain,case)
            out=ROOT/'output/designator-live'/domain/case; save(out/'prediction.json',pred); save(out/'observations.json',obs)
            rows.append({'domain':domain,'case':case,'exact_match':True,'seconds':diag['seconds']})
            print(domain,case,'exact',flush=True)
    save(ROOT/'output/ablation/designator-live.json',{'rows':rows,'actual_fresh_ocr_pages':len(rows),'all_changed_pages_replayed':True,'official_equivalence':False})
if __name__=='__main__': main()
