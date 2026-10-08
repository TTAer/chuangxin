"""Fresh OCR replay of all changed guarded predictions and fixed controls."""
import argparse
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from pcbparse.__main__ import load,save
from pcbparse.vision import ImageParser
from pcbparse.dataset import index_dataset
from pcbparse.competition import export_competition

def main():
    p=argparse.ArgumentParser(); p.add_argument('--mode',default='guarded'); p.add_argument('--transaction-mode',default='none'); p.add_argument('--record-prefix',default='designator'); a=p.parse_args()
    folder_name='designator-'+a.mode+('-catalog' if a.transaction_mode=='catalog' else '-transaction' if a.transaction_mode=='evidence' else '')
    parser=ImageParser(designator_mode=a.mode,identity_transaction_mode=a.transaction_mode,ic_identity_mode='retrieval',side_pin_model='models/candidates/side-strip-local/side-strip.onnx',side_name_anchors=True)
    samples={s['id']:s for s in index_dataset(ROOT/'data/raw/PCBnet/Dataset300')}; rows=[]
    for domain in ('gt10','pcb55'):
        summary=load(ROOT/'output'/folder_name/domain/'summary.json')
        for record in summary['records']:
            case=record['case']
            if not record['prediction_changed'] and not (domain=='gt10' and case=='1') and not (domain=='pcb55' and case=='0005'): continue
            path=ROOT/'data/raw/official/gt-0930/10GT'/case/(case+'.png') if domain=='gt10' else Path(samples[case]['image'])
            pred,obs,diag,_=parser.parse(path)
            if domain=='gt10': pred=export_competition(pred,obs.get('component_categories'))
            assert pred==load(ROOT/'output'/folder_name/domain/case/'prediction.json'),(domain,case)
            out=ROOT/'output'/(a.record_prefix+'-live')/domain/case; save(out/'prediction.json',pred); save(out/'observations.json',obs)
            rows.append({'domain':domain,'case':case,'exact_match':True,'seconds':diag['seconds']})
            print(domain,case,'exact',flush=True)
    save(ROOT/'output/ablation'/(a.record_prefix+'-live.json'),{'rows':rows,'actual_fresh_ocr_pages':len(rows),'all_changed_pages_replayed':True,'official_equivalence':False})
if __name__=='__main__': main()
