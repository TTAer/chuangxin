"""First cross-domain stress run on 25 pre-reserved AMSNet source groups."""
from pathlib import Path
import hashlib
import sys
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from pcbparse.__main__ import load,save
from pcbparse.vision import ImageParser,iou
from pcbparse.competition import canonical_type,export_competition
from pcbparse.core import assignment,f1,validate

def truth_boxes(path):
    mapping={'resistor':'r','capacitor':'c','inductor':'l','gnd':'gnd','nmos':'mosfet','pmos':'mosfet','npn':'bjt','pnp':'bjt','diode':'d','opamp':'amp','voltage':'v','current':'other'}
    boxes=[]; excluded={}
    for cls,items in load(path).items():
        base=cls.rsplit('_',1)[0]
        if base not in mapping:
            if items: excluded[cls]=len(items)
            continue
        boxes.extend({'type':mapping[base],'bbox':b} for b in items)
    return boxes,excluded

def score(pred,truth,h,threshold):
    targets={t['type'] for t in truth}; predicted=[]
    for c in pred['components'].values():
        kind=canonical_type(c['type']); b=c['bbox']
        # Evaluate the declared shared label subset; all other predictions are
        # counted separately, rather than pretending absent annotations are GT.
        if kind in targets: predicted.append({'type':kind,'bbox':[b[0],h-b[3],b[2],h-b[1]]})
    weights=[[iou(p['bbox'],g['bbox']) if p['type']==g['type'] and iou(p['bbox'],g['bbox'])>=threshold else 0 for g in truth] for p in predicted]
    tp=sum(weights[i][j]>0 for i,j in assignment(weights))
    return {'tp':tp,'predictions':len(predicted),'truth':len(truth),'F1':f1(tp,len(predicted),len(truth)),'unscored_predictions':len(pred['components'])-len(predicted)}

def main():
    protocol=load(ROOT/'data/splits/rc0-evaluation.json'); pool=load(ROOT/'data/splits/amsnet-frozen.json')
    for name,sha in protocol['runtime_sha256'].items(): assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==sha,name
    out=ROOT/'output/freeze-b-amsnet-rc0'; ledger=out/'summary.json'
    if ledger.exists(): raise RuntimeError('first-pass result already exists; do not overwrite or tune on frozen set')
    parsers={'legacy':ImageParser(),'rc0':ImageParser(designator_mode='stable',identity_transaction_mode='catalog',ic_identity_mode='retrieval',side_pin_model='models/candidates/side-strip-local/side-strip.onnx',side_name_anchors=True)}
    rows=[]
    for group in pool['groups']:
        record=group['records'][0]; image=ROOT/record['image']; case=record['case']
        assert hashlib.sha256(image.read_bytes()).hexdigest()==record['image_sha256']
        predictions={}; observations={}; row={'case':case,'source_group':group['spice_text_group_sha256'],'profiles':{}}
        for mode,parser in parsers.items():
            try:
                pred,obs,diag,_=parser.parse(image); exported=export_competition(pred,obs['component_categories']); assert not validate(exported)
                predictions[mode]=pred; observations[mode]=obs
                save(out/mode/case/'prediction.json',pred); save(out/mode/case/'observations.json',obs)
                row['profiles'][mode]={'status':'completed','seconds':diag['seconds'],'valid_export':True}
            except Exception as exc: row['profiles'][mode]={'status':'failed','error_type':type(exc).__name__,'message':str(exc)[:200]}
        # Annotations are accessed only after both pixel-only predictions.
        truth,excluded=truth_boxes(image.with_name(case+'_bbox.json')); row['excluded_annotation_classes']=excluded
        for mode,pred in predictions.items():
            h=observations[mode]['image']['height']
            row['profiles'][mode]['bbox_iou_0_3']=score(pred,truth,h,.3); row['profiles'][mode]['bbox_iou_0_5']=score(pred,truth,h,.5)
        rows.append(row); save(out/'progress.json',{'rows':rows}); print(case,{k:v['status'] for k,v in row['profiles'].items()},flush=True)
    summary={'dataset':'Freeze-B AMSNet shared-class bbox stress','groups':len(rows),'independent_pcb_blind_test':False,'official_equivalence':False,'profiles':{},'rows':rows,'first_pass':True,'rc0_image_id':protocol['image_id']}
    for mode in parsers:
        good=[r['profiles'][mode] for r in rows if r['profiles'][mode]['status']=='completed']
        summary['profiles'][mode]={'completed':len(good),'failed':len(rows)-len(good),'mean_seconds':sum(r['seconds'] for r in good)/len(good) if good else None}
        for metric in ('bbox_iou_0_3','bbox_iou_0_5'):
            tp=sum(r[metric]['tp'] for r in good); np=sum(r[metric]['predictions'] for r in good); ng=sum(r[metric]['truth'] for r in good)
            summary['profiles'][mode][metric]={'micro_F1_completed_only':f1(tp,np,ng),'tp':tp,'predictions':np,'truth':ng}
    save(ledger,summary); print(summary['profiles'],flush=True)
if __name__=='__main__': main()
