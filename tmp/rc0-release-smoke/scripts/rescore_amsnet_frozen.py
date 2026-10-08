"""Correct dataset schema adapter without rerunning or tuning frozen inference."""
from pathlib import Path
import hashlib
import sys
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from pcbparse.__main__ import load,save
from pcbparse.core import f1
from evaluation_metrics import annotation_boxes,bbox_score,CLASS_MAP
raw=load(ROOT/'output/freeze-b-amsnet-rc0/summary.json'); rows=[]
for old in raw['rows']:
    case=old['case']; base=ROOT/'data/raw/AMSNet-1.0/amsnet_1.0'/case
    truth,excluded=annotation_boxes(load(base/(case+'_bbox.json')))
    row={'case':case,'source_group':old['source_group'],'profiles':{},'excluded_annotation_classes':excluded}
    for mode,status in old['profiles'].items():
        if status['status']!='completed': row['profiles'][mode]=status; continue
        folder=ROOT/'output/freeze-b-amsnet-rc0'/mode/case
        pred=load(folder/'prediction.json'); obs=load(folder/'observations.json')
        row['profiles'][mode]={'status':'completed','seconds':status['seconds'],'valid_export':True,'bbox_iou_0_3':bbox_score(pred,truth,obs['image']['height'],.3),'bbox_iou_0_5':bbox_score(pred,truth,obs['image']['height'],.5)}
    rows.append(row)
summary=dict(raw,rows=rows,profiles={},metric_adapter_version=3,class_map=CLASS_MAP,
    annotation_adapter_sha256=hashlib.sha256((ROOT/'scripts/evaluation_metrics.py').read_bytes()).hexdigest(),
    schema_correction='v1 omitted mirrored transistor/vdd GT and absent-GT shared-class false positives; v1/v2 treated AMSNet yxyx as xyxy; convert yxyx to xyxy, frozen pixel predictions unchanged',inference_rerun=False)
for mode in ('legacy','rc0'):
    good=[r['profiles'][mode] for r in rows if r['profiles'][mode]['status']=='completed']
    summary['profiles'][mode]={'completed':len(good),'failed':len(rows)-len(good),'mean_seconds':sum(r['seconds'] for r in good)/len(good) if good else None}
    for metric in ('bbox_iou_0_3','bbox_iou_0_5'):
        tp=sum(r[metric]['tp'] for r in good); np=sum(r[metric]['predictions'] for r in good); ng=sum(r[metric]['truth'] for r in good)
        summary['profiles'][mode][metric]={'micro_F1_completed_only':f1(tp,np,ng),'tp':tp,'predictions':np,'truth':ng}
save(ROOT/'output/freeze-b-amsnet-rc0/summary-adapter-v3.json',summary)
print(summary['profiles'],flush=True)
