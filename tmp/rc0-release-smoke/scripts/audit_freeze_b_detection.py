"""Fixed detector-vs-emission funnel diagnostic; no parameter adaptation."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from pcbparse.__main__ import load,save
from pcbparse.detector import OnnxDetector
from pcbparse.dataset import CATEGORIES
from pcbparse.vision import read_image
from evaluation_metrics import annotation_boxes,bbox_score
from pcbparse.core import f1
detector=OnnxDetector(ROOT/'models/detector/pcb-yolo11n.onnx'); pool=load(ROOT/'data/splits/amsnet-frozen.json'); rows=[]
for group in pool['groups']:
    r=group['records'][0]; p=ROOT/r['image']; image=read_image(p); h=image.shape[0]; detections=detector(image)
    pred={'components':{str(i):{'type':CATEGORIES.get(d['category'],d['category'].lower()),'bbox':[d['bbox'][0],h-d['bbox'][3],d['bbox'][2],h-d['bbox'][1]]} for i,d in enumerate(detections)}}
    truth,_=annotation_boxes(load(p.with_name(p.stem+'_bbox.json')))
    rows.append({'case':r['case'],'raw_bbox_iou_0_3':bbox_score(pred,truth,h,.3),'raw_bbox_iou_0_5':bbox_score(pred,truth,h,.5)})
summary={}
for metric in ('raw_bbox_iou_0_3','raw_bbox_iou_0_5'):
    tp=sum(r[metric]['tp'] for r in rows); np=sum(r[metric]['predictions'] for r in rows); ng=sum(r[metric]['truth'] for r in rows)
    summary[metric]={'tp':tp,'predictions':np,'truth':ng,'micro_F1':f1(tp,np,ng)}
save(ROOT/'output/ablation/rc0-freeze-b-detection.json',{'rows':rows,'summary':summary,'post_first_pass_diagnostic_only':True,'no_threshold_changes':True})
print(summary,flush=True)
