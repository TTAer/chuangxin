"""Partial IC identity evaluator. No full-page or net scores are defined."""
from pcbparse.core import assignment
import numpy as np

BODY_IOU = .5
PIN_DISTANCE_PX = 3.0
SCOPE = 'visible IC identity within individually approved IC bodies only'

def iou(a,b):
    w=max(0,min(a[2],b[2])-max(a[0],b[0]))
    h=max(0,min(a[3],b[3])-max(a[1],b[1]))
    inter=w*h
    return inter/max(1e-12,(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter)

def evaluate(gt_components,pred_components):
    """Inputs use body_xyxy/key/name/pins; pins use point_xy/number/name.

    Extra predictions outside covered bodies are unscored, explicitly disclosed.
    Body misses count within the registered scope. No net metric is fabricated.
    """
    pairs=[]
    if gt_components and pred_components:
        quality=np.array([[iou(g['body_xyxy'],p['body_xyxy']) for p in pred_components] for g in gt_components])
        weights=np.where(quality>=BODY_IOU,1+quality,0)
        pairs=[(int(r),int(c)) for r,c in assignment(weights.tolist()) if quality[r,c]>=BODY_IOU]
    key_correct=model_correct=pin_pair_correct=pin_number_correct=0
    pin_gt=sum(len(g.get('pins',[])) for g in gt_components)
    for gi,pi in pairs:
        g,p=gt_components[gi],pred_components[pi]
        key_correct+=g['key']==p.get('key')
        model_correct+=g['name']==p.get('name')
        gp,pp=g.get('pins',[]),p.get('pins',[])
        if not gp or not pp:continue
        distance=np.array([[np.linalg.norm(np.array(a['point_xy'])-np.array(b['point_xy'])) for b in pp] for a in gp])
        weights=np.where(distance<=PIN_DISTANCE_PX,2-distance/PIN_DISTANCE_PX,0)
        for r,c in assignment(weights.tolist()):
            if distance[r,c]>PIN_DISTANCE_PX:continue
            number=gp[r]['number']==pp[c].get('number')
            pin_number_correct+=number
            pin_pair_correct+=number and gp[r]['name']==pp[c].get('name')
    return {'scope':SCOPE,'covered_ic_gt':len(gt_components),'matched_ic':len(pairs),
            'key_correct':key_correct,'model_correct':model_correct,
            'covered_pin_gt':pin_gt,'pin_number_correct':int(pin_number_correct),'pin_number_name_correct':int(pin_pair_correct),
            'out_of_scope_predictions_unscored':len(pred_components)-len(pairs),
            'full_page_component_recall':None,'net_score':None}
