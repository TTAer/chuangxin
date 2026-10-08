"""Fixed partial-annotation bbox metric, independent of recognition runtime."""
from pcbparse.core import assignment,f1
from pcbparse.competition import canonical_type
from pcbparse.vision import iou

CLASS_MAP={'resistor':'r','capacitor':'c','inductor':'l','gnd':'gnd','nmos':'mosfet','pmos':'mosfet','npn':'bjt','pnp':'bjt','diode':'d','opamp':'amp','voltage':'v','vdd':'v'}
ELIGIBLE=set(CLASS_MAP.values())

def annotation_boxes(raw):
    boxes=[]; excluded={}
    for cls,items in raw.items():
        family=cls.rsplit('_',1)[0].replace('-mirror','').replace('-cross','')
        if family not in CLASS_MAP:
            if items: excluded[cls]=len(items)
            continue
        # AMSNet stores [y_min, x_min, y_max, x_max]. Convert before comparing
        # with image-top-left detector [x_min, y_min, x_max, y_max] boxes.
        boxes.extend({'type':CLASS_MAP[family],'bbox':[b[1],b[0],b[3],b[2]]} for b in items)
    return boxes,excluded

def bbox_score(pred,truth,h,threshold):
    predicted=[]
    for c in pred['components'].values():
        kind=canonical_type(c['type']); b=c['bbox']
        if kind in ELIGIBLE: predicted.append({'type':kind,'bbox':[b[0],h-b[3],b[2],h-b[1]]})
    weights=[[iou(p['bbox'],g['bbox']) if p['type']==g['type'] and iou(p['bbox'],g['bbox'])>=threshold else 0 for g in truth] for p in predicted]
    tp=sum(weights[i][j]>0 for i,j in assignment(weights))
    return {'tp':tp,'predictions':len(predicted),'truth':len(truth),'F1':f1(tp,len(predicted),len(truth)),'unscored_predictions':len(pred['components'])-len(predicted)}
