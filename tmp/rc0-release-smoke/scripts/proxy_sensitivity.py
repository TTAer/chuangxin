"""Post-inference matcher sensitivity; never selects inference thresholds."""
import json
import math
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from pcbparse.__main__ import load,save
from pcbparse.competition import evaluate_qa,canonical_type
from pcbparse.core import assignment,f1
from pcbparse.vision import iou
profiles={'old_consensus':'benchmark-official-pin-consensus-live','current_default':'benchmark-recognition-gt10',
          'ic_structure':'ic-structure-gt10-geometry','side_strip_local':'side-strip-side-strip-local-gt10'}
cases=[str(i) for i in range(1,11)]; truths={c:load(ROOT/'data/raw/official/gt-0930/10GT'/c/(c+'.json')) for c in cases}
predictions={name:{c:load(ROOT/'output'/folder/c/'prediction.json') for c in cases} for name,folder in profiles.items()}
def component_metric(pred,gt,criterion,key_first):
    pc,gc=pred['components'],gt['components']; pk,gk=list(pc),list(gc)
    def identity(p,g): return p==g or canonical_type(gc[g]['type']) in ('gnd','v') or g.startswith('∅')
    def geometry(a,b):
        if criterion.startswith('iou'): return iou(a,b)>=float(criterion.split(':')[1])
        tolerance=float(criterion.split(':')[1])
        if criterion.startswith('center'): return math.hypot((a[0]+a[2]-b[0]-b[2])/2,(a[1]+a[3]-b[1]-b[3])/2)<=tolerance
        return max(math.hypot(a[0]-b[0],a[1]-b[1]),math.hypot(a[2]-b[2],a[3]-b[3]))<=tolerance
    weights=[]
    for p in pk:
        row=[]
        for g in gk:
            a,b=pc[p],gc[g]; valid=canonical_type(a['type'])==canonical_type(b['type']) and geometry(a['bbox'],b['bbox']) and (not key_first or identity(p,g))
            row.append(1+iou(a['bbox'],b['bbox']) if valid else 0)
        weights.append(row)
    matched=0
    for i,j in assignment(weights):
        p,g=pk[i],gk[j]
        if weights[i][j]<=0 or not identity(p,g): continue
        kind=canonical_type(gc[g]['type']); field='Name' if kind=='box' else 'value' if kind in ('r','c') else None
        if field and pc[p].get(field)!=gc[g].get(field): continue
        matched+=1
    return f1(matched,len(pc),len(gc))
rows=[]
for tolerance in (10,15,20,25,30):
    metrics={name:[evaluate_qa(pred[c],truths[c],bbox_tolerance=tolerance) for c in cases] for name,pred in predictions.items()}
    keys=['ComponentF1','PinF1','NetHypergraphF1','NetLineF1','Score_Accuracy_SingleCase_proxy']
    rows.append({'bbox_coordinate_tolerance':tolerance,'mean_metrics':{name:{k:sum(r[k] for r in records)/10 for k in keys} for name,records in metrics.items()}})
components=[]
for criterion in ('iou:0.3','iou:0.5','iou:0.75','corner:15','corner:25','center:10','center:20'):
    for key_first in (True,False):
        components.append({'criterion':criterion,'order':'key_first' if key_first else 'geometry_first_then_identity',
            'ComponentF1':{name:sum(component_metric(pred[c],truths[c],criterion,key_first) for c in cases)/10 for name,pred in predictions.items()}})
summary={'dataset':'public GT10; previously used in development','QA_bbox_sweep':rows,'hypothetical_component_matchers':components,
         'official_equivalence':False,'changes_inference':False,'selection_rule':'no inference parameters selected from this audit',
         'limitations':['center-only ignores extent and is a stress test, not necessarily a plausible official rule','hypothetical component matchers do not imply corresponding net-score equivalence']}
save(ROOT/'output/ablation/proxy-sensitivity.json',summary)
print(json.dumps({'bbox_sweep':[{ 'tol':r['bbox_coordinate_tolerance'],'old':r['mean_metrics']['old_consensus']['Score_Accuracy_SingleCase_proxy'],'current':r['mean_metrics']['current_default']['Score_Accuracy_SingleCase_proxy']} for r in rows],
                  'component_matchers_current_beats_old':sum(r['ComponentF1']['current_default']>r['ComponentF1']['old_consensus'] for r in components),'matchers':len(components)}),flush=True)
