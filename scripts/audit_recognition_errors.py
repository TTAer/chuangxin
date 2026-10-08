"""Post-inference attribution of component/pin losses under the QA proxy.

Diagnostic geometric matches deliberately ignore keys; they never modify an
inference output. Buckets prioritize failed fields; they are diagnostic counts,
not a claim of equivalence to the evaluator's identity-first matching.
"""
import argparse
import copy
from collections import Counter
import json
import math
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from pcbparse.competition import canonical_type,evaluate_qa
from pcbparse.core import assignment,distance,evaluate
from pcbparse.vision import iou,box_distance

def load(p): return json.loads(Path(p).read_text(encoding='utf-8'))

def audit(pred,truth,obs,qa_metrics=True):
    metrics=evaluate_qa(pred,truth) if qa_metrics else evaluate(pred,truth)
    pred,truth=copy.deepcopy(pred),copy.deepcopy(truth)
    for data in (pred,truth):
        for c in data['components'].values(): c['type']=canonical_type(c['type'])
    pc,gc=pred['components'],truth['components']; pk,gk=list(pc),list(gc)
    def gap(p,g): return max(abs(a-b) for a,b in zip(p['bbox'],g['bbox']))
    def geometric_score(a,b):
        overlap=iou(a,b)
        intersection=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
        containment=intersection/max(1,min((a[2]-a[0])*(a[3]-a[1]),(b[2]-b[0])*(b[3]-b[1])))
        center_gap=math.hypot((a[0]+a[2]-b[0]-b[2])/2,(a[1]+a[3]-b[1]-b[3])/2)
        diameter=max(a[2]-a[0],a[3]-a[1],b[2]-b[0],b[3]-b[1])
        return 1+overlap if max(abs(x-y) for x,y in zip(a,b))<=20 else overlap if overlap>.2 else .1+overlap if containment>.8 and center_gap<=diameter*.3 else 0
    scores=[[geometric_score(pc[p]['bbox'],gc[g]['bbox']) for g in gk] for p in pk]
    pairs={j:i for i,j in assignment(scores) if scores[i][j]>0}; used=set(pairs.values())
    components=[]; pins=[]
    for j,g in enumerate(gk):
        i=pairs.get(j); p=pk[i] if i is not None else None; gold=gc[g]; fields=[]
        if p is None:
            observed=any(iou(box,gold['bbox'])>.5 for box in obs.get('masked_symbol_boxes',[]))
            bucket='detected_but_identity_unassigned' if observed else 'missed_or_severely_mislocalized'
        else:
            actual=pc[p]
            if actual['type']!=gold['type']: fields.append('type')
            if not (gold['type'] in ('gnd','v') or g.startswith('∅')) and p!=g: fields.append('key')
            if gap(actual,gold)>20: fields.append('bbox')
            if gold['type']=='box' and actual.get('Name')!=gold.get('Name'): fields.append('ic_name')
            if gold['type'] in ('r','c') and actual.get('value')!=gold.get('value'): fields.append('value')
            bucket=next((b for b in ('key','type','bbox','ic_name','value') if b in fields),'correct')
        local=[t for t in obs.get('texts',[]) if box_distance(t['bbox'],gold['bbox'])<=75]
        components.append({'golden':g,'predicted':p,'bucket':bucket,'all_failed_fields':fields,'type':gold['type'],'golden_value':gold.get('value'),'predicted_value':pc[p].get('value') if p else None,'golden_name':gold.get('Name'),'predicted_name':pc[p].get('Name') if p else None,'golden_key_in_local_ocr':any(t['text']==g for t in local),'local_ocr':[t['text'] for t in local]})
        gp=truth['pins'].get(g,{}); pp=pred['pins'].get(p,{}) if p else {}
        pnums,gn=list(pp),list(gp)
        weights=[[max(0,30-distance(pp[n]['point'],gp[m]['point'])) for m in gn] for n in pnums]
        located={k:l for l,k in assignment(weights) if weights[l][k]>0}
        for k,n in enumerate(gn):
            m=pnums[located[k]] if k in located else None
            error=None if m is None else distance(pp[m]['point'],gp[n]['point'])
            if bucket in ('key','type','bbox','detected_but_identity_unassigned','missed_or_severely_mislocalized'): cause='wrong_or_missing_component_identity'
            elif m is None: cause='endpoint_missing_or_beyond_30px'
            elif error>5: cause='endpoint_beyond_5px'
            elif gold['type']=='box' and m!=n: cause='number_wrong'
            elif gold['type']=='box' and pp[m]['pinname']!=gp[n]['pinname']: cause='name_wrong'
            else: cause='correct'
            pins.append({'component':g,'golden_pin':n,'predicted_pin':m,'family':'box' if gold['type']=='box' else 'simple','bucket':cause,'distance':error})
    return {'components':components,'pins':pins,'component_buckets':dict(Counter(c['bucket'] for c in components)),'pin_buckets':{f:dict(Counter(p['bucket'] for p in pins if p['family']==f)) for f in ('box','simple')},'unpaired_predicted_components':[pk[i] for i in range(len(pk)) if i not in used],'metrics':metrics}

def main(a):
    cases=[]
    if a.dataset=='pcb55':
        base=ROOT/'output/benchmark-graph-topology-v2-55'
        for folder in sorted(base.iterdir()):
            if not (folder/'golden.json').exists(): continue
            pred=ROOT/'output/benchmark-pin-consensus-55'/folder.name/'prediction.json'
            cases.append((folder.name,load(pred),load(folder/'golden.json'),load(folder/'observations.json')))
    else:
        base=ROOT/'output/benchmark-official-pin-consensus-live'
        for folder in sorted(base.iterdir()):
            if not (folder/'prediction.json').exists(): continue
            cases.append((folder.name,load(folder/'prediction.json'),load(ROOT/f'data/raw/official/gt-0930/10GT/{folder.name}/{folder.name}.json'),load(folder/'observations.json')))
    records=[]; total=Counter(); families={f:Counter() for f in ('box','simple')}
    for case,pred,truth,obs in cases:
        record=dict(case=case,**audit(pred,truth,obs,qa_metrics=a.dataset=='gt10')); records.append(record)
        total.update(record['component_buckets'])
        for f in families: families[f].update(record['pin_buckets'][f])
    result={'dataset':a.dataset,'cases':len(records),'component_buckets':dict(total),'pin_buckets':{f:dict(c) for f,c in families.items()},'records':records,'official_equivalence':False,'diagnostic_matching':'one-to-one geometry first; identity/text attribution on matched boxes; priority buckets are not causal proofs','proxy_note':'QA proxy matches pins despite component Name/value mismatch, but key/type/bbox failures block pin matching'}
    path=Path(a.output); path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='records'},ensure_ascii=False),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--dataset',choices=['pcb55','gt10'],required=True); p.add_argument('--output',required=True); main(p.parse_args())
