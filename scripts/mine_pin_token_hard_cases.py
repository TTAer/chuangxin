"""Post-label diagnostics; no oracle identity is included in feature inputs."""
from collections import Counter
import json
import re
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
import numpy as np

def side(point,body):
    x,y=point
    return min(('left','right','top','bottom'),key=lambda s:abs({'left':x-body[0],'right':x-body[2],'top':y-body[1],'bottom':y-body[3]}[s]))

def main():
    cache=ROOT/'data/processed/pin-token-residual'
    with np.load(cache/'tokens.npz') as stored:
        data={key:stored[key] for key in ('pairs','valid','text','role','association','split')}
    records=json.loads((cache/'records.json').read_text(encoding='utf-8'))
    raw={}; hard=[]; counts=Counter(); taxonomy=Counter(); features=[]; label_risks=Counter()
    for i,r in enumerate(records):
        pairs=data['pairs'][i]; valid=data['valid'][i]; distance=pairs[:,4]
        count=int(valid.sum()); d1=distance[0]; d2=distance[1] if count>1 else d1+10
        # Fixed train-only feature recipe: text syntax, first three candidate geometry,
        # distance margin and OCR confidence. Labels never enter these features.
        features.append(np.r_[data['text'][i],pairs[:3].ravel(),valid[:3].astype(float),d1,d2,d2-d1,r['token']['confidence']].astype(np.float32))
        if r['target']==8: counts[r['split']+':non_pin']+=1; continue
        if r['role']=='pin_number' and not re.fullmatch(r'[A-Z]?[0-9]+',r['token']['text'],re.I): label_risks[r['split']+':number_format_requires_review']+=1
        if r['role']=='pin_number' and re.fullmatch(r'(?:R|C|L|D|Q|U|IC|J|P)[0-9]+',r['token']['text'],re.I): label_risks[r['split']+':number_designator_ambiguity']+=1
        counts[r['split']+':pin_tokens']+=1
        if r['target']==0: counts[r['split']+':nearest_correct']+=1; continue
        counts[r['split']+':hard']+=1
        if r['page'] not in raw:
            p=next((ROOT/'data/raw/PCBnet/Dataset300'/r['page']).glob('*_CVJsonStd.json')); raw[r['page']]=json.loads(p.read_text(encoding='utf-8'))
        c=raw[r['page']]['components'][r['component']]; b=c.get('realBBox') or c['location']; body=[b['x1'],b['y1'],b['x2'],b['y2']]
        candidates=[c['pins'][pid]['tip'] for pid in r['candidate_pin_ids']]; first=candidates[0]; true=candidates[r['target']]
        box=r['token']['bbox']; tags=[]
        if np.linalg.norm(np.asarray(first)-true)<.01: tags.append('coincident_endpoints_identity_ambiguous')
        if r['role']=='pin_number' and not re.fullmatch(r'[A-Z]?[0-9]+',r['token']['text'],re.I): tags.append('number_format_requires_review')
        if r['role']=='pin_number' and re.fullmatch(r'(?:R|C|L|D|Q|U|IC|J|P)[0-9]+',r['token']['text'],re.I): tags.append('number_designator_ambiguity')
        if side(first,body)!=side(true,body): tags.append('wrong_side')
        else: tags.append('same_side_neighbor')
        if box[3]-box[1]>1.4*max(1,box[2]-box[0]): tags.append('vertical_ocr_box')
        if len(r['token']['text'])>=8: tags.append('long_label')
        axis=1 if side(true,body) in ('left','right') else 0
        if sum(box[axis]<=p[axis]<=box[axis+2] for p in candidates)>=2: tags.append('token_span_multiple_endpoints')
        fraction=(true[axis]-body[axis])/max(1,body[axis+2]-body[axis])
        if fraction<.1 or fraction>.9: tags.append('corner_region')
        if r['target']>2: tags.append('outside_residual_top3')
        taxonomy.update(tags)
        hard.append({'index':i,'page':r['page'],'component':r['component'],'token':r['token'],'role':r['role'],'split':r['split'],'target_rank':r['target']+1,'tags':tags,'first_pin':r['candidate_pin_ids'][0],'source_pin':r['pin']})
    np.savez_compressed(cache/'residual-features.npz',features=np.stack(features),role=np.where(data['role']==1,0,np.where(data['role']==2,1,2)),target=data['association'],split=data['split'])
    result={'counts':dict(counts),'taxonomy_multilabel':dict(taxonomy),'pin_label_risk_counts':dict(label_risks),'hard_relations':hard,
        'neural_route_supported_by_training_hard_count':counts['train:hard']>=100,
        'feature_definition':'12 text syntax + first3x12 geometry + 3 masks + d1/d2/margin/confidence',
        'limitations':['conditional on uniquely aligned OCR and oracle body/endpoints','taxonomy is geometric diagnostic, not manually adjudicated error cause','unknown text roles not labelled non-pin','existing development split, not blind test']}
    (ROOT/'output/ablation/pin-token-hard-mining.json').write_text(json.dumps(result,indent=2),encoding='utf-8'); print(json.dumps({'counts':result['counts'],'taxonomy':result['taxonomy_multilabel']}),flush=True)
if __name__=='__main__': main()
