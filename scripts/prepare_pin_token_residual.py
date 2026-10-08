"""All existing official training pages: independent residual mining cache; frozen V1 untouched."""
import argparse
from collections import Counter,defaultdict
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
import cv2
import numpy as np
from pcbparse.dataset import index_dataset
from pcbparse.vision import ImageParser,read_image,iou
from experiments.pin_token_v1 import K,ROLES,candidates,text_features

def referenced_labels(raw):
    roles=defaultdict(set); owners=defaultdict(set)
    for cid,c in raw['components'].items():
        for pid,p in c.get('pins',{}).items():
            for field,role in [('pinNumberByOCR','pin_number'),('pinNameByOCR','pin_name')]:
                tid=p.get(field)
                if tid in raw['texts']:
                    roles[tid].add(role); owners[tid].add((cid,pid))
        for tid,t in c.get('associatedTexts',{}).items():
            role={'designator':'designator','searchName':'component_model','name':'component_model','Name':'component_model'}.get(t.get('details'))
            if role and tid in raw['texts']: roles[tid].add(role)
        tid=c.get('searchNameByOCR')
        if tid in raw['texts']: roles[tid].add('net_label' if c['category'] in ('NET_input','Net_output','Net_bidirection','V','Gnd') else 'component_model')
    return roles,owners

def family(raw):
    # Conservative structural signature, not a proof of design independence.
    components=raw['components']; nodes=sorted((c['category'],len(c.get('pins',{}))) for c in components.values())
    edges=sorted(sorted((components[v['cpntId']]['category'],len(components[v['cpntId']].get('pins',{}))) for v in e.get('connect',{}).values() if v.get('cpntId') in components) for e in raw['edges'].values())
    return hashlib.sha256(json.dumps([nodes,edges],sort_keys=True).encode()).hexdigest()

def main():
    p=argparse.ArgumentParser(); p.add_argument('--train-pages',type=int,default=999); p.add_argument('--validation-pages',type=int,default=999); args=p.parse_args()
    audit=json.loads((ROOT/'output/official-audit/inventory.json').read_text(encoding='utf-8'))
    items=index_dataset(ROOT/'data/raw/PCBnet/Dataset300'); lookup={r['id']:r for r in items}
    mapped=[]; counts=Counter(); seen=set()
    for row in audit['details']['train200']['records']:
        page=Path(row['image']).parent.name
        if page not in lookup or page=='0237' or row['pixels_sha256'] in seen: continue
        seen.add(row['pixels_sha256']); item=lookup[page]
        raw=json.loads(Path(item['annotation']).read_text(encoding='utf-8'))
        mapped.append(dict(item,source=row,family=family(raw),initial_split='train' if row['training_overlap'] else 'validation'))
    validation_families={r['family'] for r in mapped if r['initial_split']=='validation'}
    selection=[]
    for r in mapped:
        r['split']='validation' if r['family'] in validation_families else 'train'
        limit=args.train_pages if r['split']=='train' else args.validation_pages
        if counts[r['split']]<limit: selection.append(r); counts[r['split']]+=1
    assert {r['family'] for r in selection if r['split']=='train'}.isdisjoint({r['family'] for r in selection if r['split']=='validation'})
    out=ROOT/'data/processed/pin-token-residual'; out.mkdir(parents=True,exist_ok=True)
    parser=ImageParser(); arrays={k:[] for k in ('patch','text','pairs','valid','role','association','split')}; records=[]; skipped=Counter()
    for item in selection:
        image=read_image(item['image']); official=read_image(ROOT/item['source']['image'])
        assert np.array_equal(image,official),'CV token annotation is not pixel-aligned with official training image'
        raw=json.loads(Path(item['annotation']).read_text(encoding='utf-8')); roles,owners=referenced_labels(raw)
        cache=out/(item['id']+'-ocr.json')
        if cache.exists(): tokens=json.loads(cache.read_text(encoding='utf-8'))
        else:
            result,_=parser.ocr(image)
            tokens=[{'text':t,'confidence':float(conf),'bbox':[min(v[0] for v in b),min(v[1] for v in b),max(v[0] for v in b),max(v[1] for v in b)]} for b,t,conf in result or []]
            cache.write_text(json.dumps(tokens,ensure_ascii=False),encoding='utf-8')
        before=len(records)
        for token in tokens:
            matches=[]
            for tid,t in raw['texts'].items():
                if token['text']!=t.get('content') or len(roles[tid])!=1: continue
                b=t['location']; score=iou(token['bbox'],[b['x1'],b['y1'],b['x2'],b['y2']])
                if score>=.25: matches.append((score,tid))
            if len(matches)!=1: skipped['unmatched_or_ambiguous_ocr']+=1; continue
            tid=matches[0][1]; role=next(iter(roles[tid])); owner=owners[tid]
            if len(owner)>1: skipped['shared_token_owner']+=1; continue
            if role.startswith('pin_') and len(owner)!=1: skipped['no_unique_pin_owner']+=1; continue
            if owner: cid,pid=next(iter(owner)); c=raw['components'][cid]
            else:
                boxes=[(key,c) for key,c in raw['components'].items() if c['category']=='Box' and c.get('pins')]
                if not boxes: continue
                x,y=(token['bbox'][0]+token['bbox'][2])/2,(token['bbox'][1]+token['bbox'][3])/2
                cid,c=min(boxes,key=lambda pair:(x-(pair[1]['location']['x1']+pair[1]['location']['x2'])/2)**2+(y-(pair[1]['location']['y1']+pair[1]['location']['y2'])/2)**2); pid=None
            if c['category']!='Box' or not c.get('pins'): continue
            b=c.get('realBBox') or c['location']; body=[b['x1'],b['y1'],b['x2'],b['y2']]
            pinids=list(c['pins']); points=[c['pins'][key]['tip'] for key in pinids]
            order,pairs,valid=candidates(token['bbox'],body,points)
            target=K
            if pid is not None:
                true=pinids.index(pid)
                if true not in order: skipped['true_endpoint_not_in_geometric_top8']+=1; continue
                target=list(order).index(true)
            x1,y1,x2,y2=map(round,token['bbox']); patch=image[max(0,y1-2):min(image.shape[0],y2+2),max(0,x1-2):min(image.shape[1],x2+2)]
            if not patch.size: continue
            patch=cv2.resize(cv2.cvtColor(patch,cv2.COLOR_BGR2GRAY),(96,32))
            for key,value in [('patch',patch[None]),('text',text_features(token['text'])),('pairs',pairs),('valid',valid),('role',ROLES.index(role)),('association',target),('split',int(item['split']=='validation'))]: arrays[key].append(value)
            records.append({'page':item['id'],'component':cid,'token':token,'annotation_text_id':tid,'role':role,'pin':pid,'candidate_pin_ids':[pinids[i] for i in order],'target':target,'family':item['family'],'split':item['split']})
        print(json.dumps({'page':item['id'],'split':item['split'],'tokens_added':len(records)-before,'total':len(records)}),flush=True)
    assert records and any(r['split']=='validation' for r in records)
    np.savez_compressed(out/'tokens.npz',**{k:np.asarray(v) for k,v in arrays.items()})
    (out/'records.json').write_text(json.dumps(records,ensure_ascii=False),encoding='utf-8')
    summary={'pages':dict(counts),'token_count':len(records),'role_counts':dict(Counter(r['role'] for r in records)),'split_counts':dict(Counter(r['split'] for r in records)),
        'skipped':dict(skipped),'selection':[{'page':r['id'],'family':r['family'],'split':r['split'],'image_sha256':r['sha256']} for r in selection],
        'source':'official 200 training images, pixel-aligned CVJson explicit token references; weak/noisy labels',
        'ocr_tokens':'actual RapidOCR top1; no fabricated top-k or character confidence',
        'supervision_limitations':['unknown/unlinked token roles are excluded, not labelled other','endpoint points and component bodies are oracle annotation; local validation is not end-to-end','structural signature is conservative overlap grouping, not proof of source design independence','validation used for model selection; Freeze-A/B and 10GT not read']}
    (out/'preparation.json').write_text(json.dumps(summary,indent=2),encoding='utf-8'); print(json.dumps(summary['split_counts']),flush=True)
if __name__=='__main__': main()
