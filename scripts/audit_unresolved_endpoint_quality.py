"""Frozen cached image-only evidence followed by posthoc development GT audit."""
import copy
import hashlib
import json
import sys
from pathlib import Path
from collections import Counter
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from experiments.endpoint_quality import distance,pixel_evidence,image_gray,bbox_iou,bucket
from pcbparse.core import assignment,evaluate
from pcbparse.competition import evaluate_qa
from pcbparse.dataset import index_dataset
from pcbparse.vision import read_image
from pcbparse.graphschematic import DualGraph

def load(path): return json.loads(Path(path).read_text(encoding='utf-8'))
def save(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')

def component_pairs(obs,gt):
    pk=[k for k,c in obs['components'].items() if c['type'] in ('ic','box')]
    gk=[k for k,c in gt['components'].items() if c['type'] in ('ic','box')]
    weights=[[bbox_iou(obs['components'][p]['bbox'],gt['components'][g]['bbox']) for g in gk] for p in pk]
    return {pk[i]:gk[j] for i,j in assignment(weights) if weights[i][j]>=.3}

def table(rows):
    result={}
    for grade in ('ALL','G0','G1','G2','G3'):
        rs=rows if grade=='ALL' else [r for r in rows if r['grade']==grade]
        known=[r for r in rs if r['nearest_GT_distance'] is not None]
        hits=sum(r['nearest_GT_distance']<=5 for r in known)
        unique=sum(r.get('one_to_one_EP5',False) for r in rs)
        novel=sum(r.get('one_to_one_uncovered_geometry_EP5',False) for r in rs)
        result[grade]={'candidates':len(rs),'GT_component_matched':len(known),
            'distance_buckets':dict(Counter(bucket(r['nearest_GT_distance']) for r in rs)),
            'nearest_EP5_known':hits/len(known) if known else None,
            'nearest_EP5_all_candidates':hits/len(rs) if rs else None,
            'one_to_one_EP5_all_candidates':unique/len(rs) if rs else None,
            'one_to_one_hits':unique,'authority':'evidence-only; reused development GT, no independent calibration'}
        result[grade]['uncovered_geometry_EP5_all_candidates']=novel/len(rs) if rs else None
        result[grade]['uncovered_geometry_hits']=novel
    return result

def main():
    out=ROOT/'output/endpoint-quality'; out.mkdir(parents=True,exist_ok=True)
    protocol={'status':'development diagnostic, not blind evaluation','GT_at_pixel_evidence':False,
        'candidate_scope':'all number_unresolved IC/box; old anchor distance <6 excluded',
        'component_match':'one-to-one bbox IoU >=0.3 IC/box class family',
        'geometry_GT':'cached pin coordinates; PCB weak conversion numbers never certified identities',
        'EP5':'nearest within matched component plus one-to-one coverage; unknown GT explicitly included in all-candidate denominator',
        'G2':'7 pixel normal stroke + inward 4..40 pixel boundary (17 pixel axial support >=.88) + stem continuity >=.88',
        'G3':'G2 plus wire segment contact <=3 and physical graph contact_allowed; contact alone never certifies geometry',
        'uncovered_geometry':'GT point not within 5px of any existing predicted anchor; one-to-one matching; geometry coverage only, no certified pin identity',
        'oracle_scope':'only previously committed local OCR new pins, exact GT number; PinF1 only, no net oracle',
        'RC0_changed':False,'snap_enabled':False}
    save(out/'protocol.json',protocol)
    samples={s['id']:s for s in index_dataset(ROOT/'data/raw/PCBnet/Dataset300')}
    allrows=[]; oracle=[]; excluded=0; raw=0
    for domain in ('gt10','pcb55'):
        for folder in sorted((ROOT/'output/designator-stable-catalog'/domain).iterdir()):
            if not (folder/'observations.json').exists(): continue
            obs=load(folder/'observations.json'); case=folder.name
            imagepath=ROOT/'data/raw/official/gt-0930/10GT'/case/(case+'.png') if domain=='gt10' else Path(samples[case]['image'])
            gray=image_gray(read_image(imagepath)); rows=[]
            physical=copy.deepcopy(obs); physical.update(components={},pins={},labels=[])
            graph=DualGraph(physical)
            for tip in obs.get('pin_hypotheses',[]):
                key=tip['component']
                if tip.get('status')!='number_unresolved' or obs['components'][key]['type'] not in ('ic','box'): continue
                raw+=1
                if any(distance(tip['point'],q['point'])<6 for q in obs['pins'].get(key,{}).values()): excluded+=1; continue
                features=pixel_evidence(gray,tip['point'],obs['components'][key]['bbox'],obs['wires'],graph.contact_allowed(tip['point']))
                rows.append(dict(domain=domain,case=case,component=key,id=tip['id'],point=tip['point'],score=tip.get('score'),**features))
            # GT first opened after ALL pixel evidence for this image is finalized.
            gtpath=imagepath.with_suffix('.json') if domain=='gt10' else ROOT/'output/benchmark-recognition-pcb55'/case/'golden.json'
            gt=load(gtpath); pairs=component_pairs(obs,gt)
            for r in rows:
                g=pairs.get(r['component']); pins=gt['pins'].get(g,{}) if g else {}
                r.update(GT_component=g,nearest_GT_distance=min((distance(r['point'],p['point']) for p in pins.values()),default=None),one_to_one_EP5=False,one_to_one_uncovered_geometry_EP5=False)
            for key,g in pairs.items():
                rs=[r for r in rows if r['component']==key]; gps=list(gt['pins'].get(g,{}).values())
                weights=[[1+(5-distance(r['point'],p['point']))/5 if distance(r['point'],p['point'])<=5 else 0 for p in gps] for r in rs]
                for i,j in assignment(weights):
                    if weights[i][j]>0: rs[i]['one_to_one_EP5']=True
                uncovered=[p for p in gps if not any(distance(p['point'],q['point'])<=5 for q in obs['pins'].get(key,{}).values())]
                weights=[[1+(5-distance(r['point'],p['point']))/5 if distance(r['point'],p['point'])<=5 else 0 for p in uncovered] for r in rs]
                for i,j in assignment(weights):
                    if weights[i][j]>0: rs[i]['one_to_one_uncovered_geometry_EP5']=True
            save(out/domain/case/'audit.json',{'observations_sha256':hashlib.sha256((folder/'observations.json').read_bytes()).hexdigest(),'rows':rows})
            allrows.extend(rows)
            localfolder=ROOT/'output/local-pin-evidence'/domain/case
            report=load(localfolder/'report.json'); pred=load(localfolder/'prediction.json')
            arms={k:copy.deepcopy(pred) for k in ('geometry_only','identity_only','both')}; cases=[]
            for d in report['decisions']:
                if not d['committed']: continue
                key=d['component']; g=pairs.get(key); pn='pin_'+d['number']; gp=gt['pins'].get(g,{}).get(pn) if g else None
                row=next((r for r in rows if r['id']==d['tip']),None)
                item={'component':key,'pin':pn,'grade':row['grade'] if row else None,'GT_exact_number_found':gp is not None}
                if gp:
                    item.update(error_pixels=distance(d['point'],gp['point']),predicted_name=d['name'],GT_name=gp['pinname'])
                    arms['geometry_only']['pins'][key][pn]['point']=copy.deepcopy(gp['point'])
                    arms['identity_only']['pins'][key][pn]['pinname']=gp['pinname']
                    arms['both']['pins'][key][pn]=copy.deepcopy(gp)
                cases.append(item)
            scorer=evaluate_qa if domain=='gt10' else evaluate
            scores={'actual':scorer(pred,gt)['PinF1'],**{k:scorer(p,gt)['PinF1'] for k,p in arms.items()}}
            oracle.append({'domain':domain,'case':case,'new_commits':cases,'PinF1':scores})
            print(domain,case,len(rows),flush=True)
    summary={'pages':len(oracle),'raw_unresolved':raw,'excluded_existing_anchor_nearby':excluded,'audited_candidates':len(allrows),
        'tables':{d:table([r for r in allrows if r['domain']==d]) for d in ('gt10','pcb55')},
        'score_ge_0_7_tables':{d:table([r for r in allrows if r['domain']==d and (r['score'] or 0)>=.7]) for d in ('gt10','pcb55')},
        'oracle':{d:{k:sum(r['PinF1'][k] for r in oracle if r['domain']==d)/sum(r['domain']==d for r in oracle) for k in ('actual','geometry_only','identity_only','both')} for d in ('gt10','pcb55')},
        'oracle_commits':[dict(domain=r['domain'],case=r['case'],**p) for r in oracle for p in r['new_commits']],
        'runtime_promoted':False,'ExistingAnchorChange':0,'snap_decision':'defer: only three exact-number commit diagnostics cannot establish general oracle dominance or independent EP5'}
    save(out/'summary.json',summary); save(out/'oracle.json',oracle)
    print(json.dumps(summary,ensure_ascii=False),flush=True)
if __name__=='__main__': main()
