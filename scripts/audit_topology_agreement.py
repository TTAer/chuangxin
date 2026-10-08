"""Frozen-model error overlap, relation disagreement and candidate exposure audit.

The fixed .95 operating point is descriptive, not calibrated or permission to
commit. Unknown labels never contribute to precision denominators.
"""
import json,sys,math
from pathlib import Path
from collections import Counter
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import cv2,numpy as np,onnxruntime as ort
from pcbparse.dataset import index_dataset
from pcbparse.vision import read_image,wire_lines
from pcbparse.topology_perception import crossing_candidates,dot_evidence,three_arm_evidence,patch_tensor
from experiments.port_partition import RELATIONS

CROSS=[1,2,3,4]


def precision_lower95(tp,fp):
    """One-sided exact binomial lower bound without optional SciPy dependency."""
    if not tp:return 0.0
    n=tp+fp
    if not fp:return math.exp(math.log(.05)/n)
    lo,hi=0.0,1.0
    for _ in range(60):
        p=(lo+hi)/2
        terms=[math.lgamma(n+1)-math.lgamma(k+1)-math.lgamma(n-k+1)+k*math.log1p(-p)+(n-k)*math.log(p) for k in range(fp+1)]
        top=max(terms);prob=math.exp(top)*sum(math.exp(t-top) for t in terms)
        if prob<.05:lo=p
        else:hi=p
    return (lo+hi)/2


def metrics(chosen,target,known):
    tp=int((chosen&target&known).sum());fp=int((chosen&~target&known).sum());fn=int((~chosen&target&known).sum())
    n=int(known.sum());pos=int((known&target).sum())
    return {'tp':tp,'fp':fp,'fn':fn,'precision':tp/(tp+fp) if tp+fp else None,
            'precision_lower_95_one_sided':precision_lower95(tp,fp),
            'recall':tp/pos if pos else None,'coverage':(tp+fp)/n if n else None,'known_relations':n}


def overlap(a,b,known):
    return {'PPN':int((a&known).sum()),'MST':int((b&known).sum()),
            'intersection':int((a&b&known).sum()),'union':int(((a|b)&known).sum())}


def infer(session,images):
    logits=np.concatenate([session.run(None,{'image':part})[0] for part in np.array_split(images,max(1,int(np.ceil(len(images)/128))))])
    p=np.exp(logits-logits.max(1,keepdims=True));p/=p.sum(1,keepdims=True)
    return RELATIONS[logits.argmax(1)].astype(bool),p@RELATIONS


def legacy_states(images,ids,rows,z,index):
    old=json.loads((ROOT/'data/processed/topology/records.json').read_text(encoding='utf-8'))
    with np.load(ROOT/'data/processed/topology/patches.npz') as ar:g=ar['geometry']
    geometry={(r['page'],tuple(r['point'])):g[i] for i,r in enumerate(old)}
    opt=ort.SessionOptions();opt.intra_op_num_threads=2
    legacy=ort.InferenceSession(str(ROOT/'models/topology/topology-perception.onnx'),sess_options=opt,providers=['CPUExecutionProvider'])
    meta=json.loads((ROOT/'models/topology/topology-perception.json').read_text(encoding='utf-8'));cache={};states=[]
    for j,i in enumerate(ids):
        r=rows[i];page=r['page'];point=r['point']
        if page not in cache:
            im=read_image(index[page]['image']);mask=(cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)<220).astype(np.uint8)*255
            lines,_=wire_lines(mask,split_junctions=False);cache[page]=(mask,{(c['x'],c['y']):c for c in crossing_candidates(lines)})
        mask,mapping=cache[page];c=mapping.get(tuple(point));geom=geometry.get((page,tuple(point))) if z['split'][i]==1 else np.asarray(c['geometry'],np.float32)
        gray=cv2.cvtColor(images[j].transpose(1,2,0),cv2.COLOR_RGB2GRAY);ink=(gray<.86).astype(np.float32)
        tensor=np.concatenate([images[j],gray[None],cv2.morphologyEx(ink,cv2.MORPH_OPEN,np.ones((1,7),np.uint8))[None],cv2.morphologyEx(ink,cv2.MORPH_OPEN,np.ones((7,1),np.uint8))[None]])
        logit=legacy.run(None,{'patch':tensor[None],'geometry':np.asarray(geom,np.float32)[None]})[0].reshape(-1)[0]
        p=float(1/(1+np.exp(-np.clip(logit,-30,30))))
        state='C' if p>=meta['connect_threshold'] else 'S' if p<=meta['disconnect_threshold'] else 'U'
        if state=='U' and ((np.min(geom)*32<=3 and three_arm_evidence(mask,*point) and p>=.2) or (dot_evidence(mask,*point) and p>=.5)):state='C'
        states.append(state)
    return np.array(states)


def matrix(legacy,p,m,target,known):
    counter={f'{a}|{b}|{c}':{'relations':0,'known_positive':0,'known_negative':0,'unknown':0} for a in 'CSU' for b in 'CSU' for c in 'CSU'}
    for i in range(len(legacy)):
        for k in CROSS:
            key=f'{legacy[i]}|{p[i,k]}|{m[i,k]}'
            row=counter.setdefault(key,{'relations':0,'known_positive':0,'known_negative':0,'unknown':0})
            row['relations']+=1
            if known[i,k]:row['known_positive' if target[i,k] else 'known_negative']+=1
            else:row['unknown']+=1
    for key,row in counter.items():
        n=row['known_positive']+row['known_negative'];row['positive_fraction']=row['known_positive']/n if n else None
    return counter


def main():
    out=ROOT/'output/topology-agreement';out.mkdir(parents=True,exist_ok=True)
    protocol={'threshold':.95,'models':['PPN expanded-seed0','MST multisource-pretrained','frozen Topology v2'],
              'threshold_scope':'fixed descriptive point from prior report, no scanning, not independently calibrated',
              'permissions':'confident RC0 preserved; uncertain + both connect => proposal only; all other disagreements abstain; both split maintains separation',
              'new_training':False,'runtime_changed':False,'Freeze_A_used':False,
              'replay_entry':'require >=.99 known merge precision in both development cohorts; unknown exposure reported without post-hoc tuning; no protected existing net bridge',
              'limits':'MST has only one seed; source/family robustness not established; unknown exposure acceptable bound was not preregistered'}
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    with np.load(ROOT/'data/processed/port-partition-expanded/relations.npz') as ar:z={k:ar[k] for k in ar.files}
    rows=json.loads((ROOT/'data/processed/port-partition-expanded/records.json').read_text(encoding='utf-8'))
    ids=np.flatnonzero(z['split']!=0);images=z['images'][ids].astype(np.float32)/255
    index={r['id']:r for r in index_dataset(ROOT/'data/raw/PCBnet/Dataset300')}
    opt=ort.SessionOptions();opt.intra_op_num_threads=2
    sessions=[ort.InferenceSession(str(ROOT/p),sess_options=opt,providers=['CPUExecutionProvider']) for p in ('models/candidates/port-partition-expanded/expanded-seed0/port-graph/model.onnx','models/candidates/masked-stroke-transformer/multisource-pretrained/model.onnx')]
    pm,pw=infer(sessions[0],images);mm,mw=infer(sessions[1],images);legacy=legacy_states(images,ids,rows,z,index)
    target=z['targets'][ids].astype(bool);known=z['valid'][ids].astype(bool);known[:,[0,5]]=False
    error_records=[];summary={}
    for code in (1,2):
        pick=z['split'][ids]==code;k=known&pick[:,None];uncertain=k&(legacy[:,None]=='U')
        gate=(pw>=.95)&(mw>=.95)
        summary[str(code)]={'samples':int(pick.sum()),'map_FP_overlap':overlap(pm&~target,mm&~target,k),
            'map_FN_overlap':overlap(~pm&target,~mm&target,k),'high_weight_FP_overlap':overlap((pw>=.95)&~target,(mw>=.95)&~target,k),
            'PPN_MAP':metrics(pm,target,k),'MST_MAP':metrics(mm,target,k),'joint_MAP':metrics(pm&mm,target,k),
            'PPN_high':metrics(pw>=.95,target,k),'MST_high':metrics(mw>=.95,target,k),'joint_high':metrics(gate,target,k),
            'old_uncertain_joint_high':metrics(gate,target,uncertain),
            'map_matrix':matrix(legacy[pick],np.where(pm[pick],'C','S'),np.where(mm[pick],'C','S'),target[pick],known[pick]),
            'confidence_matrix':matrix(legacy[pick],np.where(pw[pick]>=.95,'C',np.where(pw[pick]<=.05,'S','U')),np.where(mw[pick]>=.95,'C',np.where(mw[pick]<=.05,'S','U')),target[pick],known[pick])}
    for j,i in enumerate(ids):
        shared=np.flatnonzero(known[j]&~target[j]&pm[j]&mm[j]).tolist()
        if shared:
            folder=out/'shared-error-crops';folder.mkdir(exist_ok=True);crop=folder/(str(i)+'.png')
            cv2.imencode('.png',z['images'][i].transpose(1,2,0)[:,:,::-1])[1].tofile(str(crop))
            error_records.append({'index':int(i),'page':rows[i]['page'],'point':rows[i]['point'],'cohort':int(z['split'][i]),'shared_map_false_merge_pairs':shared,'PPN_weights':pw[j].tolist(),'MST_weights':mw[j].tolist(),'human_reviewed':False,'crop':str(crop.relative_to(ROOT))})
    (out/'local-audit.json').write_text(json.dumps({'summary':summary,'shared_error_records':error_records,'independent_accuracy':False},indent=2),encoding='utf-8')
    np.savez_compressed(out/'local-predictions.npz',indices=ids,PPN_map=pm,MST_map=mm,PPN_weights=pw,MST_weights=mw,legacy=legacy)
    print(json.dumps({'local':summary},default=str),flush=True)
    exposure={}
    for domain in ('gt10','pcb55'):
        totals=Counter();pages=[]
        for folder in sorted((ROOT/'output/port-partition-guarded-replay'/domain).iterdir()):
            if not (folder/'decisions.json').exists():continue
            records=json.loads((folder/'decisions.json').read_text(encoding='utf-8'));case=folder.name
            image=read_image(ROOT/'data/raw/official/gt-0930/10GT'/case/(case+'.png') if domain=='gt10' else index[case]['image'])
            if not records:continue
            patches=np.stack([patch_tensor(image,*r['point_xy'])[:3] for r in records]);_,weights=infer(sessions[1],patches)
            decisions=[];count=Counter(candidates=len(records))
            for j,r in enumerate(records):
                p=np.array(r['marginal']);m=weights[j];known=np.array(r['known']).astype(bool);known[[0,5]]=True
                unknown=~known;both=(p>=.95)&(m>=.95);ph=(p>=.95);mh=(m>=.95)
                count['unknown_relations']+=int(unknown.sum());count['PPN_unknown_high_relations']+=int((ph&unknown).sum())
                count['MST_unknown_high_relations']+=int((mh&unknown).sum());count['joint_unknown_high_relations']+=int((both&unknown).sum())
                count['joint_unknown_high_candidates']+=bool((both&unknown).any())
                u=r['legacy_state']=='uncertain';count['old_uncertain_candidates']+=u
                count['old_uncertain_joint_unknown_relations']+=int((both&unknown).sum()) if u else 0
                split=bool((p[CROSS]<=.05).all() and (m[CROSS]<=.05).all() and (p[[0,5]]>=.95).all() and (m[[0,5]]>=.95).all())
                count['connected_joint_canonical_split']+=split and r['legacy_state']=='connected'
                if u and both[CROSS].any():count['uncertain_joint_merge_candidates']+=1
                decisions.append({'point_xy':r['point_xy'],'legacy_state':r['legacy_state'],'PPN_weights':p.tolist(),'MST_weights':m.tolist(),
                    'known':r['known'],'weak_target':r['weak_target'],'joint_unknown_high_pairs':np.flatnonzero(both&unknown).tolist(),
                    'canonical_joint_split':split,'permission':'proposal_only' if u and both[CROSS].any() else 'retain_RC0','commit':False})
            totals.update(count);pages.append({'case':case,**dict(count)})
            dest=out/domain/case;dest.mkdir(parents=True,exist_ok=True);(dest/'decisions.json').write_text(json.dumps(decisions,indent=2),encoding='utf-8')
            print(json.dumps({'domain':domain,'case':case,**dict(count)}),flush=True)
        exposure[domain]={'counts':dict(totals),'pages':pages,'unknown_not_counted_as_FP':True}
    (out/'unknown-exposure.json').write_text(json.dumps(exposure,indent=2),encoding='utf-8')
    passed=all(summary[str(c)]['joint_high']['precision'] is not None and summary[str(c)]['joint_high']['precision']>=.99 for c in (1,2))
    decision={'local_precision_gate_passed':passed,'deployment_authorized':False,'independent_calibration':False,'new_training':False,'runtime_changed':False}
    (out/'entry-decision.json').write_text(json.dumps(decision,indent=2),encoding='utf-8')


if __name__=='__main__':main()
