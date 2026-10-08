"""Same-candidate legacy baseline and weak-label FP stability audit."""
import json,hashlib,sys
from pathlib import Path
from collections import Counter
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import cv2,numpy as np,onnxruntime as ort
from pcbparse.dataset import index_dataset
from pcbparse.vision import read_image,wire_lines
from pcbparse.topology_perception import crossing_candidates,dot_evidence,three_arm_evidence
from experiments.port_partition import RELATIONS,PAIRS


def stats(pred,target,valid):
    mask=valid.astype(bool).copy();mask[:,[0,5]]=False
    tp=int(((pred==1)&(target==1)&mask).sum());fp=int(((pred==1)&(target==0)&mask).sum());fn=int(((pred==0)&(target==1)&mask).sum())
    return {'tp':tp,'fp':fp,'fn':fn,'f1':2*tp/max(1,2*tp+fp+fn),'merge_precision':tp/(tp+fp) if tp+fp else None}


def main():
    out=ROOT/'output/port-partition-release-audit';out.mkdir(parents=True,exist_ok=True)
    previous=json.loads((out/'same-candidate-audit.json').read_text(encoding='utf-8')) if (out/'same-candidate-audit.json').exists() else {}
    notes={r['index']:r for r in previous.get('error_records',[])}
    with np.load(ROOT/'data/processed/port-partition-expanded/relations.npz') as archive:z={k:archive[k] for k in archive.files}
    rows=json.loads((ROOT/'data/processed/port-partition-expanded/records.json').read_text(encoding='utf-8'))
    ids=np.flatnonzero(z['split']!=0);images=z['images'][ids].astype(np.float32)/255
    oldrecords=json.loads((ROOT/'data/processed/topology/records.json').read_text(encoding='utf-8'))
    with np.load(ROOT/'data/processed/topology/patches.npz') as ar:g=ar['geometry']
    geometry={(r['page'],tuple(r['point'])):g[i] for i,r in enumerate(oldrecords)}
    index={r['id']:r for r in index_dataset(ROOT/'data/raw/PCBnet/Dataset300')}
    opt=ort.SessionOptions();opt.intra_op_num_threads=2
    pp=[];marginals=[]
    for seed in range(3):
        model=ROOT/f'models/candidates/port-partition-expanded/expanded-seed{seed}/port-graph/model.onnx'
        sess=ort.InferenceSession(str(model),sess_options=opt,providers=['CPUExecutionProvider'])
        logits=np.concatenate([sess.run(None,{'image':part})[0] for part in np.array_split(images,7)])
        p=np.exp(logits-logits.max(1,keepdims=True));p/=p.sum(1,keepdims=True)
        pp.append(RELATIONS[logits.argmax(1)]);marginals.append(p@RELATIONS)
    legacy=ort.InferenceSession(str(ROOT/'models/topology/topology-perception.onnx'),sess_options=opt,providers=['CPUExecutionProvider'])
    meta=json.loads((ROOT/'models/topology/topology-perception.json').read_text(encoding='utf-8'))
    cache={};legacy_predictions=[];states=[];errors=[];fp_sets=[set() for _ in range(3)]
    for j,i in enumerate(ids):
        r=rows[i];page=r['page'];point=r['point']
        if page not in cache:
            image=read_image(index[page]['image']);mask=(cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)<220).astype(np.uint8)*255
            lines,_=wire_lines(mask,split_junctions=False)
            mapping={(c['x'],c['y']):c for c in crossing_candidates(lines)}
            cache[page]=(mask,mapping)
        mask,mapping=cache[page];candidate=mapping.get(tuple(point))
        geom=geometry.get((page,tuple(point))) if z['split'][i]==1 else (np.array(candidate['geometry'],np.float32) if candidate else None)
        if geom is None:raise ValueError('Candidate geometry unavailable')
        rgb=images[j].transpose(1,2,0);gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY);ink=(gray<.86).astype(np.float32)
        tensor=np.concatenate([images[j],gray[None],cv2.morphologyEx(ink,cv2.MORPH_OPEN,np.ones((1,7),np.uint8))[None],cv2.morphologyEx(ink,cv2.MORPH_OPEN,np.ones((7,1),np.uint8))[None]])
        logit=legacy.run(None,{'patch':tensor[None],'geometry':np.asarray(geom,np.float32)[None]})[0].reshape(-1)[0]
        probability=float(1/(1+np.exp(-np.clip(logit,-30,30))))
        state='connected' if probability>=meta['connect_threshold'] else 'disconnected' if probability<=meta['disconnect_threshold'] else 'uncertain'
        endpoint=bool(np.min(geom)*32<=3)
        if state=='uncertain' and ((endpoint and three_arm_evidence(mask,*point) and probability>=.2) or (dot_evidence(mask,*point) and probability>=.5)):state='connected'
        old=np.ones(6) if state=='connected' else np.array([1,0,0,0,0,1])
        legacy_predictions.append(old);states.append(state)
        for seed in range(3):
            for pair in (1,2,3,4):
                if z['valid'][i,pair] and not z['targets'][i,pair] and marginals[seed][j,pair]>=.95:
                    fp_sets[seed].add((int(i),pair))
        if any((int(i),p) in s for s in fp_sets for p in (1,2,3,4)):
            crop=out/'crops'/f'{i}.png';crop.parent.mkdir(exist_ok=True)
            cv2.imencode('.png',z['images'][i].transpose(1,2,0)[:,:,::-1])[1].tofile(str(crop))
            errors.append({'index':int(i),'page':page,'point':point,'cohort':int(z['split'][i]),'crop':str(crop.relative_to(ROOT)),
                           'target':z['targets'][i].tolist(),'valid':z['valid'][i].tolist(),
                           'marginal_weights':[m[j].tolist() for m in marginals],
                           'source_wire_groups':r['port_wire_groups'],'legacy_state':state,
                           'taxonomy':'pending_pixel_evidence_review','human_reviewed':False})
    old=np.asarray(legacy_predictions);summary={}
    for code in (1,2):
        pick=z['split'][ids]==code;target=z['targets'][ids[pick]];valid=z['valid'][ids[pick]]
        disagreement={}
        for state in ('connected','disconnected','uncertain'):
            select=pick & (np.asarray(states)==state)
            connect=(pp[0].sum(1)>2)
            disagreement[state]={'samples':int(select.sum()),'PPN_connect_samples':int((select&connect).sum()),
                                 'PPN_no_cross_merge_samples':int((select&~connect).sum())}
            disagreement[state]['known_cross_relation_metrics']=stats(pp[0][select],z['targets'][ids[select]],z['valid'][ids[select]])
        summary[str(code)]={'legacy_conservative_merge':stats(old[pick],target,valid),
                            'PPN_MAP_seed0':stats(pp[0][pick],target,valid),
                            'legacy_state_counts':dict(Counter(np.asarray(states)[pick])),
                            'disagreements':disagreement,
                            'high_weight_FP_overlap':{'per_seed':[len({k for k in s if z['split'][k[0]]==code}) for s in fp_sets],
                                                      'union':len({k for k in set.union(*fp_sets) if z['split'][k[0]]==code}),
                                                      'intersection':len({k for k in set.intersection(*fp_sets) if z['split'][k[0]]==code})}}
    for error in errors:
        for key in ('taxonomy','agent_pixel_observation'):
            if key in notes.get(error['index'],{}):error[key]=notes[error['index']][key]
    false_negatives=[]
    for code in (1,2):
        count=0
        for j,i in enumerate(ids):
            pairs=[p for p in (1,2,3,4) if z['valid'][i,p] and z['targets'][i,p] and not pp[0][j,p]]
            if z['split'][i]!=code or not pairs:continue
            r=rows[i];crop=out/'crops'/f'fn-{i}.png'
            cv2.imencode('.png',z['images'][i].transpose(1,2,0)[:,:,::-1])[1].tofile(str(crop))
            false_negatives.append({'index':int(i),'page':r['page'],'point':r['point'],'cohort':code,'pairs':pairs,'crop':str(crop.relative_to(ROOT)),'human_reviewed':False,'selection':'first six candidate rows in cohort order; illustrative, not representative prevalence estimate'})
            count+=1
            if count==6:break
    result={'scope':'same-candidate module comparison; original geometry from cached candidate extraction, auto geometry from raw pixels; dot/T context uses unmasked raw ink, NOT full RC0 masked inference',
            'legacy_uncertain_policy':'no merge, not an explicit split prediction','thresholds':meta,
            'summary':summary,'error_records':errors,'illustrative_false_negatives':false_negatives,'human_reviews':0,'runtime_changed':False}
    (out/'same-candidate-audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
