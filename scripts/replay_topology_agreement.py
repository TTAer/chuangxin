"""Isolated graph-rebuild ablation with immutable components/pins and flying labels.

No published prediction is modified. Unlike orphan extensions this rebuild can
change pin membership. Split needs a low-confidence old connection, two-axis
agreement, no visible dot and a crop free of detected symbol/OCR interference.
"""
import copy,json,sys
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import cv2,numpy as np
from pcbparse.dataset import index_dataset
from pcbparse.vision import read_image
from pcbparse.graphschematic import parse_graph
from pcbparse.core import references,point_on_segment,distance,evaluate,validate
from pcbparse.competition import evaluate_qa
from pcbparse.topology_perception import dot_evidence

CROSS=[1,2,3,4]


def canonical(data):
    return sorted((tuple(sorted(references(n['hyperGraph']))),tuple(sorted(json.dumps(e,sort_keys=True) for e in n['edges'].values()))) for n in data['nets'].values())


def mask_image(image,obs):
    h,w=image.shape[:2];mask=(cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)<220).astype(np.uint8)*255
    for a,b,c,d in obs['masked_symbol_boxes']+[t['bbox'] for t in obs.get('texts',[])]:
        a,b,c,d=map(int,(a,h-d,c,h-b));mask[max(0,b):min(h,d),max(0,a):min(w,c)]=0
    return mask


def rebuild(obs,mask,before):
    after,_=parse_graph(obs,line_mask=mask)
    # Body display boxes are applied after graph construction by frozen runtime.
    after['components']=copy.deepcopy(before['components'])
    assert after['pins']==before['pins']
    assert not validate(after)
    return after


def protected_merge(before,after):
    old=[references(n['hyperGraph']) for n in before['nets'].values()]
    new=[references(n['hyperGraph']) for n in after['nets'].values()]
    return any(sum(bool(a&b) for a in old)>1 for b in new)


def split_only_refinement(before,after):
    old=[references(n['hyperGraph']) for n in before['nets'].values()]
    new=[references(n['hyperGraph']) for n in after['nets'].values()]
    return set().union(*old)==set().union(*new) and all(any(b<=a for a in old) for b in new)


def main():
    out=ROOT/'output/topology-agreement-replay';out.mkdir(parents=True,exist_ok=True)
    entry=json.loads((ROOT/'output/topology-agreement/entry-decision.json').read_text(encoding='utf-8'))
    if not entry['local_precision_gate_passed']:raise ValueError('Local precision gate failed; do not replay')
    meta=json.loads((ROOT/'models/topology/topology-perception.json').read_text(encoding='utf-8'))
    protocol={'purpose':'uncalibrated diagnostic replay, not commit or deployment','models_frozen':True,'training':False,
        'marginal_threshold':.95,'split':'old connected with probability below frozen neural connect threshold; both models canonical LR/TB split; no dot; whole crop outside symbol/OCR boxes',
        'merge':'old uncertain; BOTH full connected legal partition at .95 on all six relations; no bridge between two existing nets',
        'guards':{'incident_wire_segments_max':8,'crop_radius':16,'existing_nets_protected_from_merge':True,'labels_preserved':True},
        'baseline':'rebuild frozen cached observations with frozen parse_graph and reconstructed mask; canonical nets must match baseline; body display boxes restored',
        'unknown':'exposure remains high and no independent calibration exists; this replay does not establish safe deployment','Freeze_A_used':False}
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    index={r['id']:r for r in index_dataset(ROOT/'data/raw/PCBnet/Dataset300')};summary={}
    for domain in ('gt10','pcb55'):
        totals=Counter();records=[]
        for folder in sorted((ROOT/'output/topology-agreement'/domain).iterdir()):
            if not (folder/'decisions.json').exists():continue
            case=folder.name;decisions=json.loads((folder/'decisions.json').read_text(encoding='utf-8'))
            cache=ROOT/'output/designator-stable-catalog'/domain/case
            obs=json.loads((cache/'observations.json').read_text(encoding='utf-8'));before=json.loads((cache/'prediction.json').read_text(encoding='utf-8'))
            path=ROOT/'data/raw/official/gt-0930/10GT'/case/(case+'.png') if domain=='gt10' else Path(index[case]['image'])
            image=read_image(path);h=image.shape[0];mask=mask_image(image,obs);count=Counter();operations=[]
            baseline=rebuild(obs,mask,before);reproduced=canonical(baseline)==canonical(before)
            count['baseline_reproduced' if reproduced else 'baseline_mismatch']+=1
            variants={};reports={}
            for mode in ('split_only','merge_only','both'):
                working=copy.deepcopy(obs);pred=copy.deepcopy(before);log=[]
                if reproduced:
                    for item in decisions:
                        point={'x':item['point_xy'][0],'y':h-item['point_xy'][1]};p=np.array(item['PPN_weights']);m=np.array(item['MST_weights'])
                        split=item['legacy_state']=='connected' and item['canonical_joint_split'] and mode!='merge_only'
                        merge=item['legacy_state']=='uncertain' and (p>=.95).all() and (m>=.95).all() and mode!='split_only'
                        if not split and not merge:continue
                        reason=None
                        if split:
                            old=next((d for d in obs.get('contact_decisions',[]) if distance(point,d['point'])<=1),None)
                            if old is None:reason='missing_old_decision'
                            elif old.get('probability',1)>=meta['connect_threshold']:reason='confident_RC0_protected'
                            elif any(a-16<=point['x']<=c+16 and b-16<=point['y']<=d+16 for a,b,c,d in obs['masked_symbol_boxes']+[t['bbox'] for t in obs.get('texts',[])]):reason='symbol_OCR_crop_interference'
                            elif dot_evidence(mask,*item['point_xy']):reason='visible_dot_protected'
                        incident=[w for w in obs['wires'] if point_on_segment(point,w,3)]
                        if reason is None and len(incident)>8:reason='incident_size_guard'
                        if merge and reason is None:
                            touched=[k for k,n in pred['nets'].items() if any(point_on_segment(point,e,3) for e in n['edges'].values())]
                            if len(touched)>1:reason='protected_existing_net_bridge'
                        proposal=copy.deepcopy(working)
                        if reason is None:
                            proposal['junctions']=[q for q in proposal.get('junctions',[]) if distance(point,q)>3]
                            proposal['contact_decisions']=[d for d in proposal.get('contact_decisions',[]) if distance(point,d['point'])>3]
                            proposal['contact_decisions'].append({'point':point,'state':'uncertain' if split else 'connected','evidence':'offline_frozen_model_agreement'})
                            if merge:proposal['junctions'].append(point)
                            candidate=rebuild(proposal,mask,before)
                            if merge and protected_merge(pred,candidate):reason='global_protected_net_bridge'
                            elif split and not split_only_refinement(pred,candidate):reason='split_changed_unrelated_merge_or_pin_coverage'
                            else:
                                changed=canonical(pred)!=canonical(candidate);working=proposal;pred=candidate
                                reason=('split_applied' if split else 'merge_applied') if changed else 'graph_no_change'
                        log.append({'point_xy':item['point_xy'],'operation':'split' if split else 'merge','reason':reason,'known':item['known'],'weak_target':item['weak_target'],'commit':False})
                variants[mode]=pred;reports[mode]=log
                for operation in log:count[mode+'_'+operation['reason']]+=1
            truth=json.loads((path.with_suffix('.json') if domain=='gt10' else ROOT/'output/benchmark-recognition-pcb55'/case/'golden.json').read_text(encoding='utf-8'))
            scorer=evaluate_qa if domain=='gt10' else evaluate
            row={'case':case,'baseline_reproduced':reproduced,'counts':dict(count),'baseline':scorer(before,truth),**{mode:scorer(pred,truth) for mode,pred in variants.items()}}
            records.append(row);totals.update(count);dest=out/domain/case;dest.mkdir(parents=True,exist_ok=True)
            for mode,pred in variants.items():(dest/(mode+'.json')).write_text(json.dumps(pred),encoding='utf-8')
            (dest/'operations.json').write_text(json.dumps(reports,indent=2),encoding='utf-8')
            print(json.dumps({'domain':domain,'case':case,**dict(count)}),flush=True)
        modes={}
        for mode in ('split_only','merge_only','both'):
            delta={key:sum(r[mode][key]-r['baseline'][key] for r in records)/len(records) for key in ('ComponentF1','PinF1','NetHypergraphF1','NetLineF1')}
            improve=sum(r[mode]['NetHypergraphF1']>r['baseline']['NetHypergraphF1']+1e-12 for r in records)
            regress=sum(r[mode]['NetHypergraphF1']<r['baseline']['NetHypergraphF1']-1e-12 for r in records)
            any_improve=sum(any(r[mode][k]>r['baseline'][k]+1e-12 for k in ('NetHypergraphF1','NetLineF1')) for r in records)
            any_regress=sum(any(r[mode][k]<r['baseline'][k]-1e-12 for k in ('NetHypergraphF1','NetLineF1')) for r in records)
            modes[mode]={'delta':delta,'net_improve_pages':improve,'net_regress_pages':regress,'any_topology_improve_pages':any_improve,'any_topology_regress_pages':any_regress,
                         'system_gate_passed':delta['NetHypergraphF1']>0 and delta['NetLineF1']>=0 and any_regress<any_improve}
        summary[domain]={'pages':len(records),'counts':dict(totals),'modes':modes,'records':records,'runtime_changed':False,'deployment_authorized':False}
        (out/domain/'summary.json').write_text(json.dumps(summary[domain],indent=2),encoding='utf-8')
    (out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')


if __name__=='__main__':main()
