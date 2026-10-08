"""Automatic masked candidates, unknown exposure and narrow offline replay.

Only orphan wire extensions into a single existing net may change the output.
All existing net pin memberships are protected. Fixed diagnostic weights are
uncalibrated: this is a research ablation, never a commit authorization.
"""
import copy,json,math,sys
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import cv2,numpy as np,onnxruntime as ort
from pcbparse.dataset import index_dataset
from pcbparse.vision import read_image
from pcbparse.topology_perception import TopologyPerception,patch_tensor
from pcbparse.core import point_on_segment,evaluate
from pcbparse.competition import evaluate_qa
from experiments.port_partition import RELATIONS,port_membership


def main():
    out=ROOT/'output/port-partition-guarded-replay';out.mkdir(parents=True,exist_ok=True)
    protocol={'model':'expanded-seed0','marginal_min':.99,'logit_margin_min':math.log(99),
              'entropy_max':.15,'decision_scope':'old uncertain only',
              'protected_net_rule':'never merge two existing output nets; existing hyperGraph strings immutable',
              'size_guard_max_incident_wire_segments':8,'permission':'offline exploratory ablation, uncalibrated weights, no deployment',
              'split_only':'uncertain is already a contact veto; high-confidence old junctions cannot be split',
              'Freeze_A_used':False}
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    topology=TopologyPerception(ROOT/'models/topology/topology-perception.onnx')
    opt=ort.SessionOptions();opt.intra_op_num_threads=2
    sess=ort.InferenceSession(str(ROOT/'models/candidates/port-partition-expanded/expanded-seed0/port-graph/model.onnx'),sess_options=opt,providers=['CPUExecutionProvider'])
    index={r['id']:r for r in index_dataset(ROOT/'data/raw/PCBnet/Dataset300')}
    summary={}
    for domain in ('gt10','pcb55'):
        records=[];totals=Counter()
        for folder in sorted((ROOT/'output/designator-stable-catalog'/domain).iterdir()):
            if not (folder/'observations.json').exists():continue
            obs=json.loads((folder/'observations.json').read_text(encoding='utf-8'))
            before=json.loads((folder/'prediction.json').read_text(encoding='utf-8'));case=folder.name
            path=ROOT/'data/raw/official/gt-0930/10GT'/case/(case+'.png') if domain=='gt10' else Path(index[case]['image'])
            image=read_image(path);h=image.shape[0]
            boxes=[[a,h-d,c,h-b] for a,b,c,d in obs['masked_symbol_boxes']]
            texts=[dict(t,bbox=[t['bbox'][0],h-t['bbox'][3],t['bbox'][2],h-t['bbox'][1]]) for t in obs.get('texts',[])]
            _,_,_,report=topology.perceive(image,boxes,texts)
            candidates=report['crossings'];pred=copy.deepcopy(before);decisions=[]
            if candidates:
                inputs=np.stack([patch_tensor(image,c['x'],c['y'])[:3] for c in candidates])
                logits=np.concatenate([sess.run(None,{'image':part})[0] for part in np.array_split(inputs,max(1,int(np.ceil(len(inputs)/128))))])
                weights=np.exp(logits-logits.max(1,keepdims=True));weights/=weights.sum(1,keepdims=True)
                marginals=weights@RELATIONS
            # Weak labels are opened only after all predictions.
            segments=[]
            if domain=='pcb55':
                raw=json.loads(Path(index[case]['annotation']).read_text(encoding='utf-8'))
                for group,edge in raw['edges'].items():
                    if edge.get('state') not in (None,'active') or edge.get('isFlywire'):continue
                    for wirepath in edge.get('keyPointPaths',{}).get('wire',[]):
                        segments.extend((group,a,b) for a,b in zip(wirepath,wirepath[1:]) if a!=b)
            bounds=np.asarray([[min(a[0],b[0]),min(a[1],b[1]),max(a[0],b[0]),max(a[1],b[1])] for _,a,b in segments]).reshape(-1,4)
            count=Counter(candidates=len(candidates))
            for j,c in enumerate(candidates):
                x,y=c['x'],c['y'];point={'x':x,'y':h-y}
                known=np.zeros(6);target=np.zeros(6)
                if len(bounds):
                    nearby=np.flatnonzero((bounds[:,0]<=x+8)&(bounds[:,2]>=x-8)&(bounds[:,1]<=y+8)&(bounds[:,3]>=y-8))
                    target,known,_=port_membership([segments[i] for i in nearby],[x,y])
                edges=(1,2,3,4);high_edges=[k for k in edges if marginals[j,k]>=.99]
                unknown_high=[k for k in high_edges if not known[k]]
                count['unknown_high_merge_relations']+=len(unknown_high)
                count['unknown_high_merge_candidates']+=bool(unknown_high)
                sorted_scores=np.sort(logits[j]);margin=float(sorted_scores[-1]-sorted_scores[-2])
                entropy=float(-(weights[j]*np.log(np.maximum(weights[j],1e-12))).sum())
                reason='old_decision_protected';changed=False
                if c['state']=='uncertain':
                    count['old_uncertain']+=1
                    if high_edges and margin>=math.log(99) and entropy<=.15:
                        count['high_margin_merge_proposals']+=1
                        incident=[i for i,w in enumerate(obs['wires']) if point_on_segment(point,w,3)]
                        netkeys=[key for key,net in pred['nets'].items() if any(point_on_segment(point,e,3) for e in net['edges'].values())]
                        if len(netkeys)>1:reason='protected_old_net_bridge';count[reason]+=1
                        elif len(incident)>8:reason='merge_size_guard';count[reason]+=1
                        elif not netkeys:reason='no_single_existing_net_target';count[reason]+=1
                        else:
                            net=pred['nets'][netkeys[0]];existing={json.dumps(e,sort_keys=True) for n in pred['nets'].values() for e in n['edges'].values()}
                            additions=[w for i,w in enumerate(obs['wires']) if i in incident and json.dumps(w,sort_keys=True) not in existing]
                            for wire in additions:
                                net['edges']['ppn_extension_'+str(len(net['edges'])+1)]=wire
                            changed=bool(additions);reason='orphan_wire_extension' if changed else 'no_new_wire';count[reason]+=1
                    else:reason='fixed_diagnostic_abstention'
                decisions.append({'point_xy':[x,y],'legacy_state':c['state'],'marginal':marginals[j].tolist(),
                                  'margin':margin,'entropy':entropy,'known':known.tolist(),'weak_target':target.tolist(),
                                  'unknown_high_merge_relations':unknown_high,'reason':reason,'output_changed':changed})
            assert pred['components']==before['components'] and pred['pins']==before['pins']
            assert {k:n['hyperGraph'] for k,n in pred['nets'].items()}=={k:n['hyperGraph'] for k,n in before['nets'].items()}
            truth=json.loads((path.with_suffix('.json') if domain=='gt10' else ROOT/'output/benchmark-recognition-pcb55'/case/'golden.json').read_text(encoding='utf-8'))
            scorer=evaluate_qa if domain=='gt10' else evaluate
            base=scorer(before,truth);after=scorer(pred,truth)
            rec={'case':case,'counts':dict(count),'baseline':base,'split_only':base,'merge_only':after,'both':after}
            records.append(rec);totals.update(count)
            dest=out/domain/case;dest.mkdir(parents=True,exist_ok=True)
            (dest/'prediction.json').write_text(json.dumps(pred),encoding='utf-8')
            (dest/'decisions.json').write_text(json.dumps(decisions,indent=2),encoding='utf-8')
            print(json.dumps({'domain':domain,'case':case,**dict(count)}),flush=True)
        keys=('ComponentF1','PinF1','NetHypergraphF1','NetLineF1')
        delta={k:sum(r['merge_only'][k]-r['baseline'][k] for r in records)/len(records) for k in keys}
        result={'pages':len(records),'counts':dict(totals),'delta':delta,'records':records,
                'improve_pages':sum(r['merge_only']['NetLineF1']>r['baseline']['NetLineF1'] for r in records),
                'regress_pages':sum(r['merge_only']['NetLineF1']<r['baseline']['NetLineF1'] for r in records),
                'gate_passed':False,'reason':'Net memberships protected; no calibrated precision or independent evidence. Not full-net correction.',
                'runtime_changed':False}
        summary[domain]=result
        (out/domain/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    (out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')


if __name__=='__main__':main()
