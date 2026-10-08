"""Image-derived input tokens first, then development-only weak target audit.

No training or Freeze-A use. Exact identity/geometry compatibility is a proxy
screen, not human certification. Inputs never include GT net IDs/membership.
"""
import json,sys
from collections import Counter,defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from pcbparse.core import distance,point_on_segment,references
from pcbparse.competition import canonical_type


def tokens(obs):
    width,height=obs['image']['width'],obs['image']['height'];refs=[];pins=[]
    for c,ps in obs['pins'].items():
        a,b,d,e=obs['components'][c]['bbox'];cx,cy=(a+d)/2,(b+e)/2
        for k,p in ps.items():
            x,y=p['point']['x'],p['point']['y'];ref=c+'.'+k[4:]
            incident=sum(point_on_segment(p['point'],edge,3) for edge in obs.get('wires',[]))
            near=[r for r in obs.get('contact_decisions',[]) if distance(p['point'],r['point'])<=3]
            # Category indicators and geometry only; no component/pin/net IDs.
            kind=canonical_type(obs['components'][c]['type'])
            pins.append([x/width,y/height,(x-cx)/width,(y-cy)/height,(d-a)/width,(e-b)/height,
                         min(incident,8)/8,float(any(r['state']=='uncertain' for r in near)),
                         float(kind=='box'),float(kind in ('gnd','v')),float(kind in ('r','c','l')),1.])
            refs.append(ref)
    wires=[]
    for edge in obs.get('wires',[]):
        a,b=sorted(edge,key=lambda p:(p['x'],p['y']));dx=(b['x']-a['x'])/width;dy=(b['y']-a['y'])/height
        wires.append([(a['x']+b['x'])/2/width,(a['y']+b['y'])/2/height,dx,dy,
                      min(a['x'],b['x'])/width,min(a['y'],b['y'])/height,max(a['x'],b['x'])/width,max(a['y'],b['y'])/height,
                      float(abs(dy)<1e-6),float(abs(dx)<1e-6),min(1,distance(a,b)/max(width,height)),1.])
    return refs,np.array(pins,np.float32).reshape(-1,12),np.array(wires,np.float32).reshape(-1,12)


def targets(obs,truth,refs):
    owners=defaultdict(list)
    for key,n in truth['nets'].items():
        for ref in references(n['hyperGraph']):owners[ref].append(key)
    labels=[];groups={};reasons=Counter()
    for ref in refs:
        c,n=ref.rsplit('.',1);p=obs['pins'][c]['pin_'+n];gold=truth['pins'].get(c,{}).get('pin_'+n)
        if gold is None:reason='identity_missing'
        elif canonical_type(obs['components'][c]['type'])!=canonical_type(truth['components'][c]['type']):reason='type_conflict'
        elif distance(p['point'],gold['point'])>5:reason='coordinate_incompatible'
        elif canonical_type(truth['components'][c]['type'])=='box' and p['pinname']!=gold['pinname']:reason='box_name_conflict'
        elif len(owners[ref])!=1:reason='missing_or_multiple_net_owner'
        else:reason='weak_compatible_identity_and_geometry'
        reasons[reason]+=1
        if reason.startswith('weak_'):
            net=owners[ref][0];groups.setdefault(net,len(groups));labels.append(groups[net])
        else:labels.append(-1)
    return np.array(labels,np.int64),dict(reasons)


def main():
    out=ROOT/'data/processed/netset-query-prototype';out.mkdir(parents=True,exist_ok=True)
    if (out/'audit.json').exists():raise ValueError('Preserve prototype audit; use separate output for rerun')
    rows=[];totals=Counter()
    for domain in ('gt10','pcb55'):
        for folder in sorted((ROOT/'output/designator-stable-catalog'/domain).iterdir()):
            if not (folder/'observations.json').exists():continue
            obs=json.loads((folder/'observations.json').read_text(encoding='utf-8'));refs,pins,wires=tokens(obs)
            dest=out/domain/folder.name;dest.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(dest/'inputs.npz',pins=pins,wires=wires)
            (dest/'evidence.json').write_text(json.dumps({'references_for_output_alignment_only':refs,'semantic_label_constraints':obs.get('labels',[]),'coordinate_system':'bottom-left normalized by image size','GT_used_in_inputs':False}),encoding='utf-8')
            truth_path=ROOT/'data/raw/official/gt-0930/10GT'/folder.name/(folder.name+'.json') if domain=='gt10' else ROOT/'output/benchmark-recognition-pcb55'/folder.name/'golden.json'
            truth=json.loads(truth_path.read_text(encoding='utf-8'));labels,reasons=targets(obs,truth,refs)
            np.savez_compressed(dest/'weak-targets.npz',labels=labels)
            known=int((labels>=0).sum());observed=len(set(labels.tolist())-{-1})
            row={'domain':domain,'case':folder.name,'input_pins':len(pins),'input_wires':len(wires),'weak_compatible_pins':known,
                 'unknown_pins':len(pins)-known,'observed_partial_nets':observed,'net_query_slots_needed':observed,'reasons':reasons}
            rows.append(row);totals.update({k:row[k] for k in ('input_pins','input_wires','weak_compatible_pins','unknown_pins','observed_partial_nets')})
    summary={'architecture':'NetSetQueryDecoder','pages':len(rows),'totals':dict(totals),'rows':rows,
             'trained':False,'independent_validation':False,'Freeze_A_used':False,'runtime_changed':False,
             'supervision':'existing development GT screened by exact identity/type and <=5px geometry; remains weak and partial, not trusted human labels',
             'input_target_separation':True,'limits':['identity coverage constrains recoverable net quality','unknown does not mean singleton or disconnected','existing 65 pages are development only, never a training/independent holdout claim','wire ownership targets and visible text tokens not implemented in this prototype']}
    (out/'audit.json').write_text(json.dumps(summary,indent=2),encoding='utf-8');print(json.dumps({'pages':len(rows),'totals':dict(totals),'maximum_observed_partial_nets':max(r['observed_partial_nets'] for r in rows)}))


if __name__=='__main__':main()
