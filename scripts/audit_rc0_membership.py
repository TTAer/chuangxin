"""Observe net membership transactions without changing frozen RC0 inference."""
from collections import Counter
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from pcbparse.__main__ import load,save
from pcbparse.core import references,assignment,distance,evaluate
from pcbparse.graphschematic import DualGraph
from pcbparse.vision import iou

def edge_key(edge): return tuple(sorted((round(p['x'],4),round(p['y'],4)) for p in edge))

def audit(before,after,old_obs,new_obs):
    pk=list(before['components']); nk=list(after['components'])
    weights=[[iou(before['components'][p]['bbox'],after['components'][n]['bbox']) if before['components'][p]['type']==after['components'][n]['type'] else 0 for n in nk] for p in pk]
    common={nk[j]:pk[i] for i,j in assignment(weights) if weights[i][j]>=.95}
    def normalize(ref):
        c,n=ref.rsplit('.',1); return common.get(c,'__new__'+c)+'.'+n
    base_nets=[references(n['hyperGraph']) for n in before['nets'].values()]
    base_edges={edge_key(e) for n in before['nets'].values() for e in n['edges'].values()}
    wires_only=dict(new_obs,components={},pins={},labels=[],pin_hypotheses=[])
    physical=DualGraph(wires_only); root_for_edge={edge_key(e):physical.uf.root(i) for i,e in enumerate(physical.wires)}
    old_refs={p for n in base_nets for p in n}; rows=[]
    for key,net in after['nets'].items():
        refs={normalize(r) for r in references(net['hyperGraph'])}; new=refs-old_refs
        groups=[i for i,b in enumerate(base_nets) if refs & b]
        additions=[e for e in net['edges'].values() if edge_key(e) not in base_edges]
        if not new and len(groups)<=1 and not additions: continue
        unknown=[r for r in new if r.startswith('__new__') and new_obs['components'][r.split('.',1)[0][7:]].get('Name') is None]
        rows.append({'net':key,'new_members':sorted(new),'previous_nets_touched':groups,'merged_previous_net_count':max(0,len(groups)-1),
                     'newly_exported_edges':len(additions),'previously_unassigned_wiregroups':sorted({root_for_edge.get(edge_key(e),'unmapped') for e in additions},key=str),
                     'new_unknown_name_members':unknown,'diagnostic_risk':len(additions)+2*max(0,len(groups)-1)+len(unknown)})
    return {'nets':rows,'merged_previous_nets':sum(r['merged_previous_net_count'] for r in rows),'newly_exported_edges':sum(r['newly_exported_edges'] for r in rows),'case_rule_used':False}

def main():
    rows=[]
    for domain in ('gt10','pcb55'):
        for folder in sorted((ROOT/'output/designator-stable-catalog'/domain).iterdir()):
            if not (folder/'prediction.json').exists(): continue
            prior=ROOT/('output/ic-identity-'+domain+'-side')/folder.name
            record=audit(load(prior/'prediction.json'),load(folder/'prediction.json'),load(prior/'observations.json'),load(folder/'observations.json'))
            rows.append(dict(domain=domain,case=folder.name,**record))
    save(ROOT/'output/ablation/rc0-membership-audit.json',{'rows':rows,'diagnostic_only':True,'no_inference_changes':True,'ground_truth_used':False,'risk_is_not_calibrated':True})
    # Exact-key strict proxy path for the known development LED example.
    base=ROOT/'output/benchmark-recognition-pcb55/0197'; truth=load(base/'golden.json'); versions={}
    for name,path in [('old_combo',ROOT/'output/ic-identity-pcb55-side/0197'),('guarded',ROOT/'output/designator-guarded/pcb55/0197'),('rc0',ROOT/'output/designator-stable-catalog/pcb55/0197')]:
        pred=load(path/'prediction.json'); pins=[]
        for c,pp in pred['pins'].items():
            for n,p in pp.items():
                g=truth['pins'].get(c,{}).get(n); strict=bool(g and g['pinname']==p['pinname'] and distance(g['point'],p['point'])<=5)
                if c.startswith('LED'): pins.append({'component':c,'pin':n,'name':p['pinname'],'gt_same_number_exists':bool(g),'strict_proxy_tp':strict})
        versions[name]={'metrics':evaluate(pred,truth),'led_pin_paths':pins}
    save(ROOT/'output/ablation/rc0-0197-scoring-path.json',{'versions':versions,'deleted_invalid_pin_numbers_cannot_be_strict_proxy_tp':True,'gt_used_only_after_inference':True,'net_assignment_metric_is_not_geometry_signature':True})
    print({'pages':len(rows),'transactions':sum(len(r['nets']) for r in rows)},flush=True)
if __name__=='__main__': main()
