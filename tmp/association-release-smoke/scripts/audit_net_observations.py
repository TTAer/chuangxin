"""Post-inference net fragmentation and upstream-observation diagnostics.

Geometry oracle matches are for attribution only and never feed inference.
These flags overlap; they are not causal labels or official evaluator matches.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from pcbparse.__main__ import load,save
from pcbparse.core import assignment,distance,references
from pcbparse.competition import canonical_type
from pcbparse.vision import iou

def audit(pred,truth,obs):
    pk=list(pred['components']); gk=list(truth['components'])
    weights=[[iou(pred['components'][p]['bbox'],truth['components'][g]['bbox'])
              if canonical_type(pred['components'][p]['type'])==canonical_type(truth['components'][g]['type']) else 0 for g in gk] for p in pk]
    pairs=[(pk[i],gk[j]) for i,j in assignment(weights) if weights[i][j]>.2]
    maps={5:{},30:{}}
    for p,g in pairs:
        pp=pred['pins'].get(p,{}); gp=truth['pins'].get(g,{})
        pn=list(pp); gn=list(gp)
        for limit in maps:
            weights=[[max(0,limit-distance(pp[n]['point'],gp[m]['point'])) for m in gn] for n in pn]
            for i,j in assignment(weights):
                if weights[i][j]>0: maps[limit][p+'.'+pn[i][4:]]=g+'.'+gn[j][4:]
    raw=[references(n['hyperGraph']) for n in pred['nets'].values()]
    mapped={limit:[{mapping[p] for p in net if p in mapping} for net in raw] for limit,mapping in maps.items()}
    all_near=set(maps[30].values()); all_strict=set(maps[5].values())
    gt_nets={k:references(n['hyperGraph']) for k,n in truth['nets'].items()}
    records=[]
    for key,gold in gt_nets.items():
        fragments=[i for i,net in enumerate(mapped[30]) if net&gold]
        best=max((mapped[30][i] for i in fragments),key=lambda n:len(n&gold),default=set())
        recovered=gold&all_near; flags=[]
        if gold-all_near: flags.append('potential_upstream_missing_or_remote_endpoint')
        if (gold&all_near)-all_strict: flags.append('potential_upstream_endpoint_precision')
        # Identity mismatch only after a geometric oracle establishes a pair.
        wrong_identity=[p for p,g in maps[30].items() if g in gold and p!=g and not truth['components'][g.rsplit('.',1)[0]]['type'] in ('gnd','v','vcc')]
        if wrong_identity: flags.append('potential_upstream_identity')
        if len(fragments)>1 and len(recovered)>=2: flags.append('potential_split')
        merged=[other for other,members in gt_nets.items() if other!=key and best&members]
        if merged: flags.append('potential_merge')
        if gold and gold==best and gold<=all_strict: flags.append('geometry_oracle_full_membership')
        if not flags: flags.append('unresolved')
        records.append({'net':key,'gold_members':len(gold),'near_endpoint_members':len(recovered),
                        'strict_endpoint_members':len(gold&all_strict),'predicted_fragments':len(fragments),
                        'best_fragment_overlap':len(best&gold),'other_gt_nets_in_best_fragment':merged,
                        'identity_mismatches':wrong_identity,'flags':flags})
    # Explicitly observable same-label split; not every split is a flying net.
    label_groups={}
    for label in obs.get('labels',[]): label_groups.setdefault(label['name'],[]).append(label['pin'])
    fly=[]
    for name,pins in label_groups.items():
        membership=[{i for i,net in enumerate(raw) if pin in net} for pin in pins]
        if len(pins)>1 and all(membership) and not set.intersection(*membership): fly.append({'label':name,'pins':pins})
    return {'nets':records,'counts':dict(Counter(flag for r in records for flag in r['flags'])),
            'same_label_split_evidence':fly,'wrong_junction':'not identifiable from this membership-only audit; needs spatial GT junction evidence'}

def main():
    p=argparse.ArgumentParser(); p.add_argument('--dataset',choices=['gt10','pcb55'],required=True); a=p.parse_args()
    official=a.dataset=='gt10'; base=ROOT/'output'/('benchmark-recognition-gt10' if official else 'benchmark-recognition-pcb55'); rows=[]; totals=Counter()
    for folder in sorted(base.iterdir()):
        if not (folder/'prediction.json').exists(): continue
        truth=ROOT/'data/raw/official/gt-0930/10GT'/folder.name/(folder.name+'.json') if official else folder/'golden.json'
        r=audit(load(folder/'prediction.json'),load(truth),load(folder/'observations.json')); totals.update(r['counts']); rows.append(dict(case=folder.name,**r))
    report={'dataset':a.dataset,'cases':len(rows),'counts':dict(totals),'records':rows,
            'method':'same-type IoU > .2 component oracle; one-to-one pin position oracle at <5 / <30 pixels; overlapping diagnostic flags',
            'causal_proof':False,'official_evaluator_matching':False,'used_for_inference':False,
            'limitations':['incomplete legacy annotations','coordinate aliases on power pins','oracle identity does not prove electrical topology','junction error cannot be inferred from membership alone']}
    save(ROOT/'output/net-observation-audit'/(a.dataset+'.json'),report); print(json.dumps({k:v for k,v in report.items() if k!='records'}),flush=True)
if __name__=='__main__': main()
