"""Category-constrained global association of OCR designators to symbols."""
import re
from .core import assignment
from .dataset import DESIGNATOR

# Categories whose output identity is a net label must not consume designators.
PREFIXES = {'R': ('R',), 'C': ('C',), 'L': ('L',),
            'Box': ('U', 'IC', 'J', 'P', 'CN'),
            'Block': ('U', 'IC', 'J', 'P', 'CN'),
            'D': ('D',), 'LED': ('D', 'LED', 'PWR'),
            'Crystal': ('X', 'Y'), 'Switch': ('SW', 'K'),
            'MOSFET': ('Q',), 'BJT': ('Q',), 'Fuse': ('F',),
            'PIN': ('J', 'P', 'CN'), 'Jumper': ('J', 'JP'),
            'AMP': ('U', 'IC'), 'Transformer': ('T',), 'Buzzer': ('BZ',)}


def associate(detections, texts, mode='legacy'):
    from .vision import box_distance, compatible
    if mode=='stable':
        return associate_stable(detections,texts)
    if mode not in ('legacy', 'typed', 'boxed', 'guarded'):
        raise ValueError('unsupported designator association mode')
    designators = [t for t in texts if DESIGNATOR.fullmatch(t['text']) or
                   (mode in ('boxed','guarded') and re.fullmatch(r'USB\d+[A-Za-z]?',t['text'],re.I))]
    scores = []
    for d in detections:
        row = []
        for t in designators:
            prefix = re.match(r'[A-Za-z]+', t['text']).group().upper()
            prefixes=PREFIXES.get(d['category'], ())
            if mode in ('boxed','guarded') and d['category'] in ('Box','Block'):
                prefixes=prefixes+('USB','LED')
            allowed = compatible(t['text'], d['category']) if mode == 'legacy' else prefix in prefixes
            # Preserve unfamiliar detector categories rather than deleting valid
            # motor/antenna/measurement symbols through an incomplete type list.
            if mode=='guarded' and d['category'] not in PREFIXES:
                allowed=d['category'] not in ('Gnd','V','NET_input','Net_output','Net_bidirection','Testpoint') and compatible(t['text'],d['category'])
            row.append(max(0, 100-box_distance(t['bbox'], d['bbox'])) if allowed else 0)
        scores.append(row)
    assigned = {i: j for i, j in assignment(scores) if scores[i][j] > 25}
    records = [{'detection': i, 'category': d['category'], 'bbox': d['bbox'],
                'designator': designators[assigned[i]]['text'] if i in assigned else None,
                'status': 'assigned' if i in assigned else 'unassigned',
                'score': scores[i][assigned[i]] if i in assigned else None}
               for i, d in enumerate(detections)]
    return designators, assigned, {'mode': mode, 'records': records}


def associate_stable(detections,texts):
    """Lock reliable existing bindings, resolve unique local pairs, then residual."""
    from .text_roles import designator_role
    from .vision import box_distance
    tokens,old,report=associate(detections,texts,'guarded')
    legacy_tokens,legacy,_=associate(detections,texts,'legacy')
    legacy_ids={id(t):i for i,t in enumerate(tokens)}
    unused_categories={'Gnd','V','NET_input','Net_output','Net_bidirection','Testpoint'}
    roles={}; scores=[]
    for i,d in enumerate(detections):
        row=[]
        for j,t in enumerate(tokens):
            prefix=re.match(r'[A-Za-z]+',t['text']).group().upper()
            allowed=prefix in PREFIXES.get(d['category'],())
            if d['category'] in ('Box','Block'): allowed=allowed or prefix in ('USB','LED')
            role=designator_role(d,t,detections,texts); roles[i,j]=role
            row.append((100-box_distance(t['bbox'],d['bbox'])+20*role['designator_score']) if allowed and role['designator_score']>=.7 else 0)
        scores.append(row)
    locked={}; used=set()
    for i,j in legacy.items():
        if detections[i]['category'] in unused_categories: continue
        target=legacy_ids[id(legacy_tokens[j])]
        # Distance alone cannot invalidate a previously exported identity.
        # Unlock only an explicit pin-column / competing-body contradiction.
        if set(roles[i,target]['reasons']) & {'repeated_gpio_column','closer_box_owns_text'}: continue
        locked[i]=target; used.add(target)
    stages={i:'existing_binding_locked' for i in locked}
    # Mutual local dominance prevents a new candidate displacing a trusted pair.
    for i in range(len(detections)):
        if i in locked: continue
        options=sorted(((s,j) for j,s in enumerate(scores[i]) if s>25 and j not in used),reverse=True)
        if not options: continue
        best,j=options[0]; competing=max((scores[k][j] for k in range(len(detections)) if k not in locked and k!=i),default=0)
        if best-competing>=8 and (len(options)==1 or best-options[1][0]>=8):
            locked[i]=j; used.add(j); stages[i]='unique_local_binding'
    remaining_i=[i for i in range(len(detections)) if i not in locked]
    remaining_j=[j for j in range(len(tokens)) if j not in used]
    residual=[[scores[i][j] for j in remaining_j] for i in remaining_i]
    for ri,rj in assignment(residual):
        if residual[ri][rj]>25:
            i,j=remaining_i[ri],remaining_j[rj]; locked[i]=j; stages[i]='residual_global_binding'
    records=[]
    for i,d in enumerate(detections):
        j=locked.get(i)
        records.append({'detection':i,'category':d['category'],'bbox':d['bbox'],'designator':tokens[j]['text'] if j is not None else None,
                        'status':'assigned' if j is not None else 'unassigned','score':scores[i][j] if j is not None else None,
                        'stage':stages.get(i,'abstain'),'text_role':roles[i,j] if j is not None else None,
                        'previous_designator':legacy_tokens[legacy[i]]['text'] if i in legacy and d['category'] not in unused_categories else None,
                        'identity_new':i not in legacy and d['category'] not in unused_categories,
                        'guarded_designator':report['records'][i]['designator']})
    return tokens,locked,{'mode':'stable','records':records,'no_steal_locked':True,'role_scores_are_heuristic':True}
