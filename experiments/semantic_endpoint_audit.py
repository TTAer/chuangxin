"""Human evidence audit; same-pin eligibility never comes from nearest GT."""
import math
import hashlib
import json
from collections import Counter
from statistics import median

CONCLUSIONS={'body','tip','contact','continuous-no-boundary','uncertain'}
CERTAINTIES={'clear','ambiguous','unknown'}
CLASSES={'same-physical-pin','wrong-pin','false-endpoint','duplicate','component-mismatch','uncertain'}

def case_fingerprint(case):
    body={k:v for k,v in case.items() if k!='case_evidence_sha256'}
    return hashlib.sha256(json.dumps(body,sort_keys=True,ensure_ascii=False).encode()).hexdigest()

def audited_semantics(cases,reviews):
    lookup={c['id']:c for c in cases};observations=[];rejected=[];seen=set()
    for review in reviews:
        reviewer=review.get('reviewer_id')
        if review.get('schema')!='pin-geometry-review-v2' or review.get('coordinate_system')!='image top-left, integer pixel' or review.get('reviewer_kind')!='human-self-declared' or not isinstance(reviewer,str) or not reviewer.strip():
            rejected.append({'reason':'review_schema_or_actor_invalid'});continue
        for entry in review.get('entries',[]):
            case=lookup.get(entry.get('id'));reason=None
            if not case:reason='unknown_case'
            elif case_fingerprint(case)!=case.get('case_evidence_sha256'):reason='case_record_tampered'
            elif (reviewer,entry['id']) in seen:reason='same_reviewer_duplicate_case'
            else:
                seen.add((reviewer,entry['id']))
                if entry.get('case_evidence_sha256')!=case['case_evidence_sha256']:reason='case_evidence_changed'
                elif not entry.get('review_complete'):reason='incomplete_review'
                elif entry.get('certainty') not in CERTAINTIES or entry.get('conclusion') not in CONCLUSIONS or entry.get('candidate_class') not in CLASSES:reason='review_vocabulary_invalid'
                elif entry.get('observed_side','unknown') not in ('unknown','left','right','top','bottom'):reason='observed_side_invalid'
                elif not isinstance(entry.get('observed_pin_number',''),str) or not isinstance(entry.get('same_pin_evidence_note',''),str):reason='identity_evidence_schema_invalid'
                else:
                    points=entry.get('observed_points',{})
                    if not isinstance(points,dict):reason='observed_points_invalid'
                    else:
                        for label,p in points.items():
                            if label not in ('body','tip','contact') or not isinstance(p,list) or len(p)!=2 or not all(type(v)==int for v in p):reason='point_invalid';break
                            origin=case['crop_origin_xy'];size=case['crop_size']
                            if not all(origin[i]<=p[i]<origin[i]+size[i] for i in range(2)):reason='point_outside_review_crop';break
            if reason:rejected.append({'id':entry.get('id'),'reviewer':reviewer,'reason':reason})
            else:observations.append(dict(entry,reviewer_id=reviewer))
    agreed=[];ambiguous=[];conflicts=[];distances=[]
    for key,case in lookup.items():
        es=[e for e in observations if e['id']==key]
        if len(es)<2:continue
        if any(e['certainty']!='clear' or e['conclusion'] in ('continuous-no-boundary','uncertain') or e['candidate_class']=='uncertain' for e in es):
            ambiguous.append({'id':key,'reviewers':[e['reviewer_id'] for e in es]});continue
        consistent=len({(e['conclusion'],e['candidate_class'],e.get('observed_pin_number','').strip(),e.get('observed_side','unknown')) for e in es})==1
        shared=set.intersection(*(set(e.get('observed_points',{})) for e in es))
        if any(math.dist(a['observed_points'][p],b['observed_points'][p])>3 for p in shared for a in es for b in es):consistent=False
        if not consistent:conflicts.append({'id':key,'reason':'review_semantics_identity_or_points_disagree'});continue
        classification=es[0]['candidate_class'];number=es[0].get('observed_pin_number','').strip()
        observed_side=es[0].get('observed_side','unknown')
        record={'id':key,'domain':case['domain'],'side':observed_side,'source_side':case['side'],'state':case['pin_state_proxy'],'class':classification,
            'side_assignment_disagreement':observed_side!='unknown' and observed_side!=case['side'],
            'reviewers':[e['reviewer_id'] for e in es],'observed_pin_number':number,'semantic_eligible':False,
            'normal_error':None,'tangent_error':None,'normalized_normal_error':None,'pin_length':None,'distance':None}
        # Use exact observed number ONLY AFTER two independent reviewers agree.
        target=case.get('GT_pins_by_number',{}).get(number)
        if classification=='same-physical-pin' and target and target.get('identity_reference_comparable') is True and number and observed_side!='unknown' and all(e.get('same_pin_evidence_note','').strip() for e in es) and all(e.get('reference_seen_before_observations') is False for e in es) and es[0]['conclusion'] in shared and case['points'].get('predicted'):
            record['semantic_eligible']=True;pred=case['points']['predicted'];gt=target['point_xy']
            normals={'left':(-1,0),'right':(1,0),'top':(0,-1),'bottom':(0,1)};nx,ny=normals[observed_side]
            dx,dy=gt[0]-pred[0],gt[1]-pred[1]
            record.update(distance=math.dist(pred,gt),normal_error=dx*nx+dy*ny,tangent_error=-dx*ny+dy*nx)
            # First review's observed length; no candidate/body-gap substitute.
            body=es[0]['observed_points'].get('body');tip=es[0]['observed_points'].get('tip')
            if body is not None and tip is not None and 'body' in shared and 'tip' in shared:
                lengths=[math.dist(e['observed_points']['body'],e['observed_points']['tip']) for e in es]
                if all(v>0 for v in lengths):
                    length=median(lengths)
                    ratios=[record['normal_error']/v for v in lengths]
                    record.update(pin_length=length,pin_length_review_values=lengths,
                        normalized_normal_error=record['normal_error']/length,
                        normalized_normal_error_interval=[min(ratios),max(ratios)])
            for definition in shared:
                distances.append({'id':key,'domain':case['domain'],'side':observed_side,'state':case['pin_state_proxy'],
                    'definition':definition,'distance':median(math.dist(e['observed_points'][definition],gt) for e in es),
                    'distance_by_reviewer':{e['reviewer_id']:math.dist(e['observed_points'][definition],gt) for e in es}})
        agreed.append(record)
    groups={}
    for domain in ('gt10','pcb55','Freeze-A1'):
        rs=[r for r in agreed if r['domain']==domain and r['semantic_eligible']]
        def stat(rows):
            normalized=[r['normalized_normal_error'] for r in rows if r['normalized_normal_error'] is not None]
            return {'same_pin_cases':len(rows),'semantic_eligible_P5':sum(r['distance']<=5 for r in rows)/len(rows) if rows else None,
                'median_abs_normal':median(abs(r['normal_error']) for r in rows) if rows else None,
                'median_abs_tangent':median(abs(r['tangent_error']) for r in rows) if rows else None,
                'measured_pin_lengths':len(normalized),'median_normalized_normal':median(normalized) if normalized else None}
        groups[domain]={'ALL':stat(rs),'by_side':{side:stat([r for r in rs if r['side']==side]) for side in ('left','right','top','bottom')},
            'by_state':{s:stat([r for r in rs if r['state']==s]) for s in sorted({r['state'] for r in rs})}}
    def definition_stats(rows):
        return {label:{'reviewed_same_pin_cases':sum(r['definition']==label for r in rows),
            'median_GT_distance':median(r['distance'] for r in rows if r['definition']==label) if any(r['definition']==label for r in rows) else None} for label in ('body','tip','contact')}
    definition_tables={d:{'ALL':definition_stats([r for r in distances if r['domain']==d]),
        'by_side':{side:definition_stats([r for r in distances if r['domain']==d and r['side']==side]) for side in ('left','right','top','bottom')}} for d in ('gt10','pcb55','Freeze-A1')}
    paired=len(agreed)+len(ambiguous)+len(conflicts)
    return {'schema':'semantic-endpoint-audit-v2','accepted_observation_entries':len(observations),'rejected':rejected,
        'observation_certainty_counts':dict(Counter(e['certainty'] for e in observations)),
        'two_reviewer_agreements':agreed,'ambiguous_or_unknown':ambiguous,'conflicts':conflicts,
        'candidate_classes':dict(Counter(r['class'] for r in agreed)),'semantic_statistics':groups,'observed_definition_distances':distances,
        'definition_distance_statistics':definition_tables,
        'reviewer_agreement':{'paired_cases':paired,'clear_agreements':len(agreed),'ambiguous_or_unknown':len(ambiguous),
            'conflicts':len(conflicts),'clear_agreement_rate':len(agreed)/paired if paired else None},
        'coordinate_convention_frozen':False,'verified_designs':0,'runtime_changed':False,
        'limitations':'Reviewer identity/independence are self-declared; review records are evidence, not automatic certification. No reviewer or GT is manufactured.'}
