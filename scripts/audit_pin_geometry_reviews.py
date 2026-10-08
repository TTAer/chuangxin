"""Validate exported review evidence; never writes GT or infers models."""
import argparse
import json
import math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def assess_reviews(cases,reviews):
    byid={r['id']:r for r in cases};accepted=[];rejected=[]
    for file_index,review in enumerate(reviews):
        if review.get('schema')!='pin-point-definition-review-v1' or review.get('coordinate_system')!='image top-left, integer pixel' or not review.get('reviewer_id') or review.get('reviewer_kind')!='human-self-declared':
            rejected.append({'file_index':file_index,'reason':'review_schema_actor_or_coordinate_invalid'});continue
        seen=set()
        for e in review.get('entries',[]):
            case=byid.get(e.get('id'));reason=None
            if not case:reason='unknown_case'
            elif e['id'] in seen:reason='duplicate_case_in_same_review'
            else:
                seen.add(e['id'])
                for key in ('source_image_sha256','source_GT_sha256','source_annotation_sha256','crop_sha256'):
                    if e.get(key)!=case.get(key):reason='source_hash_mismatch';break
                if not e.get('review_complete'):reason='review_not_complete'
                if e.get('coordinate_convention') in (None,'unresolved','ambiguous'):reason='point_semantics_not_resolved'
                for key,p in e.get('manual_points',{}).items():
                    if key not in ('body_edge','stroke_outer_tip','wire_contact','logical_anchor') or not isinstance(p,list) or len(p)!=2 or not all(type(v)==int for v in p):reason='manual_point_schema_invalid';break
                    origin=case['crop_origin_xy'];size=case['crop_size']
                    if not all(origin[i]<=p[i]<origin[i]+size[i] for i in range(2)):reason='manual_point_outside_inspected_crop';break
            if reason:rejected.append({'file_index':file_index,'id':e.get('id'),'reason':reason})
            else:accepted.append(dict(e,reviewer_id=review['reviewer_id']))
    confirmed=[];conflicts=[]
    for key in byid:
        rs=[e for e in accepted if e['id']==key]
        # One reviewer appearing in multiple files is still one review authority.
        unique={r['reviewer_id']:r for r in rs};rs=list(unique.values())
        if len(rs)<2:continue
        points=set.intersection(*(set(r.get('manual_points',{})) for r in rs))
        required=rs[0]['coordinate_convention']
        agreement=all(r['coordinate_convention']==required for r in rs) and required in points
        if agreement:
            agreement=all(math.dist(a['manual_points'][p],b['manual_points'][p])<=3 for p in points for a in rs for b in rs)
        item={'id':key,'reviewers':list(unique),'shared_point_definitions':sorted(points)}
        (confirmed if agreement else conflicts).append(item)
    distances={}
    for domain in ('gt10','pcb55','Freeze-A1'):
        distances[domain]={}
        for definition in ('body_edge','stroke_outer_tip','wire_contact','logical_anchor'):
            values=[]
            for e in accepted:
                case=byid[e['id']];gt=case['points']['dataset_GT'];point=e.get('manual_points',{}).get(definition)
                if case['domain']==domain and gt is not None and point is not None:values.append(math.dist(point,gt))
            values.sort();n=len(values)
            distances[domain][definition]={'review_point_measurements':n,'median_to_dataset_GT':(values[(n-1)//2]+values[n//2])/2 if n else None}
    return {'status':'review_evidence_audit_only','accepted_review_entries':len(accepted),'rejected_entries':rejected,
        'two_reviewer_local_agreements':confirmed,'review_conflicts':conflicts,'manual_GT_distances':distances,
        'verified_designs':0,'GT_modified':False,'first_inference_authorized':False,
        'limitations':'Self-declared exported review is not automatically independently certified GT; whole-page revision, history and licenses still required.'}

def main():
    p=argparse.ArgumentParser();p.add_argument('--reviews',nargs='*',default=[]);p.add_argument('--output',default='output/pin-geometry-inspector/review-audit.json');a=p.parse_args()
    cases=json.loads((ROOT/'output/pin-geometry-inspector/cases.json').read_text(encoding='utf-8'))
    result=assess_reviews(cases,[json.loads(Path(x).read_text(encoding='utf-8')) for x in a.reviews])
    target=ROOT/a.output;target.parent.mkdir(parents=True,exist_ok=True);target.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:result[k] for k in ('accepted_review_entries','two_reviewer_local_agreements','verified_designs','GT_modified')}))
if __name__=='__main__':main()
