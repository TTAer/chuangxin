"""Source/pixel identity evidence only. Never imports or runs a parser/model."""
import hashlib
import json
from pathlib import Path
from audit_freeze_a_sources import sexp, children, first

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'output/freeze-a1-identity'

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def normalized(text):
    # Semantic display notation only, never a claimed literal OCR transcript.
    return text.replace('~{', '').replace('_{', '').replace('}', '').replace('_', '')

def center(box):
    return [(box[0]+box[2])/2, (box[1]+box[3])/2]

def match_token(words, text, anchor, used, max_distance):
    hits = []
    for i, w in enumerate(words):
        if i in used or normalized(w['text']) != normalized(text):
            continue
        xy = center(w['bbox_xyxy'])
        distance = sum((xy[j]-anchor[j])**2 for j in (0, 1))**.5
        if distance <= max_distance:
            hits.append((distance, i, w))
    hits.sort(key=lambda row: row[0])
    if not hits or (len(hits)>1 and abs(hits[0][0]-hits[1][0])<1):
        return None
    distance, i, w = hits[0]
    used.add(i)
    return dict(w, distance_px=distance, association='exact semantic text + fixed local anchor, one-to-one')

def source_instance(ann, comp):
    tree = sexp((ROOT/ann['source_cad']).read_text(encoding='utf-8'))
    matches = [s for s in children(tree, 'symbol')
               if (first(s, 'lib_id') or ['', None])[1] == comp['lib_id']
               and int((first(s, 'unit') or ['', 1])[1]) == comp['unit']
               and {p[1]:p[2] for p in children(s, 'property')}.get('Reference') == comp['source_property_reference']]
    # Multiple same-model instances are not resolved by list order.
    if len(matches) != 1:
        return {'status':'ambiguous_instance', 'matching_instances':len(matches)}
    s=matches[0]
    props={p[1]:p[2] for p in children(s,'property')}
    return {'status':'unique_property_and_unit_match', 'reference':props.get('Reference'),
            'unit':comp['unit'], 'dnp':(first(s,'dnp') or ['', 'unknown'])[1],
            'mirror':comp['mirror'], 'rotation_degrees':comp['rotation_degrees'],
            'display_reference_expected':props.get('Reference','')+(chr(64+comp['unit']) if comp['unit']>1 else ''),
            'value':props.get('Value')}

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    audit=json.loads((ROOT/'data/processed/FreezeA-A1-candidates/audit.json').read_text(encoding='utf-8'))
    reviews=json.loads((ROOT/'data/research/freeze-a1-agent-pixel-review.json').read_text(encoding='utf-8'))
    review_by_id={r['id']:r for r in reviews['records']}
    records=[];seen=set();page_hashes=set();number_count=name_count=0
    for row in audit['pages']:
        path=ROOT/row['annotation'];ann=json.loads(path.read_text(encoding='utf-8'))
        if sha(path)!=row['annotation_sha256'] or sha(ROOT/ann['image'])!=ann['image_sha256']:
            raise ValueError('Frozen candidate source changed')
        words=[];wordseen=set()
        for w in ann['pdf_text_layer_unassigned']:
            sig=(w['text'],tuple(round(v,3) for v in w['bbox_xyxy']))
            if sig not in wordseen: words.append(w);wordseen.add(sig)
        for j,c in enumerate(ann['components']):
            sig=(ann['image_sha256'],tuple(round(v,2) for b in c['body_rectangles_xyxy'] for v in b))
            if sig in seen:continue
            seen.add(sig);idx=len(records);review=review_by_id[idx];used=set();tokens=[]
            box=c['body_rectangles_xyxy'][0];anchor=center(box)
            for role,text in [('designator',review['observed_key']),('model',review['observed_model'])]:
                if text:
                    token=match_token(words,text,anchor,used,max(box[2]-box[0],box[3]-box[1])/2+90)
                    if token:tokens.append(dict(token,role=role))
            pins=[]
            for p in c['pins']:
                mid=[(p['point_xy'][k]+p['body_contact_xy'][k])/2 for k in (0,1)]
                num=match_token(words,p['number'],mid,used,26)
                name=match_token(words,p['name'],p['body_contact_xy'],used,44)
                number_count+=bool(num);name_count+=bool(name)
                pins.append(dict(p,native_number_token=num,native_name_token=name,
                    agent_number_visible=review['status']=='agent_pixel_matched' and p['number'] not in review.get('number_unreadable_or_hidden',[]),
                    agent_name_visible=review['status']=='agent_pixel_matched' and p['number'] not in review.get('name_unreadable_or_hidden',[])))
            source=source_instance(ann,c)
            licenses=[]
            for l in ann['license_files']:
                text=(ROOT/l['path']).read_text(encoding='utf-8',errors='replace')
                licenses.append(dict(l,hardware_cc_by_sa_40_mentioned=('Attribution-ShareAlike 4.0' in text or 'CC BY-SA 4.0' in text or 'by-sa/4.0' in text),scope_approved=False))
            records.append({'id':idx,'annotation':str(path.relative_to(ROOT)),'annotation_sha256':sha(path),
                'image':ann['image'],'image_sha256':ann['image_sha256'],'source_cad':ann['source_cad'],'source_cad_sha256':ann['source_cad_sha256'],
                'source_pdf':ann['source_pdf'],'source_pdf_sha256':ann['source_pdf_sha256'],'source_pdf_page':ann['source_pdf_page'],
                'repository':ann['repository'],'component_index':j,'body_rectangles_xyxy':c['body_rectangles_xyxy'],
                'candidate_key':c['key'],'candidate_model':c['name'],'observed_review':review,'source_instance':source,
                'native_identity_tokens':tokens,'pins':pins,'license_records':licenses,
                'identity_scope_gt_candidate':({'body_xyxy':box,'key':review['observed_key'],'name':review['observed_model'],
                    'pins':[{'point_xy':p['point_xy'],'number':p['number'],'name':p['name']} for p in pins
                            if p['agent_number_visible'] and p['agent_name_visible']]} if review['status']=='agent_pixel_matched' else None),
                'scope':'partial IC visible identity; not full page, not net truth','independent_human_review':False,
                'design_family_exclusion_verified':False,'family_exclusion_reason':'Historical training/development design provenance incomplete.',
                'eligible_for_blind_scoring':False})
            page_hashes.add(ann['image_sha256'])
    matched=[r for r in records if r['observed_review']['status']=='agent_pixel_matched']
    summary={'schema':'freeze-a1-identity-source-evidence-v1','unique_pixel_pages':len(page_hashes),'unique_ic_candidates':len(records),
        'pin_segments':sum(len(r['pins']) for r in records),'agent_pixel_reviewed_instances':len(records),
        'agent_matched_ic_instances':len(matched),'agent_matched_pages':len({r['image_sha256'] for r in matched}),
        'agent_matched_repository_families':len({r['repository'] for r in matched}),
        'agent_visible_pin_numbers':sum(p['agent_number_visible'] for r in records for p in r['pins']),
        'agent_visible_pin_names':sum(p['agent_name_visible'] for r in records for p in r['pins']),
        'native_text_number_associations':number_count,'native_text_name_associations':name_count,
        'native_text_ic_identity_associations':sum(len(r['native_identity_tokens']) for r in records),
        'native_text_pages':len({r['image_sha256'] for r in records if r['native_identity_tokens']}),
        'verified_independent_designs':0,'independent_human_reviews':0,'eligible_pages':0,'inference_runs':0,
        'status':'NO_GO_SOURCE_AND_FAMILY_REVIEW_PENDING','records':records,
        'limitations':['Agent visual comparison is not independent human verification.','Semantic overbar/subscript notation is retained separately from literal token text.','Matching text evidence does not verify page revision or design exclusion.','No models or parser inference were run.']}
    (OUT/'identity-audit.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in summary.items() if k not in ('records','limitations')},indent=2))

if __name__=='__main__':main()
