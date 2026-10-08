"""Whole-sheet inventories and unfilled semantic review forms; not GT."""
import hashlib
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.audit_freeze_a_sources import sexp,children,first
OUT=ROOT/'output/freeze-a1-calibration'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,d):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d,ensure_ascii=False,indent=2),encoding='utf-8')

def whole_sheet(tree):
    rows=[]
    for symbol in children(tree,'symbol'):
        lid=first(symbol,'lib_id')
        if not lid:continue
        properties={p[1]:p[2] for p in children(symbol,'property')}
        refs=[]
        def walk(n):
            if n and n[0]=='reference':refs.append(n[1])
            for c in n:
                if isinstance(c,list):walk(c)
        walk(first(symbol,'instances') or [])
        rows.append({'uuid':(first(symbol,'uuid') or [None,None])[1],
            'property_reference':properties.get('Reference'),'project_references':sorted(set(refs)),
            'CAD_value':properties.get('Value'),'lib_id':lid[1],
            'unit':int((first(symbol,'unit') or [None,'1'])[1]),
            'dnp':(first(symbol,'dnp') or [None,'no'])[1],
            'rotation_degrees':float((first(symbol,'at') or [None,0,0,0])[3]),
            'mirror':(first(symbol,'mirror') or [None,None])[1],
            'visible_key':None,'pixel_presence':None,'pixel_model':None,
            'manual_instance_review_complete':False})
    return rows

def review_gate(review,expected):
    """Additional semantic/review prerequisites, not a substitute for A1 gate."""
    reasons=[]
    for k in ('image_sha256','source_cad_sha256','source_pdf_sha256'):
        if review.get(k)!=expected.get(k):reasons.append(k+'_mismatch')
    if review.get('status')!='complete':reasons.append('review_pending')
    if not review.get('whole_page_component_instances_reviewed'):reasons.append('whole_page_instances_pending')
    if not review.get('whole_page_revision_verified'):reasons.append('whole_page_revision_pending')
    convention=review.get('point_convention',{})
    if convention.get('status')!='verified' or convention.get('definition') not in ('stroke_outer_tip','wire_contact','body_edge','logical_anchor') or not convention.get('evidence_records'):
        reasons.append('point_definition_pending')
    if convention.get('coordinate_system')!='image_top_left_integer_pixel':reasons.append('coordinate_convention_pending')
    for k in ('hidden_pin_rule','DNP_rule','multi_unit_rule','visible_key_rule'):
        if not review.get(k):reasons.append(k+'_pending')
    if not review.get('pin_number_name_human_checked'):reasons.append('visible_identity_pending')
    passes=review.get('review_passes',[])
    valid=[r for r in passes if r.get('status')=='complete' and r.get('reviewer_id') and r.get('reviewer_kind')=='human' and r.get('evidence_record')]
    if len({r['reviewer_id'] for r in valid})<2:reasons.append('independent_second_review_pending')
    return {'semantic_review_complete':not reasons,'reasons':reasons,'blind_ready':False,
        'point_definition':convention.get('definition'),'coordinate_system':convention.get('coordinate_system'),
        'note':'Also requires original A1 license/history-family/source/evaluator freeze gate; this result never authorizes inference'}

def main():
    audit=json.loads((ROOT/'output/freeze-a1-identity/identity-audit.json').read_text(encoding='utf-8'))
    groups={}
    for r in audit['records']:groups.setdefault(r['image_sha256'],r)
    entries=[]
    for imagehash,r in groups.items():
        folder=OUT/imagehash[:12];tree=sexp((ROOT/r['source_cad']).read_text(encoding='utf-8'));rows=whole_sheet(tree)
        snapshot={k:r[k] for k in ('repository','image','image_sha256','source_cad','source_cad_sha256','source_pdf','source_pdf_sha256','source_pdf_page')}
        inventory=dict(snapshot,scope='all placed symbols on corresponding CAD sheet; pixel page coverage/revision not approved',
            symbols=rows,placed_symbol_count=len(rows),DNP_count=sum(x['dnp']=='yes' for x in rows),
            multi_unit_instances=sum(x['unit']>1 for x in rows),blind_ready=False)
        save(folder/'whole-sheet-inventory.json',inventory)
        template=dict(snapshot,schema='freeze-a1-semantic-review-v1',status='pending',
            whole_page_component_instances_reviewed=False,whole_page_revision_verified=False,
            point_convention={'status':'unresolved','definition':None,'coordinate_system':'image_top_left_integer_pixel','evidence_records':[]},
            hidden_pin_rule=None,DNP_rule=None,multi_unit_rule=None,visible_key_rule=None,
            pin_number_name_human_checked=False,review_passes=[],verified_design_id=None)
        # Keep edited review forms intact across regeneration.
        p=folder/'review-template.json'
        if not p.exists():save(p,template)
        assessed=review_gate(json.loads(p.read_text(encoding='utf-8')),snapshot)
        entries.append(dict(image_sha256=imagehash,repository=r['repository'],placed_symbols=len(rows),
            inventory=str((folder/'whole-sheet-inventory.json').relative_to(ROOT)),review_form=str(p.relative_to(ROOT)),**assessed))
    save(OUT/'summary.json',{'pages':len(entries),'CAD_placed_symbols':sum(e['placed_symbols'] for e in entries),
        'semantic_review_complete':sum(e['semantic_review_complete'] for e in entries),'verified_designs':0,'first_inference_runs':0,'entries':entries})
    print(json.dumps({'pages':len(entries),'CAD_placed_symbols':sum(e['placed_symbols'] for e in entries),'semantic_review_complete':sum(e['semantic_review_complete'] for e in entries)}))
if __name__=='__main__':main()
