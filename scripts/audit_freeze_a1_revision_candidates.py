"""Fixed-scale whole-sheet pin/vector compatibility; no revision certification."""
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
import pymupdf
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.audit_freeze_a_sources import sexp,children,first
from scripts.prepare_freeze_a1_candidates import transform,hidden
OUT=ROOT/'output/freeze-a1-revision-audit'
def load(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,d):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d,ensure_ascii=False,indent=2),encoding='utf-8')

def vector_error(expected,lines):
    return float(np.min(np.max(np.abs(lines-np.asarray(expected)),axis=1))) if len(lines) else None

def all_pin_segments(tree):
    libs={v[1]:v for v in children(first(tree,'lib_symbols') or [],'symbol')};rows=[]
    for symbol in children(tree,'symbol'):
        lid=first(symbol,'lib_id')
        if not lid:continue
        properties={p[1]:p[2] for p in children(symbol,'property')};unit=int((first(symbol,'unit') or ['',1])[1]);at=(first(symbol,'at') or ['',0,0,0])[1:]
        mirror=(first(symbol,'mirror') or ['',None])[1];parts=[]
        for part in children(libs.get(lid[1],[]),'symbol'):
            try:part_unit=int(part[1].rsplit('_',2)[-2])
            except (ValueError,IndexError):continue
            if part_unit in (0,unit):parts.append(part)
        row={'uuid':(first(symbol,'uuid') or ['',None])[1],'reference':properties.get('Reference'),'model':properties.get('Value'),
            'unit':unit,'dnp':(first(symbol,'dnp') or ['','no'])[1],'rotation':float(at[2]),'mirror':mirror,'pins':[],
            'hidden_pins_design_only':[],'zero_length_pins':[],'library_found':lid[1] in libs}
        for part in parts:
            for p in children(part,'pin'):
                number=first(p,'number')[1];name=first(p,'name')[1]
                if hidden(p):row['hidden_pins_design_only'].append({'number':number,'name':name});continue
                length=float(first(p,'length')[1])
                if length<=0:row['zero_length_pins'].append(number);continue
                x,y,angle=map(float,first(p,'at')[1:]);a=np.radians(angle)
                tip=transform(x,y,at,mirror);base=transform(x+length*np.cos(a),y+length*np.sin(a),at,mirror)
                row['pins'].append({'number':number,'CAD_name':name,'outer_tip_xy':tip,'body_base_xy':base,'pin_length_px':float(np.linalg.norm(np.array(tip)-base))})
        rows.append(row)
    return rows

def main():
    audit=load(ROOT/'output/freeze-a1-identity/identity-audit.json');groups={}
    for r in audit['records']:groups.setdefault(r['image_sha256'],[]).append(r)
    pages=[]
    for imagehash,records in groups.items():
        r=records[0];cad=ROOT/r['source_cad'];pdf=ROOT/r['source_pdf']
        if sha(cad)!=r['source_cad_sha256'] or sha(pdf)!=r['source_pdf_sha256']:raise ValueError('Registered source snapshot changed')
        rows=all_pin_segments(sexp(cad.read_text(encoding='utf-8')))
        with pymupdf.open(pdf) as doc:
            page=doc[r['source_pdf_page']];lines=[]
            for drawing in page.get_drawings():
                for item in drawing['items']:
                    if item[0]=='l':
                        v=np.array(tuple(item[1])+tuple(item[2]))*2;lines.extend([v,v[[2,3,0,1]]])
            text_words=len(page.get_text('words'))
        lines=np.array(lines)
        for row in rows:
            for p in row['pins']:
                error=vector_error(p['outer_tip_xy']+p['body_base_xy'],lines)
                p.update(PDF_vector_max_error_px=error,PDF_segment_match=error is not None and error<=.25)
        pins=[p for row in rows for p in row['pins']];matched=sum(p['PDF_segment_match'] for p in pins)
        pageout={'image_sha256':imagehash,'repository':r['repository'],'source_cad':r['source_cad'],'source_cad_sha256':r['source_cad_sha256'],
            'source_pdf':r['source_pdf'],'source_pdf_sha256':r['source_pdf_sha256'],'PDF_page':r['source_pdf_page'],
            'scope':'all CAD placed symbols on one source sheet against selected PDF page, visible nonzero pin segments only; not whole symbol/text/net correctness',
            'placed_instances':len(rows),'visible_nonzero_CAD_pin_segments':len(pins),'PDF_pin_segments_matching':matched,
            'pin_segment_match_fraction':matched/len(pins) if pins else None,'DNP_instances':sum(row['dnp']=='yes' for row in rows),
            'multi_unit_instances':sum(row['unit']>1 for row in rows),'native_PDF_word_count':text_words,
            'IC_identity_conflicts_or_pending':sum(rec['observed_review']['status']!='agent_pixel_matched' for rec in records),
            'license_evidence_file_count':len(r['license_records']),'source_revision_verified':False,'verified_design':False,'instances':rows}
        save(OUT/(imagehash[:12]+'.json'),pageout);pages.append({k:v for k,v in pageout.items() if k!='instances'})
    # Source-evidence ranking only, not a metric-tuned source selection.
    ranked=sorted(pages,key=lambda r:(r['IC_identity_conflicts_or_pending']>0,r['license_evidence_file_count']==0,r['multi_unit_instances']>0,r['DNP_instances']>0,-(r['pin_segment_match_fraction'] or 0),-bool(r['native_PDF_word_count']),r['placed_instances']))
    selected=[];seen=set()
    for p in ranked:
        family='sparkfun/PCA9554-navigation-family' if p['repository'].endswith(('Qwiic_Directional_Pad','Qwiic_Navigation_Switch')) else p['repository']
        if family in seen or p['IC_identity_conflicts_or_pending']:continue
        seen.add(family);selected.append(dict(p,candidate_design_family=family,selection_status='review_priority_only_not_verified'))
        if len(selected)==5:break
    summary={'pages':len(pages),'placed_instances':sum(r['placed_instances'] for r in pages),
        'CAD_pin_segments':sum(r['visible_nonzero_CAD_pin_segments'] for r in pages),
        'PDF_segment_matches':sum(r['PDF_pin_segments_matching'] for r in pages),
        'fixed_scale_pixels_per_mm':144/25.4,'tolerance_px':.25,'registration_fit':False,'model_runs':0,
        'shortlist':selected,'pages_detail':pages,'verified_designs':0,
        'limitations':['Pin/vector equality alone cannot establish PDF/CAD whole-page revision identity.','Hidden pins remain design-only; DNP/unit metadata are not visible truth.','Source identity/text/family exclusion and two independent human reviews remain pending.']}
    save(OUT/'summary.json',summary)
    print(json.dumps({k:v for k,v in summary.items() if k not in ('pages_detail','shortlist')},ensure_ascii=False));print(json.dumps([(r['repository'],r['pin_segment_match_fraction'],r['placed_instances'],r['DNP_instances'],r['multi_unit_instances']) for r in selected]))
if __name__=='__main__':main()
