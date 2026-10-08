"""Source-only, fixed-transform CAD/PDF registration. No inference or fitting.

Output is partial candidate evidence, never automatically approved pixel GT.
"""
import hashlib
import json
import math
from pathlib import Path
import numpy as np
import pymupdf
from audit_freeze_a_sources import sexp, children, first

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'data/processed/FreezeA-A1-candidates'
SCALE = 144 / 25.4
TOLERANCE = .25  # pixel; fixed before first matching, no fit/subpixel optimization

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def hidden(n):
    return 'hide' in n or bool(first(n, 'hide') and first(n, 'hide')[1:] != ['no'])

def transform(x, y, at, mirror):
    x, y = float(x), float(y)
    # KiCad mirror x/y denotes reflected page x/y after local y inversion.
    if mirror == 'x': y = -y
    if mirror == 'y': x = -x
    a = math.radians(float(at[2]))
    return [(float(at[0]) + x*math.cos(a)-y*math.sin(a))*SCALE,
            (float(at[1]) - x*math.sin(a)-y*math.cos(a))*SCALE]

def instances(tree):
    libs = {s[1]:s for s in children(first(tree,'lib_symbols') or [],'symbol')}
    output=[]
    for s in children(tree,'symbol'):
        lid=first(s,'lib_id')
        if not lid or lid[1] not in libs: continue
        props={p[1]:p[2] for p in children(s,'property')}
        refs=[]
        def walk(n):
            if n and n[0]=='reference': refs.append(n[1])
            for c in n:
                if isinstance(c,list): walk(c)
        walk(first(s,'instances') or [])
        keys=sorted(set([props.get('Reference','')]+refs))
        if not any(k.startswith('U') for k in keys): continue
        at=first(s,'at')[1:]; unit=int(first(s,'unit')[1]); mirror=(first(s,'mirror') or ['',None])[1]
        parts=[p for p in children(libs[lid[1]],'symbol') if int(p[1].rsplit('_',2)[-2]) in (0,unit)]
        rects=[r for p in parts for r in children(p,'rectangle')]
        if not rects: continue
        bodies=[]
        for r in rects:
            a=first(r,'start')[1:]; b=first(r,'end')[1:]
            corners=[transform(x,y,at,mirror) for x in (a[0],b[0]) for y in (a[1],b[1])]
            bodies.append([min(v[0] for v in corners),min(v[1] for v in corners),max(v[0] for v in corners),max(v[1] for v in corners)])
        pins=[]; excluded=[]
        for part in parts:
            for p in children(part,'pin'):
                number=first(p,'number')[1]; name=first(p,'name')[1]
                if hidden(p): excluded.append(number); continue
                x,y,angle=map(float,first(p,'at')[1:]); length=float(first(p,'length')[1])
                if length<=0: excluded.append(number); continue
                inner=[x+length*math.cos(math.radians(angle)),y+length*math.sin(math.radians(angle))]
                pins.append({'number':number,'name':name,'point_xy':transform(x,y,at,mirror),'body_contact_xy':transform(*inner,at,mirror),
                             'visible_number_name_reviewed':False})
        if not pins: continue
        output.append({'source_property_reference':props.get('Reference'),'source_project_references':refs,'reference_candidates':keys,
                       'key':keys[0] if len(keys)==1 else None,'name':props.get('Value'),'lib_id':lid[1],'unit':unit,
                       'rotation_degrees':float(at[2]),'mirror':mirror,'body_rectangles_xyxy':bodies,'pins':pins,
                       'excluded_hidden_or_zero_length_pins':excluded,'visible_identity_reviewed':False})
    return output

def distance(expected, actual):
    if not len(actual): return float('inf')
    return float(np.min(np.max(np.abs(actual-np.array(expected)),axis=1)))

def register(components,page):
    drawings=page.get_drawings(); rectangles=[]; lines=[]
    for d in drawings:
        for item in d['items']:
            if item[0]=='re': rectangles.append(np.array(tuple(item[1]))*2)
            elif item[0]=='l':
                line=np.array(tuple(item[1])+tuple(item[2]))*2
                lines.extend([line,line[[2,3,0,1]]])
    rectangles=np.array(rectangles); lines=np.array(lines); accepted=[]; rejected=[]
    for c in components:
        body_errors=[distance(r,rectangles) for r in c['body_rectangles_xyxy']]
        pin_errors=[distance(p['point_xy']+p['body_contact_xy'],lines) for p in c['pins']]
        rec=dict(c,body_registration_errors_px=body_errors,pin_registration_errors_px=pin_errors)
        if max(body_errors+pin_errors)<=TOLERANCE: accepted.append(rec)
        else: rejected.append(rec)
    return accepted,rejected

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    intake=json.loads((ROOT/'data/splits/freeze-a-source-intake.json').read_text(encoding='utf-8'))
    pages=[]; attempts=[]; mismatches=[]; seen_cad=set(); source_hashes={}
    for source in intake['sources']:
        for row in source['files']:
            p=ROOT/row['path']; actual=sha(p); source_hashes[row['path']]=actual
            if actual!=row['sha256']: mismatches.append(row['path'])
    if mismatches: raise ValueError('intake source changed: '+str(mismatches))
    for source in intake['sources']:
        files=[ROOT/r['path'] for r in source['files']]
        pdfs=[p for p in files if p.suffix.lower()=='.pdf' and any(w in p.name.lower() for w in ('schematic','rev_','rev.')) and 'bom' not in p.name.lower()]
        for cad in [p for p in files if p.suffix=='.kicad_sch']:
            if sha(cad) in seen_cad: continue
            seen_cad.add(sha(cad))
            comps=instances(sexp(cad.read_text(encoding='utf-8')))
            best=None
            for pdf in pdfs:
                try:
                    with pymupdf.open(pdf) as doc:
                        for i,page in enumerate(doc):
                            good,bad=register(comps,page)
                            attempts.append({'cad':str(cad.relative_to(ROOT)),'pdf':str(pdf.relative_to(ROOT)),'page':i,'accepted_ic_instances':len(good),'rejected_ic_instances':len(bad)})
                            if good and (best is None or len(good)>len(best[2])): best=(pdf,i,good,bad)
                except Exception as e: attempts.append({'cad':str(cad.relative_to(ROOT)),'pdf':str(pdf.relative_to(ROOT)),'error':str(e)})
            if best is None: continue
            pdf,i,good,bad=best; folder=OUT/f'{source["repository"].replace("/","--")}--{sha(cad)[:10]}'; folder.mkdir(exist_ok=True)
            with pymupdf.open(pdf) as doc:
                page=doc[i]; image=folder/'page.png'; page.get_pixmap(matrix=pymupdf.Matrix(2,2),alpha=False).save(image)
                words=[{'text':w[4],'bbox_xyxy':[v*2 for v in w[:4]],'source':'PDF text layer; unassigned/unreviewed'} for w in page.get_text('words')]
            annotation={'schema':'freeze-a1-partial-cad-pdf-registration-v1','repository':source['repository'],'source_url':source.get('source','https://github.com/'+source['repository']),
                'source_cad':str(cad.relative_to(ROOT)),'source_cad_sha256':sha(cad),'source_pdf':str(pdf.relative_to(ROOT)),'source_pdf_sha256':sha(pdf),'source_pdf_page':i,
                'image':str(image.relative_to(ROOT)),'image_sha256':sha(image),'components':good,'rejected_components':bad,'pdf_text_layer_unassigned':words,
                'transform':{'scale_pixels_per_mm':SCALE,'offset_xy':[0,0],'fixed_tolerance_px':TOLERANCE,'no_parameter_fit':True},
                'license_files':[{'path':str(p.relative_to(ROOT)),'sha256':sha(p)} for p in files if 'license' in p.name.lower()],
                'annotation_scope':'partial IC body and visible nonzero pin geometry only; CAD identity strings are unreviewed candidates',
                'full_page_annotation_complete':False,'number_name_pixel_gt_complete':False,'independent_human_review_complete':False,
                'design_independence_verified':False,'eligible_for_blind_scoring':False,'training_use_allowed':False,'inference_runs':0}
            ann=folder/'annotation.json'; ann.write_text(json.dumps(annotation,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
            from PIL import Image,ImageDraw
            im=Image.open(image)
            for j,c in enumerate(good):
                box=c['body_rectangles_xyxy'][0]; cropbox=(max(0,int(box[0])-90),max(0,int(box[1])-90),min(im.width,int(box[2])+90),min(im.height,int(box[3])+90))
                im.crop(cropbox).save(folder/f'ic-{j:02d}-review.png')
                overlay=im.copy(); draw=ImageDraw.Draw(overlay)
                for r in c['body_rectangles_xyxy']: draw.rectangle(r,outline='blue',width=2)
                for p in c['pins']: draw.line(p['point_xy']+p['body_contact_xy'],fill='magenta',width=2)
                overlay.crop(cropbox).save(folder/f'ic-{j:02d}-overlay.png')
            pages.append({'repository':source['repository'],'annotation':str(ann.relative_to(ROOT)),'annotation_sha256':sha(ann),'registered_ic_instances':len(good),'registered_pin_segments':sum(len(c['pins']) for c in good),'rotations':sorted(set(c['rotation_degrees'] for c in good)),'mirrors':sorted(set(str(c['mirror']) for c in good))})
    annotations=[json.loads((ROOT/p['annotation']).read_text(encoding='utf-8')) for p in pages]
    unique_ic={}; unique_pages=set()
    for ann in annotations:
        unique_pages.add((ann['source_pdf_sha256'],ann['source_pdf_page']))
        for c in ann['components']:
            signature=(ann['image_sha256'],tuple(round(x,2) for r in c['body_rectangles_xyxy'] for x in r))
            unique_ic.setdefault(signature,c)
    report={'status':'PARTIAL_REGISTRATION_CANDIDATES_NOT_BLIND_READY','pages':pages,'cad_pdf_pair_count':len(pages),'unique_pdf_pages':len(unique_pages),'candidate_repository_families':len(set(p['repository'] for p in pages)),
        'registered_ic_instances':sum(p['registered_ic_instances'] for p in pages),'registered_pin_segments':sum(p['registered_pin_segments'] for p in pages),
        'unique_pixel_ic_bodies':len(unique_ic),'unique_pixel_pin_segments':sum(len(c['pins']) for c in unique_ic.values()),
        'maximum_registration_error_px':max((e for a in annotations for c in a['components'] for e in c['body_registration_errors_px']+c['pin_registration_errors_px']),default=None),
        'rotations_degrees':sorted(set(c['rotation_degrees'] for c in unique_ic.values())),
        'mirror_counts':{m:sum(str(c['mirror'])==m for c in unique_ic.values()) for m in sorted(set(str(c['mirror']) for c in unique_ic.values()))},
        'source_files_verified':len(source_hashes),'source_sha256':source_hashes,'registration_attempts':attempts,'source_hash_mismatches':mismatches,
        'verified_independent_designs':0,'complete_pixel_gt_pages':0,'inference_runs':0,
        'limitations':['Repository families and pages are not verified independent designs.','Fixed PDF/CAD coordinates only; no fitting or geometry-driven source identity resolution.','Numbers/names come from CAD and require pixel visibility/transcription review.','No full-page labels; do not score full-page recall.','PDF text layer may be empty because visible text is stroked vectors.','Licenses recorded but scope/redistribution/historical design exclusion require review.','Partial registration can occur on stale PDF revisions; reject until page-level provenance reviewed.']}
    (OUT/'audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k not in ('pages','source_sha256','registration_attempts','limitations')},indent=2))

if __name__=='__main__': main()
