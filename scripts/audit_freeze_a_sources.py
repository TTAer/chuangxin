"""Source-only CAD inventory and honest Freeze-A readiness audit.

No recognizer runs here. CAD fields remain source-coordinate evidence and
cannot be scored as pixel annotations before renderer registration/review.
"""
import hashlib
import json
from pathlib import Path
import re

ROOT=Path(__file__).resolve().parents[1]

def sexp(text):
    # Supports quoted KiCad strings and escapes without interpreting source code.
    tokens=re.findall(r'"(?:\\.|[^"\\])*"|[()]|[^\s()]+',text)
    stack=[]; roots=[]
    for token in tokens:
        if token=='(':
            node=[]
            if stack: stack[-1].append(node)
            else: roots.append(node)
            stack.append(node)
        elif token==')':
            if not stack: raise ValueError('unbalanced KiCad source')
            stack.pop()
        else:
            if not stack: raise ValueError('token outside source tree')
            stack[-1].append(json.loads(token) if token.startswith('"') else token)
    if stack or len(roots)!=1: raise ValueError('invalid KiCad source tree')
    return roots[0]

def children(node,key): return [v for v in node if isinstance(v,list) and v and v[0]==key]
def first(node,key): return next(iter(children(node,key)),None)

def cad_inventory(path):
    tree=sexp(path.read_text(encoding='utf-8'))
    library=first(tree,'lib_symbols'); libs={v[1]:v for v in children(library or [],'symbol')}
    def pins(node):
        output=[]
        for p in children(node,'pin'):
            num=first(p,'number'); name=first(p,'name'); at=first(p,'at')
            output.append({'number':num[1] if num else None,'name':name[1] if name else None,'symbol_local_at_mm':at[1:] if at else None,'electrical_type':p[1]})
        for s in children(node,'symbol'): output.extend(pins(s))
        return output
    components=[]
    for instance in children(tree,'symbol'):
        libid=first(instance,'lib_id'); attrs={p[1]:p[2] for p in children(instance,'property')}
        if not libid: continue
        components.append({'key':attrs.get('Reference'),'name':attrs.get('Value'),'lib_id':libid[1],'instance_at_mm':(first(instance,'at') or [None])[1:],'mirror':(first(instance,'mirror') or [None])[1:],'unit':(first(instance,'unit') or [None,None])[1],'source_pins':pins(libs.get(libid[1],[])),'visible_text_reviewed':False,'pixel_coordinates_verified':False})
    return {'components':components,'paper':(first(tree,'paper') or [None])[1:],'source_coordinate_units':'mm; symbol local pins require unit/rotation/mirror and renderer registration','not_ground_truth_for_pixel_scoring':True}

def main():
    manifest=json.loads((ROOT/'data/splits/freeze-a-source-intake.json').read_text(encoding='utf-8'))
    prior={}; prior_count=0
    for base in (ROOT/'data/raw',ROOT/'data/processed',ROOT/'data/examples'):
        for path in base.rglob('*'):
            if not path.is_file() or 'FreezeA-manufacturer' in path.parts or path.suffix.lower() not in ('.png','.jpg','.jpeg','.pdf','.kicad_sch','.sch'): continue
            prior.setdefault(hashlib.sha256(path.read_bytes()).hexdigest(),[]).append(str(path.relative_to(ROOT))); prior_count+=1
    family=[]; records=[]; cad_errors=[]
    for source in manifest['sources']:
        files=source['files']; acquired=source['status']=='acquired'
        duplicates=[]; mismatches=[]; cad=[]; seen=set()
        for row in files:
            p=ROOT/row['path']; actual=hashlib.sha256(p.read_bytes()).hexdigest()
            if actual!=row['sha256']: mismatches.append(row['path'])
            if actual in prior: duplicates.append({'path':row['path'],'prior_files':prior[actual]})
            if p.suffix=='.kicad_sch' and actual not in seen:
                seen.add(actual)
                try:
                    inv=cad_inventory(p); inv.update(source_path=row['path'],source_sha256=actual,repository=source['repository']); records.append(inv)
                    cad.append({'path':row['path'],'components':len(inv['components']),'pins':sum(len(v['source_pins']) for v in inv['components'])})
                except Exception as e: cad_errors.append({'path':row['path'],'error':str(e)})
        family.append({'repository':source['repository'],'acquired':acquired,'candidate_family_only':True,'source_files':len(files),'pdf_files':sum(r['path'].endswith('.pdf') for r in files),'modern_cad':cad,'exact_prior_duplicates':duplicates,'hash_mismatches':mismatches,'design_level_exclusion_verified':False,'license_review_complete':False,'pixel_gt_complete':False})
    report={'status':'NOT_READY_SOURCE_INTAKE','prior_files_hashed':prior_count,'acquired_candidate_families':sum(x['acquired'] for x in family),'unique_kicad_sheets':len(records),'cad_components':sum(len(r['components']) for r in records),'source_pin_records_including_multi_unit_definitions':sum(len(c['source_pins']) for r in records for c in r['components']),'exact_duplicate_files':sum(len(r['exact_prior_duplicates']) for r in family),'verified_independent_designs':0,'verified_pixel_gt_pages':0,'inference_runs':0,'families':family,'cad_errors':cad_errors,'limitations':['Zero byte overlap is not design-level independence.','Repository/revision/sheet counts are not independent design counts.','CAD pin definitions include non-rendered units/hidden pins; visibility and point transforms await review.','Source coordinate CAD values must not be treated as registered pixel ground truth.','Unknown provenance in historical image corpora prevents automatic semantic-design exclusion.']}
    (ROOT/'data/splits/freeze-a-cad-evidence.json').write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding='utf-8')
    (ROOT/'data/splits/freeze-a-readiness-audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k not in ('families','cad_errors','limitations')},indent=2))
if __name__=='__main__': main()
