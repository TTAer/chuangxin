"""Keep the 25 cases fixed; add review vocabulary and exact-number references."""
import argparse
import hashlib
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.audit_unresolved_endpoint_quality import component_pairs
from experiments.semantic_endpoint_audit import audited_semantics,case_fingerprint
from pcbparse.dataset import text_value
OUT=ROOT/'output/pin-geometry-review-v2'
def load(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,d):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d,ensure_ascii=False,indent=2),encoding='utf-8')
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--reviews',nargs='*',default=[]);args=parser.parse_args()
    if args.reviews:
        # Import against the existing review cohort, never regenerate it after seeing reviews.
        cases=load(OUT/'cases.json')
        for c in cases:
            if sha(ROOT/c['image'])!=c['source_image_sha256'] or sha(OUT/c['crop'])!=c['crop_sha256']:
                raise ValueError('Reviewed pixels changed')
            if c['domain'] in ('gt10','pcb55'):
                p=ROOT/'data/raw/official/gt-0930/10GT'/c['case']/(c['case']+'.json') if c['domain']=='gt10' else ROOT/'output/benchmark-recognition-pcb55'/c['case']/'golden.json'
                if sha(p)!=c['source_GT_sha256']:raise ValueError('Reviewed GT reference changed')
                if c.get('raw_number_provenance_sha256'):
                    image=ROOT/c['image'];rawpath=image.with_name(image.stem+'_CVJsonStd.json')
                    if sha(rawpath)!=c['raw_number_provenance_sha256']:raise ValueError('Raw identity provenance changed')
        result=audited_semantics(cases,[load(Path(p)) for p in args.reviews])
        result['review_input_sha256']={str(p):sha(Path(p)) for p in args.reviews}
        save(OUT/'review-import-audit.json',result)
        print(json.dumps({'review_entries':result['accepted_observation_entries'],'paired_cases':result['reviewer_agreement']['paired_cases'],'verified_designs':0}))
        return
    source=ROOT/'output/pin-geometry-inspector/cases.json';cases=load(source)
    for c in cases:
        if sha(ROOT/c['image'])!=c['source_image_sha256'] or sha(ROOT/'output/pin-geometry-inspector'/c['crop'])!=c['crop_sha256']:
            raise ValueError('Source image or reviewed crop changed')
        c['GT_pins_by_number']={}
        if c['domain'] in ('gt10','pcb55'):
            domain,case=c['domain'],c['case'];obs=load(ROOT/'output/designator-stable-catalog'/domain/case/'observations.json')
            p=ROOT/'data/raw/official/gt-0930/10GT'/case/(case+'.json') if domain=='gt10' else ROOT/'output/benchmark-recognition-pcb55'/case/'golden.json'
            if sha(p)!=c['source_GT_sha256']:raise ValueError('GT reference changed before review')
            gt=load(p);g=component_pairs(obs,gt).get(c['component']);h=obs['image']['height']
            c['GT_pins_by_number']={n[4:]:{'point_xy':[r['point']['x'],h-r['point']['y']],'name':r['pinname'],
                'identity_reference_comparable':domain=='gt10','number_provenance':'official_GT_key' if domain=='gt10' else 'unverified_conversion_fallback'} for n,r in gt['pins'].get(g,{}).items()}
            if domain=='pcb55' and g:
                image=ROOT/c['image'];rawpath=image.with_name(image.stem+'_CVJsonStd.json');raw=load(rawpath)
                c['raw_number_provenance_sha256']=sha(rawpath)
                box=gt['components'][g]['bbox'];owners=[]
                for owner in raw['components'].values():
                    b=owner['location'];candidate=[b['x1'],h-b['y2'],b['x2'],h-b['y1']]
                    if max(abs(a-b) for a,b in zip(box,candidate))<=.01:owners.append(owner)
                if len(owners)==1:
                    for pin in owners[0].get('pins',{}).values():
                        # Never use generated pin_specs fallback as visible identity.
                        number=text_value(raw,pin.get('pinNumberByVLM') or pin.get('pinNumberByOCR'))
                        reference=c['GT_pins_by_number'].get(number)
                        if reference and max(abs(a-b) for a,b in zip(reference['point_xy'],pin['tip']))<=.01:
                            reference.update(identity_reference_comparable=True,number_provenance='weak_explicit_source_number; still not independently certified GT')
        c['observed_truth']={'body':None,'tip':None,'contact':None,'number':None,'name':None,'certainty':'unknown','review_complete':False}
        c['design_truth']={'CAD_number':c.get('CAD_number'),'CAD_name':c.get('CAD_name'),'CAD_outer_tip':c['points'].get('CAD_outer_tip'),
            'scope':'CAD geometry/properties only; invisible design data never auto-populates observed truth'}
        c['crop']='../pin-geometry-inspector/'+c['crop']
        c['skeleton_crop']='../pin-geometry-inspector/'+c['skeleton_crop']
        c['wire_crop']='../pin-geometry-inspector/'+c['wire_crop']
        c['case_evidence_sha256']=case_fingerprint(c)
    save(OUT/'cases.json',cases)
    protocol={'case_count':len(cases),'case_source_sha256':sha(source),'no_new_samples':True,'initial_view':'pixels-only; GT/prediction hidden until explicitly revealed',
        'conclusions':['body','tip','contact','continuous-no-boundary','uncertain'],'certainty':['clear','ambiguous','unknown'],
        'semantic_eligible':'two distinct human reviewers agree on same physical pin, visible number, target definition and observed point within 3px, with nonempty same-pin evidence and observations before reference reveal; then exact-number GT',
        'normalization':'only agreed manually observed body-to-tip length, never predicted body-gap',
        'human_review_count':0,'coordinate_convention_frozen':False,'verified_designs':0}
    save(OUT/'protocol.json',protocol)
    for pass_name,ordered in [('A',cases),('B',list(reversed(cases)))]:
        template_review={'schema':'pin-geometry-review-v2','coordinate_system':'image top-left, integer pixel',
            'reviewer_id':'','reviewer_kind':'human-self-declared','status':'blank-human-review-template',
            'pass':pass_name,'entries':[{'id':c['id'],'case_evidence_sha256':c['case_evidence_sha256'],
                'observed_points':{},'observed_side':'unknown','certainty':'unknown','conclusion':'uncertain','candidate_class':'uncertain',
                'observed_pin_number':'','same_pin_evidence_note':'','review_complete':False,'reference_seen_before_observations':False} for c in ordered]}
        path=OUT/('pass-'+pass_name+'-blank-template.json')
        if not path.exists():save(path,template_review)
    template=(ROOT/'scripts/pin_geometry_review_v2_template.html').read_text(encoding='utf-8')
    text=json.dumps({'cases':cases,'protocol':protocol},ensure_ascii=False).replace('<','\\u003c')
    (OUT/'index.html').write_text(template.replace('__DATA__',text),encoding='utf-8')
    result=audited_semantics(cases,[load(Path(p)) for p in args.reviews]);save(OUT/'semantic-audit.json',result)
    print(json.dumps({'cases':len(cases),'review_entries':result['accepted_observation_entries'],'verified_designs':0}))
if __name__=='__main__':main()
