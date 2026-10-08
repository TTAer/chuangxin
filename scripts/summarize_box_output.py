"""Align each catalog-available GT component with emitted key and exact model."""
from collections import Counter
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from pcbparse.__main__ import load,save
from pcbparse.semantic_library import SemanticLibrary,norm
from pcbparse.vision import iou

def main():
    lib=SemanticLibrary(); audit=load(ROOT/'output/ablation/box-lifecycle.json'); result={}
    for domain in ('gt10','pcb55'):
        result[domain]={}
        for mode,folder in [('default','benchmark-recognition-'+domain),('identity-side','ic-identity-'+domain+'-side'),('typed','designator-typed/'+domain),('boxed','designator-boxed/'+domain),('guarded','designator-guarded/'+domain)]:
            counts=Counter(); rows=[]
            for gt in audit['domains'][domain]['gt_records']:
                r=gt['detection']; obs=load(ROOT/'output'/folder/gt['case']/'observations.json'); h=obs['image']['height']
                emitted=[]
                if r:
                    for key,c in obs['components'].items():
                        b=c['bbox']; b=[b[0],h-b[3],b[2],h-b[1]]
                        if c['type'] in ('ic','box') and iou(b,r['bbox'])>=.5: emitted.append((key,c))
                correct=[(k,c) for k,c in emitted if k==gt['gt_key']]
                exact=any(c.get('Name') and norm(c['Name'])==norm(gt['gt_name']) for k,c in correct) if gt.get('gt_name') else False
                confirmed=[]
                for k,c in correct:
                    model,_=lib.identify(c.get('Name'),c['bbox'],obs.get('texts',[]))
                    if model: confirmed.append(model)
                counts['gt']+=1; counts['detected']+=bool(r); counts['emitted']+=bool(emitted); counts['correct_key']+=bool(correct); counts['correct_key_and_exact_gt_name']+=exact
                if gt['catalog_available']:
                    counts['catalog_gt']+=1; counts['catalog_detected']+=bool(r); counts['catalog_emitted']+=bool(emitted); counts['catalog_correct_key']+=bool(correct)
                    counts['catalog_correct_key_and_confirmed']+=bool(confirmed)
                    counts['catalog_correct_key_and_confirmed_exact_gt_model']+=any(norm(m)==norm(gt['gt_name']) for m in confirmed)
                rows.append({'case':gt['case'],'gt_key':gt['gt_key'],'catalog_available':gt['catalog_available'],'detected':bool(r),'emitted_keys':[k for k,c in emitted],'correct_key':bool(correct),'exact_gt_name':exact,'confirmed_models':confirmed})
            result[domain][mode]={'counts':dict(counts),'records':rows}; print(domain,mode,dict(counts),flush=True)
    save(ROOT/'output/ablation/box-output-oracle.json',{'domains':result,'ground_truth_used_only_after_inference':True,'official_equivalence':False,'denominators_aligned_by_gt_component':True})
if __name__=='__main__': main()
