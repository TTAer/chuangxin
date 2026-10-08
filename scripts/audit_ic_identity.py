"""GT-only development diagnostics after candidate inference; not runtime input."""
from collections import Counter
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from pcbparse.__main__ import load,save
from pcbparse.semantic_library import SemanticLibrary,norm,box_gap
from pcbparse.dataset import index_dataset
from pcbparse.detector import OnnxDetector
from pcbparse.vision import read_image,iou
lib=SemanticLibrary(); detector=OnnxDetector(ROOT/'models/detector/pcb-yolo11n.onnx')
samples={s['id']:s for s in index_dataset(ROOT/'data/raw/PCBnet/Dataset300')}; domains={}
for domain in ('gt10','pcb55'):
    counts=Counter(); records=[]; predicted=[]
    for folder in sorted((ROOT/('output/benchmark-recognition-'+domain)).iterdir()):
        if not (folder/'observations.json').exists(): continue
        path=ROOT/'data/raw/official/gt-0930/10GT'/folder.name/(folder.name+'.png') if domain=='gt10' else Path(samples[folder.name]['image'])
        image=read_image(path); h=image.shape[0]
        detections=[d for d in detector(image) if d['category'] in ('Box','Block')]
        obs=load(folder/'observations.json'); recovered=load(ROOT/('output/ic-identity-'+domain)/folder.name/'observations.json')
        for key,c in obs['components'].items():
            if c['type'] not in ('ic','box'): continue
            before,old=lib.identify(c.get('Name'),c['bbox'],obs.get('texts',[]))
            after,_=lib.identify(recovered['components'][key].get('Name'),c['bbox'],recovered.get('texts',[]))
            counts['emitted_ic']+=1; counts['strict_identity_before']+=bool(before); counts['strict_identity_after']+=bool(after)
            predicted.append({'case':folder.name,'component':key,'name':c.get('Name'),'recovered_name':recovered['components'][key].get('Name'),
                              'strict_before':before,'strict_after':after,'initial_status':old['status']})
        truth=load(path.with_suffix('.json') if domain=='gt10' else folder/'golden.json')
        gt=[(key,c) for key,c in truth['components'].items() if c['type'] in ('ic','box')]
        pairs=[]
        for i,(key,c) in enumerate(gt):
            b=c['bbox']; box=[b[0],h-b[3],b[2],h-b[1]]
            pairs.extend((iou(box,d['bbox']),i,j) for j,d in enumerate(detections))
        matched={}; used=set()
        for score,i,j in sorted(pairs,reverse=True):
            if score<.3 or i in matched or j in used: continue
            matched[i]=j; used.add(j)
        for i,(key,c) in enumerate(gt):
            name=c.get('Name'); available=bool(name) and any(norm(name)==norm(m) for m in lib.entries)
            detection=detections[matched[i]] if i in matched else None
            emitted=[]
            if detection:
                for observed_key,observed in obs['components'].items():
                    if observed['type'] not in ('ic','box'): continue
                    b=observed['bbox']; box=[b[0],h-b[3],b[2],h-b[1]]
                    if iou(box,detection['bbox'])>=.5: emitted.append(observed_key)
            counts['gt_ic']+=1; counts['gt_model_in_library']+=available; counts['geometry_detected_gt_ic']+=bool(detection)
            counts['geometry_matched_emitted_gt_ic']+=bool(emitted); counts['correct_key_gt_ic']+=key in emitted
            records.append({'case':folder.name,'gt_component':key,'gt_name':name,'catalog_exact_model_available':available,
                            'detected_iou_ge_0_3':bool(detection),'emitted_geometry_keys':emitted,'correct_key':key in emitted})
    domains[domain]={'counts':dict(counts),'gt_records':records,'emitted_records':predicted}
save(ROOT/'output/ablation/ic-identity-audit.json',{'domains':domains,'post_inference_gt_audit':True,'official_equivalence':False,
     'limitations':['geometry detection uses development IoU >=0.3, not official match','GT exact model availability ignores aliases','emitted ic includes connectors and some non-chip boxes']})
print({d:r['counts'] for d,r in domains.items()},flush=True)
