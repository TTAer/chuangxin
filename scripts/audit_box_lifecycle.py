"""Post-inference per-GT and per-detection Box lifecycle; no GT in association."""
from collections import Counter
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from pcbparse.__main__ import load,save
from pcbparse.dataset import index_dataset
from pcbparse.detector import OnnxDetector
from pcbparse.designator_association import associate
from pcbparse.semantic_library import SemanticLibrary,norm
from pcbparse.vision import read_image,iou,box_distance

def main():
    lib=SemanticLibrary(); detector=OnnxDetector(ROOT/'models/detector/pcb-yolo11n.onnx')
    samples={s['id']:s for s in index_dataset(ROOT/'data/raw/PCBnet/Dataset300')}; domains={}
    for domain in ('gt10','pcb55'):
        counts=Counter(); rows=[]; boxes=[]
        for folder in sorted((ROOT/('output/benchmark-recognition-'+domain)).iterdir()):
            if not (folder/'observations.json').exists(): continue
            path=ROOT/'data/raw/official/gt-0930/10GT'/folder.name/(folder.name+'.png') if domain=='gt10' else Path(samples[folder.name]['image'])
            obs=load(folder/'observations.json'); h=obs['image']['height']
            texts=[dict(t,bbox=[t['bbox'][0],h-t['bbox'][3],t['bbox'][2],h-t['bbox'][1]]) for t in obs['texts']]
            auxiliary={'aitee_edge','aitee_junction','current_source','voltage_source','ident-i','ident-i-dir','ident-res','ident-v','ident-v-dir'}
            detections=[d for d in detector(read_image(path)) if d['category'] not in auxiliary]
            _,_,old=associate(detections,texts); _,_,new=associate(detections,texts,'typed')
            for r in old['records']:
                if r['category'] in ('Gnd','V','NET_input','Net_output','Net_bidirection','Testpoint') and r['designator']:
                    counts['unused_designators_consumed']+=1
            records=[]
            for i,d in enumerate(detections):
                if d['category'] not in ('Box','Block'): continue
                emitted=[]
                for k,c in obs['components'].items():
                    b=c['bbox']; b=[b[0],h-b[3],b[2],h-b[1]]
                    if iou(b,d['bbox'])>=.5: emitted.append({'key':k,'type':c['type'],'name':c.get('Name')})
                record={'case':folder.name,**old['records'][i],'typed_designator':new['records'][i]['designator'],'emitted':emitted}
                records.append((i,record)); boxes.append(record)
            # Truth is read only after both association variants are complete.
            truth=load(path.with_suffix('.json') if domain=='gt10' else folder/'golden.json')
            gt=[(k,c) for k,c in truth['components'].items() if c['type'] in ('ic','box')]
            candidates=[]
            for g,(k,c) in enumerate(gt):
                b=c['bbox']; b=[b[0],h-b[3],b[2],h-b[1]]
                candidates.extend((iou(b,detections[i]['bbox']),g,i) for i,r in records)
            matched={}; used=set()
            for score,g,i in sorted(candidates,reverse=True):
                if score<.3 or g in matched or i in used: continue
                matched[g]=i; used.add(i)
            by_index=dict(records)
            for g,(key,c) in enumerate(gt):
                r=by_index.get(matched.get(g)); available=bool(c.get('Name')) and any(norm(c['Name'])==norm(m) for m in lib.entries)
                emitted=r['emitted'] if r else []; correct=[e for e in emitted if e['key']==key]
                identified=any(lib.identify(e['name'],obs['components'][e['key']]['bbox'],obs.get('texts',[]))[0] for e in correct)
                stage='not_detected' if not r else 'no_designator' if not r['designator'] else 'not_emitted' if not emitted else 'wrong_key' if not correct else 'identity_unconfirmed' if not identified else 'identity_confirmed'
                observable=any(t['text']==key for t in texts)
                local_observable=bool(r and any(t['text']==key and box_distance(t['bbox'],r['bbox'])<75 for t in texts))
                counts['gt_key_observed_in_ocr']+=observable
                counts['no_designator_key_absent_from_ocr']+=stage=='no_designator' and not observable
                counts['no_designator_key_present_locally']+=stage=='no_designator' and local_observable
                counts['no_designator_generated_C_key']+=stage=='no_designator' and key.startswith('C_')
                counts[stage]+=1; counts['gt_ic']+=1
                counts['typed_correct_assignment']+=bool(r and r['typed_designator']==key)
                counts['legacy_correct_assignment']+=bool(r and r['designator']==key)
                if available:
                    counts['library_gt']+=1; counts['library_detected']+=bool(r); counts['library_emitted']+=bool(emitted); counts['library_correct_key']+=bool(correct); counts['library_confirmed_correct_key']+=bool(identified)
                rows.append({'case':folder.name,'gt_key':key,'gt_name':c.get('Name'),'catalog_available':available,'gt_key_in_ocr':observable,'gt_key_local_ocr':local_observable,'stage':stage,'detection':r})
        domains[domain]={'counts':dict(counts),'gt_records':rows,'box_records':boxes}
        print(domain,dict(counts),flush=True)
    save(ROOT/'output/ablation/box-lifecycle.json',{'domains':domains,'post_inference_gt_only':True,'official_equivalence':False,'matching_iou':.3,'emitted_iou':.5,'typed_counts_are_assignment_only':True})
if __name__=='__main__': main()
