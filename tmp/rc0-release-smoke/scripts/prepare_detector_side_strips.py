"""Paired oracle/detector-box IC bands on existing official page splits only."""
from collections import Counter
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
import cv2
import numpy as np
from pcbparse.detector import OnnxDetector
from pcbparse.vision import read_image
from pcbparse.side_strips import band,SIDES

def iou(a,b):
    area=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
    return area/max(1,(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-area)

def main():
    detector=OnnxDetector(ROOT/'models/detector/pcb-yolo11n.onnx'); rng=np.random.default_rng(20261007)
    audit=json.loads((ROOT/'output/official-audit/inventory.json').read_text(encoding='utf-8'))
    arrays={k:[] for k in ('oracle','detector')}; records={k:[] for k in arrays}; matched=[]; totals=Counter(); seen=set()
    for item in audit['details']['train200']['records']:
        if item['pixels_sha256'] in seen: continue
        seen.add(item['pixels_sha256']); image=read_image(ROOT/item['image']); h,w=image.shape[:2]
        gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY); raw=json.loads((ROOT/item['annotation']).read_text(encoding='utf-8'))
        split='train' if item['training_overlap'] else 'validation'
        gt=[]
        for key,c in raw['components'].items():
            if c['type'] not in ('box','ic'): continue
            b=c['bbox']; points=[(p['point']['x'],h-p['point']['y']) for p in raw['pins'].get(key,{}).values()]
            if not points or min(b[2]-b[0],b[3]-b[1])<20: continue
            box=[min(b[0],min(x for x,y in points)),min(h-b[3],min(y for x,y in points)),max(b[2],max(x for x,y in points)),max(h-b[1],max(y for x,y in points))]
            gt.append((key,box,points))
        totals[split+':eligible_ic']+=len(gt)
        detections=[d for d in detector(image) if d['category'] in ('Box','Block')]
        pairs=sorted(((iou(g[1],d['bbox']),i,j) for i,g in enumerate(gt) for j,d in enumerate(detections)),reverse=True)
        used_g=set(); used_d=set()
        for score,i,j in pairs:
            if score<.3 or i in used_g or j in used_d: continue
            used_g.add(i); used_d.add(j); key,oracle,points=gt[i]; predicted=detections[j]['bbox']
            totals[split+':matched_ic']+=1
            matched.append({'page':item['case'],'component':key,'split':split,'iou':score,'oracle':oracle,'detector':predicted,
                            'edge_delta_pixels':[predicted[n]-oracle[n] for n in range(4)]})
            # The same bounded edge perturbation is applied to both paired crops.
            jitter=rng.uniform(-6,6,4).tolist() if split=='train' else None
            for variant,base in [('oracle',oracle),('detector',predicted)]:
                for augmentation,shift in enumerate(([0]*4,jitter) if jitter else ([0]*4,)):
                    box=[base[n]+shift[n] for n in range(4)]
                    if box[2]-box[0]<20 or box[3]-box[1]<20: continue
                    for side in SIDES:
                        lo,hi=(box[1],box[3]) if side in ('left','right') else (box[0],box[2])
                        for start in range(int(lo)-16,int(hi)+1,224):
                            target=[]
                            for x,y in points:
                                distances=[abs(x-box[0]),abs(x-box[2]),abs(y-box[1]),abs(y-box[3])]
                                if SIDES[int(np.argmin(distances))]!=side: continue
                                axis=(y if side in ('left','right') else x)-start
                                normal={'left':x-box[0],'right':box[2]-x,'top':y-box[1],'bottom':box[3]-y}[side]
                                if 0<=axis<256:
                                    totals[variant+':'+split+':target_instances']+=1
                                    if abs(normal)<30: target.append([axis,normal])
                                    else: totals[variant+':'+split+':out_of_normal_band']+=1
                            arrays[variant].append(band(gray,box,side,start))
                            records[variant].append({'page':'official:'+item['case'],'component':key,'split':split,'source':'official',
                                'side':side,'start':start,'box':box,'targets':target,'augmentation':augmentation})
    for variant in arrays:
        out=ROOT/('data/processed/side-strips-'+variant+'-paired'); out.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(out/'bands.npz',gray=np.stack(arrays[variant])); (out/'records.json').write_text(json.dumps(records[variant]),encoding='utf-8')
        report={'variant':variant,'bands':len(records[variant]),'counts':dict(Counter(r['split'] for r in records[variant])),
                'matched_components':matched,'totals':dict(totals),'split':'existing official 160/40 pages before crops',
                'matching':'one-to-one geometric IoU >= 0.3; no designator required; supervision preparation only',
                'train_augmentation':'original plus shared independent four-edge U(-6,6) perturbation; validation unperturbed',
                'gt10_training':False,'frozen_amsnet_used':False,'coordinate_step_source_pixels':1,
                'caveat':'matched IC subset only; counts and optimization steps may differ; targets beyond 30px normal omitted and counted'}
        (out/'preparation.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
        print(json.dumps({k:v for k,v in report.items() if k!='matched_components'}),flush=True)
if __name__=='__main__': main()
