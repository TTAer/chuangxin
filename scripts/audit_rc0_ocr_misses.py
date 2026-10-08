"""Post-inference development audit; local OCR experiment is not enabled in RC0."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from pcbparse.__main__ import load,save
from pcbparse.vision import ImageParser,read_image
from pcbparse.dataset import index_dataset
import cv2
parser=ImageParser(); samples={s['id']:s for s in index_dataset(ROOT/'data/raw/PCBnet/Dataset300')}; audit=load(ROOT/'output/ablation/box-lifecycle.json'); rows=[]
for r in audit['domains']['pcb55']['gt_records']:
    if r['stage']!='no_designator' or r['gt_key_in_ocr'] or r['gt_key'].startswith('C_'): continue
    image=read_image(samples[r['case']]['image']); h,w=image.shape[:2]; box=r['detection']['bbox']
    x1,y1,x2,y2=[round(v) for v in box]; x1=max(0,x1-75); y1=max(0,y1-75); x2=min(w,x2+75); y2=min(h,y2+75)
    crop=cv2.resize(image[y1:y2,x1:x2],None,fx=2,fy=2,interpolation=cv2.INTER_CUBIC); raw,_=parser.ocr(crop)
    texts=[{'text':text,'confidence':float(conf),'bbox':[(min(p[0] for p in bb)/2+x1),(min(p[1] for p in bb)/2+y1),(max(p[0] for p in bb)/2+x1),(max(p[1] for p in bb)/2+y1)]} for bb,text,conf in raw or []]
    key=r['gt_key']; exact=any(t['text']==key for t in texts); glued=[t['text'] for t in texts if t['text'].startswith(key) and t['text']!=key]
    row={'case':r['case'],'gt_key':key,'local_ocr_exact_key':exact,'local_ocr_glued_candidates':glued,'local_texts':texts,
         'category':'high_resolution_local_exact_recovery' if exact else 'possible_glued_token' if glued else 'unresolved_requires_image_review',
         'absence_in_ocr_does_not_prove_absence_in_image':True,'character_confidence_available':False}
    rows.append(row); print(r['case'],key,row['category'],flush=True)
save(ROOT/'output/ablation/rc0-ocr-misses.json',{'samples':len(rows),'rows':rows,'post_inference_gt_diagnostic_only':True,'rc0_unchanged':True,'cannot_separate_text_detector_vs_recognizer_without_review':True})
