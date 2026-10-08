"""Render fixed development hard cases for agent draft / subsequent human review."""
import json
import hashlib
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from PIL import Image, ImageDraw
from pcbparse.vision import read_image

def main():
    source=ROOT/'output/ablation/pin-token-hard-mining.json'
    mined=json.loads(source.read_text(encoding='utf-8'))
    records=json.loads((ROOT/'data/processed/pin-token-residual/records.json').read_text(encoding='utf-8'))
    hard=[r for r in mined['hard_relations'] if r['split']=='validation']; assert len(hard)==56
    out=ROOT/'output/ablation/hard56-review'; out.mkdir(exist_ok=True)
    entries=[]
    for j,h in enumerate(hard):
        r=records[h['index']]; folder=ROOT/'data/raw/PCBnet/Dataset300'/h['page']
        raw=json.loads(next(folder.glob('*_CVJsonStd.json')).read_text(encoding='utf-8'))
        c=raw['components'][r['component']]; image=Image.open(next(folder.glob('*.png'))).convert('RGB')
        points=[c['pins'][pid]['tip'] for pid in r['candidate_pin_ids']]
        box=r['token']['bbox']; t=points[r['target']]; n=points[0]
        # Local view includes token and both compared endpoints, not a guessed label crop.
        x1=max(0,int(min(box[0],t[0],n[0]))-65); y1=max(0,int(min(box[1],t[1],n[1]))-65)
        x2=min(image.width,int(max(box[2],t[0],n[0]))+65); y2=min(image.height,int(max(box[3],t[1],n[1]))+65)
        crop=image.crop((x1,y1,x2,y2)); d=ImageDraw.Draw(crop)
        d.rectangle((box[0]-x1,box[1]-y1,box[2]-x1,box[3]-y1),outline='blue',width=2)
        for p,color,label in [(n,'red','N'),(t,'green','W')]:
            x,y=p[0]-x1,p[1]-y1; d.ellipse((x-4,y-4,x+4,y+4),outline=color,width=2);d.text((x+5,y+3),label,fill=color)
        crop.thumbnail((395,245)); card=Image.new('RGB',(410,285),'white'); card.paste(crop,((410-crop.width)//2,36))
        d=ImageDraw.Draw(card); d.text((5,2),f"{j+1}: id={h['index']} page={h['page']} {r['token']['text']}",fill='black');d.text((5,17),f"{r['role']} N=nearest(red), W=weak target(green)",fill='black')
        path=out/f'{j+1:02d}-{h["index"]}.png'; card.save(path)
        entries.append(dict(h,review_order=j+1,crop=str(path.relative_to(ROOT)),candidate_endpoint_points=points,weak_target_point=t,nearest_point=n,annotation_text_id=r['annotation_text_id'],category='unknown',agent_visual_review=False,note='Pending agent draft review; source ownership and endpoints are weak annotation, not independently verified.'))
    for k in range(4):
        sheet=Image.new('RGB',(1640,1140),'#ddd')
        for q,e in enumerate(entries[k*14:(k+1)*14]): sheet.paste(Image.open(ROOT/e['crop']),((q%4)*410,(q//4)*285))
        sheet.save(out/f'contact-{k+1}.png')
    result={'status':'agent-draft-not-human-review','raw_hard_count':56,'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'raw56_HRR_unchanged':9/56,'reviewed_correctable_HRR':None,'source_scope':'actual exact uniquely matched OCR token; oracle weak component body/endpoints; development validation only','categories':['geometry-fixable','ocr-fixable','candidate-missing','label-ambiguous','unknown'],'entries':entries}
    (ROOT/'output/ablation/hard56-review-draft.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print('Rendered 56 immutable cases into 4 contact sheets.')
if __name__=='__main__': main()
