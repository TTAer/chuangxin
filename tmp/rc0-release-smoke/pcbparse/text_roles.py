"""Auditable text-role scores; heuristic evidence, not calibrated probability."""
import re

def designator_role(detection,token,detections,texts):
    from .vision import box_distance
    box=detection['bbox']; tb=token['bbox']; text=token['text']
    prefix=re.match(r'[A-Za-z]+',text).group().upper()
    size=max(5,tb[3]-tb[1]); gap=box_distance(tb,box)
    reasons=[]; score=.95
    if token.get('confidence',1)<.9: score-=.2; reasons.append('ocr_below_0_9')
    if gap>max(30,2*size): score-=.5; reasons.append('outside_local_text_scale')
    if prefix=='P':
        tx=(tb[0]+tb[2])/2; ty=(tb[1]+tb[3])/2
        for owner in detections:
            ob=owner['bbox']
            if owner['category'] not in ('Box','Block') or max(ob[2]-ob[0],ob[3]-ob[1])<160: continue
            if box_distance(tb,ob)>30 or not ob[1]-size<=ty<=ob[3]+size: continue
            aligned=[t for t in texts if re.fullmatch(r'P\d+',t['text'],re.I)
                     and abs((t['bbox'][0]+t['bbox'][2])/2-tx)<2*size
                     and ob[1]-size<=(t['bbox'][1]+t['bbox'][3])/2<=ob[3]+size]
            if len(aligned)>=4:
                score=min(score,.15); reasons.append('repeated_gpio_column'); break
    if detection['category'] in ('Box','Block') and gap>12:
        others=[o for o in detections if o is not detection and o['category'] in ('Box','Block')]
        if any(box_distance(tb,o['bbox'])+12<gap for o in others):
            score=min(score,.2); reasons.append('closer_box_owns_text')
    return {'designator_score':max(0,score),'reasons':reasons,'gap':gap,'text_height':size,'calibrated_probability':False}
