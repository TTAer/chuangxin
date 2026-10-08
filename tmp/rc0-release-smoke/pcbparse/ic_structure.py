"""Four-side, monotone OCR association for image-observed IC pin endpoints.

The catalog supplies allowed identities, never coordinates or pin order. Side
order comes from observed endpoints and OCR. No missing endpoint is fabricated.
All coordinates follow the observations' bottom-left convention.
"""
import copy
import math
import re
import cv2
import numpy as np
from .semantic_library import center, similarity


def side_of(point, box):
    x, y = point['x'], point['y']
    distances = {'left': abs(x-box[0]), 'right': abs(x-box[2]),
                 'bottom': abs(y-box[1]), 'top': abs(y-box[3])}
    return min(distances, key=distances.get)


def monotone_matches(weights):
    """Maximum supported one-to-one ordered alignment with free skips."""
    n = len(weights); m = len(weights[0]) if n else 0
    dp = [[0.]*(m+1) for _ in range(n+1)]
    trace = {}
    for i in range(1, n+1):
        for j in range(1, m+1):
            candidates = [(dp[i-1][j], 'skip_pin'), (dp[i][j-1], 'skip_text')]
            if weights[i-1][j-1] > 0:
                candidates.append((dp[i-1][j-1]+weights[i-1][j-1], 'match'))
            dp[i][j], trace[i,j] = max(candidates, key=lambda a: a[0])
    result = []; i = n; j = m
    while i and j:
        action = trace[i,j]
        if action == 'match': result.append((i-1,j-1)); i -= 1; j -= 1
        elif action == 'skip_pin': i -= 1
        else: j -= 1
    return list(reversed(result))


class StructureAwareICParser:
    def __init__(self, library):
        self.library = library

    def refine_geometry(self, observations, image):
        """Use numbered OCR, named OCR and a real wire crossing at the outer box.

        The detector's outer box estimates the stub termination along the normal;
        the actual horizontal stroke estimates the transverse position. This is
        an optional estimator, not a guarantee of symbolic pin-tip semantics.
        """
        result = copy.deepcopy(observations)
        report = {'module':'StructureAwareICPinParser','edits':0,'records':[],
                  'fabricated_endpoints':0,'order_source':'local numbered OCR and named OCR; not catalog numbering'}
        height=image.shape[0]; ink=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)<180
        horizontal=cv2.morphologyEx(ink.astype(np.uint8),cv2.MORPH_OPEN,np.ones((1,7),np.uint8))
        edits=[]
        for key,c in result['components'].items():
            if c['type'] not in ('ic','box'): continue
            box=c['bbox']; model,_=self.library.identify(c.get('Name'),box,result.get('texts',[]))
            if not model: continue
            catalog=self.library.entries[model]['pins']
            for side in ('left','right'):
                anchors=[]
                for pin_key,pin in result['pins'][key].items():
                    if side_of(pin['point'],box)!=side: continue
                    number=pin_key[4:]
                    numeric=[t for t in result.get('texts',[]) if t['text']==number and t.get('confidence',1)>=.9
                             and side_of(dict(zip(('x','y'),center(t['bbox']))),box)==side
                             and math.dist(center(t['bbox']),(pin['point']['x'],pin['point']['y']))<60]
                    if len(numeric)==1: anchors.append((pin_key,pin,numeric[0]))
                anchors.sort(key=lambda p:center(p[2]['bbox'])[1])
                names=[]
                for t in result.get('texts',[]):
                    tx,ty=center(t['bbox'])
                    if t.get('confidence',1)<.9 or not (box[0]<tx<box[2] and box[1]<ty<box[3]): continue
                    if side_of({'x':tx,'y':ty},box)==side: names.append(t)
                names.sort(key=lambda t:center(t['bbox'])[1])
                weights=[]; proposals={}
                for i,(pin_key,pin,nt) in enumerate(anchors):
                    number=pin_key[4:]; allowed=catalog.get(number,{}).get('names',[])
                    _,ny=center(nt['bbox']); extent=nt['bbox'][3]-nt['bbox'][1]; row=[]
                    for j,text in enumerate(names):
                        _,ty=center(text['bbox']); delta=abs(ty-ny); limit=max(12,extent*1.1)
                        ranked=sorted([(similarity(text['text'],name),name) for name in allowed],reverse=True)
                        score=0
                        if delta<=limit and ranked and ranked[0][0]>=.94 and (len(ranked)==1 or ranked[0][0]-ranked[1][0]>=.06):
                            xx=box[0] if side=='left' else box[2]
                            ix=int(round(xx)); iy=int(round(height-ty)); a=max(0,ix-4); b=min(image.shape[1],ix+5)
                            low=max(0,iy-8); high=min(height,iy+9)
                            if a<b and low<high:
                                supported=np.flatnonzero(horizontal[low:high,a:b].sum(axis=1)>=max(3,(b-a)*.7))
                                groups=np.split(supported,np.flatnonzero(np.diff(supported)>1)+1) if len(supported) else []
                                if len(groups)==1 and len(groups[0])<=5:
                                    point={'x':float(xx),'y':height-(low+float(np.mean(groups[0])))}
                                    if math.dist((point['x'],point['y']),(pin['point']['x'],pin['point']['y']))<=48:
                                        score=ranked[0][0]+.2*(1-delta/limit)
                                        proposals[i,j]=(point,ranked[0][1])
                        row.append(score)
                    weights.append(row)
                for i,j in monotone_matches(weights):
                    pin_key,pin,_=anchors[i]; text=names[j]; point,name=proposals[i,j]; before=copy.deepcopy(pin)
                    pin.update(point=point,pinname=name)
                    if pin!=before: edits.append({'component':key,'pin':pin_key,'side':side,'before':before,'after':copy.deepcopy(pin),'numeric_ocr':pin_key[4:],'name_ocr':text['text'],'image_stroke_support':True})
        report['geometry_edits']=edits; report['geometry_edit_count']=len(edits)
        report['edits']+=len(edits); report['coordinates_unchanged']=not bool(edits)
        report['normal_estimator']='detector outer-box side; transverse coordinate from observed straight stroke'
        return result,report

    def refine(self, observations, resolve_numbers=True):
        result = copy.deepcopy(observations); records = []
        for key, component in result['components'].items():
            if component['type'] not in ('ic', 'box'): continue
            model, identity = self.library.identify(component.get('Name'), component['bbox'], result.get('texts', []))
            if not model: continue
            catalog = self.library.entries[model]['pins']; box = component['bbox']
            pins = result['pins'][key]
            for side in ('left', 'right', 'bottom', 'top'):
                axis = 1 if side in ('left', 'right') else 0
                def coord(p): return p['point']['y' if axis else 'x']
                observed = [{'id':n, 'number':n[4:], 'point':p['point'], 'pinname':p['pinname']} for n,p in pins.items()
                            if side_of(p['point'],box)==side]
                if resolve_numbers:
                    observed += [dict(p,number=None,pinname='') for p in result.get('pin_hypotheses', [])
                                 if p['component']==key and side_of(p['point'],box)==side]
                observed.sort(key=coord)
                texts = []
                for t in result.get('texts', []):
                    tx,ty = center(t['bbox']); bw=box[2]-box[0]; bh=box[3]-box[1]
                    if not (box[0]<=tx<=box[2] and box[1]<=ty<=box[3]): continue
                    if side_of({'x':tx,'y':ty},box)!=side: continue
                    depth = {'left':tx-box[0], 'right':box[2]-tx, 'bottom':ty-box[1], 'top':box[3]-ty}[side]
                    if depth>min(110, (bw if axis else bh)*.3) or t.get('confidence',1)<.85: continue
                    if re.fullmatch(r'[A-Za-z]{1,3}\d+|\d+', t['text']):
                        # GPIO0-style pin names are permitted if present in catalog.
                        if not any(similarity(t['text'],name)>=.94 for item in catalog.values() for name in item['names']): continue
                    texts.append(t)
                texts.sort(key=lambda t:center(t['bbox'])[axis])
                weights=[]; choices={}
                for i,p in enumerate(observed):
                    number=p['number']
                    if number is None:
                        # Recover identity only from a single, local numeric OCR token.
                        numeric=[]
                        for t in result.get('texts', []):
                            n=t['text']; tx,ty=center(t['bbox'])
                            if not re.fullmatch(r'\d+',n) or n not in catalog or 'pin_'+n in pins or t.get('confidence',1)<.9: continue
                            if side_of({'x':tx,'y':ty},box)!=side: continue
                            distance=math.hypot(tx-p['point']['x'],ty-p['point']['y'])
                            if distance<=30: numeric.append((distance,n))
                        numeric.sort()
                        if numeric and (len(numeric)==1 or numeric[1][0]-numeric[0][0]>=8): number=numeric[0][1]
                    row=[]
                    for j,t in enumerate(texts):
                        delta=abs(center(t['bbox'])[axis]-coord(p))
                        extent=t['bbox'][3]-t['bbox'][1] if axis else t['bbox'][2]-t['bbox'][0]
                        limit=min(18,max(8,extent*.6))
                        allowed=catalog.get(number,{}).get('names',[])
                        names=sorted([(similarity(t['text'],name),name) for name in allowed],reverse=True)
                        supported=names and names[0][0]>=.94 and (len(names)==1 or names[0][0]-names[1][0]>=.06)
                        score=names[0][0]+.2*(1-delta/limit) if supported and delta<limit else 0
                        row.append(score)
                        if score: choices[i,j]=(number,names[0][1],t['text'],delta)
                    weights.append(row)
                for i,j in monotone_matches(weights):
                    p=observed[i]; number,name,text,delta=choices[i,j]
                    if p['number'] is None:
                        if 'pin_'+number in pins: continue
                        pins['pin_'+number]={'pinname':name,'point':copy.deepcopy(p['point'])}
                        p['resolved']=True
                        result['pin_hypotheses']=[q for q in result.get('pin_hypotheses',[]) if q['id']!=p['id']]
                        kind='number_and_name'
                    else:
                        if pins[p['id']]['pinname']==name: continue
                        # Existing nonempty contradictory text is retained unless empty
                        # or already a catalog alias for the same numbered pin.
                        before=pins[p['id']]['pinname']
                        if before and before not in catalog[number]['names']: continue
                        pins[p['id']]['pinname']=name; kind='name'
                    records.append({'component':key,'side':side,'tip':p['id'],'number':number,'name':name,
                                    'ocr':text,'transverse_error':delta,'kind':kind,'model':model})
        return result, {'module':'StructureAwareICPinParser','edits':len(records),'records':records,
                        'coordinates_unchanged':True,'fabricated_endpoints':0,'order_source':'observed endpoints and OCR; not catalog numbering'}
