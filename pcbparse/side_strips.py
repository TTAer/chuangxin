"""Canonical four-side image bands and optional high-recall learned proposals."""
import json
from pathlib import Path
import cv2
import numpy as np

SIDES=('left','right','top','bottom')

def number_side_points(points,box,texts):
    import re
    from .ic_structure import monotone_matches,side_of
    from .semantic_library import center,box_gap
    result=[]; used=set()
    def image_side(point):
        side=side_of(point,box)
        return {'top':'bottom','bottom':'top'}.get(side,side)
    for side in SIDES:
        axis=1 if side in ('left','right') else 0
        ps=sorted([p for p in points if p['side']==side],key=lambda p:p['y' if axis else 'x'])
        numbers=[t for t in texts if re.fullmatch(r'\d+',t['text']) and t.get('confidence',1)>=.9
                 and box_gap(t['bbox'],box)<24 and image_side(dict(zip(('x','y'),center(t['bbox']))))==side]
        numbers.sort(key=lambda t:center(t['bbox'])[axis]); weights=[]
        for p in ps:
            row=[]; position=p['y' if axis else 'x']; normal=p['x' if axis else 'y']
            for t in numbers:
                bounds=t['bbox']; axial_gap=max(bounds[axis]-position,position-bounds[axis+2],0)
                normal_gap=abs(center(bounds)[1-axis]-normal)
                row.append(max(0,45-normal_gap-2*axial_gap) if normal_gap<35 and axial_gap<12 else 0)
            weights.append(row)
        for i,j in monotone_matches(weights):
            if weights[i][j]<10 or numbers[j]['text'] in used: continue
            p=ps[i]; number=numbers[j]['text']; used.add(number)
            result.append({'number':number,'name':'','x':p['x'],'y':p['y']})
    return result

def recover_name_anchors(points,box,texts,catalog,numbered,existing_numbers=()):
    """Recover missing number from unique local name evidence, never numeric order."""
    import math
    from .ic_structure import monotone_matches,side_of
    from .semantic_library import center,similarity
    recovered=[]; used={p['number'] for p in numbered}|set(existing_numbers)
    for side in SIDES:
        axis=1 if side in ('left','right') else 0
        ps=sorted([p for p in points if p['side']==side and not any(math.hypot(p['x']-q['x'],p['y']-q['y'])<6 for q in numbered)],key=lambda p:p['y' if axis else 'x'])
        names=[]
        for t in texts:
            tx,ty=center(t['bbox'])
            if t.get('confidence',1)<.9 or not (box[0]<tx<box[2] and box[1]<ty<box[3]): continue
            ts=side_of({'x':tx,'y':ty},box); ts={'top':'bottom','bottom':'top'}.get(ts,ts)
            depth={'left':tx-box[0],'right':box[2]-tx,'top':ty-box[1],'bottom':box[3]-ty}[side]
            if ts!=side or depth>min(110,(box[2]-box[0] if axis else box[3]-box[1])*.3): continue
            ranked=sorted([(max((similarity(t['text'],name) for name in info['names']),default=0),n,info['names']) for n,info in catalog.items()],reverse=True)
            if not ranked or ranked[0][0]<.94 or ranked[0][1] in used: continue
            if len(ranked)>1 and ranked[0][0]-ranked[1][0]<.08: continue
            score,n,allowed=ranked[0]; name=max(allowed,key=lambda x:similarity(t['text'],x))
            names.append((t,n,name,score))
        names.sort(key=lambda t:center(t[0]['bbox'])[axis]); weights=[]
        for p in ps:
            row=[]
            for t,n,name,score in names:
                delta=abs(p['y' if axis else 'x']-center(t['bbox'])[axis])
                row.append(score+.2*(1-delta/10) if delta<10 else 0)
            weights.append(row)
        for i,j in monotone_matches(weights):
            t,n,name,score=names[j]
            # Also require a separated best geometric candidate for this text.
            distances=sorted(abs(p['y' if axis else 'x']-center(t['bbox'])[axis]) for p in ps)
            if n in used or (len(distances)>1 and distances[1]-distances[0]<4): continue
            p=ps[i]; used.add(n)
            recovered.append({'number':n,'name':name,'x':p['x'],'y':p['y'],'evidence':'unique_name_ocr'})
    return recovered

def perpendicular_stroke(image,point):
    """Require a seven-pixel normal stroke near an inferred identity endpoint."""
    x,y=int(round(point['x'])),int(round(point['y'])); gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    if not (0<=x<gray.shape[1] and 0<=y<gray.shape[0]): return False
    if point.get('side') in ('left','right'):
        patch=gray[max(0,y-2):y+3,max(0,x-4):x+5]
        if patch.shape[1]<7: return False
        opened=cv2.morphologyEx((patch<180).astype(np.uint8),cv2.MORPH_OPEN,np.ones((1,7),np.uint8))
    else:
        patch=gray[max(0,y-4):y+5,max(0,x-2):x+3]
        if patch.shape[0]<7: return False
        opened=cv2.morphologyEx((patch<180).astype(np.uint8),cv2.MORPH_OPEN,np.ones((7,1),np.uint8))
    return bool(opened.size and np.any(opened))

def refine_observations(observations,image,locator,library,structure=True,conservative=True,name_anchors=False):
    import copy
    import math
    from .ic_structure import StructureAwareICParser
    result=copy.deepcopy(observations); h=image.shape[0]; edits=[]; funnel=[]
    for key,c in result['components'].items():
        if c['type'] not in ('ic','box'): continue
        b=c['bbox']; box=[b[0],h-b[3],b[2],h-b[1]]
        texts=[dict(t,bbox=[t['bbox'][0],h-t['bbox'][3],t['bbox'][2],h-t['bbox'][1]]) for t in result.get('texts',[])]
        located=locator(image,box)
        numbered=number_side_points(located,box,texts)
        model,identity=library.identify(c.get('Name'),c['bbox'],result.get('texts',[])) if library else (None,{'status':'library_disabled'})
        recovered=recover_name_anchors(located,box,texts,library.entries[model]['pins'],numbered,
                                      (n[4:] for n in result['pins'].get(key,{}))) if name_anchors and model else []
        for p in recovered:
            nearest=min(located,key=lambda q:(p['x']-q['x'])**2+(p['y']-q['y'])**2)
            p['side']=nearest['side']
        recovered=[p for p in recovered if perpendicular_stroke(image,p)]
        numbered+=recovered
        funnel.append({'component':key,'library_model':model,'identity_status':identity.get('status','unknown'),'name_anchored':len(recovered),
                       'locator':dict(getattr(locator,'last_report',{})), 'proposals':len(located),
                       'numbered_candidates':len(numbered),'existing_pins':len(result['pins'].get(key,{}))})
        if not numbered: continue
        pins={}
        for p in numbered:
            n=str(p['number']); pins['pin_'+n]={'pinname':p['name'],'point':{'x':p['x'],'y':h-p['y']}}
        before=copy.deepcopy(result['pins'][key]); result['pins'][key]=pins
        result['pin_hypotheses']=[p for p in result.get('pin_hypotheses',[]) if p['component']!=key]
        for i,p in enumerate(located):
            if any(math.hypot(p['x']-q['x'],p['y']-q['y'])<3 for q in numbered): continue
            result['pin_hypotheses'].append({'id':key+'@strip_'+str(i),'component':key,'point':{'x':p['x'],'y':h-p['y']},'score':p['score'],'source':'side_strip','status':'number_unresolved'})
        if before!=pins: edits.append({'component':key,'before_count':len(before),'after_count':len(pins),'proposals':len(located)})
    semantic=None; structured=None
    if library:
        if edits: result,semantic=library.refine(result)
        if structure: result,structured=StructureAwareICParser(library).refine(result)
    accepted=[]
    if conservative:
        proposed=result; result=copy.deepcopy(observations)
        if library:
            for key,c in proposed['components'].items():
                if c['type'] not in ('ic','box'): continue
                model,_=library.identify(c.get('Name'),c['bbox'],proposed.get('texts',[]))
                if not model: continue
                catalog=library.entries[model]['pins']; existing=result['pins'][key]
                for n,p in proposed['pins'][key].items():
                    if p['pinname'] not in catalog.get(n[4:],{}).get('names',[]): continue
                    original=existing.get(n)
                    if original and math.dist((original['point']['x'],original['point']['y']),(p['point']['x'],p['point']['y']))>48: continue
                    if not original and any(math.dist((q['point']['x'],q['point']['y']),(p['point']['x'],p['point']['y']))<6 for q in existing.values()): continue
                    if original!=p:
                        existing[n]=copy.deepcopy(p); accepted.append({'component':key,'pin':n,'action':'refine' if original else 'add','catalog_name':p['pinname']})
        # Keep unresolved proposals for diagnostics only; no unnumbered pin is exported.
    for row in funnel:
        changes=[e for e in accepted if e['component']==row['component']]
        row['accepted_add']=sum(e['action']=='add' for e in changes)
        row['accepted_refine']=sum(e['action']=='refine' for e in changes)
    return result,{'module':'SideStripProposalAssociation','components':edits,'semantic':semantic,'structure':structured,'accepted':accepted,'funnel':funnel,'conservative':conservative,'name_anchors':name_anchors,'ground_truth_used':False}
def band(gray,box,side,start):
    normal=np.arange(64,dtype=np.float32)[:,None]-32
    axis=np.arange(256,dtype=np.float32)[None,:]+start
    if side=='left': x=box[0]+normal; y=axis
    elif side=='right': x=box[2]-normal; y=axis
    elif side=='top': x=axis; y=box[1]+normal
    elif side=='bottom': x=axis; y=box[3]-normal
    else: raise ValueError('unknown side')
    x=np.broadcast_to(x,(64,256)).copy(); y=np.broadcast_to(y,(64,256)).copy()
    return cv2.remap(gray,x,y,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=255)

def tensor(patch):
    coord=np.broadcast_to((np.arange(64,dtype=np.float32)[:,None]-32)/32,(64,256))
    return np.ascontiguousarray(np.stack((patch.astype(np.float32)/255,(patch<180).astype(np.float32),coord)))

def world(box,side,start,axis,normal):
    if side=='left': return float(box[0]+normal),float(start+axis)
    if side=='right': return float(box[2]-normal),float(start+axis)
    if side=='top': return float(start+axis),float(box[1]+normal)
    if side=='bottom': return float(start+axis),float(box[3]-normal)
    raise ValueError('unknown side')

class SideStripLocator:
    def __init__(self,path,threads=2):
        import onnxruntime as ort
        path=Path(path); self.meta=json.loads(path.with_suffix('.json').read_text(encoding='utf-8'))
        options=ort.SessionOptions(); options.intra_op_num_threads=threads; options.inter_op_num_threads=threads
        self.session=ort.InferenceSession(str(path),sess_options=options,providers=['CPUExecutionProvider'])
        self.input=self.session.get_inputs()[0].name

    def __call__(self,image,box,threshold=None):
        gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY); candidates=[]
        threshold=self.meta['threshold'] if threshold is None else threshold
        report={'threshold':float(threshold),'bands':0,'threshold_peaks':0,'in_axis_range':0,'stroke_valid':0}
        for side in SIDES:
            lo,hi=(box[1],box[3]) if side in ('left','right') else (box[0],box[2])
            for start in range(int(lo)-16,int(hi)+1,224):
                patch=band(gray,box,side,start); output=self.session.run(None,{self.input:tensor(patch)[None]})[0][0]
                report['bands']+=1
                probability=1/(1+np.exp(-np.clip(output[0],-30,30)))
                maxima=cv2.dilate(probability[None],np.ones((1,7),np.uint8))[0]
                for axis in np.flatnonzero((probability>=threshold)&(probability>=maxima-1e-7)):
                    report['threshold_peaks']+=1
                    if not lo-5<=start+axis<=hi+5: continue
                    report['in_axis_range']+=1
                    normal=float(np.tanh(output[1,axis])*32)
                    x,y=world(box,side,start,float(axis),normal)
                    ix,iy=int(round(x)),int(round(y)); a=max(0,ix-3); b=min(gray.shape[1],ix+4); c=max(0,iy-3); d=min(gray.shape[0],iy+4)
                    if a>=b or c>=d or np.count_nonzero(gray[c:d,a:b]<180)<3: continue
                    report['stroke_valid']+=1
                    candidates.append({'x':x,'y':y,'score':float(probability[axis]),'side':side,'source':'side_strip'})
        selected=[]
        for p in sorted(candidates,key=lambda p:p['score'],reverse=True):
            if not any((p['x']-q['x'])**2+(p['y']-q['y'])**2<36 for q in selected): selected.append(p)
        report['after_nms']=len(selected); self.last_report=report
        return selected
