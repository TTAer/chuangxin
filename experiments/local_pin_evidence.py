"""Add-only experimental identity evidence; frozen parser never imports this."""
import copy
import math
import re
import cv2
from pcbparse.core import point_on_segment,references
from pcbparse.graphschematic import DualGraph
from pcbparse.identity_transaction import wire_signature
from pcbparse.semantic_library import center,norm

def edge_key(edge):
    return tuple(sorted(tuple(round(p[k],6) for k in ('x','y')) for p in edge))

def commit_additions(before,observations,proposals):
    """Existing identities/nets immutable; new memberships need physical contact."""
    result=copy.deepcopy(before); physical=copy.deepcopy(observations)
    physical.update(components={},pins={},labels=[])
    graph=DualGraph(physical); root_nets={}; segment_roots={edge_key(e):graph.uf.root(i) for i,e in enumerate(graph.wires)}
    for net_id,net in before['nets'].items():
        for edge in net['edges'].values():
            root=segment_roots.get(edge_key(edge))
            if root is not None: root_nets.setdefault(root,set()).add(net_id)
    decisions=[]; serial=1
    for p in proposals:
        key=p['component']; number=p['number']; pin='pin_'+number; point=p['point']
        reason=None
        if pin in result['pins'].get(key,{}): reason='existing_identity_locked'
        elif any(math.dist(tuple(point[k] for k in ('x','y')),tuple(q['point'][k] for k in ('x','y')))<6 for q in result['pins'].get(key,{}).values()): reason='existing_geometry_anchor_locked'
        wires=[i for i in graph.index.query([point['x']-3,point['y']-3,point['x']+3,point['y']+3]) if point_on_segment(point,graph.wires[i],3) and graph.contact_allowed(point)]
        roots={graph.uf.root(i) for i in wires}
        nets=set().union(*(root_nets.get(root,set()) for root in roots)) if roots else set()
        if reason is None and (len(roots)!=1 or len(nets)>1): reason='no_unique_physical_contact_or_old_net_conflict'
        if reason:
            decisions.append(dict(p,committed=False,reason=reason)); continue
        root=next(iter(roots)); ref=key+'.'+number
        if nets: net_id=next(iter(nets))
        else:
            while 'net_local_'+str(serial) in result['nets']: serial+=1
            net_id='net_local_'+str(serial); serial+=1
            edges=[edge for i,edge in enumerate(graph.wires) if graph.uf.root(i)==root]
            result['nets'][net_id]={'hyperGraph':'()','edges':{'edge_'+str(i+1):copy.deepcopy(edge) for i,edge in enumerate(edges)}}
            root_nets[root]={net_id}
        result['pins'].setdefault(key,{})[pin]={'pinname':p['name'],'point':copy.deepcopy(point)}
        members=set(references(result['nets'][net_id]['hyperGraph'])); members.add(ref)
        result['nets'][net_id]['hyperGraph']='('+','.join(sorted(members))+')'
        decisions.append(dict(p,committed=True,net=net_id))
    for key,pins in before['pins'].items():
        assert all(result['pins'][key][number]==value for number,value in pins.items())
    assert result['components']==before['components']
    for key,net in before['nets'].items():
        assert result['nets'][key]['edges']==net['edges']
        old=set(references(net['hyperGraph'])); new=set(references(result['nets'][key]['hyperGraph']))
        assert old<=new
    return result,decisions

class LocalPinEvidence:
    def __init__(self,library):
        from rapidocr_onnxruntime import RapidOCR
        self.ocr=RapidOCR(intra_op_num_threads=2,inter_op_num_threads=2,det_limit_type='max',det_limit_side_len=736)
        self.library=library

    def collect(self,observations,image):
        h,w=image.shape[:2]; tokens=[]; proposals=[]; counts={'unresolved':0,'eligible_geometry':0,'crops':0,'catalog_components':0}; records=[]
        for key,c in observations['components'].items():
            if c['type'] not in ('ic','box'): continue
            model,identity=self.library.identify(c.get('Name'),c['bbox'],observations.get('texts',[]))
            tips=[p for p in observations.get('pin_hypotheses',[]) if p['component']==key and p.get('status')=='number_unresolved']
            counts['unresolved']+=len(tips)
            if not model: continue
            counts['catalog_components']+=1; catalog=self.library.entries[model]['pins']; old=observations['pins'].get(key,{})
            tips=[p for p in tips if p.get('score',0)>=.7 and not any(math.dist(tuple(p['point'][k] for k in ('x','y')),tuple(q['point'][k] for k in ('x','y')))<6 for q in old.values())]
            counts['eligible_geometry']+=len(tips)
            local=[]; box=c['bbox']
            for tip in tips:
                x,y=tip['point']['x'],h-tip['point']['y']; vertical=min(abs(tip['point']['y']-box[1]),abs(tip['point']['y']-box[3]))<min(abs(x-box[0]),abs(x-box[2]))
                rx,ry=(24,48) if vertical else (48,24)
                x1,y1,x2,y2=max(0,round(x-rx)),max(0,round(y-ry)),min(w,round(x+rx)),min(h,round(y+ry))
                if x1>=x2 or y1>=y2: continue
                patch=cv2.resize(image[y1:y2,x1:x2],None,fx=2,fy=2,interpolation=cv2.INTER_CUBIC)
                output,_=self.ocr(patch); counts['crops']+=1
                for bb,text,confidence in output or []:
                    b=[min(p[0] for p in bb)/2+x1,h-(max(p[1] for p in bb)/2+y1),max(p[0] for p in bb)/2+x1,h-(min(p[1] for p in bb)/2+y1)]
                    t={'text':text,'bbox':b,'confidence':float(confidence),'component':key}
                    if not any(t['text']==q['text'] and math.dist(center(b),center(q['bbox']))<4 for q in local): local.append(t)
            tokens.extend(local)
            used=set()
            for tip in tips:
                point=tip['point']; x,y=point['x'],point['y']; vertical=min(abs(y-box[1]),abs(y-box[3]))<min(abs(x-box[0]),abs(x-box[2])); axis=0 if vertical else 1
                choices=[]
                for t in local:
                    number=t['text'].strip()
                    if not re.fullmatch(r'[A-Z]?[0-9]+',number) or t['confidence']<.95 or number not in catalog or 'pin_'+number in old: continue
                    b=t['bbox']; axial=max(b[axis]-point['xy'[axis]],point['xy'[axis]]-b[axis+2],0)
                    distances=sorted(math.dist(center(b),tuple(p['point'][k] for k in ('x','y'))) for p in tips)
                    if axial>6 or math.dist(center(b),(x,y))>40 or len(distances)>1 and distances[1]-distances[0]<4: continue
                    if math.dist(center(b),(x,y))>distances[0]+.01: continue
                    names=[(n,q) for n in catalog[number]['names'] for q in local if norm(q['text'])==norm(n) and q['confidence']>=.9 and math.dist(center(q['bbox']),(x,y))<=48 and abs(center(q['bbox'])[axis]-point['xy'[axis]])<=8]
                    if not names: continue
                    choices.append((number,names[0][0],t))
                if len({p[0] for p in choices})!=1:
                    records.append({'component':key,'tip':tip['id'],'status':'no_unique_number_and_visible_name'}); continue
                number,name,text=choices[0]
                if number in used: continue
                used.add(number); proposals.append({'component':key,'number':number,'name':name,'point':copy.deepcopy(point),'tip':tip['id'],'model':model,'ocr_number':text,'authority':'proposal'})
        return proposals,{'counts':counts,'tokens':tokens,'records':records,'proposals':len(proposals),'physical_signature':wire_signature(observations),'gt_used':False}
