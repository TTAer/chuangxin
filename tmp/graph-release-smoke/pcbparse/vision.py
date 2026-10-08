"""Image-only offline baseline: OCR, learned symbol templates and wire geometry."""
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
import time

import cv2
import numpy as np

from .core import assignment, build, validate
from .dataset import CATEGORIES, DESIGNATOR, index_dataset, pin_specs


def read_image(path):
    a=cv2.imdecode(np.fromfile(str(path),dtype=np.uint8),cv2.IMREAD_COLOR)
    if a is None: raise ValueError('cannot decode image: '+str(path))
    return a


def write_image(path, image):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    cv2.imencode(path.suffix,image)[1].tofile(str(path))


def masks(image):
    b,g,r=cv2.split(image.astype(np.int16))
    red=(r-g>35)&(r-b>25)&(r>90)
    green=(g-r>20)&(g-b>10)&(g>45)
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    # Colored schematics allow separating symbols from blue text and green wire.
    colored=int(red.sum())>max(20,image.shape[0]*image.shape[1]*.0002)
    symbol=red if colored else gray<170
    return symbol.astype(np.uint8)*255,green.astype(np.uint8)*255,colored


def iou(a,b):
    x=max(0,min(a[2],b[2])-max(a[0],b[0])); y=max(0,min(a[3],b[3])-max(a[1],b[1]))
    intersection=x*y
    return intersection/((a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-intersection+1e-9)


def train_templates(root, output, per_category=14):
    items=index_dataset(root); candidates={}; skipped=Counter(); successful=0
    for item in items:
        if item['split']!='train': continue
        raw=json.loads(Path(item['annotation']).read_text(encoding='utf-8'))
        image=read_image(item['image']); symbol,_,colored=masks(image)
        for c in raw['components'].values():
            cat=c['category']
            if cat in ('Box','Block'): continue
            b=c['location']; x1,y1,x2,y2=[int(b[k]) for k in ('x1','y1','x2','y2')]
            x1=max(0,x1); y1=max(0,y1); x2=min(image.shape[1],x2); y2=min(image.shape[0],y2)
            w,h=x2-x1,y2-y1
            if min(w,h)<4 or max(w,h)>140: skipped['size']+=1; continue
            crop=symbol[y1:y2,x1:x2].copy()
            # Exclude annotated text only when constructing TRAINING templates.
            for t in c.get('associatedTexts',{}).values():
                q=t.get('location',{})
                if not q: continue
                ax=max(0,int(q['x1'])-x1); ay=max(0,int(q['y1'])-y1)
                bx=min(w,int(q['x2'])-x1); by=min(h,int(q['y2'])-y1)
                if ax<bx and ay<by: crop[ay:by,ax:bx]=0
            density=np.count_nonzero(crop)/(w*h)
            if not .015<density<.65: skipped['density']+=1; continue
            signature=(cat,round(w/6),round(h/6),colored)
            if signature in candidates: continue
            specs=[]
            for _,number,name,p in pin_specs(raw,c):
                # Symbol-specific pin numbering transfers; text/designator/value do not.
                specs.append({'number':number,'name':name,'x':(p['tip'][0]-x1)/w,'y':(p['tip'][1]-y1)/h})
            if not specs: skipped['no_pins']+=1; continue
            candidates[signature]=(crop,{'category':cat,'pins':specs,'train_id':item['id'],'sha256':item['sha256']})
            successful+=1
    selected=[]; counts=Counter()
    # Round-robin diversity across orientation and color rather than first page only.
    ordered=sorted(candidates.items(),key=lambda kv:hashlib.sha256(str(kv[0]).encode()).hexdigest())
    for signature,entry in ordered:
        cat=signature[0]
        if counts[cat]>=per_category: continue
        selected.append(entry); counts[cat]+=1
    out=Path(output); out.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(out/'templates.npz',**{'t'+str(i):x[0] for i,x in enumerate(selected)})
    metadata={'version':1,'training_method':'binary symbol templates from training split only','templates':[x[1] for x in selected],'counts':dict(counts),'train_cases':sum(i['split']=='train' for i in items),'validation_cases':sum(i['split']=='validation' for i in items),'skipped':dict(skipped)}
    (out/'templates.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding='utf-8')
    (out/'dataset-index.json').write_text(json.dumps(items,ensure_ascii=False,indent=2),encoding='utf-8')
    return {k:v for k,v in metadata.items() if k!='templates'}


def wire_lines(mask, min_length=8, split_junctions=True):
    """Consolidate collinear stroke rows, then split at perpendicular contacts."""
    result=[]
    for vertical in (False,True):
        m=mask.T if vertical else mask
        kernel_length=min_length if min_length%2 else min_length+1
        opened=cv2.morphologyEx(m,cv2.MORPH_OPEN,np.ones((1,kernel_length),np.uint8))
        n,labels,stats,centers=cv2.connectedComponentsWithStats(opened,8)
        for x,y,w,h,area in stats[1:]:
            if w<min_length or h>max(8,w*.3): continue
            mid=float(y+(h-1)/2)
            a=(mid,float(x)) if vertical else (float(x),mid)
            b=(mid,float(x+w-1)) if vertical else (float(x+w-1),mid)
            result.append((a,b))
    if not split_junctions: return result,[]
    split=[set((0.,1.)) for _ in result]; junctions=[]
    for i,(a,b) in enumerate(result):
        for j in range(i):
            c,d=result[j]
            if (a[0]==b[0])==(c[0]==d[0]): continue
            h=(a,b) if a[1]==b[1] else (c,d)
            v=(c,d) if a[1]==b[1] else (a,b)
            x,y=v[0][0],h[0][1]
            if h[0][0]-2<=x<=h[1][0]+2 and v[0][1]-2<=y<=v[1][1]+2:
                # A thick filled dot gives interior crossing evidence; endpoints
                # are T/turn evidence. Thin interior crosses remain unsplit.
                endpoint=min(math.hypot(x-p[0],y-p[1]) for p in (a,b,c,d))<=3
                ix,iy=int(round(x)),int(round(y)); patch=mask[max(0,iy-2):iy+3,max(0,ix-2):ix+3]
                dot=patch.size==25 and np.count_nonzero(patch)>=19
                if not endpoint and not dot: continue
                junctions.append((x,y))
                for k in (i,j):
                    p,q=result[k]; denom=(q[0]-p[0])**2+(q[1]-p[1])**2
                    if denom: split[k].add(max(0,min(1,((x-p[0])*(q[0]-p[0])+(y-p[1])*(q[1]-p[1]))/denom)))
    segments=[]
    for (a,b),positions in zip(result,split):
        t=sorted(positions)
        for s,e in zip(t,t[1:]):
            p=(a[0]+s*(b[0]-a[0]),a[1]+s*(b[1]-a[1])); q=(a[0]+e*(b[0]-a[0]),a[1]+e*(b[1]-a[1]))
            if math.dist(p,q)>=2: segments.append((p,q))
    return segments,junctions


class ImageParser:
    def __init__(self, model='models/symbols', threshold=.78, threads=2, detector='models/detector/pcb-yolo11n.onnx', pin_model='auto', local_pin_ocr=False, text_constraints=False, graph_mode='auto', topology_model='auto'):
        from rapidocr_onnxruntime import RapidOCR
        config_path=Path(__file__).resolve().parents[1]/'models/pipeline.json'
        config=json.loads(config_path.read_text(encoding='utf-8')) if config_path.exists() else {}
        if graph_mode=='auto': graph_mode=config.get('graph_mode','legacy')
        if pin_model=='auto':
            pin_model=config.get('pin_model')
            local_pin_ocr=local_pin_ocr or config.get('local_pin_ocr',False)
            text_constraints=text_constraints or config.get('text_constraints',False)
        if pin_model=='none': pin_model=None
        self.ocr=RapidOCR(intra_op_num_threads=threads,inter_op_num_threads=threads,det_limit_side_len=1600)
        self.threshold=threshold
        self.local_pin_ocr=local_pin_ocr
        self.text_constraints=text_constraints
        self.detector_path=str(detector) if detector else None
        self.pin_model_path=str(pin_model) if pin_model else None
        if graph_mode not in ('legacy','association','full'): raise ValueError('unknown graph mode')
        self.graph_mode=graph_mode; self.last_graph_report=None; self.last_review_queue=None
        self.topology=None
        if graph_mode=='full':
            from .topology_perception import TopologyPerception
            if topology_model=='auto':
                topology_model='models/topology/topology-perception.onnx' if Path('models/topology/topology-perception.onnx').exists() else None
            if topology_model=='none': topology_model=None
            self.topology=TopologyPerception(topology_model,threads)
        path=Path(model)
        if not (path/'templates.json').exists(): raise ValueError('symbol model missing; run train-templates first')
        self.metadata=json.loads((path/'templates.json').read_text(encoding='utf-8'))
        with np.load(path/'templates.npz',allow_pickle=False) as arrays:
            self.templates=[arrays['t'+str(i)] for i in range(len(self.metadata['templates']))]
        self.detector=None
        if detector:
            from .detector import OnnxDetector
            self.detector=OnnxDetector(detector,threads=threads)
        self.pin_locator=None
        if pin_model:
            from .pins import PinLocator
            self.pin_locator=PinLocator(pin_model,threads=threads)

    def texts(self,image):
        # Small pin numbers need more pixels; offsets are mapped back to original.
        scale=2 if max(image.shape[:2])<1600 else 1
        large=cv2.resize(image,None,fx=scale,fy=scale,interpolation=cv2.INTER_CUBIC) if scale!=1 else image
        result,_=self.ocr(large)
        texts=[]
        for box,text,confidence in result or []:
            points=np.array(box)/scale
            texts.append({'text':text,'confidence':float(confidence),'bbox':[float(points[:,0].min()),float(points[:,1].min()),float(points[:,0].max()),float(points[:,1].max())]})
        return texts

    def detect(self,image,texts):
        symbol,green,colored=masks(image); search=symbol.copy()
        if self.detector is not None:
            auxiliary={'aitee_edge','aitee_junction','current_source','voltage_source','ident-i','ident-i-dir','ident-res','ident-v','ident-v-dir'}
            detections=[d for d in self.detector(image) if d['category'] not in auxiliary]
            # Match TRAINING symbol geometry only inside detected boxes. Ground
            # truth files are never read during image inference.
            for d in detections:
                if d['category'] in ('Box','Block'): continue
                x1,y1,x2,y2=map(round,d['bbox']); crop=symbol[y1:y2,x1:x2]
                if crop.size==0: continue
                best=-1; specs=[]
                target=cv2.resize(crop,(48,48),interpolation=cv2.INTER_NEAREST).astype(np.float32)
                target-=target.mean()
                for template,meta in zip(self.templates,self.metadata['templates']):
                    if meta['category']!=d['category']: continue
                    t=cv2.resize(template,(48,48),interpolation=cv2.INTER_NEAREST).astype(np.float32); t-=t.mean()
                    sim=float(np.sum(t*target)/(np.linalg.norm(t)*np.linalg.norm(target)+1e-6))
                    if sim>best: best=sim; specs=meta['pins']
                d['pins']=specs
            return detections,symbol,green,colored
        if not colored:
            for t in texts:
                x1,y1,x2,y2=map(int,t['bbox']); search[max(0,y1):y2,max(0,x1):x2]=0
        candidates=[]
        for template,meta in zip(self.templates,self.metadata['templates']):
            for scale in (.75,1.,1.3):
                h,w=template.shape; tw,th=round(w*scale),round(h*scale)
                if min(tw,th)<4 or tw>=search.shape[1] or th>=search.shape[0]: continue
                resized=cv2.resize(template,(tw,th),interpolation=cv2.INTER_NEAREST)
                scores=cv2.matchTemplate(search,resized,cv2.TM_CCOEFF_NORMED)
                peaks=cv2.dilate(scores,np.ones((max(3,th//2),max(3,tw//2)),np.uint8))
                ys,xs=np.where((scores>=self.threshold)&(scores>=peaks-1e-6))
                ranked=sorted(zip(ys,xs),key=lambda p:scores[p],reverse=True)[:40]
                for y,x in ranked:
                    candidates.append({'bbox':[float(x),float(y),float(x+tw),float(y+th)],'score':float(scores[y,x]),'category':meta['category'],'pins':meta['pins']})
        # IC body rectangles are too variable for small symbol templates.
        horizontal=cv2.morphologyEx(search,cv2.MORPH_OPEN,np.ones((1,18),np.uint8))
        vertical=cv2.morphologyEx(search,cv2.MORPH_OPEN,np.ones((18,1),np.uint8))
        contours,_=cv2.findContours(horizontal|vertical,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            x,y,w,h=cv2.boundingRect(contour)
            if min(w,h)<20 or w*h>image.shape[0]*image.shape[1]*.65: continue
            perimeter=cv2.arcLength(contour,True); approx=cv2.approxPolyDP(contour,.025*perimeter,True)
            if len(approx)!=4: continue
            if abs(cv2.contourArea(contour)-w*h)/(w*h)>.25: continue
            # Ignore border/title boxes without nearby designator evidence.
            relevant=[t for t in texts if DESIGNATOR.fullmatch(t['text']) and re.match(r'^(U|IC|J|P|CN)',t['text'],re.I) and box_distance(t['bbox'],[x,y,x+w,y+h])<40]
            if relevant: candidates.append({'bbox':[float(x),float(y),float(x+w-1),float(y+h-1)],'score':.92,'category':'Box','pins':[]})
        chosen=[]
        for c in sorted(candidates,key=lambda c:c['score'],reverse=True):
            if any(iou(c['bbox'],d['bbox'])>.3 for d in chosen): continue
            # Reject tiny fragments within already selected IC bodies.
            center=((c['bbox'][0]+c['bbox'][2])/2,(c['bbox'][1]+c['bbox'][3])/2)
            if any(d['category']=='Box' and d['bbox'][0]<center[0]<d['bbox'][2] and d['bbox'][1]<center[1]<d['bbox'][3] for d in chosen): continue
            chosen.append(c)
        return chosen,symbol,green,colored

    def parse(self,path):
        started=time.perf_counter(); image=read_image(path); h,w=image.shape[:2]
        texts=self.texts(image); detections,symbol,green,colored=self.detect(image,texts)
        designators=[t for t in texts if DESIGNATOR.fullmatch(t['text'])]
        scores=[]
        for d in detections:
            scores.append([max(0,100-box_distance(t['bbox'],d['bbox'])) if compatible(t['text'],d['category']) else 0 for t in designators])
        assigned={i:j for i,j in assignment(scores) if scores[i][j]>25}
        entries=[]; warnings=[]
        for i,d in enumerate(detections):
            b=d['bbox']; cat=d['category']; nearby=sorted(texts,key=lambda t:box_distance(t['bbox'],b))
            designator=designators[assigned[i]]['text'] if i in assigned else None
            nettext=next((t['text'] for t in nearby if not DESIGNATOR.fullmatch(t['text']) and box_distance(t['bbox'],b)<45 and not re.fullmatch(r'\d+',t['text'])),None)
            if cat=='Gnd': key='GND'
            elif cat=='V':
                power=next((t['text'] for t in nearby if re.fullmatch(r'(?:[+-]?\d+(?:\.\d+)?V|VCC|VDD\w*|VSS\w*|VBUS|VBAT|GND)',t['text'],re.I) and box_distance(t['bbox'],b)<60),None)
                key=power or 'VCC'
            elif cat in ('NET_input','Net_output','Net_bidirection','Testpoint'): key=nettext or 'LABEL'
            elif designator: key=designator
            else:
                # Do not emit unsupported component identities just to inflate counts.
                warnings.append('unassigned '+cat+' symbol'); continue
            key=re.sub(r'[\s,()]','_',key)
            value=next((t['text'] for t in nearby if is_value(t['text']) and box_distance(t['bbox'],b)<70),None)
            if cat in ('R','C') and value is None: value=''; warnings.append('value missing: '+key)
            name=next((t['text'] for t in nearby if not DESIGNATOR.fullmatch(t['text']) and not re.fullmatch(r'\d+',t['text']) and not is_value(t['text']) and box_distance(t['bbox'],b)<45),None) if cat=='Box' else None
            if self.text_constraints:
                if cat=='R' and value: value=correct_resistance_ocr(value)
                if cat=='Box':
                    name=select_ic_name(nearby,b)
                    value=None
            ps=self.ic_pins(d,symbol,texts) if cat in ('Box','Block') else []
            located=[]
            if not ps:
                for pin in d['pins']:
                    ps.append({'number':pin['number'],'name':pin['name'],'x':b[0]+pin['x']*(b[2]-b[0]),'y':b[1]+pin['y']*(b[3]-b[1])})
            if self.pin_locator:
                located=self.pin_locator(image,b)
                if cat in ('Box','Block'):
                    pin_texts=self.local_texts(image,b,texts) if self.local_pin_ocr else texts
                    learned=self.number_located_pins(located,b,pin_texts)
                    if learned: ps=learned
                else:
                    ps=refine_pin_locations(ps,located,b)
            entries.append({'key':key,'bbox':b,'category':cat,'value':value,'name':name,'pins':ps,'score':d['score'],'unresolved_tips':[p for p in located if cat in ('Box','Block') and not any(math.dist((p['x'],p['y']),(q['x'],q['y']))<3 for q in ps)]})
        counts=Counter(e['key'] for e in entries); serial=Counter(); components={}; pins={}; labels=[]; pin_hypotheses=[]
        for e in sorted(entries,key=lambda e:(e['bbox'][1],e['bbox'][0])):
            k=e['key']; serial[k]+=1
            if counts[k]>1: k+='_'+str(serial[k])
            b=e['bbox']; components[k]={'type':CATEGORIES.get(e['category'],e['category'].lower()),'Name':e['name'],'value':e['value'],'bbox':[b[0],h-b[3],b[2],h-b[1]]}
            pins[k]={}
            for j,p in enumerate(e['unresolved_tips']):
                pin_hypotheses.append({'id':k+'@tip_'+str(j),'component':k,'point':{'x':float(p['x']),'y':float(h-p['y'])},'score':p['score'],'source':'learned_tip_heatmap','status':'number_unresolved'})
            for j,p in enumerate(e['pins']):
                n=str(p['number']); n=n if re.fullmatch(r'[^\s,().]+',n) else str(j+1)
                if 'pin_'+n in pins[k]: warnings.append('duplicate pin number: '+k+'.'+n); continue
                pins[k]['pin_'+n]={'pinname':p['name'],'point':{'x':float(p['x']),'y':float(h-p['y'])}}
            if e['category'] in ('Gnd','V','NET_input','Net_output','Net_bidirection','Testpoint') and pins[k]:
                n=next(iter(pins[k]))[4:]; labels.append({'name':e['key'],'pin':k+'.'+n})
        if colored:
            line_mask=green
        else:
            line_mask=symbol.copy()
            for t in texts:
                x1,y1,x2,y2=map(int,t['bbox']); line_mask[max(0,y1):y2,max(0,x1):x2]=0
            for e in entries:
                x1,y1,x2,y2=map(int,e['bbox']); line_mask[max(0,y1):y2,max(0,x1):x2]=0
        topology_report=None
        if self.topology:
            segments,junctions,line_mask,topology_report=self.topology.perceive(image,[d['bbox'] for d in detections],texts)
        else: segments,junctions=wire_lines(line_mask)
        def point(p): return {'x':float(p[0]),'y':float(h-p[1])}
        obs={'components':components,'pins':pins,'wires':[[point(a),point(b)] for a,b in segments],'junctions':[point(p) for p in junctions],'labels':labels,'coordinate_system':'bottom-left','image':{'width':w,'height':h}}
        obs['texts']=[dict(t,bbox=[t['bbox'][0],h-t['bbox'][3],t['bbox'][2],h-t['bbox'][1]]) for t in texts]
        obs['pin_hypotheses']=pin_hypotheses
        if topology_report:
            obs['contact_decisions']=[{'point':point((c['x'],c['y'])),'state':c['state'],'probability':c['probability'],'evidence':c['evidence']} for c in topology_report['crossings']]
        if self.graph_mode!='legacy':
            from .graphschematic import parse_graph
            from .corrections import review_queue
            result,self.last_graph_report=parse_graph(obs,line_mask=line_mask)
            self.last_graph_report['topology_perception']=topology_report
            self.last_review_queue=review_queue(obs,self.last_graph_report,hashlib.sha256(Path(path).read_bytes()).hexdigest())
        else: result=build(obs,tolerance=3)
        errors=validate(result)
        if errors: raise ValueError('; '.join(errors))
        diagnostics={'input':str(Path(path).resolve()),'image_sha256':hashlib.sha256(Path(path).read_bytes()).hexdigest(),'seconds':time.perf_counter()-started,'backend':'yolo-onnx-ocr' if self.detector else 'offline-templates-ocr','learned_pin_branch':bool(self.pin_locator),'local_pin_ocr':self.local_pin_ocr,'text_constraints':self.text_constraints,'detector':self.detector_path,'pin_model':self.pin_model_path,'components':len(components),'pins':sum(map(len,pins.values())),'nets':len(result['nets']),'wires':len(segments),'ocr_regions':len(texts),'colored':colored,'warnings':warnings,'tokens':0,'official_equivalence':False}
        diagnostics['graph_mode']=self.graph_mode
        if self.last_graph_report:
            diagnostics['graphschematic']={'accepted_associations':len(self.last_graph_report['association']['accepted']),'circuit_issues':len(self.last_graph_report['circuit_issues']),'llm_status':'queued_not_called','review_tasks':len(self.last_review_queue['tasks']),'topology_policy':'electrical topology > geometric proximity > color feature' if self.topology else 'legacy extraction; geometry association ablation'}
        return result,obs,diagnostics,texts

    def local_texts(self,image,box,global_texts):
        h,w=image.shape[:2]
        x1,y1,x2,y2=box; a=max(0,int(x1)-15); b=max(0,int(y1)-15); c=min(w,int(x2)+16); d=min(h,int(y2)+16)
        if min(c-a,d-b)<5: return global_texts
        crop=image[b:d,a:c]
        scale=min(4.,1200/max(crop.shape[:2]))
        result,_=self.ocr(cv2.resize(crop,None,fx=scale,fy=scale,interpolation=cv2.INTER_CUBIC))
        local=[]
        for polygon,text,confidence in result or []:
            points=np.asarray(polygon,dtype=float)/scale; points[:,0]+=a; points[:,1]+=b
            local.append({'text':text,'confidence':float(confidence),'bbox':[float(points[:,0].min()),float(points[:,1].min()),float(points[:,0].max()),float(points[:,1].max())]})
        # Replace overlapping global regions rather than create competing copies.
        return local+[t for t in global_texts if not any(iou(t['bbox'],q['bbox'])>.3 for q in local)]

    def number_located_pins(self,points,box,texts):
        numbers=[t for t in texts if re.fullmatch(r'[A-Za-z]?\d+',t['text']) and not DESIGNATOR.fullmatch(t['text']) and box_distance(t['bbox'],box)<15]
        if not points or not numbers: return []
        scores=[]
        for p in points:
            # Pin text follows the stub direction. Penalize transverse offset.
            vertical=min(abs(p['y']-box[1]),abs(p['y']-box[3]))<min(abs(p['x']-box[0]),abs(p['x']-box[2]))
            row=[]
            for t in numbers:
                tx,ty=center(t['bbox']); distance=math.hypot(tx-p['x'],ty-p['y'])
                transverse=abs(tx-p['x']) if vertical else abs(ty-p['y'])
                row.append(max(0,35-distance-2*transverse) if distance<30 else 0)
            scores.append(row)
        used=set(); result=[]
        for i,j in assignment(scores):
            if scores[i][j]<=5: continue
            number=numbers[j]['text']
            if number in used: continue
            p=points[i]; used.add(number)
            names=[t for t in texts if not re.fullmatch(r'\d+',t['text']) and not DESIGNATOR.fullmatch(t['text']) and box[0]<=center(t['bbox'])[0]<=box[2] and box[1]<=center(t['bbox'])[1]<=box[3]]
            names.sort(key=lambda t:math.dist(center(t['bbox']),(p['x'],p['y'])))
            name=names[0]['text'] if names and math.dist(center(names[0]['bbox']),(p['x'],p['y']))<50 else ''
            result.append({'number':number,'name':name,'x':p['x'],'y':p['y']})
        return result

    def ic_pins(self,d,symbol,texts):
        x1,y1,x2,y2=map(int,d['bbox']); h,w=symbol.shape; points=[]
        # YOLO predicts the outer box including pins. Locate the body rectangle
        # inside it before scanning pin stubs.
        crop=symbol[max(0,y1):min(h,y2),max(0,x1):min(w,x2)]
        contours,_=cv2.findContours(crop,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
        bodies=[]
        for contour in contours:
            bx,by,bw,bh=cv2.boundingRect(contour)
            if min(bw,bh)<15: continue
            approx=cv2.approxPolyDP(contour,.02*cv2.arcLength(contour,True),True)
            if len(approx)==4 and cv2.contourArea(contour)>bw*bh*.65: bodies.append((bw*bh,bx,by,bw,bh))
        if bodies:
            _,bx,by,bw,bh=max(bodies); x2=x1+bx+bw-1; y2=y1+by+bh-1; x1+=bx; y1+=by
        # Scan red/black pin stubs extending from body sides. A run's outside
        # endpoint is the tip, not the point at the body border.
        for side in ('left','right','top','bottom'):
            if side in ('left','right'):
                ax=max(0,x1-22) if side=='left' else x2+2; bx=x1-1 if side=='left' else min(w,x2+23)
                crop=symbol[max(0,y1+3):min(h,y2-2),max(0,ax):max(0,bx)]
                if crop.size==0: continue
                active=np.count_nonzero(crop,axis=1)>=max(3,crop.shape[1]*.25)
            else:
                ay=max(0,y1-22) if side=='top' else y2+2; by=y1-1 if side=='top' else min(h,y2+23)
                crop=symbol[max(0,ay):max(0,by),max(0,x1+3):min(w,x2-2)]
                if crop.size==0: continue
                active=np.count_nonzero(crop,axis=0)>=max(3,crop.shape[0]*.25)
            for start,end in runs(active):
                pos=(start+end-1)/2
                if end-start>5: continue
                if side in ('left','right'):
                    yy=y1+3+pos; row=crop[int(pos)]; xs=np.flatnonzero(row)
                    xx=ax+(xs.min() if side=='left' else xs.max())
                else:
                    xx=x1+3+pos; col=crop[:,int(pos)]; ys=np.flatnonzero(col)
                    yy=ay+(ys.min() if side=='top' else ys.max())
                points.append((float(xx),float(yy),side))
        used=set(); result=[]
        for x,y,side in points:
            near=sorted(texts,key=lambda t:math.hypot(center(t['bbox'])[0]-x,center(t['bbox'])[1]-y))
            number=next((t['text'] for t in near if re.fullmatch(r'[A-Za-z]?\d+',t['text']) and not DESIGNATOR.fullmatch(t['text']) and math.dist(center(t['bbox']),(x,y))<25),None)
            if number is None: continue # Avoid confidently inventing IC numbering.
            if number in used: continue
            name=next((t['text'] for t in near if not re.fullmatch(r'\d+',t['text']) and not DESIGNATOR.fullmatch(t['text']) and x1<=center(t['bbox'])[0]<=x2 and y1<=center(t['bbox'])[1]<=y2 and math.dist(center(t['bbox']),(x,y))<60),'')
            result.append({'number':number,'name':name,'x':x,'y':y}); used.add(number)
        return result


def runs(active):
    positions=np.flatnonzero(np.diff(np.r_[False,active,False].astype(np.int8)))
    return list(zip(positions[::2],positions[1::2]))


def center(b): return ((b[0]+b[2])/2,(b[1]+b[3])/2)


def refine_pin_locations(priors,located,box):
    """One-to-one refinement preserves numbering and rejects remote candidates."""
    result=[dict(p) for p in priors]
    limit=max(8,min(25,max(box[2]-box[0],box[3]-box[1])*.35))
    scores=[[max(0,limit-math.hypot(p['x']-q['x'],p['y']-q['y'])) for q in located] for p in priors]
    if not priors or not located: return result
    for i,j in assignment(scores):
        if scores[i][j]>0: result[i].update(x=located[j]['x'],y=located[j]['y'])
    return result


def box_distance(a,b):
    return math.hypot(max(b[0]-a[2],a[0]-b[2],0),max(b[1]-a[3],a[1]-b[3],0))


def compatible(text,cat):
    prefix=re.match(r'[A-Za-z]+',text).group().upper()
    allowed={'R':('R',),'C':('C',),'L':('L',),'Box':('U','IC','J','P','CN'),'D':('D',),'LED':('D','LED','PWR'),'Crystal':('X','Y'),'Switch':('SW','K'),'MOSFET':('Q',),'BJT':('Q',),'Fuse':('F',),'PIN':('J','P','CN'),'Testpoint':('TP',)}
    return prefix in allowed.get(cat,(prefix,))


def is_value(text):
    return bool(re.fullmatch(r'(?:\d+(?:\.\d+)?|\d+[RrKkMm]\d+)[\s\wµμΩω%./+-]*',text)) and not bool(re.fullmatch(r'\d{1,2}',text))


def correct_resistance_ocr(text):
    # Q is a common OCR confusion for the ohm glyph; require a resistor value.
    return re.sub(r'Q$','Ω',text) if re.fullmatch(r'\d+(?:\.\d+)?[kKmM]?Q',text) else text


def select_ic_name(texts,box):
    candidates=[]
    for t in texts:
        text=t['text']; x,y=center(t['bbox'])
        if DESIGNATOR.fullmatch(text) or box_distance(t['bbox'],box)>=50: continue
        if re.fullmatch(r'(?:[+-]?\d+(?:\.\d+)?V|VCC|VDD\w*|VSS\w*|VBUS|VBAT|GND)',text,re.I): continue
        if re.match(r'^(?:GPIO|IO|PB|PA|PC|PD|PIN)\d',text,re.I): continue
        frequency=bool(re.fullmatch(r'\d+(?:\.\d+)?[kMG]?Hz',text,re.I))
        model=bool(re.search(r'[A-Za-z]',text) and re.search(r'\d',text) and len(text)>=4)
        if not (model or frequency): continue
        if not frequency and re.fullmatch(r'\d+(?:\.\d+)?[kKmMuUnNpPµμ]?[FfΩωQ]',text): continue
        inside=box[0]<=x<=box[2] and box[1]<=y<=box[3]
        # Signal names at an IC border are not its central model label.
        if inside and min(x-box[0],box[2]-x)<(box[2]-box[0])*.18: continue
        candidates.append((box_distance(t['bbox'],box),text))
    return min(candidates)[1] if candidates else None


def overlay(image_path,result,output):
    image=read_image(image_path); h=image.shape[0]
    colors=[(20,120,220),(220,120,20),(150,30,200),(20,170,50)]
    for i,net in enumerate(result['nets'].values()):
        for a,b in net['edges'].values():
            cv2.line(image,(round(a['x']),round(h-a['y'])),(round(b['x']),round(h-b['y'])),colors[i%len(colors)],2)
    for key,c in result['components'].items():
        x1,y1,x2,y2=c['bbox']; a=(round(x1),round(h-y2)); b=(round(x2),round(h-y1))
        cv2.rectangle(image,a,b,(210,0,200),1); cv2.putText(image,key,(a[0],max(12,a[1]-3)),cv2.FONT_HERSHEY_SIMPLEX,.4,(180,0,160),1,cv2.LINE_AA)
    for ps in result['pins'].values():
        for pin in ps.values():
            p=pin['point']; cv2.circle(image,(round(p['x']),round(h-p['y'])),3,(0,150,255),1)
    write_image(output,image)
