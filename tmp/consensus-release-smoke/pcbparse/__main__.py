import argparse
import copy
import json
from pathlib import Path

from .core import build, evaluate, validate


def load(path):
    def unique(pairs):
        result = {}
        for k,v in pairs:
            if k in result: raise ValueError('duplicate JSON key: '+k)
            result[k]=v
        return result
    return json.loads(Path(path).read_text(encoding='utf-8-sig'),object_pairs_hook=unique)


def save(path, data):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def convert_coordinates(obs):
    result=copy.deepcopy(obs)
    origin=result.get('coordinate_system','bottom-left')
    if origin not in ('top-left','bottom-left'): raise ValueError('unsupported coordinate_system')
    if origin=='top-left':
        height=result['image']['height']
        if not isinstance(height,(int,float)) or height<=0: raise ValueError('positive image height required')
        # Continuous image-edge coordinates: origin at image boundary, not pixel center.
        for c in result['components'].values():
            x1,y1,x2,y2=c['bbox']; c['bbox']=[x1,height-y2,x2,height-y1]
        for text in result.get('texts',[]):
            x1,y1,x2,y2=text['bbox']; text['bbox']=[x1,height-y2,x2,height-y1]
        result['masked_symbol_boxes']=[[b[0],height-b[3],b[2],height-b[1]] for b in result.get('masked_symbol_boxes',[])]
        points=[v['point'] for ps in result['pins'].values() for v in ps.values()]
        points += [p for e in result.get('wires',[]) for p in e]
        points += result.get('junctions',[])
        points += [d['point'] for d in result.get('contact_decisions',[])]
        points += [d['point'] for d in result.get('pin_hypotheses',[])]
        seen=set()
        for p in points:
            if id(p) not in seen: p['y']=height-p['y']; seen.add(id(p))
    result['coordinate_system']='bottom-left'
    return result


def demo(directory):
    directory=Path(directory)
    c={k:{'type':'resistor','Name':None,'value':'10k','bbox':b} for k,b in [('R1',[20,40,60,60]),('R2',[140,40,180,60])]}
    pins={'R1':{'pin_1':{'pinname':'1','point':{'x':20,'y':50}},'pin_2':{'pinname':'2','point':{'x':60,'y':50}}},'R2':{'pin_1':{'pinname':'1','point':{'x':140,'y':50}},'pin_2':{'pinname':'2','point':{'x':180,'y':50}}}}
    obs={'provenance':'synthetic geometry; not an image recognition benchmark','image':{'width':200,'height':100},'coordinate_system':'bottom-left','components':c,'pins':pins,'wires':[[{'x':60,'y':50},{'x':140,'y':50}]],'junctions':[],'labels':[]}
    # Ground truth is independently specified, not produced by build().
    gt={'components':c,'pins':pins,'nets':{'left':{'hyperGraph':'(R1.1)','edges':{}},'middle':{'hyperGraph':'(R1.2,R2.1)','edges':{'edge_1':[{'x':60,'y':50},{'x':140,'y':50}]}},'right':{'hyperGraph':'(R2.2)','edges':{}}}}
    pred=build(obs)
    save(directory/'observations.json',obs); save(directory/'golden.json',gt)
    save(directory/'prediction.json',pred); save(directory/'metrics.json',evaluate(pred,gt))
    svg='<svg xmlns="http://www.w3.org/2000/svg" width="800" height="400" viewBox="0 0 200 100"><rect width="200" height="100" fill="white"/><path d="M60 50 H140" stroke="green" fill="none"/><g stroke="black" fill="none"><rect x="20" y="40" width="40" height="20"/><rect x="140" y="40" width="40" height="20"/></g><g font-size="8" font-family="sans-serif"><text x="25" y="30">R1 10k</text><text x="140" y="30">R2 10k</text><text x="30" y="85">Synthetic topology test</text></g></svg>'
    (directory/'schematic.svg').write_text(svg,encoding='utf-8')
    return {'output':str(directory), 'metrics':evaluate(pred,gt)}


def main():
    parser=argparse.ArgumentParser(description='Image-only PCB schematic parsing and development evaluation')
    subs=parser.add_subparsers(dest='command',required=True)
    p=subs.add_parser('build'); p.add_argument('observations'); p.add_argument('--output',required=True); p.add_argument('--tolerance',type=float,default=2)
    p=subs.add_parser('validate'); p.add_argument('json')
    p=subs.add_parser('evaluate'); p.add_argument('prediction'); p.add_argument('golden'); p.add_argument('--output')
    p=subs.add_parser('demo'); p.add_argument('--output',default='output/demo')
    p=subs.add_parser('train-templates'); p.add_argument('data'); p.add_argument('--output',default='models/symbols')
    p=subs.add_parser('convert-dataset'); p.add_argument('annotation'); p.add_argument('--output',required=True); p.add_argument('--diagnostics'); p.add_argument('--bbox-source',choices=['location','realBBox'],default='location')
    p=subs.add_parser('graph-build'); p.add_argument('observations'); p.add_argument('--output',required=True); p.add_argument('--graph-output',required=True); p.add_argument('--review-output'); p.add_argument('--image'); p.add_argument('--corrections'); p.add_argument('--no-optimize',action='store_true')
    p=subs.add_parser('infer'); p.add_argument('image'); p.add_argument('--output',required=True); p.add_argument('--model',default='models/symbols'); p.add_argument('--detector',default='models/detector/pcb-yolo11n.onnx'); p.add_argument('--diagnostics'); p.add_argument('--observations'); p.add_argument('--overlay'); p.add_argument('--ocr')
    p=subs.add_parser('batch'); p.add_argument('images'); p.add_argument('--output',default='output/submission'); p.add_argument('--model',default='models/symbols'); p.add_argument('--detector',default='models/detector/pcb-yolo11n.onnx'); p.add_argument('--case-timeout',type=float,default=115); p.add_argument('--total-timeout',type=float,default=6900)
    for name in ('infer','batch'):
        subs.choices[name].add_argument('--pin-model',default='auto',help='auto selects models/pipeline.json; none disables learned pins')
        subs.choices[name].add_argument('--local-pin-ocr',action='store_true')
        subs.choices[name].add_argument('--text-constraints',action='store_true')
        subs.choices[name].add_argument('--graph-mode',choices=['auto','legacy','association','full'],default='auto')
        subs.choices[name].add_argument('--topology-model',default='auto')
        subs.choices[name].add_argument('--semantic-library',default='auto',help='auto uses pipeline config; none disables official library semantics')
        subs.choices[name].add_argument('--pin-ensemble',default='auto',help='optional auxiliary pin ONNX; none disables topology-preserving consensus')
    subs.choices['infer'].add_argument('--graph-output')
    subs.choices['infer'].add_argument('--review-output')
    args=parser.parse_args()
    try:
        if args.command=='build':
            result=build(convert_coordinates(load(args.observations)),args.tolerance)
            errors=validate(result)
            if errors: raise ValueError('; '.join(errors))
            save(args.output,result); print('Saved '+args.output)
        elif args.command=='validate':
            errors=validate(load(args.json)); print(json.dumps({'valid':not errors,'errors':errors},ensure_ascii=False,indent=2))
            return int(bool(errors))
        elif args.command=='evaluate':
            result=evaluate(load(args.prediction),load(args.golden))
            if args.output: save(args.output,result)
            print(json.dumps(result,ensure_ascii=False,indent=2))
        elif args.command=='demo': print(json.dumps(demo(args.output),ensure_ascii=False,indent=2))
        elif args.command=='train-templates':
            from .vision import train_templates
            print(json.dumps(train_templates(args.data,args.output),ensure_ascii=False,indent=2))
        elif args.command=='convert-dataset':
            from .dataset import convert_raw
            result,diagnostics=convert_raw(load(args.annotation),args.bbox_source)
            save(args.output,result)
            if args.diagnostics: save(args.diagnostics,diagnostics)
            print('Saved '+args.output)
        elif args.command=='graph-build':
            from .graphschematic import parse_graph
            from .corrections import review_queue,validate_proposals,apply_semantic_choices
            from .vision import read_image
            import cv2
            import hashlib
            obs=convert_coordinates(load(args.observations)); line_mask=None; image_hash=None
            if args.image:
                image=read_image(args.image); h,w=image.shape[:2]
                if obs.get('image',{}).get('height',h)!=h or obs.get('image',{}).get('width',w)!=w: raise ValueError('image/observation dimensions mismatch')
                line_mask=(cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)<220).astype('uint8')*255
                for box in obs.get('masked_symbol_boxes',[c['bbox'] for c in obs['components'].values()])+[t['bbox'] for t in obs.get('texts',[])]:
                    x1,y1,x2,y2=box; a,b,c,d=map(int,(x1,h-y2,x2,h-y1)); line_mask[max(0,b):min(h,d),max(0,a):min(w,c)]=0
                image_hash=hashlib.sha256(Path(args.image).read_bytes()).hexdigest()
            result,report=parse_graph(obs,line_mask=line_mask,optimize=not args.no_optimize)
            queue=review_queue(obs,report,image_hash)
            if args.corrections:
                try:
                    payload=load(args.corrections); forced,blocked=validate_proposals(payload,queue)
                    corrected,edits=apply_semantic_choices(obs,payload,queue)
                    corrected_result,corrected_report=parse_graph(corrected,line_mask=line_mask,optimize=True,forced=forced,blocked=blocked)
                    result,report=corrected_result,corrected_report
                    report['llm'].update(status='validated_proposal_replay',semantic_edits=edits)
                except ValueError as exc:
                    report['llm'].update(status='proposal_rejected_local_fallback',reason=str(exc)[:300])
            save(args.output,result); save(args.graph_output,report)
            if args.review_output: save(args.review_output,queue)
            print(json.dumps({'framework':'GraphSchematic','accepted':len(report['association']['accepted']),'issues':len(report['circuit_issues']),'llm':report['llm']},ensure_ascii=False))
        elif args.command=='infer':
            from .vision import ImageParser,overlay
            image_parser=ImageParser(args.model,detector=args.detector,pin_model=args.pin_model,local_pin_ocr=args.local_pin_ocr,text_constraints=args.text_constraints,graph_mode=args.graph_mode,topology_model=args.topology_model,semantic_library=args.semantic_library,pin_ensemble=args.pin_ensemble)
            result,obs,diagnostics,texts=image_parser.parse(args.image)
            save(args.output,result)
            if args.diagnostics: save(args.diagnostics,diagnostics)
            if args.observations: save(args.observations,obs)
            if args.ocr: save(args.ocr,texts)
            if args.overlay: overlay(args.image,result,args.overlay)
            if args.graph_output:
                if image_parser.last_graph_report is None: raise ValueError('--graph-output requires association/full graph mode')
                save(args.graph_output,image_parser.last_graph_report)
            if args.review_output:
                if image_parser.last_review_queue is None: raise ValueError('--review-output requires association/full graph mode')
                save(args.review_output,image_parser.last_review_queue)
            print(json.dumps(diagnostics,ensure_ascii=False,indent=2))
        elif args.command=='batch':
            from .batch import run_batch
            result=run_batch(args.images,args.output,args.model,args.detector,args.case_timeout,args.total_timeout,pin_model=args.pin_model,local_pin_ocr=args.local_pin_ocr,text_constraints=args.text_constraints,graph_mode=args.graph_mode,topology_model=args.topology_model,semantic_library=args.semantic_library,pin_ensemble=args.pin_ensemble)
            print(json.dumps(result,ensure_ascii=False,indent=2))
            return int(bool(result['failed'] or result['unfinished']))
        return 0
    except (ValueError,KeyError,TypeError,OSError) as e:
        parser.exit(2,'Error: '+str(e)+'\n')


if __name__=='__main__':
    raise SystemExit(main())
