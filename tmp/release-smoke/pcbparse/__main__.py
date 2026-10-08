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
        points=[v['point'] for ps in result['pins'].values() for v in ps.values()]
        points += [p for e in result.get('wires',[]) for p in e]
        points += result.get('junctions',[])
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
    p=subs.add_parser('infer'); p.add_argument('image'); p.add_argument('--output',required=True); p.add_argument('--model',default='models/symbols'); p.add_argument('--detector',default='models/detector/pcb-yolo11n.onnx'); p.add_argument('--diagnostics'); p.add_argument('--observations'); p.add_argument('--overlay'); p.add_argument('--ocr')
    p=subs.add_parser('batch'); p.add_argument('images'); p.add_argument('--output',default='output/submission'); p.add_argument('--model',default='models/symbols'); p.add_argument('--detector',default='models/detector/pcb-yolo11n.onnx'); p.add_argument('--case-timeout',type=float,default=115); p.add_argument('--total-timeout',type=float,default=6900)
    for name in ('infer','batch'):
        subs.choices[name].add_argument('--pin-model',default='auto',help='auto selects models/pipeline.json; none disables learned pins')
        subs.choices[name].add_argument('--local-pin-ocr',action='store_true')
        subs.choices[name].add_argument('--text-constraints',action='store_true')
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
        elif args.command=='infer':
            from .vision import ImageParser,overlay
            result,obs,diagnostics,texts=ImageParser(args.model,detector=args.detector,pin_model=args.pin_model,local_pin_ocr=args.local_pin_ocr,text_constraints=args.text_constraints).parse(args.image)
            save(args.output,result)
            if args.diagnostics: save(args.diagnostics,diagnostics)
            if args.observations: save(args.observations,obs)
            if args.ocr: save(args.ocr,texts)
            if args.overlay: overlay(args.image,result,args.overlay)
            print(json.dumps(diagnostics,ensure_ascii=False,indent=2))
        elif args.command=='batch':
            from .batch import run_batch
            result=run_batch(args.images,args.output,args.model,args.detector,args.case_timeout,args.total_timeout,pin_model=args.pin_model,local_pin_ocr=args.local_pin_ocr,text_constraints=args.text_constraints)
            print(json.dumps(result,ensure_ascii=False,indent=2))
            return int(bool(result['failed'] or result['unfinished']))
        return 0
    except (ValueError,KeyError,TypeError,OSError) as e:
        parser.exit(2,'Error: '+str(e)+'\n')


if __name__=='__main__':
    raise SystemExit(main())
