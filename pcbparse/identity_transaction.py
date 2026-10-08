"""Separate newly recovered component identity from permission to affect nets."""
import copy
import hashlib
import json
import math
from .core import point_on_segment
from .graphschematic import DualGraph
from .semantic_library import similarity

def wire_signature(obs):
    payload={'wires':obs.get('wires',[]),'junctions':obs.get('junctions',[]),'contact_decisions':obs.get('contact_decisions',[])}
    return hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':')).encode()).hexdigest()

def guard_new_components(observations,image,library,mode='evidence'):
    """Existing pins/wires are immutable; unsupported new pins become hypotheses."""
    result=copy.deepcopy(observations); records=[]
    if mode not in ('evidence','catalog'): raise ValueError('unsupported identity transaction guard')
    new={r['output_key'] for r in result.get('designator_association',{}).get('records',[]) if r.get('output_key') and r.get('identity_new')}
    base=copy.deepcopy(result)
    for key in new: base['pins'][key]={}
    geometry=DualGraph(base); original=wire_signature(result)
    grounded=[(p,geometry.uf.root(geometry.ref_index[ref])) for ref,p,_ in geometry.pins]
    h,w=image.shape[:2]
    import cv2
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    for key in sorted(new):
        component=result['components'][key]; pins=result['pins'].get(key,{})
        model,_=library.identify(component.get('Name'),component['bbox'],result.get('texts',[])) if library else (None,{})
        if mode=='catalog' and not model:
            records.append({'component':key,'confidence_level':'existing_perception','model':None,'pins_before':len(pins),'pins_after':len(pins),'status':'no_confirmed_catalog_preserve_upstream_perception'})
            continue
        catalog=library.entries[model]['pins'] if model else None
        fixed_cardinality={'resistor':2,'capacitor':2,'inductor':2,'diode':2,'led':2,'fuse':2}
        accepted={}; decisions=[]; electrical_count=0
        for number,pin in pins.items():
            point=pin['point']; reasons=[]; n=number[4:]
            if catalog is not None:
                allowed=catalog.get(n,{}).get('names',[])
                if not allowed: reasons.append('number_not_in_confirmed_catalog')
                elif max(similarity(pin.get('pinname',''),name) for name in allowed)<.9: reasons.append('pin_name_catalog_conflict')
            elif component['type'] in fixed_cardinality:
                expected=fixed_cardinality[component['type']]
                if len(pins)!=expected or n not in {str(i+1) for i in range(expected)}:
                    reasons.append('unsupported_symbol_pin_schema')
            else:
                numbers=[t for t in result.get('texts',[]) if t['text']==n and n.isdigit() and t.get('confidence',1)>=.9]
                if not any(math.hypot((t['bbox'][0]+t['bbox'][2])/2-point['x'],(t['bbox'][1]+t['bbox'][3])/2-point['y'])<=32 for t in numbers):
                    reasons.append('no_local_numeric_pin_evidence')
            x,y=round(point['x']),round(h-point['y'])
            patch=gray[max(0,y-3):min(h,y+4),max(0,x-3):min(w,x+4)]
            if not patch.size or int((patch<180).sum())<3: reasons.append('no_endpoint_stroke')
            wires=[i for i in geometry.index.query([point['x']-3,point['y']-3,point['x']+3,point['y']+3]) if point_on_segment(point,geometry.wires[i],3) and geometry.contact_allowed(point)]
            if not wires: reasons.append('no_confirmed_wire_contact')
            roots={geometry.uf.root(i) for i in wires}
            roots.update(root for other,root in grounded if math.hypot(point['x']-other['x'],point['y']-other['y'])<=3)
            if len(roots)>1: reasons.append('would_bridge_existing_wire_groups')
            identity_valid=not any(reason in ('number_not_in_confirmed_catalog','pin_name_catalog_conflict','unsupported_symbol_pin_schema','no_local_numeric_pin_evidence','no_endpoint_stroke') for reason in reasons)
            if identity_valid: accepted[number]=pin
            ref=key+'.'+n
            result.setdefault('electrical_pin_permissions',{})[ref]=identity_valid and not reasons
            if not reasons:
                electrical_count+=1
                grounded.append((point,next(iter(roots))))
            else:
                result.setdefault('pin_hypotheses',[]).append({'id':key+'@transaction_'+number,'component':key,'point':point,'status':'electrical_abstention','evidence':reasons,'source':'identity_transaction_guard'})
            decisions.append({'pin':number,'accepted':identity_valid,'electrical_allowed':not reasons,'reasons':reasons,'wire_ids':sorted(wires)})
        result['pins'][key]=accepted
        records.append({'component':key,'confidence_level':'electrical_confirmed' if electrical_count else 'pin_grounded' if accepted else 'identity_confirmed','model':model,'pins_before':len(pins),'pins_after':len(accepted),'electrical_pins':electrical_count,'decisions':decisions})
    assert original==wire_signature(result)
    return result,{'module':'IdentityTransactionGuard','mode':mode,'records':records,'physical_wire_signature':original,'physical_wires_unchanged':True,'existing_pins_unchanged':True,'ground_truth_used':False}
