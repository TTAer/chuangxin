"""Development adapter for public CVJsonStd, not the official evaluator."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

from .core import validate

CATEGORIES={'Box':'ic','Block':'ic','R':'resistor','C':'capacitor','L':'inductor','Gnd':'gnd','V':'vcc','NET_input':'vbus_in','Net_output':'vbus_out','Net_bidirection':'vbus_in','PIN':'connector','D':'diode','LED':'led','Crystal':'crystal','Switch':'switch','Testpoint':'testpoint','MOSFET':'mosfet','BJT':'bjt','Fuse':'fuse','Jumper':'jumper','AMP':'amplifier'}
DESIGNATOR=re.compile(r'^(?:R|C|L|U|IC|D|Q|J|P|CN|SW|X|Y|F|TP|LED|PWR|K|T|BZ)\d+[A-Za-z]?$',re.I)


def natural(s): return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)',s)]


def text_value(raw,value):
    if value is None: return None
    if isinstance(value,str) and value in raw['texts']: return raw['texts'][value]['content']
    return str(value)


def identity(raw,component):
    texts=list(component.get('associatedTexts',{}).values())
    designators=[t['content'] for t in texts if t.get('details')=='designator']
    if not designators: designators=[t['content'] for t in texts if DESIGNATOR.fullmatch(t.get('content',''))]
    name=text_value(raw,component.get('searchNameByVLM') or component.get('searchNameByOCR'))
    if not name:
        name=next((t['content'] for t in texts if t.get('details') in ('searchName','name','Name')),None)
    value=next((t['content'] for t in texts if t.get('details')=='value'),None)
    category=component['category']
    key=designators[0] if designators else name if category in ('NET_input','Net_output','Net_bidirection','V','Testpoint') and name else 'GND' if category=='Gnd' else 'VCC' if category=='V' else component['cpntId']
    # Text annotation is incomplete in public data; do not pretend fallbacks are golden truth.
    warnings=[]
    if key==component['cpntId']: warnings.append('missing_designator')
    if category in ('R','C') and value is None: warnings.append('missing_value'); value=''
    if category=='Box' and name is None: warnings.append('missing_ic_name')
    return key,name,value,warnings


def pin_specs(raw,component):
    result=[]; used=set(); next_number=1
    for pid,p in sorted(component.get('pins',{}).items(),key=lambda kv:natural(kv[0])):
        number=text_value(raw,p.get('pinNumberByVLM') or p.get('pinNumberByOCR'))
        numbers=re.split(r'[,，]',number) if number and re.fullmatch(r'[A-Za-z0-9]+(?:[,，][A-Za-z0-9]+)*',number) else []
        if not numbers:
            while str(next_number) in used: next_number+=1
            numbers=[str(next_number)]; next_number+=1
        name=text_value(raw,p.get('pinNameByVLM') or p.get('pinNameByOCR')) or ''
        for n in numbers:
            if n in used: continue
            used.add(n); result.append((pid,n,name,p))
    return result


def convert_raw(raw, bbox_source='location'):
    h=raw['canvas']['height']; components={}; pins={}; mapping={}; diagnostics=[]
    entries=[]
    for cid,c in sorted(raw['components'].items(),key=lambda kv:natural(kv[0])):
        if c.get('state') not in ('active',None): continue
        key,name,value,warnings=identity(raw,c)
        entries.append((cid,c,key,name,value,warnings))
    counts=Counter(x[2] for x in entries); seen=Counter()
    for cid,c,key,name,value,warnings in entries:
        seen[key]+=1
        key=key+'_'+str(seen[key]) if counts[key]>1 else key
        b=c.get(bbox_source) or c['location']
        components[key]={'type':CATEGORIES.get(c['category'],c['category'].lower()),'Name':name,'value':value,'bbox':[b['x1'],h-b['y2'],b['x2'],h-b['y1']]}
        pins[key]={}
        for pid,n,pname,p in pin_specs(raw,c):
            pins[key]['pin_'+n]={'pinname':pname,'point':{'x':p['tip'][0],'y':h-p['tip'][1]}}
            mapping.setdefault((cid,pid),[]).append(key+'.'+n)
        if warnings: diagnostics.append({'id':cid,'key':key,'warnings':warnings})
    # Merge source edges sharing references or explicit net label. Source edge IDs
    # describe physical groups; output nets describe logical connectivity.
    groups=[]
    for eid,e in raw['edges'].items():
        refs=set()
        for pid,conn in e.get('connect',{}).items(): refs.update(mapping.get((conn.get('cpntId'),pid),[]))
        if not refs: continue
        segments=[]
        paths=e.get('keyPointPaths',{}).get('wire',[])
        for path in paths:
            for a,b in zip(path,path[1:]):
                segments.append([{'x':a[0],'y':h-a[1]},{'x':b[0],'y':h-b[1]}])
        if e.get('isFlywire') and not segments:
            for conn in e.get('connect',{}).values():
                p=conn.get('location')
                if p: segments.append([{'x':p[0],'y':h-p[1]},{'x':p[0],'y':h-p[1]}])
        label=e.get('netlabel') or None
        groups.append({'refs':refs,'segments':segments,'labels':{label} if label else set()})
    changed=True
    while changed:
        changed=False
        for i in range(len(groups)):
            for j in range(i):
                if groups[i]['refs']&groups[j]['refs'] or groups[i]['labels']&groups[j]['labels']:
                    groups[j]['refs']|=groups[i]['refs']; groups[j]['labels']|=groups[i]['labels']; groups[j]['segments']+=groups[i]['segments']
                    groups.pop(i); changed=True; break
            if changed: break
    nets={}
    for i,g in enumerate(groups):
        nets['net_'+str(i+1)]={'hyperGraph':'('+','.join(sorted(g['refs']))+')','edges':{'edge_'+str(j+1):e for j,e in enumerate(g['segments'])}}
    result={'components':components,'pins':pins,'nets':nets}
    errors=validate(result)
    if errors: raise ValueError('; '.join(errors))
    return result,diagnostics


def index_dataset(root):
    items=[]
    for p in sorted(x for x in Path(root).rglob('*') if x.suffix.lower() in ('.png','.jpg','.jpeg')):
        annotation=p.with_name(p.stem+'_CVJsonStd.json')
        if not annotation.exists(): continue
        digest=hashlib.sha256(p.read_bytes()).hexdigest()
        # All exact duplicates share the same split. No inference reads annotations.
        split='validation' if int(digest[:8],16)%5==0 else 'train'
        raw=json.loads(annotation.read_text(encoding='utf-8'))
        items.append({'id':p.parent.name,'image':str(p.resolve()),'annotation':str(annotation.resolve()),'sha256':digest,'split':split,'components':len(raw['components']),'style':raw.get('nowCls')})
    return items
