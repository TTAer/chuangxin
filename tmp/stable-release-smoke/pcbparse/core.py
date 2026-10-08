import math
import re
from collections import defaultdict


def distance(a, b):
    return math.hypot(a['x'] - b['x'], a['y'] - b['y'])


def point_on_segment(p, edge, tolerance):
    a, b = edge
    dx, dy = b['x'] - a['x'], b['y'] - a['y']
    length = dx * dx + dy * dy
    t = max(0, min(1, ((p['x']-a['x'])*dx + (p['y']-a['y'])*dy) / length)) if length else 0
    return distance(p, {'x': a['x']+t*dx, 'y': a['y']+t*dy}) <= tolerance


def references(graph):
    if not isinstance(graph, str) or not graph.startswith('(') or not graph.endswith(')'):
        raise ValueError('hyperGraph must be a parenthesized string')
    items = graph[1:-1].split(',') if graph[1:-1] else []
    if any(not re.fullmatch(r'[^\s,()]+\.[^\s,().]+', x) for x in items):
        raise ValueError('invalid component.pin reference')
    if len(items) != len(set(items)):
        raise ValueError('duplicate pin reference')
    return set(items)


def validate(data):
    errors = []
    if not isinstance(data, dict):
        return ['root must be an object']
    for k in ('components', 'pins', 'nets'):
        if not isinstance(data.get(k), dict):
            errors.append(k + ' must be an object')
    if errors:
        return errors
    def point(p):
        return isinstance(p, dict) and all(isinstance(p.get(k), (float, int)) and not isinstance(p[k], bool) and math.isfinite(p[k]) for k in ('x','y'))
    for key, c in data['components'].items():
        if not isinstance(c, dict):
            errors.append(f'{key}: component must be object'); continue
        if not isinstance(c.get('type'), str) or not c['type']:
            errors.append(f'{key}: type required')
        if 'Name' not in c or c['Name'] is not None and not isinstance(c['Name'], str):
            errors.append(f'{key}: Name must be string/null')
        if c.get('type') in ('resistor','capacitor') and not isinstance(c.get('value'), str):
            errors.append(f'{key}: value required')
        b = c.get('bbox')
        if not isinstance(b, list) or len(b)!=4 or not all(isinstance(x,(int,float)) and not isinstance(x,bool) and math.isfinite(x) for x in b) or not (b[0]<b[2] and b[1]<b[3]):
            errors.append(f'{key}: invalid bbox')
    known = set()
    for key, pins in data['pins'].items():
        if key not in data['components']:
            errors.append(f'{key}: unknown component')
        if not isinstance(pins, dict):
            errors.append(f'{key}: pins must be object'); continue
        for number, p in pins.items():
            if not re.fullmatch(r'pin_[^\s,().]+', number):
                errors.append(f'{key}.{number}: invalid pin key')
            known.add(key + '.' + number[4:])
            if not isinstance(p,dict) or not isinstance(p.get('pinname'),str) or not point(p.get('point')):
                errors.append(f'{key}.{number}: invalid pin')
    used = set()
    for key, net in data['nets'].items():
        if not isinstance(net,dict):
            errors.append(f'{key}: net must be object'); continue
        try:
            refs = references(net.get('hyperGraph'))
            if refs-known: errors.append(f'{key}: unknown pins {sorted(refs-known)}')
            if refs & used: errors.append(f'{key}: pins assigned to multiple nets')
            used |= refs
        except ValueError as e:
            errors.append(f'{key}: {e}')
        edges = net.get('edges')
        if not isinstance(edges,dict):
            errors.append(f'{key}: edges must be object'); continue
        for ek, e in edges.items():
            if not isinstance(e,list) or len(e)!=2 or not all(point(p) for p in e):
                errors.append(f'{key}.{ek}: edge must have exactly two finite points')
    return errors


def build(observations, tolerance=2):
    """Build from detector/manual observations already in bottom-left coordinates.

    Endpoint-to-segment contact connects T junctions. Interior crossings require
    an explicit junction. Labels merge logic without inventing geometric wires.
    """
    if not math.isfinite(tolerance) or tolerance < 0: raise ValueError('tolerance must be finite and nonnegative')
    components, pins = observations['components'], observations['pins']
    initial = {'components':components,'pins':pins,'nets':{}}
    errors = validate(initial)
    if errors: raise ValueError('; '.join(errors))
    wires = observations.get('wires', [])
    if not isinstance(wires,list): raise ValueError('wires must be a list')
    geometry_check = dict(initial, nets={'check':{'hyperGraph':'()', 'edges':{str(i):e for i,e in enumerate(wires)}}})
    errors = validate(geometry_check)
    if errors: raise ValueError('; '.join(errors))
    for p in observations.get('junctions',[]):
        check = dict(initial,nets={'check':{'hyperGraph':'()', 'edges':{'dot':[p,p]}}})
        if validate(check): raise ValueError('invalid junction point')
    pp = [(c+'.'+k[4:], v['point']) for c, ps in pins.items() for k,v in ps.items()]
    parent = list(range(len(wires)+len(pp)))
    def root(i):
        while parent[i]!=i:
            parent[i]=parent[parent[i]]; i=parent[i]
        return i
    def union(i,j): parent[root(j)] = root(i)
    for i,a in enumerate(wires):
        for j in range(i):
            b = wires[j]
            contact = any(point_on_segment(p,b,tolerance) for p in a) or any(point_on_segment(p,a,tolerance) for p in b)
            junction = any(point_on_segment(p,a,tolerance) and point_on_segment(p,b,tolerance) for p in observations.get('junctions',[]))
            if contact or junction: union(i,j)
    for i,(_,p) in enumerate(pp):
        idx = len(wires)+i
        for j,e in enumerate(wires):
            if point_on_segment(p,e,tolerance): union(idx,j)
        for j,(_,q) in enumerate(pp[:i]):
            if distance(p,q)<=tolerance: union(idx,len(wires)+j)
    by_ref = {r:len(wires)+i for i,(r,_) in enumerate(pp)}
    labels = {}
    for label in observations.get('labels',[]):
        r, name = label['pin'], label['name']
        if r not in by_ref: raise ValueError('unknown label pin: '+r)
        if not isinstance(name,str) or not name: raise ValueError('label name required')
        if name in labels: union(by_ref[r],labels[name])
        else: labels[name]=by_ref[r]
    groups = defaultdict(lambda:{'refs':[], 'edges':[]})
    for i,(r,_) in enumerate(pp): groups[root(len(wires)+i)]['refs'].append(r)
    for i,e in enumerate(wires): groups[root(i)]['edges'].append(e)
    nets = {}
    for g in groups.values():
        if not g['refs']: continue
        nets['net_'+str(len(nets)+1)] = {'hyperGraph':'('+','.join(sorted(g['refs']))+')', 'edges':{'edge_'+str(i+1):e for i,e in enumerate(g['edges'])}}
    return dict(initial,nets=nets)


def f1(tp, pred, truth):
    return 2*tp/(pred+truth) if pred+truth else 1.0


def assignment(weights):
    """Maximum-weight one-to-one assignment using padded Hungarian algorithm."""
    rows = len(weights); cols = len(weights[0]) if rows else 0
    n = max(rows,cols)
    u,v,p,way = [[0]*(n+1) for _ in range(4)]
    for i in range(1,n+1):
        p[0]=i; j0=0; mins=[math.inf]*(n+1); used=[False]*(n+1)
        while True:
            used[j0]=True; i0=p[j0]; delta=math.inf; j1=0
            for j in range(1,n+1):
                if used[j]: continue
                cost = -weights[i0-1][j-1] if i0<=rows and j<=cols else 0
                cur=cost-u[i0]-v[j]
                if cur<mins[j]: mins[j]=cur; way[j]=j0
                if mins[j]<delta: delta=mins[j]; j1=j
            for j in range(n+1):
                if used[j]: u[p[j]]+=delta; v[j]-=delta
                else: mins[j]-=delta
            j0=j1
            if p[j0]==0: break
        while j0:
            j1=way[j0]; p[j0]=p[j1]; j0=j1
    return [(p[j]-1,j-1) for j in range(1,n+1) if 0<p[j]<=rows and j<=cols]


def evaluate(pred, truth, bbox_tolerance=20, point_tolerance=5):
    """Development proxy. Official net matching/aggregation rules remain pending."""
    for data in (pred,truth):
        errors=validate(data)
        if errors: raise ValueError('; '.join(errors))
    pc,gc=pred['components'],truth['components']; ct=0
    for k,c in pc.items():
        g=gc.get(k)
        if not g or c['type']!=g['type']: continue
        text='Name' if g['type']=='ic' else 'value' if g['type'] in ('resistor','capacitor') else None
        if text and c.get(text)!=g.get(text): continue
        ct+=all(abs(a-b)<=bbox_tolerance for a,b in zip(c['bbox'],g['bbox']))
    def flatten(data): return {c+'.'+k:v for c,ps in data['pins'].items() for k,v in ps.items()}
    pp,gp=flatten(pred),flatten(truth)
    pt=sum(k in gp and v['pinname']==gp[k]['pinname'] and distance(v['point'],gp[k]['point'])<=point_tolerance for k,v in pp.items())
    pn,gn=list(pred['nets'].values()),list(truth['nets'].values())
    ps,gs=[references(n['hyperGraph']) for n in pn],[references(n['hyperGraph']) for n in gn]
    weights=[[f1(len(a&b),len(a),len(b)) for b in gs] for a in ps]
    matches=[(i,j) for i,j in assignment(weights) if ps[i]&gs[j]]
    nt=sum(len(ps[i]&gs[j]) for i,j in matches); lt=0
    for i,j in matches:
        pe,ge=list(pn[i]['edges'].values()),list(gn[j]['edges'].values())
        def close(a,b):
            return min(max(distance(a[0],b[0]),distance(a[1],b[1])),max(distance(a[0],b[1]),distance(a[1],b[0])))<=point_tolerance
        w=[[int(close(a,b)) for b in ge] for a in pe]
        lt+=sum(w[a][b] for a,b in assignment(w))
    result={'ComponentF1':f1(ct,len(pc),len(gc)), 'PinF1':f1(pt,len(pp),len(gp)), 'NetHypergraphF1':f1(nt,sum(map(len,ps)),sum(map(len,gs))), 'NetLineF1':f1(lt,sum(len(n['edges']) for n in pn),sum(len(n['edges']) for n in gn))}
    result['Score_Accuracy_SingleCase_proxy']=100*sum(result[k]*w for k,w in zip(result,(.3,.25,.35,.1)))
    result['official_equivalence']=False
    return result
