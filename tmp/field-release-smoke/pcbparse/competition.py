"""Competition type adapter and Q&A-informed proxy, not the official evaluator."""
import copy
from .core import assignment, distance, f1, references, validate

ALIASES={'ic':'box','resistor':'r','capacitor':'c','inductor':'l','vcc':'v','vbus_in':'net_input','vbus_out':'net_output','connector':'pin','diode':'d','amplifier':'amp','battery':'battary','transformer':'transfomer','antenna':'ant'}
TYPES=set('gnd c r v box net_input net_short net_output net_bidirection d led pin switch l mosfet other bjt testpoint crystal buzzer battary fuse block amp jumper motor esd dc transfomer ant circle_header opt seg'.split())|{'m3螺丝'}

def canonical_type(value):
    value=ALIASES.get(value,value)
    for family in ('crystal','amp','bjt','mosfet'):
        if value.startswith(family+'_'): return family
    return value

def export_competition(data,source_categories=None):
    result=copy.deepcopy(data)
    for key,c in result['components'].items():
        c['type']=canonical_type(c['type'])
        category=(source_categories or {}).get(key)
        if category=='Net_bidirection': c['type']='net_bidirection'
        if category=='Block': c['type']='block'
        if c['type'] not in TYPES: raise ValueError('type missing from updated official list: '+c['type'])
        if c['type']!='box': c['Name']=key
    errors=validate(result)
    if errors: raise ValueError('; '.join(errors))
    return result

def evaluate_qa(pred,truth,bbox_tolerance=20,point_tolerance=5):
    """Q&A matching semantics; net aggregation and thresholds remain proxy choices."""
    pred,truth=export_competition(pred),export_competition(truth)
    pc,gc=pred['components'],truth['components']; pkeys,gkeys=list(pc),list(gc)
    def identity(pk,gk):
        p,g=pc[pk],gc[gk]
        anonymous=g['type'] in ('gnd','v') or gk.startswith('∅')
        return p['type']==g['type'] and (pk==gk or anonymous) and all(abs(a-b)<=bbox_tolerance for a,b in zip(p['bbox'],g['bbox']))
    weights=[[1+1/(1+sum(abs(a-b) for a,b in zip(pc[p]['bbox'],gc[g]['bbox']))) if identity(p,g) else 0 for g in gkeys] for p in pkeys]
    pairs=[(pkeys[i],gkeys[j]) for i,j in assignment(weights) if weights[i][j]>0]
    ct=0; pin_map={}; pt=0
    for pk,gk in pairs:
        p,g=pc[pk],gc[gk]
        text='Name' if g['type']=='box' else 'value' if g['type'] in ('r','c') else None
        ct+=text is None or p.get(text)==g.get(text)
        pp,gp=pred['pins'].get(pk,{}),truth['pins'].get(gk,{})
        pnums,gnums=list(pp),list(gp)
        w=[]
        for pn in pnums:
            row=[]
            for gn in gnums:
                gap=distance(pp[pn]['point'],gp[gn]['point'])
                valid=gap<=point_tolerance and (g['type']!='box' or pn==gn and pp[pn]['pinname']==gp[gn]['pinname'])
                row.append(1+1/(1+gap) if valid else 0)
            w.append(row)
        for i,j in assignment(w):
            if w[i][j]>0:
                pt+=1; pin_map[pk+'.'+pnums[i][4:]]=gk+'.'+gnums[j][4:]
    pn,gn=list(pred['nets'].values()),list(truth['nets'].values())
    ps=[{pin_map.get(r,'__unmatched_prediction__'+r) for r in references(n['hyperGraph'])} for n in pn]
    gs=[references(n['hyperGraph']) for n in gn]
    w=[[f1(len(a&b),len(a),len(b)) for b in gs] for a in ps]
    matches=[(i,j) for i,j in assignment(w) if ps[i]&gs[j]]
    nt=sum(len(ps[i]&gs[j]) for i,j in matches); lt=0
    for i,j in matches:
        pe,ge=list(pn[i]['edges'].values()),list(gn[j]['edges'].values())
        def close(a,b):
            return min(max(distance(a[0],b[0]),distance(a[1],b[1])),max(distance(a[0],b[1]),distance(a[1],b[0])))<=point_tolerance
        weights=[[int(close(a,b)) for b in ge] for a in pe]
        lt+=sum(weights[a][b] for a,b in assignment(weights))
    result={'ComponentF1':f1(ct,len(pc),len(gc)),'PinF1':f1(pt,sum(map(len,pred['pins'].values())),sum(map(len,truth['pins'].values()))),'NetHypergraphF1':f1(nt,sum(map(len,ps)),sum(map(len,gs))),'NetLineF1':f1(lt,sum(len(n['edges']) for n in pn),sum(len(n['edges']) for n in gn))}
    result['Score_Accuracy_SingleCase_proxy']=100*sum(result[k]*w for k,w in zip(result,(.3,.25,.35,.1)))
    result.update(official_equivalence=False,matching_profile='Q&A-informed proxy; coordinate pin remapping, Hungarian segment matching')
    return result
