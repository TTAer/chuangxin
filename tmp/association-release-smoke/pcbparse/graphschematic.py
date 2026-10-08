"""Geometry-to-Electrical dual graphs and bounded constraint-guided association.

Physical wire edges never represent semantic flying connections. Uncertain
associations are separate hypotheses; a beam searches coupled circuit factors.
"""
from collections import defaultdict
import copy
import math
import re
from .core import distance, point_on_segment, validate


class UnionFind:
    def __init__(self,n): self.parent=list(range(n))
    def root(self,i):
        while self.parent[i]!=i:
            self.parent[i]=self.parent[self.parent[i]]; i=self.parent[i]
        return i
    def union(self,i,j): self.parent[self.root(j)]=self.root(i)
    def clone(self):
        result=UnionFind(0); result.parent=self.parent.copy(); return result


class SpatialIndex:
    def __init__(self,cell=32): self.cell=cell; self.bins=defaultdict(set)
    def cells(self,box):
        a,b,c,d=box
        for x in range(math.floor(a/self.cell),math.floor(c/self.cell)+1):
            for y in range(math.floor(b/self.cell),math.floor(d/self.cell)+1): yield x,y
    def insert(self,i,box):
        for key in self.cells(box): self.bins[key].add(i)
    def query(self,box):
        result=set()
        for key in self.cells(box): result.update(self.bins.get(key,()))
        return result


def point_box(p,r): return [p['x']-r,p['y']-r,p['x']+r,p['y']+r]
def projection(p,edge):
    a,b=edge; dx=b['x']-a['x']; dy=b['y']-a['y']; norm=dx*dx+dy*dy
    t=max(0,min(1,((p['x']-a['x'])*dx+(p['y']-a['y'])*dy)/norm)) if norm else 0
    return {'x':a['x']+t*dx,'y':a['y']+t*dy}


def power_domain(name):
    value=name.strip().upper()
    if value in ('GND','GROUND','0V'): return 'ground'
    match=re.fullmatch(r'\+?(\d+(?:\.\d+)?)V',value)
    if match: return 'ground' if float(match[1])==0 else 'voltage:'+str(float(match[1]))
    if re.fullmatch(r'(?:VCC|VDD\w*|VBUS|VBAT)',value): return 'positive-power'
    return None


def incompatible(domains):
    return ('ground' in domains and any(x!='ground' for x in domains)) or len({x for x in domains if x.startswith('voltage:')})>1


class DualGraph:
    def __init__(self,observations,tolerance=3):
        if not math.isfinite(tolerance) or tolerance<0: raise ValueError('invalid graph tolerance')
        self.obs=copy.deepcopy(observations); self.tolerance=tolerance
        self.wires=self.obs.get('wires',[])
        initial={'components':self.obs['components'],'pins':self.obs['pins'],'nets':{'check':{'hyperGraph':'()','edges':{str(i):e for i,e in enumerate(self.wires)}}}}
        errors=validate(initial)
        if errors: raise ValueError('; '.join(errors))
        self.pins=[(c+'.'+k[4:],p['point'],c) for c,ps in self.obs['pins'].items() for k,p in ps.items()]
        self.ref_index={r:len(self.wires)+i for i,(r,_,_) in enumerate(self.pins)}
        self.uf=UnionFind(len(self.wires)+len(self.pins)); self.index=SpatialIndex()
        self.nodes={}; self.edges=[]; self.contacts=[]
        self.contact_decisions=self.obs.get('contact_decisions',[])
        for i,edge in enumerate(self.wires):
            a,b=edge
            self.index.insert(i,[min(a['x'],b['x'])-tolerance,min(a['y'],b['y'])-tolerance,max(a['x'],b['x'])+tolerance,max(a['y'],b['y'])+tolerance])
            for j,p in enumerate(edge): self.nodes[f'w{i}:{j}']={'kind':'wire_endpoint','point':p}
            self.edges.append({'id':'wire_'+str(i),'kind':'physical_wire','source':f'w{i}:0','target':f'w{i}:1','geometry':edge})
        # End contact/T contact, then explicit electrical junctions. Interior
        # crossings are not connected just because projected lines intersect.
        for i,edge in enumerate(self.wires):
            for endpoint in edge:
                for j in sorted(self.index.query(point_box(endpoint,tolerance))):
                    if j!=i and self.contact_allowed(endpoint) and point_on_segment(endpoint,self.wires[j],tolerance):
                        self.uf.union(i,j); self.contacts.append({'kind':'wire_contact','wires':[i,j],'point':endpoint})
        for i,p in enumerate(self.obs.get('junctions',[])):
            if not all(isinstance(p.get(k),(int,float)) and math.isfinite(p[k]) for k in ('x','y')): raise ValueError('invalid junction')
            self.nodes['j'+str(i)]={'kind':'junction','point':p,'evidence':'topology_perception_or_observation'}
            ids=[j for j in sorted(self.index.query(point_box(p,tolerance))) if point_on_segment(p,self.wires[j],tolerance)]
            self.contacts.append({'kind':'junction_incidence','junction':'j'+str(i),'wires':ids,'point':p})
            for j in ids[1:]: self.uf.union(ids[0],j)
        pin_index=SpatialIndex(); self.locally_attached=set()
        for i,(ref,p,c) in enumerate(self.pins):
            idx=self.ref_index[ref]; self.nodes[ref]={'kind':'pin_endpoint','component':c,'point':p}
            for j in sorted(self.index.query(point_box(p,tolerance))):
                if self.contact_allowed(p) and point_on_segment(p,self.wires[j],tolerance):
                    self.uf.union(idx,j); self.locally_attached.add(ref)
                    self.contacts.append({'kind':'pin_wire_contact','pin':ref,'wire':j,'point':p})
            for j in sorted(pin_index.query(point_box(p,tolerance))):
                if distance(p,self.pins[j][1])<=tolerance: self.uf.union(idx,len(self.wires)+j)
            pin_index.insert(i,point_box(p,0))
        self.semantic_links=[]; names={}; self.domains=defaultdict(set)
        for i,label in enumerate(self.obs.get('labels',[])):
            ref,name=label['pin'],label['name']
            if ref not in self.ref_index or not isinstance(name,str) or not name: raise ValueError('invalid label reference')
            anchor=self.nodes[ref]['point']
            texts=[t for t in self.obs.get('texts',[]) if t['text']==name]
            texts.sort(key=lambda t:distance(anchor,{'x':(t['bbox'][0]+t['bbox'][2])/2,'y':(t['bbox'][1]+t['bbox'][3])/2}))
            nearest=texts[0] if texts else None
            label_point={'x':(nearest['bbox'][0]+nearest['bbox'][2])/2,'y':(nearest['bbox'][1]+nearest['bbox'][3])/2} if nearest else anchor
            if nearest and distance(anchor,label_point)>80: nearest=None; label_point=anchor
            self.nodes['label_'+str(i)]={'kind':'label','text':name,'pin':ref,'point':label_point,'evidence':'local_ocr' if nearest else 'symbol_or_netlabel_semantics','bbox':nearest['bbox'] if nearest else None}
            if name in names:
                self.uf.union(self.ref_index[ref],self.ref_index[names[name]])
                self.semantic_links.append({'kind':'flying_label_equivalence','name':name,'pins':[names[name],ref]})
            else: names[name]=ref
        for label in self.obs.get('labels',[]):
            domain=power_domain(label['name'])
            if domain: self.domains[self.uf.root(self.ref_index[label['pin']])].add(domain)
        for hypothesis in self.obs.get('pin_hypotheses',[]):
            self.nodes[hypothesis['id']]={'kind':'pin_endpoint','component':hypothesis['component'],'point':hypothesis['point'],'status':'number_unresolved','evidence':hypothesis}

    def contact_allowed(self,point):
        return not any(d['state']!='connected' and distance(point,d['point'])<=self.tolerance for d in self.contact_decisions)

    def export(self,uf=None):
        uf=uf or self.uf; groups=defaultdict(lambda:{'pins':[],'wires':[]})
        for ref,_,_ in self.pins: groups[uf.root(self.ref_index[ref])]['pins'].append(ref)
        for i,e in enumerate(self.wires): groups[uf.root(i)]['wires'].append((i,e))
        nets={}; hyperedges=[]
        for group in groups.values():
            if not group['pins']: continue
            key='net_'+str(len(nets)+1)
            nets[key]={'hyperGraph':'('+','.join(sorted(group['pins']))+')','edges':{'edge_'+str(j+1):e for j,(_,e) in enumerate(group['wires'])}}
            hyperedges.append({'id':key,'pins':sorted(group['pins']),'physical_wire_ids':['wire_'+str(i) for i,_ in group['wires']]})
        result={'components':self.obs['components'],'pins':self.obs['pins'],'nets':nets}
        return result,{'kind':'electrical_hypergraph','hyperedges':hyperedges,'semantic_links':self.semantic_links}

    def candidates(self,line_mask=None,max_gap=6):
        candidates=[]
        for ref,p,c in self.pins:
            if ref in self.locally_attached: continue
            box=self.obs['components'][c]['bbox']; cx=(box[0]+box[2])/2; cy=(box[1]+box[3])/2
            options={}
            for j in sorted(self.index.query(point_box(p,max_gap))):
                q=projection(p,self.wires[j]); gap=distance(p,q)
                if gap>max_gap or self.uf.root(self.ref_index[ref])==self.uf.root(j): continue
                dx,dy=q['x']-p['x'],q['y']-p['y']
                ox,oy=p['x']-cx,p['y']-cy
                norm=math.hypot(ox,oy)*max(gap,1e-9)
                alignment=max(0,(dx*ox+dy*oy)/norm) if norm else 0
                support=0.; available=line_mask is not None
                if available:
                    h,w=line_mask.shape; samples=9
                    for k in range(samples):
                        t=(k+.5)/samples; x=round(p['x']+t*dx); y=round(h-(p['y']+t*dy))
                        support+=int(0<=x<w and 0<=y<h and line_mask[y,x]>0)/samples
                score=.4*(1-gap/max_gap)+.45*support+.15*alignment
                option={'id':f'{ref}@wire_{j}','pin':ref,'wire':j,'point':q,'gap':gap,'image_support':support,'outward_alignment':alignment,'score':score,'image_evidence_available':available,'topology_allowed':self.contact_allowed(p) and self.contact_allowed(q)}
                root=self.uf.root(j)
                if root not in options or score>options[root]['score']: options[root]=option
            candidates.extend(sorted(options.values(),key=lambda v:(-v['score'],v['id']))[:3])
        return candidates

    def associate(self,candidates,beam_width=16,threshold=.62,forced=()):
        grouped=defaultdict(list)
        for candidate in candidates:
            if candidate['topology_allowed'] and candidate['image_evidence_available'] and candidate['image_support']>=.55 and candidate['score']>=threshold:
                grouped[candidate['pin']].append(candidate)
        # Objective: association evidence minus passive-short penalties, subject
        # to explicit power-domain constraints. A bounded beam is approximate.
        beams=[(0.,self.uf.clone(),{k:set(v) for k,v in self.domains.items()},[])]
        rejected=[]
        for ref,options in sorted(grouped.items()):
            expanded=[]
            for objective,uf,domains,chosen in beams:
                force_options=[o for o in options if o['id'] in forced]
                if not force_options: expanded.append((objective,uf,domains,chosen))
                for option in force_options or options:
                    a=uf.root(self.ref_index[ref]); b=uf.root(option['wire'])
                    merged=domains.get(a,set())|domains.get(b,set())
                    if incompatible(merged) and not incompatible(domains.get(a,set())) and not incompatible(domains.get(b,set())):
                        rejected.append({'candidate':option['id'],'reason':'new_explicit_power_domain_conflict'}); continue
                    component=next(c for r,_,c in self.pins if r==ref)
                    comp=self.obs['components'][component]; siblings=[r for r,_,c in self.pins if c==component and r!=ref]
                    penalty=0.
                    if comp['type'] in ('resistor','capacitor','inductor') and len(siblings)==1:
                        value=str(comp.get('value') or '').upper()
                        if uf.root(self.ref_index[siblings[0]])==b and value not in ('0','0R','0Ω'):
                            penalty=.2
                    next_uf=uf.clone(); next_uf.union(a,b); next_domains={k:set(v) for k,v in domains.items()}
                    next_domains.pop(a,None); next_domains.pop(b,None); next_domains[next_uf.root(a)]=merged
                    gain=option.get('association_gain',option['score']-threshold)
                    expanded.append((objective+gain-penalty,next_uf,next_domains,chosen+[option]))
            if not expanded: raise ValueError('forced correction violates global constraints')
            beams=sorted(expanded,key=lambda s:(-s[0],tuple(o['id'] for o in s[3])))[:beam_width]
        best=beams[0]
        credible=[state for state in beams if best[0]-state[0]<=.04]
        consensus=set.intersection(*(set(o['id'] for o in state[3]) for state in credible)) if credible else set()
        # Forced choices still undergo the exact same image and circuit gates.
        consensus.update(forced)
        accepted=[o for o in best[3] if o['id'] in consensus]
        final=self.uf.clone()
        for option in accepted: final.union(self.ref_index[option['pin']],option['wire'])
        return final,{'optimizer':'bounded_global_beam','beam_width':beam_width,'objective':best[0],'accepted':accepted,'rejected':list({x['candidate']:x for x in rejected}.values()),'ambiguous_candidates':[o for o in candidates if o['id'] not in consensus]}


def circuit_checks(result,labels):
    issues=[]; refs_to_net={}
    for key,net in result['nets'].items():
        from .core import references
        for ref in references(net['hyperGraph']): refs_to_net[ref]=key
    expected={'resistor':2,'capacitor':2,'inductor':2,'fuse':2,'gnd':1,'vcc':1}
    for c,component in result['components'].items():
        if component['type'] in expected and len(result['pins'][c])!=expected[component['type']]:
            issues.append({'kind':'pin_cardinality','component':c,'expected':expected[component['type']],'observed':len(result['pins'][c]),'action':'review_only'})
    domains=defaultdict(set)
    for label in labels:
        domain=power_domain(label['name'])
        if domain and label['pin'] in refs_to_net: domains[refs_to_net[label['pin']]].add(domain)
    for key,values in domains.items():
        if incompatible(values): issues.append({'kind':'power_domain_conflict','net':key,'domains':sorted(values),'action':'review_only; preserve observed physical wires'})
    return issues


def parse_graph(observations,line_mask=None,optimize=True,forced=(),blocked=(),association_model=None):
    graph=DualGraph(observations)
    candidates=graph.candidates(line_mask)
    known={o['id']:o for o in candidates}
    if any(i not in known for i in forced): raise ValueError('correction references unknown association candidate')
    if any(not known[i]['topology_allowed'] or not known[i]['image_evidence_available'] or known[i]['image_support']<.55 or known[i]['score']<.62 for i in forced): raise ValueError('correction lacks sufficient local visual evidence/topology permission')
    if any(i not in known for i in blocked) or set(forced)&set(blocked): raise ValueError('invalid rejected association IDs')
    model_report=None
    selected=[o for o in candidates if o['id'] not in blocked]
    if association_model and optimize:
        selected,model_report=association_model.filter(graph,selected)
        if any(i not in {c['id'] for c in selected} for i in forced): raise ValueError('correction conflicts with learned association veto')
    if optimize: uf,association=graph.associate(selected,forced=forced)
    else: uf=graph.uf; association={'optimizer':'disabled','accepted':[],'ambiguous_candidates':candidates,'rejected':[]}
    result,electrical=graph.export(uf)
    errors=validate(result)
    if errors: raise ValueError('; '.join(errors))
    report={'framework':'GraphSchematic','geometry_graph':{'nodes':graph.nodes,'edges':graph.edges,'contacts':graph.contacts,'hypotheses':candidates},'electrical_hypergraph':electrical,'association':association,'circuit_issues':circuit_checks(result,observations.get('labels',[])),'llm':{'status':'validated_proposal_replay' if forced or blocked else 'not_called','accepted_corrections':list(forced),'rejected_associations':list(blocked)},'official_equivalence':False}
    report['learned_association']=model_report
    return result,report
