"""Two-source endpoint consensus that preserves electrical partition evidence."""
import copy
from collections import defaultdict
from .core import distance,references
from .graphschematic import DualGraph,circuit_checks

def replay_electrical_associations(observations,baseline_report,line_mask=None):
    """Refine coordinates while retaining prior confirmed electrical relations."""
    graph=DualGraph(observations)
    association=copy.deepcopy(baseline_report['association'])
    for option in association['accepted']:
        graph.uf.union(graph.ref_index[option['pin']],option['wire'])
    result,electrical=graph.export()
    association['replayed_for_pin_consensus']=True
    association['evidence_coordinates']='baseline visual evidence; no new association accepted from coordinate averaging'
    candidates=graph.candidates(line_mask)
    report={'framework':'GraphSchematic','geometry_graph':{'nodes':graph.nodes,'edges':graph.edges,'contacts':graph.contacts,'hypotheses':candidates},'electrical_hypergraph':electrical,'association':association,'circuit_issues':circuit_checks(result,observations.get('labels',[])),'llm':copy.deepcopy(baseline_report.get('llm',{'status':'not_called'})),'official_equivalence':False}
    return result,report

def partition(graph):
    groups=defaultdict(lambda:{'pins':set(),'wires':set()})
    for ref in graph.ref_index: groups[graph.uf.root(graph.ref_index[ref])]['pins'].add(ref)
    for i in range(len(graph.wires)): groups[graph.uf.root(i)]['wires'].add(i)
    return {ref:(frozenset(g['pins']),frozenset(g['wires'])) for g in groups.values() for ref in g['pins']}

def electrical_signature(result):
    def point(p): return p['x'],p['y']
    def edge(e): return tuple(sorted(map(point,e)))
    return sorted((tuple(sorted(references(n['hyperGraph']))),tuple(sorted(edge(e) for e in n['edges'].values()))) for n in result['nets'].values())

def fuse_observations(base,auxiliary,max_disagreement=6):
    """Fuse only matching identities; graph changes roll back their proposals."""
    if base['components']!=auxiliary['components'] or base.get('wires',[])!=auxiliary.get('wires',[]):
        raise ValueError('pin consensus requires identical components and physical wires')
    if base.get('image')!=auxiliary.get('image'): raise ValueError('image dimensions differ')
    result=copy.deepcopy(base); proposed=[]; rejected=[]
    for component,pins in base['pins'].items():
        for number,pin in pins.items():
            ref=component+'.'+number[4:]; other=auxiliary['pins'].get(component,{}).get(number)
            if not other or other['pinname']!=pin['pinname']: continue
            gap=distance(pin['point'],other['point'])
            if gap<=1e-6: continue
            if gap>max_disagreement:
                rejected.append({'pin':ref,'reason':'model_disagreement','gap':gap}); continue
            result['pins'][component][number]['point']={k:(pin['point'][k]+other['point'][k])/2 for k in ('x','y')}
            proposed.append({'pin':ref,'component':component,'number':number,'gap':gap,'before':pin['point'],'after':result['pins'][component][number]['point']})
    initial=partition(DualGraph(base)); changed=partition(DualGraph(result))
    accepted=[]
    for proposal in proposed:
        if initial[proposal['pin']]!=changed[proposal['pin']]:
            result['pins'][proposal['component']][proposal['number']]=copy.deepcopy(base['pins'][proposal['component']][proposal['number']])
            rejected.append(dict(proposal,reason='electrical_partition_change'))
        else: accepted.append(proposal)
    if partition(DualGraph(result))!=initial:
        rejected.extend(dict(p,reason='joint_partition_rollback') for p in accepted)
        accepted=[]; result=copy.deepcopy(base)
    return result,{'module':'TopologyPreservingPinConsensus','max_disagreement':max_disagreement,'fusion':'equal coordinate average; no learned confidence calibration','accepted':accepted,'rejected':rejected,'preserves_pin_identity':True,'preserves_physical_wires':True,'preserves_observed_partition':True}
