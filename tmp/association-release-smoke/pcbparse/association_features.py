"""Label-free geometry/electrical candidate graph shared by training and ONNX."""
from collections import defaultdict
import math
import numpy as np
from .graphschematic import projection
from .core import distance

NODE_FEATURES = 10
EDGE_FEATURES = 12

def candidate_graph(graph, candidates):
    """Contract observed wire groups; two relations: component and incidence.

    No target net, target pin identity, case ID or OCR text is encoded. Wire
    groups come only from the current observed geometry/topology decisions.
    """
    if not candidates: raise ValueError('no association candidates')
    relevant = {graph.nodes[c['pin']]['component'] for c in candidates}
    pins = [(r, p, c) for r, p, c in graph.pins if c in relevant]
    roots = sorted({graph.uf.root(c['wire']) for c in candidates})
    ids = {r: i for i, (r, _, _) in enumerate(pins)}
    wires = {r: len(pins) + i for i, r in enumerate(roots)}
    count = len(pins) + len(roots)
    if count > 2048: raise ValueError('association graph exceeds 2048-node budget')
    x = np.zeros((count, NODE_FEATURES), np.float32)
    component = np.eye(count, dtype=np.float32)
    incidence = np.eye(count, dtype=np.float32)
    members = defaultdict(list)
    for ref, p, c in pins:
        members[c].append(ids[ref])
        comp = graph.obs['components'][c]; b = comp['bbox']; kind = comp['type']
        x[ids[ref]] = [1, 0, min(4, (b[2]-b[0])/100), min(4, (b[3]-b[1])/100), min(4, len(graph.obs['pins'][c])/20), 0, 0, kind in ('resistor','capacitor','inductor','r','c','l'), kind in ('vcc','gnd','v'), kind in ('ic','box')]
    for group in members.values(): component[np.ix_(group, group)] = 1
    grouped = defaultdict(list)
    for i, edge in enumerate(graph.wires):
        root = graph.uf.root(i)
        if root in wires: grouped[root].extend(edge)
    for root, points in grouped.items():
        xx = [p['x'] for p in points]; yy = [p['y'] for p in points]
        x[wires[root]] = [0, 1, min(4, (max(xx)-min(xx))/100), min(4, (max(yy)-min(yy))/100), 0, min(4, len(points)/40), min(4, math.hypot(max(xx)-min(xx),max(yy)-min(yy))/100), 0, 0, 0]
    pairs = []; features = []
    for c in candidates:
        a, b = ids[c['pin']], wires[graph.uf.root(c['wire'])]
        incidence[a,b] = incidence[b,a] = 1
        pairs.append([a,b]); p = graph.nodes[c['pin']]['point']; q = c['point']
        edge = graph.wires[c['wire']]; dx = edge[1]['x']-edge[0]['x']; dy = edge[1]['y']-edge[0]['y']; length = max(1, math.hypot(dx,dy))
        box = graph.obs['components'][graph.nodes[c['pin']]['component']]['bbox']
        features.append([c['gap']/6, c['image_support'], c['outward_alignment'], (q['x']-p['x'])/6, (q['y']-p['y'])/6, abs(dx)/length, abs(dy)/length, min(2,min(distance(q,z) for z in edge)/6), (p['x']-(box[0]+box[2])/2)/max(1,box[2]-box[0]), (p['y']-(box[1]+box[3])/2)/max(1,box[3]-box[1]), c['score'], float(c['topology_allowed'])])
    # Row-normalized relation matrices support permutation-equivariant mean
    # aggregation and avoid optional graph-library runtime dependencies.
    component /= component.sum(1, keepdims=True)
    incidence /= incidence.sum(1, keepdims=True)
    return {'nodes': x, 'component_relation': component, 'incidence_relation': incidence, 'pairs': np.asarray(pairs,np.int64), 'edge_features': np.asarray(features,np.float32)}
