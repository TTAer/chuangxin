"""Coordinate semantics diagnostics, never an endpoint correction module."""
import math
import numpy as np

NORMAL={'left':(-1,0),'right':(1,0),'top':(0,-1),'bottom':(0,1)}

def error_vector(pred_xy,gt_xy,side):
    dx,dy=gt_xy[0]-pred_xy[0],gt_xy[1]-pred_xy[1]
    nx,ny=NORMAL[side]; tx,ty=-ny,nx
    return {'dx':dx,'dy':dy,'normal':dx*nx+dy*ny,'tangent':dx*tx+dy*ty,
            'distance':math.hypot(dx,dy),'orientation':'left/right' if nx else 'top/bottom'}

def project(point,a,b):
    vx,vy=b[0]-a[0],b[1]-a[1]; den=vx*vx+vy*vy
    t=max(0,min(1,((point[0]-a[0])*vx+(point[1]-a[1])*vy)/den)) if den else 0
    q=[a[0]+t*vx,a[1]+t*vy]
    return q,math.dist(point,q)

def nearest_wire_contact(point,wires,limit=3):
    candidates=[project(point,a,b) for a,b in wires]
    if not candidates: return None
    q,d=min(candidates,key=lambda r:r[1])
    return {'point_xy':q,'distance_from_pred':d,'definition':'nearest parsed segment projection; pin/wire boundary unresolved'} if d<=limit else None

def skeleton(mask):
    """Zhang-Suen thinning of local binary pixels; borders are not real tips."""
    img=np.pad(mask.astype(bool),1)
    for _ in range(1000):
        changed=False
        for step in (0,1):
            p=[img[:-2,1:-1],img[:-2,2:],img[1:-1,2:],img[2:,2:],
               img[2:,1:-1],img[2:,:-2],img[1:-1,:-2],img[:-2,:-2]]
            neighbors=sum(q.astype(np.uint8) for q in p)
            transitions=sum((~p[i]&p[(i+1)%8]).astype(np.uint8) for i in range(8))
            triples=(~(p[0]&p[2]&p[4])&~(p[2]&p[4]&p[6])) if step==0 else (~(p[0]&p[2]&p[6])&~(p[0]&p[4]&p[6]))
            remove=img[1:-1,1:-1]&(neighbors>=2)&(neighbors<=6)&(transitions==1)&triples
            if np.any(remove): img[1:-1,1:-1][remove]=False; changed=True
        if not changed: break
    return img[1:-1,1:-1]

def skeleton_endpoints(skel):
    p=np.pad(skel.astype(np.uint8),1); neighbors=np.zeros_like(skel,dtype=np.uint8)
    for dy in range(3):
        for dx in range(3):
            if dy!=1 or dx!=1: neighbors+=p[dy:dy+skel.shape[0],dx:dx+skel.shape[1]]
    ys,xs=np.where(skel&(neighbors==1))
    return [[int(x),int(y)] for x,y in zip(xs,ys) if 2<x<skel.shape[1]-3 and 2<y<skel.shape[0]-3]

def stats(rows):
    known=[r['error_vector'] for r in rows if r.get('error_vector')]
    if not known: return {'candidates':len(rows),'matched_geometry':0}
    def median(key,absolute=False):
        return float(np.median([abs(v[key]) if absolute else v[key] for v in known]))
    return {'candidates':len(rows),'matched_geometry':len(known),'EP5_nearest':sum(v['distance']<=5 for v in known)/len(rows),
        'median_distance':median('distance'),'median_normal':median('normal'),
        'median_abs_normal':median('normal',True),'median_abs_tangent':median('tangent',True),
        'normal_dominant_fraction':sum(abs(v['normal'])>2*abs(v['tangent']) and abs(v['normal'])>5 for v in known)/len(known)}
