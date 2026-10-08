"""Read-only pixel evidence; grades do not grant runtime authority."""
import math
import cv2
import numpy as np
from pcbparse.core import point_on_segment


def distance(a, b):
    return math.hypot(a['x']-b['x'], a['y']-b['y'])


def pixel_evidence(gray_bottom_left, point, box, wires, contact_allowed=True):
    """G2 needs a perpendicular boundary AND continuous normal stem.

    A wire crossing by itself can never upgrade an endpoint to G2/G3.
    Fixed pixel kernels are development evidence, not calibrated confidence.
    """
    gray=gray_bottom_left; h,w=gray.shape
    x,y=(int(round(point[k])) for k in ('x','y'))
    sides=[(abs(x-box[0]),'left'),(abs(x-box[2]),'right'),
           (abs(y-box[1]),'bottom'),(abs(y-box[3]),'top')]
    side=min(sides)[1]; horizontal=side in ('left','right')
    # Canonical coordinates: columns follow the pin normal, rows its axial direction.
    a,n=(y,x) if horizontal else (x,y)
    ink=gray<180 if horizontal else gray.T<180
    ah,nw=ink.shape
    def normal_support(start,end):
        start,end=sorted((start,end))
        if not (0<=start<=end<nw and 0<=a<ah): return 0.
        band=ink[max(0,a-2):min(ah,a+3),start:end+1]
        return float(np.mean(np.any(band,axis=0))) if band.size else 0.
    stroke=normal_support(n-3,n+3)==1.
    inward=1 if side in ('left','bottom') else -1
    boundary=None
    if stroke:
        for gap in range(4,41):
            bn=n+inward*gap
            if not (1<=bn<nw-1 and 8<=a<ah-8): continue
            bx,by=(bn,a) if horizontal else (a,bn)
            if not (box[0]<=bx<=box[2] and box[1]<=by<=box[3]): continue
            line=ink[a-8:a+9,bn-1:bn+2]
            # Long boundary parallel to body side; connected pin normal to it.
            if np.mean(np.any(line,axis=1))>=.88 and normal_support(n,bn)>=.88:
                boundary={'x':float(bx),'y':float(by),'gap':gap}; break
    contact=contact_allowed and any(point_on_segment(point,e,3) for e in wires)
    grade='G3' if boundary and contact else 'G2' if boundary else 'G1' if stroke else 'G0'
    return {'grade':grade,'side':side,'normal_stroke':bool(stroke),
            'body_boundary_candidate':boundary,'wire_segment_contact':contact,
            'wire_endpoint_distance':min((distance(point,p) for e in wires for p in e),default=None),
            'runtime_commit_authorized':False}


def image_gray(image):
    return cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)[::-1]


def bbox_iou(a,b):
    area=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
    union=max(0,a[2]-a[0])*max(0,a[3]-a[1])+max(0,b[2]-b[0])*max(0,b[3]-b[1])-area
    return area/union if union else 0.


def bucket(d):
    if d is None: return 'unmatched_component_or_no_GT_pin'
    for limit in (3,5,8,12):
        if d<=limit: return '<='+str(limit)
    return '>12'
