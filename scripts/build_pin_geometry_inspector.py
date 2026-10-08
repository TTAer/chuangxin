"""Build read-only vector/recall audit and pixel review tool. No inference."""
import copy
import hashlib
import json
import math
import sys
from pathlib import Path
from collections import defaultdict
import cv2
import numpy as np
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from experiments.pin_geometry_inspector import error_vector,nearest_wire_contact,skeleton,skeleton_endpoints,stats
from scripts.audit_unresolved_endpoint_quality import component_pairs
from pcbparse.core import assignment
from pcbparse.dataset import index_dataset
from pcbparse.vision import read_image
from audit_freeze_a_sources import sexp,children,first

OUT=ROOT/'output/pin-geometry-inspector'
def load(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def save(path,obj):path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def xy(p,h):return [p['x'],h-p['y']]
def match(points,targets,tol=5):
    weights=[[1+(tol-math.dist(p,q))/tol if math.dist(p,q)<=tol else 0 for q in targets] for p in points]
    return [(i,j) for i,j in assignment(weights) if weights[i][j]>0]

def add_crop(row,image,wires,texts):
    h,w=image.shape[:2]; p=row['points']['predicted'] or row['points']['CAD_outer_tip']
    x1,y1=max(0,int(p[0])-80),max(0,int(p[1])-80);x2,y2=min(w,x1+160),min(h,y1+160)
    crop=image[y1:y2,x1:x2]; g=cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY)
    sk=skeleton(g<180); endpoints=skeleton_endpoints(sk)
    near=min(endpoints,key=lambda q:math.dist([q[0]+x1,q[1]+y1],p)) if endpoints else None
    row['points']['skeleton_endpoint_candidate']=[near[0]+x1,near[1]+y1] if near and math.dist([near[0]+x1,near[1]+y1],p)<=20 else None
    # Text endpoints and crop artifacts are not silently called symbol pin tips.
    row['skeleton_semantics']='all local ink includes wire/text/body; degree-1 point is a candidate, never certified pin tip'
    folder=OUT/'crops';folder.mkdir(exist_ok=True)
    path=folder/(row['id']+'.png');cv2.imencode('.png',crop)[1].tofile(str(path))
    overlay=crop.copy();overlay[sk]=[220,170,0]
    cv2.imencode('.png',overlay)[1].tofile(str(folder/(row['id']+'-skeleton.png')))
    wire_mask=np.zeros_like(g,dtype=np.uint8)
    for a,b in wires:cv2.line(wire_mask,(round(a[0]-x1),round(a[1]-y1)),(round(b[0]-x1),round(b[1]-y1)),255,1)
    wover=crop.copy();wover[wire_mask>0]=[100,200,50]
    cv2.imencode('.png',wover)[1].tofile(str(folder/(row['id']+'-wire.png')))
    unique_text={}
    for t in texts:
        signature=(t['text'],tuple(round(v,3) for v in t['bbox_xyxy']))
        unique_text.setdefault(signature,t)
    row.update(crop='crops/'+path.name,crop_sha256=sha(path),crop_origin_xy=[x1,y1],crop_size=[x2-x1,y2-y1],
        skeleton_crop='crops/'+row['id']+'-skeleton.png',wire_crop='crops/'+row['id']+'-wire.png',
        local_text_candidates=[t for t in unique_text.values() if x1<=sum(t['bbox_xyxy'][::2])/2<=x2 and y1<=sum(t['bbox_xyxy'][1::2])/2<=y2])
    ref=row['points']['dataset_GT']
    row['distance_table_to_dataset_GT']={k:math.dist(v,ref) if v is not None and ref is not None else None for k,v in row['points'].items() if k!='dataset_GT'}

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    protocol={'status':'coordinate-definition calibration only; no model training/inference/correction',
        'coordinate_system':'image top-left x right y down; legacy cached bottom-left converted once using h-y',
        'sampling':'raster first two candidates per domain/orientation/state at score>=.7; fixed source order, no GT/error selection; three prior OCR commits separate focused cases',
        'FreezeA_sampling':'first two visible locally matched CAD pins per side; no prediction or certified dataset GT',
        'GT_definition':'unresolved; no tip/contact convention assigned automatically',
        'recall':'within matched component, one-to-one baseline cover then one-to-one new cover; all-GT and matched-component denominators separate',
        'tip_policy':'raster outer tip unknown until manual semantic review; continuous pin/wire stroke cannot supply unique boundary',
        'official_tolerance':'development proxy 5px; not official evaluator equivalence'}
    save(OUT/'protocol.json',protocol)
    samples={s['id']:s for s in index_dataset(ROOT/'data/raw/PCBnet/Dataset300')}
    rows=[];inspector=[];picked=defaultdict(int);recall={};joints=[]
    for domain in ('gt10','pcb55'):
        counts={'GT_pins_all_IC_box':0,'GT_pins_matched_components':0,'baseline_covered_all':0,'previously_uncovered_all':0,'previously_uncovered_matched':0,'newly_covered':0,'proposed_candidates':0}
        for folder in sorted((ROOT/'output/designator-stable-catalog'/domain).iterdir()):
            if not (folder/'observations.json').exists():continue
            case=folder.name;obs=load(folder/'observations.json');h=obs['image']['height']
            imagepath=ROOT/'data/raw/official/gt-0930/10GT'/case/(case+'.png') if domain=='gt10' else Path(samples[case]['image'])
            gtpath=imagepath.with_suffix('.json') if domain=='gt10' else ROOT/'output/benchmark-recognition-pcb55'/case/'golden.json'
            gt=load(gtpath);pairs=component_pairs(obs,gt);wires=[[xy(p,h) for p in e] for e in obs['wires']]
            audit=load(ROOT/'output/endpoint-quality'/domain/case/'audit.json')['rows']
            all_gt=sum(len(gt['pins'].get(g,{})) for g,c in gt['components'].items() if c['type'] in ('ic','box'))
            matched=covered=new=0
            for key,g in pairs.items():
                gps=[xy(p['point'],h) for p in gt['pins'].get(g,{}).values()];old=[xy(p['point'],h) for p in obs['pins'].get(key,{}).values()]
                taken={j for _,j in match(old,gps)};missing=[p for j,p in enumerate(gps) if j not in taken]
                candidates=[xy(r['point'],h) for r in audit if r['component']==key]
                matched+=len(gps);covered+=len(taken);new+=len(match(candidates,missing))
            for field,n in [('GT_pins_all_IC_box',all_gt),('GT_pins_matched_components',matched),('baseline_covered_all',covered),('previously_uncovered_all',all_gt-covered),('previously_uncovered_matched',matched-covered),('newly_covered',new),('proposed_candidates',len(audit))]:counts[field]+=n
            texts=[dict(t,bbox_xyxy=[t['bbox'][0],h-t['bbox'][3],t['bbox'][2],h-t['bbox'][1]]) for t in obs.get('texts',[])]
            report=load(ROOT/'output/local-pin-evidence'/domain/case/'report.json')
            image=None
            for r in audit:
                p=xy(r['point'],h);key=r['component'];g=pairs.get(key);gps=list(gt['pins'].get(g,{}).items()) if g else []
                gp=min(gps,key=lambda q:math.dist(p,xy(q[1]['point'],h))) if gps else None
                # Evidence names low bottom-left y as 'bottom', already a physical side.
                # Reflect coordinates, not the semantic side a second time.
                side=r['side']
                typ='connector' if key.startswith(('P','J')) else 'passive-like box' if key.startswith(('R','C','L')) else 'large box IC' if (obs['components'][key]['bbox'][2]-obs['components'][key]['bbox'][0])*(obs['components'][key]['bbox'][3]-obs['components'][key]['bbox'][1])>=10000 else 'small IC'
                contact=nearest_wire_contact(p,wires)
                nearby_label=any(math.dist(p,[(t['bbox_xyxy'][0]+t['bbox_xyxy'][2])/2,(t['bbox_xyxy'][1]+t['bbox_xyxy'][3])/2])<=32 and t['text'] in {a.get('name') for a in obs.get('labels',[])} for t in texts)
                state='parsed-wire-contact' if r['wire_segment_contact'] else 'flying-label-adjacent' if nearby_label else 'no-parsed-contact (unconnected not proven)'
                row={'id':domain+'-'+case+'-'+r['id'].replace('@','_'),'domain':domain,'case':case,'component':key,'candidate_id':r['id'],
                    'component_class_proxy':typ,'pin_state_proxy':state,'side':side,'grade':r['grade'],'score':r['score'],
                    'points':{'predicted':p,'dataset_GT':xy(gp[1]['point'],h) if gp else None,
                        'body_edge_candidate':xy(r['body_boundary_candidate'],h) if r['body_boundary_candidate'] else None,
                        'stroke_outer_tip':None,'parsed_wire_contact_candidate':contact['point_xy'] if contact else None,'CAD_outer_tip':None},
                    'GT_pin_nearest':gp[0] if gp else None,'error_vector':error_vector(p,xy(gp[1]['point'],h),side) if gp else None,
                    'matching_scope':'nearest pin inside one-to-one IoU matched component; vectors are diagnostic, not identities',
                    'official_point_semantics_confirmed':False,'human_reviewed':False,'image':str(imagepath.relative_to(ROOT)),
                    'source_image_sha256':sha(imagepath),'source_GT_sha256':sha(gtpath)}
                rows.append(row)
                focus=any(d['tip']==r['id'] and d['committed'] for d in report['decisions'])
                group=(domain,row['error_vector']['orientation'] if row['error_vector'] else ('left/right' if side in ('left','right') else 'top/bottom'),state)
                chosen=(r['score'] or 0)>=.7 and picked[group]<2
                if chosen or focus:
                    if chosen:picked[group]+=1
                    if image is None:image=read_image(imagepath)
                    item=copy.deepcopy(row);item['selection']='fixed feature stratum' if chosen else 'prior OCR commit focused diagnostic'
                    add_crop(item,image,wires,texts);inspector.append(item)
            for d in report['decisions']:
                g=pairs.get(d['component']);gp=gt['pins'].get(g,{}).get('pin_'+d['number']) if g else None
                dist=math.dist(xy(d['point'],h),xy(gp['point'],h)) if gp else None
                joints.append({'domain':domain,'case':case,'component':d['component'],'number':d['number'],'committed_in_prior_offline_experiment':d['committed'],
                    'number_in_GT_matched_component':gp is not None,'name_exact_GT':gp is not None and d['name']==gp['pinname'],
                    'distance_to_exact_number_GT':dist,'geometry_EP5':dist is not None and dist<=5,
                    'geometry_and_exact_identity':gp is not None and d['name']==gp['pinname'] and dist<=5,
                    'prior_topology_and_add_only_checks_passed':bool(d['committed']),
                    'runtime_authority':'proposal-only; semantics not calibrated','topology_prior_decision':d.get('reason','unique physical contact, old facts locked')})
        counts.update(candidate_precision_5=counts['newly_covered']/counts['proposed_candidates'] if counts['proposed_candidates'] else None,
            recoverable_pin_recall_5_all=counts['newly_covered']/counts['previously_uncovered_all'] if counts['previously_uncovered_all'] else None,
            recoverable_pin_recall_5_matched_components=counts['newly_covered']/counts['previously_uncovered_matched'] if counts['previously_uncovered_matched'] else None)
        recall[domain]=counts
    # CAD/PDF anatomical reference only: no RC0 inference and no official GT.
    fa=load(ROOT/'output/freeze-a1-identity/identity-audit.json');cadrows=[];cadpicked=defaultdict(int)
    for record in fa['records']:
        if record['observed_review']['status']!='agent_pixel_matched':continue
        tree=sexp((ROOT/record['source_cad']).read_text(encoding='utf-8'))
        cwires=[]
        for wire in children(tree,'wire'):
            pts=first(wire,'pts')
            ends=children(pts or [],'xy')
            if len(ends)==2:cwires.append([[float(v)*144/25.4 for v in e[1:]] for e in ends])
        image=None
        for pin in record['pins']:
            tip=pin['point_xy'];body=pin['body_contact_xy'];dx,dy=tip[0]-body[0],tip[1]-body[1]
            side=('left' if dx<0 else 'right') if abs(dx)>abs(dy) else ('top' if dy<0 else 'bottom')
            contact=nearest_wire_contact(tip,cwires,.25)
            row={'id':'freezeA-'+str(record['id'])+'-'+pin['number'],'domain':'Freeze-A1','case':str(record['id']),
                'component':record['observed_review']['observed_key'],'component_class_proxy':'CAD registered IC','side':side,
                'pin_state_proxy':'CAD wire contact candidate' if contact else 'CAD no wire contact; pixel verification pending',
                'points':{'predicted':None,'dataset_GT':None,'body_edge_candidate':body,'stroke_outer_tip':tip,'CAD_outer_tip':tip,'parsed_wire_contact_candidate':contact['point_xy'] if contact else None},
                'CAD_pin_length_pixels':math.dist(tip,body),'CAD_number':pin['number'],'CAD_name':pin['name'],
                'official_point_semantics_confirmed':False,'human_reviewed':False,'source_image_sha256':record['image_sha256'],
                'source_annotation_sha256':record['annotation_sha256'],'image':record['image'],'error_vector':None}
            cadrows.append(row)
            if cadpicked[side]<2:
                cadpicked[side]+=1
                if image is None:image=read_image(ROOT/record['image'])
                item=copy.deepcopy(row);item['selection']='first two per CAD side, locally matched identity'
                add_crop(item,image,cwires,load(ROOT/record['annotation'])['pdf_text_layer_unassigned']);inspector.append(item)
    vectors={}
    for domain in ('gt10','pcb55'):
        rs=[r for r in rows if r['domain']==domain]
        vectors[domain]={'ALL':stats(rs)}
        for field in ('side','component_class_proxy','pin_state_proxy','grade'):
            vectors[domain][field]={v:stats([r for r in rs if r[field]==v]) for v in sorted({r[field] for r in rs})}
    summary={'runtime_changed':False,'training_runs':0,'inference_on_FreezeA':0,'raster_candidates':len(rows),'inspector_cases':len(inspector),
        'recall':recall,'error_vectors':vectors,'FreezeA_anatomical_reference':{'pins':len(cadrows),'median_pin_length_px':float(np.median([r['CAD_pin_length_pixels'] for r in cadrows])) if cadrows else None,'dataset_GT_points':0,'verified_designs':0},
        'joint_proposals':len(joints),'joint_geometry_identity_matches':sum(r['geometry_and_exact_identity'] for r in joints),
        'joint_geometry_identity_and_topology_matches':sum(r['geometry_and_exact_identity'] and r['prior_topology_and_add_only_checks_passed'] for r in joints),
        'coordinate_convention_status':'UNRESOLVED; no snap or endpoint conversion authorized by evidence'}
    summary['definition_distances']={}
    for domain in ('gt10','pcb55','Freeze-A1'):
        rs=[r for r in rows+cadrows if r['domain']==domain]
        dt={}
        for key in ('predicted','body_edge_candidate','stroke_outer_tip','parsed_wire_contact_candidate'):
            values=[math.dist(r['points'][key],r['points']['dataset_GT']) for r in rs if r['points'].get(key) is not None and r['points']['dataset_GT'] is not None]
            dt[key]={'total_cases':len(rs),'available_GT_distances':len(values),'median_distance':float(np.median(values)) if values else None,'fraction_le_5_among_available':sum(v<=5 for v in values)/len(values) if values else None}
        summary['definition_distances'][domain]=dt
    save(OUT/'vector-audit.json',rows);save(OUT/'cad-anatomy.json',cadrows);save(OUT/'joint-audit.json',joints);save(OUT/'cases.json',inspector);save(OUT/'summary.json',summary)
    template=(ROOT/'scripts/pin_geometry_inspector_template.html').read_text(encoding='utf-8')
    encoded=json.dumps({'protocol':protocol,'cases':inspector},ensure_ascii=False).replace('<','\\u003c')
    (OUT/'index.html').write_text(template.replace('__DATA__',encoded),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
