"""Expand real developer data using pixel-only proposals, then weak labels.

Freeze-A is never read; the original 133 development samples are unchanged.
"""
import hashlib
import json
from pathlib import Path
import sys
from collections import Counter
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import cv2
import numpy as np
from pcbparse.dataset import index_dataset
from pcbparse.vision import read_image,wire_lines
from pcbparse.topology_perception import crossing_candidates,patch_tensor
from experiments.port_partition import port_membership,compatible_partitions


def resolve_training_crop_conflicts(images,targets,valid,split,original_count):
    groups={};conflicts=[];valid=valid.copy()
    for i in np.flatnonzero(split==0):
        groups.setdefault(hashlib.sha256(images[i].tobytes()).hexdigest(),[]).append(int(i))
    for ids in groups.values():
        if len(ids)<2:continue
        t=targets[ids];v=valid[ids].astype(bool)
        inconsistent=np.any((t==1)&v,axis=0)&np.any((t==0)&v,axis=0)
        if inconsistent.any():
            for i in ids:
                if i>=original_count:valid[i,inconsistent]=0
            conflicts.append({'indices':ids,'conflicting_pair_indices':np.flatnonzero(inconsistent).tolist(),
                              'resolution':'new conflicting relations unknown; original cohort preserved'})
    return valid,conflicts


def main():
    out=ROOT/'data/processed/port-partition-expanded'
    out.mkdir(parents=True,exist_ok=True)
    if (out/'relations.npz').exists():raise ValueError('Expansion exists; preserve it')
    base=ROOT/'data/processed/port-partition-v1'
    z=np.load(base/'relations.npz')
    rows=json.loads((base/'records.json').read_text(encoding='utf-8'))
    images=list(z['images']);targets=list(z['targets']);valids=list(z['valid']);split=list(z['split'])
    rows=[dict(r,source='original_conditional_cohort') for r in rows]
    train_sha={r['image_sha256'] for r in rows if r['split']=='train'}
    val_sha={r['image_sha256'] for r in rows if r['split']=='validation'}
    if train_sha&val_sha:raise ValueError('Original split leakage')
    original_points={}
    for r in rows:original_points.setdefault(r['image_sha256'],[]).append(r['point'])
    pages=[];seen=set();crop_seen=set();counts=Counter()
    for item in index_dataset(ROOT/'data/raw/PCBnet/Dataset300'):
        digest=item['sha256']
        if item['id']=='0237' or digest in seen:continue
        seen.add(digest)
        is_train=item['split']=='train'
        if is_train and digest in val_sha:raise ValueError('Expanded image leakage')
        # Propose candidates and materialize pixels BEFORE annotation is opened.
        image=read_image(item['image'])
        mask=(cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)<220).astype(np.uint8)*255
        lines,_=wire_lines(mask,split_junctions=False)
        candidates=crossing_candidates(lines)
        candidates=sorted(candidates,key=lambda c:hashlib.sha256(
            f"{digest}/{c['x']}/{c['y']}".encode()).hexdigest())[:512]
        proposed=[(c,np.rint(patch_tensor(image,c['x'],c['y'])[:3]*255).clip(0,255).astype(np.uint8)) for c in candidates]
        raw=json.loads(Path(item['annotation']).read_text(encoding='utf-8'))
        segments=[]
        for group,edge in raw['edges'].items():
            if edge.get('state') not in (None,'active') or edge.get('isFlywire'):continue
            for path in edge.get('keyPointPaths',{}).get('wire',[]):
                segments.extend((group,a,b) for a,b in zip(path,path[1:]) if a!=b)
        bounds=np.asarray([[min(a[0],b[0]),min(a[1],b[1]),max(a[0],b[0]),max(a[1],b[1])] for _,a,b in segments]).reshape(-1,4)
        page=Counter(proposed=len(proposed));annsha=hashlib.sha256(Path(item['annotation']).read_bytes()).hexdigest()
        for c,crop in proposed:
            point=[c['x'],c['y']]
            if is_train and any(np.linalg.norm(np.asarray(point)-p)<=3 for p in original_points.get(digest,[])):
                page['near_original_training_center_skipped']+=1;continue
            x,y=point
            nearby=np.flatnonzero((bounds[:,0]<=x+8)&(bounds[:,2]>=x-8)&(bounds[:,1]<=y+8)&(bounds[:,3]>=y-8))
            target,valid,owners=port_membership([segments[i] for i in nearby],point)
            if valid.sum()<3:page['insufficient_or_ambiguous_supervision']+=1;continue
            if not compatible_partitions(target,valid).any():raise ValueError('Illegal source relation')
            # Exact crop duplicates stay within a split. Keep relation conflict as unknown.
            key=(0 if is_train else 2,hashlib.sha256(crop.tobytes()).hexdigest(),tuple(target[valid.astype(bool)]),tuple(valid))
            if key in crop_seen:page['duplicate_crop_target']+=1;continue
            crop_seen.add(key)
            images.append(crop);targets.append(target);valids.append(valid);split.append(0 if is_train else 2)
            rows.append({'page':item['id'],'split':'train' if is_train else 'automatic_validation',
                         'image_sha256':digest,'annotation_sha256':annsha,'point':point,
                         'port_wire_groups':owners,'source':'pixel_only_automatic_candidate'})
            page['accepted']+=1
        pages.append({'page':item['id'],'split':item['split'],'image_sha256':digest,**dict(page)})
        counts.update(page)
        print(json.dumps({'page':item['id'],'split':item['split'],**dict(page)}),flush=True)
    arr=np.asarray(split)
    # Byte-equality of all four original cohort arrays, not only its counts.
    n=len(z['images'])
    for original,expanded in [(z['images'],np.asarray(images[:n])),(z['targets'],np.asarray(targets[:n])),(z['valid'],np.asarray(valids[:n])),(z['split'],arr[:n])]:
        if not np.array_equal(original,expanded):raise ValueError('Original cohort changed')
    training_sha={r['image_sha256'] for r in rows if r['split']=='train'}
    evaluation_sha={r['image_sha256'] for r in rows if r['split']!='train'}
    if training_sha&evaluation_sha:raise ValueError('Expanded split leakage')
    fixed_valid,conflicts=resolve_training_crop_conflicts(np.asarray(images),np.asarray(targets),np.asarray(valids),arr,n)
    (out/'label-conflict-audit.json').write_text(json.dumps({'conflicts':conflicts},indent=2),encoding='utf-8')
    np.savez_compressed(out/'relations.npz',images=np.asarray(images),targets=np.asarray(targets),valid=fixed_valid,split=arr)
    (out/'records.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
    meta={'schema':'port-partition-expanded-v1','sample_counts':{str(i):int((arr==i).sum()) for i in (0,1,2)},
          'split_codes':{'0':'train','1':'unchanged original development validation','2':'automatic candidate development diagnostic, not checkpoint selection'},
          'pages':pages,'counts':dict(counts),'original_arrays_identical':True,
          'original_sha256':hashlib.sha256((base/'relations.npz').read_bytes()).hexdigest(),
          'candidate_creation':'raw grayscale ink, no GT mask; all patches created before reading annotation; deterministic cap512/page',
          'supervision':'weak physical wire groups; unknown pairs ignored; no flying-label/pin-point supervision',
          'Freeze_A_used':False,'GT10_used':False,'independent_test':False,'image_split_overlap':0,
          'identical_training_crop_conflicts_resolved':len(conflicts),
          'limitations':'Automatic image candidates have no detector/OCR masking. Diagnostic drops insufficiently supervised proposals, so metrics are conditional and do not measure candidate precision/recall.'}
    (out/'preparation.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in meta.items() if k!='pages'}),flush=True)


if __name__=='__main__':main()
