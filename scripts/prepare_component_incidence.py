"""Lossless component incidence targets + candidate coordinate correspondence audit.

Source annotations are NOT pin partitions. This script never creates pin IDs,
never unions components through devices, and never emits training admission.
"""
import json, hashlib, sys
from pathlib import Path
from itertools import combinations
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from PIL import Image
from pcbparse.core import assignment

def pairs(nets,keep):
    return {tuple(x) for net in nets for x in combinations(sorted({int(c) for c,_ in net}&keep),2)}

def main():
    source=ROOT/'data/raw/CircuitDigitization-source'
    truth_path=source/'ground_truth/real_nets_verified.json'
    truth=json.loads(truth_path.read_text(encoding='utf-8'))
    modality=json.loads((source/'docs/research/experiments/revision2/image_modality.json').read_text())['net31']
    upstream=json.loads((source/'docs/research/experiments/revision_evidence/drafter_map_n31_provenance.json').read_text())
    local=ROOT/'data/raw/CGHD-v16-subset';index={r['sample']:r for r in json.loads((local/'records.json').read_text())}
    compiled=[];alignment=[]
    for key,r in truth.items():
        if key.startswith('_'):continue
        keep=set(r['electrical_idxs']);positive=pairs(r['nets'],keep);allpairs=set(combinations(sorted(keep),2));negative=allpairs-positive
        witnesses=[]
        for a,b,c in combinations(sorted(keep),3):
            edges={(a,b),(a,c),(b,c)}
            if len(edges&positive)==2:witnesses.append([a,b,c])
        compiled.append({'image_id':key,'components':r['components'],'electrical_ids':sorted(keep),
                         'nets_component_incidence':r['nets'],'positive_component_pairs':sorted(positive),'negative_component_pairs':sorted(negative),
                         'nontransitivity_witnesses':witnesses[:5], 'nontransitive_triple_count':len(witnesses),
                         'upstream_path_author_claim':upstream['benchmark_image_to_upstream_path'].get(key),
                         'pin_identity_supervision':False,'training_admissible':False})
        stem=key.removesuffix('_jpg');record=index.get(stem)
        if record is None:continue
        if not record['instances']:continue
        labels=json.loads((local/record['instances']).read_text());image_path=local/record['image']
        with Image.open(image_path) as im:w,h=im.size
        # Diagnose coordinates only. Class identity has not been mapped/verified.
        shapes=[s for s in labels['shapes'] if s['label']!='text' and len(s['points'])>1]
        centers=np.array([(np.min(s['points'],0)+np.max(s['points'],0))/2 for s in shapes])/np.array([w,h])
        ids=sorted(keep);target=np.array([[r['components'][str(i)]['cx'],r['components'][str(i)]['cy']] for i in ids])
        variants=[]
        for swap in (False,True):
            for flipx in (False,True):
                for flipy in (False,True):
                    points=centers[:,::-1].copy() if swap else centers.copy()
                    if flipx:points[:,0]=1-points[:,0]
                    if flipy:points[:,1]=1-points[:,1]
                    distances=np.linalg.norm(target[:,None,:]-points[None,:,:],axis=-1)
                    matches=assignment((-distances).tolist());costs=[float(distances[i,j]) for i,j in matches]
                    variants.append({'swap_xy':swap,'flip_x':flipx,'flip_y':flipy,'mean_center_distance':float(np.mean(costs)),
                                     'max_center_distance':max(costs),'matched':len(matches),
                                     'proposed_component_matches':[{'author_id':ids[i],'local_shape_index':j,'local_label':shapes[j]['label'],'author_type':r['components'][str(ids[i])]['type'],'distance':float(distances[i,j])} for i,j in matches]})
        variants.sort(key=lambda x:x['mean_center_distance'])
        aliases={'voltage-DC':'voltage.dc','IC-NE555':'integrated_circuit.ne555','IC':'integrated_circuit',
                 'capacitor-polarized':'capacitor.polarized','capacitor-unpolarized':'capacitor.unpolarized',
                 'transistor-FET':'transistor.fet','transistor-BJT':'transistor.bjt','resistor':'resistor','diode':'diode'}
        best=variants[0]
        type_agreement=sum(aliases.get(x['author_type'])==x['local_label'] for x in best['proposed_component_matches'])
        observed_path=local/(record['segmentation'] if modality[stem] else record['image'])
        alignment.append({'image_id':key,'author_modality_stroke_map':modality[stem],
                          'local_modality_candidate':str(observed_path.relative_to(ROOT)),
                          'local_candidate_exists':observed_path.exists(),
                          'local_candidate_sha256':hashlib.sha256(observed_path.read_bytes()).hexdigest() if observed_path.exists() else None,
                          'best_coordinate_proposal':best,'runner_up_mean_distance':variants[1]['mean_center_distance'],
                          'class_correspondence_count':type_agreement,'expected_electrical_count':len(ids),
                          'class_identity_verified':False,'pixel_identity_verified':False,'admitted':False})
    # Semantic fixture: sharing a resistor does not connect both of its nets.
    test=pairs([[[0,'e'],[1,'e']],[[1,'e'],[2,'e']]],{0,1,2})
    assert test=={(0,1),(1,2)} and (0,2) not in test
    out=ROOT/'output/component-incidence-20261007';out.mkdir(parents=True,exist_ok=True)
    (out/'targets.json').write_text(json.dumps({'scope':'author component-pair relation, NOT electrical pin net partition','source_sha256':hashlib.sha256(truth_path.read_bytes()).hexdigest(),'records':compiled},indent=2),encoding='utf-8')
    result={'records':len(compiled),'positive_pairs':sum(len(r['positive_component_pairs']) for r in compiled),
            'negative_pairs':sum(len(r['negative_component_pairs']) for r in compiled),
            'nontransitive_records':sum(r['nontransitive_triple_count']>0 for r in compiled),'coordinate_candidates':alignment,
            'training_admitted':0,'pin_labels_created':0,'model_training_executed':False,'fixture_passed':True}
    (out/'audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    # Local stroke-map display with matched IDs; no wire GT overlay as an input.
    from PIL import ImageDraw
    sheet=Image.new('RGB',(1280,1360),'#eeeeee')
    for i,row in enumerate(alignment):
        image=Image.open(ROOT/row['local_modality_candidate']).convert('RGB').resize((640,640))
        draw=ImageDraw.Draw(image);entry=truth[row['image_id']]
        for match in row['best_coordinate_proposal']['proposed_component_matches']:
            comp=entry['components'][str(match['author_id'])];x=int(comp['cx']*640);y=int(comp['cy']*640)
            draw.ellipse((x-7,y-7,x+7,y+7),outline='red',width=2);draw.text((x+9,y),str(match['author_id']),fill='red')
        x0=(i%2)*640;y0=(i//2)*680;sheet.paste(image,(x0,y0));caption=ImageDraw.Draw(sheet)
        caption.text((x0+10,y0+646),row['image_id']+' | PROPOSAL, not human verified',fill='black')
    sheet.save(out/'coordinate-proposals.png')
    print(json.dumps({**{k:v for k,v in result.items() if k!='coordinate_candidates'},'alignment_summary':[{k:r[k] for k in ('image_id','author_modality_stroke_map')}|{'best_mean':r['best_coordinate_proposal']['mean_center_distance'],'matched':r['best_coordinate_proposal']['matched']} for r in alignment]}))

if __name__=='__main__':main()
