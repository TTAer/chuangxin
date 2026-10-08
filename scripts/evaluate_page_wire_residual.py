"""New native-page holdouts for two matched loss trials; no pressure selection."""
import sys,json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,cv2,torch
from PIL import Image
from experiments.radial_evidence_net import RadialEvidenceNet,crossing_patches
from experiments.spatial_evidence_net import SpatialEvidenceNet
from experiments.contrast_edge_circuit_reader import ContrastOnlyReader
from scripts.prepare_circuit_program import compile_page
from scripts.evaluate_ordered_holdout import render_irregular
from scripts.ordered_line_rectification_v2 import register_ordered
from scripts.train_circuit_program import metrics
from experiments.page_wire_residual import page_scores
from experiments.stroke_residual import decisions

TRIALS=['risk','residual']


def trial_folder(trial):
    return ROOT/f'output/training/topology-cost-v1-{trial}'


def main():
    out=ROOT/'output/page-wire-residual-holdout';out.mkdir(parents=True,exist_ok=True)
    if (out/'protocol.json').exists():raise ValueError('Preserve holdout')
    development=json.loads((ROOT/'output/page-wire-residual-strict-development/report.json').read_text());assert development['gate_passed']
    thresholds=development['thresholds'];selected='residual'
    protocol={'date':'2026-10-08','scope':'new disjoint native synthetic programs only, no real accuracy',
              'selected_trial_dev_only':selected,'selection':'third train/dev attempt passed; thresholds fixed before new pages, old risk weights only a frozen diagnostic comparator',
              'primary':'256 jitter+/-4 scale.55:.85 blur.9:1.2 width1:7 noise5',
              'uniform':'128 same style without jitter','hard':'128 jitter+/-4 blur1.2:1.6; diagnostic only',
              'rotation':'128 unsupported +/-8:15degree pages',
              'gate':'residual primary coverage>=.9 F1>=.98 exact-net>=.95 net gain>=.005 over frozen baseline; FP not increased, broken programs0; rotation accept<=.05',
              'loss_trials':TRIALS,'no_pressure_selection':True,
              'thresholds':thresholds,'new_optimizer_steps':0,'assumption':'page-end full-length wires and shared page linewidth; not general schematic guarantee',
              'risk_checkpoint_hash':hashlib.sha256((trial_folder('risk')/'best.pt').read_bytes()).hexdigest()}
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    seen=set()
    for bank in ('circuit-program-v1','circuit-program-v2','factorized-circuit-v1','separable-circuit-v1','registered-circuit-v1','contrast-edge-v1','radial-evidence-v1'):
        for p in (ROOT/f'data/processed/{bank}').glob('*.npz'):
            with np.load(p) as z:
                if 'program' in z:seen.update(hashlib.sha256(g.tobytes()).hexdigest() for g in z['program'])
    for folder in ('registration-v2-holdout','guard-v2-holdout','ordered-rectification-holdout','radial-consensus-holdout','evidence-factorial-holdout','topology-cost-holdout'):
        for p in (ROOT/f'output/{folder}').glob('*-inputs.npz'):
            with np.load(p) as z:seen.update(hashlib.sha256(g.tobytes()).hexdigest() for g in z['program'])
    torch.set_num_threads(4);base=ContrastOnlyReader().cuda().eval();base.load_state_dict(torch.load(ROOT/'output/training/contrast-edge-v1-contrast/best.pt',weights_only=True,map_location='cuda'));models={}
    for trial in ['risk']:
        model=SpatialEvidenceNet().cuda().eval();model.load_state_dict(torch.load(trial_folder(trial)/'best.pt',weights_only=True,map_location='cuda'));models[trial]=model
    report={'protocol':protocol,'cohorts':{}}
    for name,n,start in (('primary',256,1310000),('uniform',128,1320000),('hard',128,1330000),('rotation',128,1340000)):
        raw=[];gold=[];aligned=[];details=[];seeds=[];seed=start
        while len(gold)<n:
            _,g=compile_page(seed);signature=hashlib.sha256(g.tobytes()).hexdigest()
            if signature in seen:seed+=1;continue
            seen.add(signature);r=np.random.default_rng(seed+85001);xy=36+36*np.arange(6);j=0 if name=='uniform' else 4;xs=xy+r.integers(-j,j+1,6);ys=xy+r.integers(-j,j+1,6)
            sx,sy=r.uniform(.55,.85,2);tx=128*(1-sx)+r.uniform(-10,10);ty=128*(1-sy)+r.uniform(-10,10)
            im=cv2.warpAffine(render_irregular(g,int(r.integers(1,8)),xs,ys),np.array([[sx,0,tx],[0,sy,ty]],np.float32),(256,256),borderValue=255)
            if name=='rotation':im=cv2.warpAffine(im,cv2.getRotationMatrix2D((128,128),float(r.choice([-1,1])*r.uniform(8,15)),1),(256,256),borderValue=255)
            im=cv2.GaussianBlur(im,(7,7),float(r.uniform(1.2,1.6) if name=='hard' else r.uniform(.9,1.2)))
            im=np.clip(im.astype(float)*r.uniform(.5,1)+r.normal(0,5,im.shape),0,255).astype(np.uint8)
            a,d=register_ordered(im);raw.append(im[None]);aligned.append(a[None]);details.append(d);gold.append(g);seeds.append(seed);seed+=1
        raw=np.stack(raw);aligned=np.stack(aligned);gold=np.stack(gold);ready=np.array([d['ready'] for d in details]);rows=[]
        with torch.no_grad():
            for i in range(0,n,16):
                x=torch.tensor(1-aligned[i:i+16].astype(np.float32)/255,device='cuda')
                with torch.autocast('cuda',dtype=torch.float16):rows.append(base.generate(x).cpu().numpy())
        baseline=np.concatenate(rows);predictions={'baseline':baseline};patches=[];mapping=[]
        for i,d in enumerate(details):
            p,slots=crossing_patches(raw[i,0],aligned[i,0],d,'raw')
            if len(slots):patches.append(p);mapping.extend((i,s) for s in slots)
        patches=np.concatenate(patches) if patches else np.empty((0,1,33,33),np.float32)
        for trial,model in models.items():
            pred=baseline.copy();outputs=[]
            with torch.no_grad():
                for j in range(0,len(patches),128):outputs.extend(model(torch.tensor(patches[j:j+128],device='cuda')).argmax(-1).cpu().numpy().tolist())
            for (i,s),v in zip(mapping,outputs):
                if pred[i,s]!=7:pred[i,s]=v+8
            predictions[trial]=pred
        corrected=baseline.copy()
        for i,d in enumerate(details):
            scores,slots=page_scores(raw[i,0],aligned[i,0],d);ds=decisions(scores,**thresholds)
            for slot,value in zip(slots,ds):
                if value!=7 and corrected[i,slot]!=7:corrected[i,slot]=value
        predictions['residual']=corrected
        results={};before=(baseline==gold).all(1)
        for trial,p in predictions.items():
            after=(p==gold).all(1);results[trial]={'ready_pages':int(ready.sum()),'coverage':float(ready.mean()),'all_model':metrics(p,gold),
                                                'ready_model':metrics(p[ready],gold[ready]) if ready.any() else None,
                                                'fixed_program_pages':int((~before&after).sum()),'broken_program_pages':int((before&~after).sum())}
        results['geometry_identity_errors']=[i for i,(d,g) in enumerate(zip(details,gold)) if d['ready'] and (len(d['observed_rows'])!=g[0] or len(d['observed_columns'])!=g[1])]
        report['cohorts'][name]=results;np.savez_compressed(out/f'{name}-inputs.npz',observed_images=raw,program=gold,seeds=seeds);np.savez_compressed(out/f'{name}-predictions.npz',**predictions)
        (out/f'{name}-geometry.json').write_text(json.dumps(details,indent=2),encoding='utf-8');Image.fromarray(raw[0,0]).save(out/f'{name}-first.png')
        print(json.dumps({'cohort':name,**results}),flush=True)
    primary=report['cohorts']['primary'];d=primary['residual']
    report['gate_passed']=d['coverage']>=.9 and d['all_model']['net_pair']['micro_F1']>=.98 and d['all_model']['exact_net_partition']>=.95 and d['all_model']['exact_net_partition']-primary['baseline']['all_model']['exact_net_partition']>=.005 and d['all_model']['net_pair']['FP']<=primary['baseline']['all_model']['net_pair']['FP'] and d['broken_program_pages']==0 and report['cohorts']['rotation']['residual']['coverage']<=.05
    (out/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')


if __name__=='__main__':main()
