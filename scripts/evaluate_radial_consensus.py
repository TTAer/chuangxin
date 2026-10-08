"""New final holdout for fixed two-view high-confidence association; no pressure tuning."""
import sys,json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,cv2,torch
from PIL import Image
from scripts.prepare_circuit_program import compile_page
from scripts.evaluate_ordered_holdout import render_irregular
from scripts.ordered_line_rectification_v2 import register_ordered
from experiments.radial_evidence_net import RadialEvidenceNet,crossing_patches
from experiments.contrast_edge_circuit_reader import ContrastOnlyReader
from scripts.train_circuit_program import metrics


def consensus_patch_decisions(raw,aligned,d,models,device):
    outputs=[];slots=[]
    if not d['ready']:return [],[],[]
    for mode in ('raw','canonical'):
        p,slots=crossing_patches(raw,aligned,d,mode)
        with torch.no_grad():outputs.append(models[mode](torch.tensor(p,device=device)).softmax(-1).cpu().numpy())
    labels=[v.argmax(-1) for v in outputs];confidence=[v.max(-1) for v in outputs]
    accepted=(labels[0]==labels[1])&(confidence[0]>=.99)&(confidence[1]>=.99)
    return slots,labels[0],accepted


def main():
    out=ROOT/'output/radial-consensus-holdout';out.mkdir(parents=True,exist_ok=True)
    if (out/'protocol.json').exists():raise ValueError('Preserve holdout')
    protocol={'scope':'new programs after direct replacement failed; no retuning of prior holdout',
              'rule':'both raw and canonical radial labels agree and both softmax>=.99, else preserve frozen base crossing',
              'threshold':'fixed .99 before new pages, not optimized; softmax not calibrated probability',
              'primary':'256 new native jitter+/-4 scale.55:.85 widths1:7 blur.9:1.2 noise5',
              'rotation':'128 new unsupported +/-8:15deg pages',
              'gate':'primary coverage>=.90 F1>=.98 exact-net>=.95; >=.01 net-exact gain over base; no exact-program regression pages; rotation acceptance<=.05',
              'models':'same dev-selected RadialEvidenceNet raw/canonical, frozen ContrastOnlyReader','new_training':False,
              'selection':'no threshold or weight selection from these pages','correlated_views':True}
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    seen=set()
    for folder in ('data/processed/circuit-program-v1','data/processed/circuit-program-v2','data/processed/factorized-circuit-v1','data/processed/separable-circuit-v1','data/processed/registered-circuit-v1','data/processed/contrast-edge-v1','data/processed/radial-evidence-v1','output/registration-v2-holdout','output/guard-v2-holdout','output/ordered-rectification-holdout'):
        for p in (ROOT/folder).glob('*.npz'):
            with np.load(p) as z:
                if 'program' in z:seen.update(hashlib.sha256(g.tobytes()).hexdigest() for g in z['program'])
    torch.set_num_threads(4);base=ContrastOnlyReader().cuda().eval();base.load_state_dict(torch.load(ROOT/'output/training/contrast-edge-v1-contrast/best.pt',weights_only=True,map_location='cuda'));models={}
    for mode in ('raw','canonical'):
        models[mode]=RadialEvidenceNet().cuda().eval();models[mode].load_state_dict(torch.load(ROOT/f'output/training/radial-evidence-v1-{mode}/best.pt',weights_only=True,map_location='cuda'))
    report={'protocol':protocol,'cohorts':{}}
    for name,n,start in (('primary',256,1010000),('rotation',128,1020000)):
        raw=[];gold=[];seeds=[];aligned=[];details=[];seed=start
        while len(gold)<n:
            _,g=compile_page(seed);sig=hashlib.sha256(g.tobytes()).hexdigest()
            if sig in seen:seed+=1;continue
            seen.add(sig);r=np.random.default_rng(seed+55001);base_xy=36+36*np.arange(6);xs=base_xy+r.integers(-4,5,6);ys=base_xy+r.integers(-4,5,6)
            sx,sy=r.uniform(.55,.85,2);tx=128*(1-sx)+r.uniform(-10,10);ty=128*(1-sy)+r.uniform(-10,10)
            im=cv2.warpAffine(render_irregular(g,int(r.integers(1,8)),xs,ys),np.array([[sx,0,tx],[0,sy,ty]],np.float32),(256,256),borderValue=255)
            if name=='rotation':im=cv2.warpAffine(im,cv2.getRotationMatrix2D((128,128),float(r.choice([-1,1])*r.uniform(8,15)),1),(256,256),borderValue=255)
            im=cv2.GaussianBlur(im,(5,5),float(r.uniform(.9,1.2)));im=np.clip(im.astype(float)*r.uniform(.5,1)+r.normal(0,5,im.shape),0,255).astype(np.uint8)
            a,d=register_ordered(im);raw.append(im[None]);gold.append(g);aligned.append(a[None]);details.append(d);seeds.append(seed);seed+=1
        raw=np.stack(raw);gold=np.stack(gold);aligned=np.stack(aligned);ready=np.array([d['ready'] for d in details]);rows=[]
        with torch.no_grad():
            for i in range(0,n,16):
                x=torch.tensor(1-aligned[i:i+16].astype(np.float32)/255,device='cuda')
                with torch.autocast('cuda',dtype=torch.float16):rows.append(base.generate(x).cpu().numpy())
        baseline=np.concatenate(rows);pred=baseline.copy();changes=[]
        for i,d in enumerate(details):
            slots,labels,accepted=consensus_patch_decisions(raw[i,0],aligned[i,0],d,models,'cuda')
            for s,v,ok in zip(slots,labels,accepted):
                if ok and pred[i,s]!=7 and pred[i,s]!=int(v)+8:
                    changes.append({'case_index':i,'slot':s,'from':int(pred[i,s]),'to':int(v)+8,'correct_before':bool(pred[i,s]==gold[i,s]),'correct_after':bool(int(v)+8==gold[i,s])});pred[i,s]=int(v)+8
        before=(baseline==gold).all(1);after=(pred==gold).all(1)
        results={m:{'ready_pages':int(ready.sum()),'coverage':float(ready.mean()),'all_model':metrics(p,gold),'ready_model':metrics(p[ready],gold[ready]) if ready.any() else None} for m,p in (('baseline',baseline),('consensus',pred))}
        results['correction_audit']={'changed_slots':len(changes),'changed_pages':len(set(d['case_index'] for d in changes)),
                                      'program_fixed_pages':int((~before&after).sum()),'program_broken_pages':int((before&~after).sum()),
                                      'changed_slot_correct_after':sum(d['correct_after'] for d in changes),'changes':changes}
        report['cohorts'][name]=results
        np.savez_compressed(out/f'{name}-inputs.npz',observed_images=raw,program=gold,seeds=seeds)
        np.savez_compressed(out/f'{name}-predictions.npz',baseline=baseline,consensus=pred)
        (out/f'{name}-geometry.json').write_text(json.dumps(details,indent=2),encoding='utf-8')
        Image.fromarray(raw[0,0]).save(out/f'{name}-first.png');print(json.dumps({'cohort':name,**results}),flush=True)
    p=report['cohorts']['primary'];d=p['consensus'];report['gate_passed']=d['coverage']>=.9 and d['all_model']['net_pair']['micro_F1']>=.98 and d['all_model']['exact_net_partition']>=.95 and d['all_model']['exact_net_partition']-p['baseline']['all_model']['exact_net_partition']>=.01 and p['correction_audit']['program_broken_pages']==0 and report['cohorts']['rotation']['consensus']['coverage']<=.05
    (out/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')


if __name__=='__main__':main()
