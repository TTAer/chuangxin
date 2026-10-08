"""Development-only registration comparison; no old pressure access or model changes."""
import sys,json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,torch
from experiments.factorized_circuit_reader import FactorizedCircuitReader
from scripts.circuit_registration_v2 import register_image_v2
from scripts.train_circuit_program import metrics


def main():
    out=ROOT/'output/registration-v2-development';out.mkdir(parents=True,exist_ok=True)
    if (out/'report.json').exists():raise ValueError('Preserve development report')
    protocol={'scope':'old train/validation only; no old pressure reading',
              'algorithm':'two analog overhang bands, weighted centroids, least-squares lattice, consistency checks',
              'weights':'frozen registered-circuit-v1 best.pt',
              'gate':'coverage>=.95, line count identity errors=0, full development exact net partition>=.95',
              'no_threshold_search':True,'GT_transform_usage':'posthoc coordinate diagnostic only',
              'new_training':False,'RC0_changed':False}
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    model=FactorizedCircuitReader().cuda().eval();torch.set_num_threads(4)
    checkpoint=ROOT/'output/training/registered-circuit-v1/best.pt';model.load_state_dict(torch.load(checkpoint,weights_only=True,map_location='cuda'))
    report={'protocol':protocol,'checkpoint_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),'cohorts':{}}
    for name in ('train','validation'):
        with np.load(ROOT/f'data/processed/registered-circuit-v1/{name}.npz') as z:raw,gold,truth=z['observed_images'],z['program'],z['transforms']
        details=[];aligned=[]
        for im in raw[:,0]:a,d=register_image_v2(im);aligned.append(a[None]);details.append(d)
        aligned=np.stack(aligned);ready=np.array([d['ready'] for d in details]);prediction=[]
        with torch.no_grad():
            for i in range(0,len(aligned),16):
                x=torch.tensor(1-aligned[i:i+16].astype(np.float32)/255,device='cuda')
                with torch.autocast('cuda',dtype=torch.float16):prediction.append(model.generate(x).cpu().numpy())
        pred=np.concatenate(prediction);errors=0;coordinate=[]
        for i,d in enumerate(details):
            if not d['ready']:continue
            errors+=int(len(d['observed_rows'])!=gold[i,0] or len(d['observed_columns'])!=gold[i,1])
            for axis,scale,translation,n in ((0,'scale_x','translation_x',gold[i,1]),(1,'scale_y','translation_y',gold[i,0])):
                original=36+36*np.arange(int(n));observed=original*truth[i,axis,axis]+truth[i,axis,2]
                coordinate.extend(np.abs((observed-d[translation])/d[scale]-original))
        m=metrics(pred,gold)
        report['cohorts'][name]={'coverage':float(ready.mean()),'ready_pages':int(ready.sum()),'line_count_identity_errors':errors,
                               'mean_abs_canonical_coordinate_error':float(np.mean(coordinate)),'max_abs_canonical_coordinate_error':float(np.max(coordinate)),
                               'all_model':m,'ready_model':metrics(pred[ready],gold[ready]) if ready.any() else None}
        (out/f'{name}-registration.json').write_text(json.dumps(details,indent=2),encoding='utf-8')
        np.savez_compressed(out/f'{name}-predictions.npz',program=pred,ready=ready)
    d=report['cohorts']['validation'];report['gate_passed']=d['coverage']>=.95 and d['line_count_identity_errors']==0 and d['all_model']['exact_net_partition']>=.95
    (out/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({'gate_passed':report['gate_passed'],'cohorts':report['cohorts']}))


if __name__=='__main__':main()
