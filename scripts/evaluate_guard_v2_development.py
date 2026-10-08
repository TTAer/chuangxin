"""Train/dev-only gate for new multicontrast geometry, with frozen contrast reader."""
import sys,json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,torch
from experiments.contrast_edge_circuit_reader import ContrastOnlyReader
from scripts.circuit_geometry_guard_v2 import register_guarded_v2
from scripts.train_circuit_program import metrics


def main():
    out=ROOT/'output/guard-v2-development';out.mkdir(parents=True,exist_ok=True)
    if (out/'report.json').exists():raise ValueError('Preserve development trial')
    protocol={'scope':'past train/validation only, no past pressure or shear pages',
              'method':'strong directions preserved; smoothed lower-contrast direction consensus; weak line component bounds only if registration fails',
              'model':'frozen ContrastOnlyReader best.pt','gate':'development coverage>=.98 and full exact net partition>=.98',
              'new_training':False,'GT_geometry_supplied':False,'no_threshold_search':True}
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    torch.set_num_threads(4);model=ContrastOnlyReader().cuda().eval();checkpoint=ROOT/'output/training/contrast-edge-v1-contrast/best.pt';model.load_state_dict(torch.load(checkpoint,weights_only=True,map_location='cuda'))
    report={'protocol':protocol,'checkpoint_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),'cohorts':{}}
    for name in ('train','validation'):
        with np.load(ROOT/f'data/processed/contrast-edge-v1/{name}.npz') as z:raw,gold=z['observed_images'],z['program']
        records=[register_guarded_v2(im) for im in raw[:,0]];images=np.stack([a[None] for a,d in records]);details=[d for a,d in records];ready=np.array([d['ready'] for d in details]);prediction=[]
        with torch.no_grad():
            for i in range(0,len(images),16):
                x=torch.tensor(1-images[i:i+16].astype(np.float32)/255,device='cuda')
                with torch.autocast('cuda',dtype=torch.float16):prediction.append(model.generate(x).cpu().numpy())
        p=np.concatenate(prediction);identity=sum(len(d['observed_rows'])!=g[0] or len(d['observed_columns'])!=g[1] for d,g in zip(details,gold) if d['ready'])
        report['cohorts'][name]={'coverage':float(ready.mean()),'ready_pages':int(ready.sum()),'ready_line_count_identity_errors':int(identity),
                               'all_model':metrics(p,gold),'ready_model':metrics(p[ready],gold[ready]) if ready.any() else None}
        (out/f'{name}-geometry.json').write_text(json.dumps(details,indent=2),encoding='utf-8');np.savez_compressed(out/f'{name}-predictions.npz',program=p,ready=ready)
        print(json.dumps({'cohort':name,**report['cohorts'][name]}),flush=True)
    d=report['cohorts']['validation'];report['gate_passed']=d['coverage']>=.98 and d['all_model']['exact_net_partition']>=.98
    (out/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')


if __name__=='__main__':main()
