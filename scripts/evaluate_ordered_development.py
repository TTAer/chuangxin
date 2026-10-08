"""Development-only anomaly trace and fixed-model gate for bilateral rectification."""
import sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,torch
from scripts.ordered_line_rectification import register_ordered
from experiments.contrast_edge_circuit_reader import ContrastOnlyReader
from scripts.train_circuit_program import metrics


def main():
    out=ROOT/'output/ordered-rectification-development';out.mkdir(parents=True,exist_ok=True)
    if (out/'report.json').exists():raise ValueError('Preserve development')
    protocol={'scope':'old train/validation only; no old pressure pixels for tuning',
              'method':'bilateral profiles, ordered peaks, monotone axis warp; spacing not forced uniform',
              'model':'frozen ContrastOnlyReader','gate':'dev coverage>=.95 and exact net partition>=.95, no ready line count identity errors',
              'no_training':True,'anomaly_trace':'old train/dev12 identity cases; GT used only for report'}
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    model=ContrastOnlyReader().cuda().eval();torch.set_num_threads(4);model.load_state_dict(torch.load(ROOT/'output/training/contrast-edge-v1-contrast/best.pt',weights_only=True,map_location='cuda'))
    report={'protocol':protocol,'cohorts':{}}
    for name in ('train','validation'):
        with np.load(ROOT/f'data/processed/contrast-edge-v1/{name}.npz') as z:raw,gold=z['observed_images'],z['program']
        old=json.loads((ROOT/f'output/guard-v2-development/{name}-geometry.json').read_text());records=[register_ordered(im) for im in raw[:,0]]
        images=np.stack([a[None] for a,d in records]);details=[d for a,d in records];ready=np.array([d['ready'] for d in details]);rows=[]
        with torch.no_grad():
            for i in range(0,len(images),16):
                x=torch.tensor(1-images[i:i+16].astype(np.float32)/255,device='cuda')
                with torch.autocast('cuda',dtype=torch.float16):rows.append(model.generate(x).cpu().numpy())
        pred=np.concatenate(rows);new_errors=[];trace=[]
        for i,(previous,d,g) in enumerate(zip(old,details,gold)):
            old_bad=previous['ready'] and (len(previous['observed_rows'])!=g[0] or len(previous['observed_columns'])!=g[1])
            new_bad=d['ready'] and (len(d['observed_rows'])!=g[0] or len(d['observed_columns'])!=g[1])
            if new_bad:new_errors.append(i)
            if old_bad:trace.append({'case_index':i,'old_observed_counts':[len(previous['observed_rows']),len(previous['observed_columns'])],
                                     'gold_counts':g[:2].tolist(),'new_ready':d['ready'],'new_observed_counts':[len(d.get('observed_rows',[])),len(d.get('observed_columns',[]))],
                                     'new_identity_error':bool(new_bad),'new_program_exact':bool(np.array_equal(pred[i],g))})
        report['cohorts'][name]={'ready_pages':int(ready.sum()),'coverage':float(ready.mean()),'ready_line_count_error_cases':new_errors,
                               'all_model':metrics(pred,gold),'ready_model':metrics(pred[ready],gold[ready]) if ready.any() else None,
                               'prior_anomaly_trace':trace}
        (out/f'{name}-geometry.json').write_text(json.dumps(details,indent=2),encoding='utf-8');np.savez_compressed(out/f'{name}-predictions.npz',program=pred,ready=ready)
        print(json.dumps({'cohort':name,'ready_pages':int(ready.sum()),'identity_errors':len(new_errors),'exact_net':report['cohorts'][name]['all_model']['exact_net_partition']}),flush=True)
    d=report['cohorts']['validation'];report['gate_passed']=d['coverage']>=.95 and d['all_model']['exact_net_partition']>=.95 and not d['ready_line_count_error_cases']
    (out/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')


if __name__=='__main__':main()
