"""One final paired holdout evaluation, preserve neural prefix and interval decisions."""
import sys,json,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,torch
from experiments.radial_evidence_net import RadialEvidenceNet,crossing_patches
from experiments.contrast_edge_circuit_reader import ContrastOnlyReader
from scripts.train_circuit_program import metrics


def main():
    bank=ROOT/'data/processed/radial-evidence-v1';out=ROOT/'output/radial-evidence-evaluation';out.mkdir(parents=True,exist_ok=True)
    if (out/'protocol.json').exists():raise ValueError('Preserve evaluation')
    protocol=json.loads((bank/'protocol.json').read_text())
    for mode in ('raw','canonical'):assert json.loads((ROOT/f'output/training/radial-evidence-v1-{mode}/summary.json').read_text())['full_training_executed']
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    torch.set_num_threads(4);base=ContrastOnlyReader().cuda().eval();base.load_state_dict(torch.load(ROOT/'output/training/contrast-edge-v1-contrast/best.pt',map_location='cuda',weights_only=True))
    models={}
    for mode in ('raw','canonical'):
        models[mode]=RadialEvidenceNet().cuda().eval();models[mode].load_state_dict(torch.load(ROOT/f'output/training/radial-evidence-v1-{mode}/best.pt',map_location='cuda',weights_only=True))
    report={'protocol':protocol,'cohorts':{}};start=time.perf_counter()
    for name in ('validation','stress','rotation'):
        with np.load(bank/f'{name}.npz') as z:raw,aligned,gold=z['observed_images'],z['images'],z['program']
        details=json.loads((bank/f'{name}-geometry.json').read_text());ready=np.array([d['ready'] for d in details]);rows=[]
        with torch.no_grad():
            for i in range(0,len(gold),16):
                x=torch.tensor(1-aligned[i:i+16].astype(np.float32)/255,device='cuda')
                with torch.autocast('cuda',dtype=torch.float16):rows.append(base.generate(x).cpu().numpy())
        baseline=np.concatenate(rows);predictions={'baseline':baseline}
        for mode,model in models.items():
            pred=baseline.copy();patch_list=[];mapping=[]
            for i,d in enumerate(details):
                patches,slots=crossing_patches(raw[i,0],aligned[i,0],d,mode)
                if len(slots):patch_list.append(patches);mapping.extend((i,s) for s in slots)
            if patch_list:
                patches=np.concatenate(patch_list);outputs=[]
                with torch.no_grad():
                    for j in range(0,len(patches),512):outputs.extend(model(torch.tensor(patches[j:j+512],device='cuda')).argmax(-1).cpu().numpy().tolist())
                for (i,s),v in zip(mapping,outputs):
                    if pred[i,s]!=7:pred[i,s]=v+8
            predictions[mode]=pred
        results={}
        for mode,p in predictions.items():
            exact=(p==gold).all(1)
            results[mode]={'ready_pages':int(ready.sum()),'coverage':float(ready.mean()),'all_model':metrics(p,gold),
                           'ready_model':metrics(p[ready],gold[ready]) if ready.any() else None,
                           'program_changed_pages':int(np.any(p!=baseline,axis=1).sum()),
                           'program_fixed_pages':int((exact&~(baseline==gold).all(1)).sum()),'program_broken_pages':int((~exact&(baseline==gold).all(1)).sum())}
        results['geometry_diagnostic']={'ready_identity_error_cases':[i for i,(d,g) in enumerate(zip(details,gold)) if d['ready'] and (len(d['observed_rows'])!=g[0] or len(d['observed_columns'])!=g[1])],
                                        'scope':'GT only for retrospective count identity diagnostic; never fed to models'}
        report['cohorts'][name]=results;np.savez_compressed(out/f'{name}-predictions.npz',**predictions)
        print(json.dumps({'cohort':name,**results}),flush=True)
    stress=report['cohorts']['stress'];unsupported=report['cohorts']['rotation']['raw']['coverage']
    for mode in ('raw','canonical'):
        d=stress[mode];d['gate_passed']=d['coverage']>=.9 and d['all_model']['net_pair']['micro_F1']>=.98 and d['all_model']['exact_net_partition']>=.95 and d['all_model']['exact_net_partition']-stress['baseline']['all_model']['exact_net_partition']>=.01 and unsupported<=.05
    report['seconds']=time.perf_counter()-start
    (out/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')


if __name__=='__main__':main()
