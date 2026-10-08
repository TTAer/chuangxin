"""Fixed Full Oracle feasibility runs; later ladder levels require gate success."""
import json,sys,time,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,torch
from experiments.netset_query import NetSetQueryDecoder
from experiments.netset_oracle_loss import oracle_loss
from pcbparse.core import assignment,f1


def net_metrics(outputs,pin_labels,wire_labels):
    p=outputs['pin_logits'][0].detach().cpu().numpy();w=outputs['wire_logits'][0].detach().cpu().numpy()
    pred=p.argmax(1);wirepred=w.argmax(1);q=p.shape[1]-1
    gold=np.asarray(pin_labels);groups=sorted(set(gold.tolist()));slots=sorted(set(pred.tolist())-{q})
    sets=[set(np.flatnonzero(pred==s).tolist()) for s in slots];truth=[set(np.flatnonzero(gold==g).tolist()) for g in groups]
    weights=[[f1(len(a&b),len(a),len(b)) for b in truth] for a in sets]
    matches=assignment(weights);mapping={slots[i]:groups[j] for i,j in matches if sets[i]&truth[j]}
    hits=sum(len(sets[i]&truth[j]) for i,j in matches);exact=sum(sets[i]==truth[j] for i,j in matches)
    a=pred[:,None]==pred[None,:];b=gold[:,None]==gold[None,:];upper=np.triu(np.ones_like(a,dtype=bool),1);a&=(pred[:,None]!=q)&(pred[None,:]!=q)
    tp=int((a&b&upper).sum());fp=int((a&~b&upper).sum());fn=int((~a&b&upper).sum())
    shifted=p-p.max(1,keepdims=True);prob=np.exp(shifted);prob/=prob.sum(1,keepdims=True);mass=prob[:,:q]
    norms=np.sqrt((mass*mass).sum(0));cos=mass.T@mass/np.maximum(norms[:,None]*norms[None,:],1e-12)
    soft=[tuple(np.flatnonzero(mass[:,i]>=.1).tolist()) for i in range(q)];active=[s for s in soft if s]
    return {'hungarian_pin_micro_F1':f1(hits,int((pred!=q).sum()),len(gold)),
            'exact_net_set_F1':f1(exact,len(sets),len(truth)), 'pairwise_connected_F1':2*tp/max(1,2*tp+fp+fn),
            'wire_ownership_micro_F1':sum(mapping.get(int(s),-999)==int(t) for s,t in zip(wirepred,wire_labels))/max(1,len(wire_labels)),
            'active_pin_queries':len(slots),'true_nets':len(groups),'existence_queries_at_0p5':int((outputs['net_existence_logits'][0].detach().cpu().numpy()>=0).sum()),
            'duplicate_soft_sets_at_0p1':len(active)-len(set(active)),'soft_active_queries_at_0p1':len(active),
            'query_query_mean_cosine':float((cos.sum()-np.trace(cos))/(q*(q-1))),
            'hard_duplicate_sets':0,'hard_duplicate_note':'exclusive categorical pin assignment makes identical nonempty hard sets impossible; soft overlap diagnoses redundancy'}


def main():
    torch.set_num_threads(4);torch.backends.cudnn.enabled=False;torch.backends.mha.set_fastpath_enabled(False)
    torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.cuda.set_per_process_memory_fraction(.04)
    bank=ROOT/'data/processed/netset-oracle';manifest=json.loads((bank/'manifest.json').read_text(encoding='utf-8'))
    out=ROOT/'output/training/netset-oracle';out.mkdir(parents=True,exist_ok=True)
    if (out/'protocol.json').exists():raise ValueError('Preserve completed/partial oracle run')
    protocol={'architecture':'frozen NetSetQueryDecoder v1','queries':128,'width':64,'seed':20261007,'epochs':24,'device':'cuda','batch':'one complete variable-length page',
        'variants':['full-pin-oracle','full-pin-wire-oracle'],'data_sha256':hashlib.sha256((bank/'manifest.json').read_bytes()).hexdigest(),
        'selection':'highest development_holdout Hungarian pin micro F1; not independent accuracy',
        'first_gate_preregistered':{'mean_holdout_pin_micro_F1_min':.8,'mean_holdout_pairwise_F1_min':.7,'pin_micro_gain_over_both_trivial_baselines_min':.05},
        'if_gate_fails':'stop coverage training and real-observation training, keep as unvalidated feasibility branch',
        'later_mask_coverages':[1,.75,.5,.3,.2,.1],'runtime_changed':False,'GT10_used':False,'Freeze_A_used':False,
        'wire_metric':'ownership classification within pin-set matched nets, NOT official endpoint LineF1','soft_duplicates_threshold':.1}
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    data=[]
    for row in manifest['rows']:
        with np.load(bank/row['case']/'oracle.npz') as z:d={k:z[k] for k in z.files}
        d['row']=row;data.append(d)
    train=[d for d in data if d['row']['split']=='train'];val=[d for d in data if d['row']['split']=='development_holdout']
    def tensors(d):return (torch.tensor(d['pins'],device='cuda')[None],torch.tensor(d['wires'],device='cuda')[None],torch.tensor(d['pin_labels'],device='cuda'),torch.tensor(d['wire_labels'],device='cuda'))
    baselines=[]
    for d in val:
        n=len(d['pin_labels']);groups=set(d['pin_labels'].tolist());largest=max(int((d['pin_labels']==g).sum()) for g in groups)
        baselines.append({'case':d['row']['case'],'singleton_micro_F1':len(groups)/n,'all_one_micro_F1':largest/n})
    reports=[];started=time.perf_counter()
    for joint in (False,True):
        variant='full-pin-wire-oracle' if joint else 'full-pin-oracle';run=out/variant;run.mkdir(exist_ok=True)
        torch.manual_seed(20261007);rng=np.random.default_rng(20261007);model=NetSetQueryDecoder().cuda();opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.001)
        best=-1;history=[]
        for epoch in range(24):
            model.train();losses=[]
            for i in rng.permutation(len(train)):
                p,w,pl,wl=tensors(train[i]);outputs=model(p,w);loss=oracle_loss(outputs,pl,wl,joint=joint,complete=True)
                if not torch.isfinite(loss):raise ValueError('Nonfinite loss')
                opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);opt.step();losses.append(float(loss.detach()))
            model.eval();rows=[]
            with torch.no_grad():
                for d in val:
                    p,w,pl,wl=tensors(d);row=net_metrics(model(p,w),d['pin_labels'],d['wire_labels']);rows.append({'case':d['row']['case'],**row})
            numeric=[k for k,v in rows[0].items() if isinstance(v,(float,int))]
            mean={k:float(np.mean([r[k] for r in rows])) for k in numeric};record={'epoch':epoch+1,'loss':float(np.mean(losses)),'mean':mean,'seconds':time.perf_counter()-started};history.append(record)
            if mean['hungarian_pin_micro_F1']>best:
                best=mean['hungarian_pin_micro_F1'];torch.save(model.state_dict(),run/'best.pt')
                (run/'best.json').write_text(json.dumps({'record':record,'pages':rows},indent=2),encoding='utf-8')
            (run/'history.json').write_text(json.dumps(history,indent=2),encoding='utf-8');print(json.dumps({'variant':variant,**record}),flush=True)
        result=json.loads((run/'best.json').read_text(encoding='utf-8'));mean=result['record']['mean'];dumb=max(np.mean([r['singleton_micro_F1'] for r in baselines]),np.mean([r['all_one_micro_F1'] for r in baselines]))
        passed=mean['hungarian_pin_micro_F1']>=.8 and mean['pairwise_connected_F1']>=.7 and mean['hungarian_pin_micro_F1']>dumb+.05
        reports.append({'variant':variant,'best':result,'first_gate_passed':bool(passed),'baseline_max_mean':float(dumb)})
    summary={'runs':reports,'baselines':baselines,'train_pages':len(train),'holdout_pages':len(val),'first_gate_passed':any(r['first_gate_passed'] for r in reports),
             'coverage_curve_status':'eligible_to_plan' if any(r['first_gate_passed'] for r in reports) else 'not_executed_first_gate_failed',
             'observation_oracle_status':'not_executed_pending_first_gate','weak_label_training_status':'not_executed',
             'seconds':time.perf_counter()-started,'CUDA_peak_allocated_bytes':torch.cuda.max_memory_allocated(),'CUDA_peak_reserved_bytes':torch.cuda.max_memory_reserved(),
             'runtime_changed':False,'independent_validation':False,'official_equivalence':False}
    (out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8');print(json.dumps({'first_gate_passed':summary['first_gate_passed'],'seconds':summary['seconds']}),flush=True)


if __name__=='__main__':main()
