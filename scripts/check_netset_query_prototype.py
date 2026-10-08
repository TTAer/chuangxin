"""Real-page forward/backward and invariance checks; no fit or quality score."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,torch
from experiments.netset_query import NetSetQueryDecoder,partial_net_loss


def main():
    torch.set_num_threads(4);torch.manual_seed(20261006);torch.backends.mha.set_fastpath_enabled(False)
    out=ROOT/'output/netset-query-prototype';out.mkdir(parents=True,exist_ok=True)
    bank=ROOT/'data/processed/netset-query-prototype'
    audit=json.loads((bank/'audit.json').read_text(encoding='utf-8'))
    row=max(audit['rows'],key=lambda r:r['weak_compatible_pins']);page=bank/row['domain']/row['case']
    with np.load(page/'inputs.npz') as z:p=torch.tensor(z['pins'])[None];w=torch.tensor(z['wires'])[None]
    with np.load(page/'weak-targets.npz') as z:labels=torch.tensor(z['labels'])[None]
    model=NetSetQueryDecoder().eval()
    with torch.no_grad():
        result=model(p,w);perm=torch.randperm(p.shape[1]);wireperm=torch.randperm(w.shape[1])
        permuted=model(p[:,perm],w[:,wireperm]);error=float((permuted['pin_logits']-result['pin_logits'][:,perm]).abs().max())
        wire_error=float((permuted['wire_logits']-result['wire_logits'][:,wireperm]).abs().max())
    assert error<1e-4,'Input order changed predicted memberships'
    assert wire_error<1e-4,'Input order changed predicted wire ownership'
    loss=partial_net_loss(result,labels)
    renamed=labels.clone();renamed[labels>=0]=97-labels[labels>=0]
    label_error=float((partial_net_loss(result,renamed)-loss).abs())
    assert label_error<1e-5,'Arbitrary GT net IDs changed set loss'
    changed={k:v.clone() for k,v in result.items()};changed['pin_logits'][labels<0]+=torch.randn_like(changed['pin_logits'][labels<0])*100
    unknown_error=float((partial_net_loss(changed,labels)-loss).abs())
    assert unknown_error<1e-6,'Unknown labels entered supervised loss'
    model.train();live=model(p,w);live_loss=partial_net_loss(live,labels);live_loss.backward()
    assert all(torch.isfinite(x.grad).all() for x in model.parameters() if x.grad is not None)
    empty=partial_net_loss(live,torch.full_like(labels,-1));assert float(empty.detach())==0
    r={'architecture':'NetSetQueryDecoder','parameters':sum(p.numel() for p in model.parameters()),'net_queries':model.queries,
       'real_development_input':{'domain':row['domain'],'case':row['case'],'pins':p.shape[1],'wires':w.shape[1],'weak_compatible_pins':row['weak_compatible_pins']},
       'output_shapes':{k:list(v.shape) for k,v in result.items()},'input_permutation_max_error':error,'wire_permutation_max_error':wire_error,'target_net_renaming_loss_error':label_error,
       'unknown_label_loss_error':unknown_error,'all_unknown_loss':0,'finite_backward':True,'training_run':False,'optimizer_steps':0,'quality_score':None,
       'limitations':'abstain and net-existence probabilities uncalibrated; no supervised wire loss; no padding mask/visible text encoder yet','runtime_changed':False}
    (out/'architecture-check.json').write_text(json.dumps(r,indent=2),encoding='utf-8');print(json.dumps(r))


if __name__=='__main__':main()
