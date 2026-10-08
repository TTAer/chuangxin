"""Targeted invariants of the executed Oracle loss; no optimizer steps."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from experiments.netset_oracle_loss import oracle_loss


def main():
    torch.manual_seed(7)
    outputs={'pin_logits':torch.randn(1,20,9,requires_grad=True),'wire_logits':torch.randn(1,25,9,requires_grad=True),'net_existence_logits':torch.randn(1,8,requires_grad=True)}
    pins=torch.arange(20)%3;wires=torch.arange(25)%3
    base=oracle_loss(outputs,pins,wires,joint=True)
    rename=torch.tensor([91,17,46]);renamed=oracle_loss(outputs,rename[pins],rename[wires],joint=True)
    perm=torch.randperm(8);columns=torch.cat((perm,torch.tensor([8])))
    shifted={k:(v[:,:,columns] if k!='net_existence_logits' else v[:,perm]) for k,v in outputs.items()}
    permuted=oracle_loss(shifted,pins,wires,joint=True)
    partial_pins=pins.clone();partial_wires=wires.clone();partial_pins[-8:]=-1;partial_wires[-7:]=-1
    partial=oracle_loss(outputs,partial_pins,partial_wires,joint=True,complete=False)
    altered={k:v.clone() for k,v in outputs.items()}
    altered['pin_logits'][:,-8:]=torch.randn(1,8,9)*100
    altered['wire_logits'][:,-7:]=torch.randn(1,7,9)*100
    masked=oracle_loss(altered,partial_pins,partial_wires,joint=True,complete=False)
    partial.backward()
    unknown_gradient=max(float(outputs['pin_logits'].grad[:,-8:].abs().max()),float(outputs['wire_logits'].grad[:,-7:].abs().max()))
    errors={'GT_label_rename_loss_delta':float((base-renamed).abs().detach()),'query_permutation_loss_delta':float((base-permuted).abs().detach()),
            'unknown_token_perturbation_loss_delta':float((partial-masked).abs().detach()),'unknown_token_gradient_max':unknown_gradient}
    assert all(v<1e-5 for v in errors.values()),errors
    try:oracle_loss(outputs,torch.arange(20),wires,joint=True)
    except ValueError:overflow=True
    else:overflow=False
    assert overflow
    result={'checks':errors,'query_overflow_rejected':overflow,'optimizer_steps':0,'scope':'invariants only; not a partial-supervision training result'}
    (ROOT/'output/training/netset-oracle/loss-checks.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result))


if __name__=='__main__':main()
