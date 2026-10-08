"""Verify masked-token isolation and finite backward pass on the trained encoder."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from experiments.masked_stroke_transformer import MaskedStrokeTransformer

torch.set_num_threads(2);torch.backends.mha.set_fastpath_enabled(False)
model=MaskedStrokeTransformer().eval()
model.load_state_dict(torch.load(ROOT/'output/training/masked-stroke-transformer/pretrained.pt',weights_only=True,map_location='cpu'))
torch.manual_seed(20261006);x=torch.rand(2,3,48,48);changed=x.clone();changed[:,:,:8,:8]=1-x[:,:,:8,:8]
mask=torch.zeros(2,36,dtype=torch.bool);mask[:,0]=True
with torch.no_grad():
    delta=float((model.reconstruct(x,mask)-model.reconstruct(changed,mask)).abs().max())
assert delta==0,'Masked patch pixels leaked into encoder'
model.train();loss=model(x).square().mean();loss.backward()
assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
r={'masked_pixel_invariance_max_error':delta,'finite_backward':True,'parameters':sum(p.numel() for p in model.parameters())}
(ROOT/'output/training/masked-stroke-transformer/architecture-check.json').write_text(json.dumps(r,indent=2),encoding='utf-8')
print(json.dumps(r))
