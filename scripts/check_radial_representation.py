"""Check advertised quarter-turn invariance and rejected-image extraction without GT."""
import sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch,numpy as np
from experiments.radial_evidence_net import RadialEvidenceNet,crossing_patches

torch.manual_seed(20261021);model=RadialEvidenceNet().eval();x=torch.rand(8,1,33,33)
with torch.no_grad():
    original=model(x);delta=max(float((original-model(torch.rot90(x,k,(-2,-1)))).abs().max()) for k in (1,2,3))
assert delta<1e-6,delta
p,s=crossing_patches(np.full((256,256),255,np.uint8),np.full((256,256),255,np.uint8),{'ready':False},'raw')
assert len(p)==0 and not s
out=ROOT/'output/radial-evidence-representation-check.json'
if out.exists():raise ValueError('Preserve check')
report={'quarter_turn_invariance_max_logit_delta':delta,'rejected_image_yields_no_patches':True,
        'parameters':sum(p.numel() for p in model.parameters()),'scope':'random inputs and initialized model; not accuracy evidence'}
out.write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report))
