"""Validate weighted microbatch gradients including a short last microbatch."""
import sys,json,copy
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from experiments.spatial_evidence_net import SpatialEvidenceNet

dest=ROOT/'output/evidence-accumulation-check.json'
if dest.exists():raise ValueError('Preserve check')
torch.manual_seed(20261081);torch.set_num_threads(4);full=SpatialEvidenceNet();micro=copy.deepcopy(full)
x=torch.rand(10,1,33,33);g=torch.arange(10)%2
torch.nn.functional.cross_entropy(full(x),g).backward()
for j in range(0,10,4):
    (torch.nn.functional.cross_entropy(micro(x[j:j+4]),g[j:j+4])*len(g[j:j+4])/len(g)).backward()
delta=max(float((a.grad-b.grad).abs().max()) for a,b in zip(full.parameters(),micro.parameters()));assert delta<2e-6
result={'max_gradient_difference':delta,'short_last_microbatch_tested':True,'same_logical_batch_weighting':True,'scope':'CPU initialized model property check, not bitwise-identical GPU training guarantee'}
dest.write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result))
