"""Exhibit information discarded by ring means without asserting an electrical-label collision."""
import sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from experiments.radial_evidence_net import RadialEvidenceNet

out=ROOT/'output/evidence-compression-check.json'
if out.exists():raise ValueError('Preserve check')
model=RadialEvidenceNet();a=torch.zeros(1,1,33,33);b=a.clone();a[0,0,16,23]=1;b[0,0,17,23]=1
fa=a.flatten(1)@model.masks.T;fb=b.flatten(1)@model.masks.T
delta=float((fa-fb).abs().max());core=float(((a-b).flatten(1)@model.core_mask).abs().max());assert delta==0 and core==0
report={'different_pixel_count':int((a!=b).sum()),'radial_feature_max_delta':delta,'center_feature_delta':core,
        'scope':'two arbitrary pixel patches share compressed evidence; not two proven valid schematics with different electrical labels',
        'causal_claim':'non-injectivity is demonstrated; it does not prove compression caused prior holdout errors'}
out.write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report))
