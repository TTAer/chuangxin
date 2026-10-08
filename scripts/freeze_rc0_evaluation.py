"""Freeze evaluation protocol and runtime hashes before any new stress scores."""
import hashlib
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from pcbparse.__main__ import load,save
source=load(ROOT/'output/ablation/stable-image-sources.json')
for name,sha in source['sha256'].items():
    assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==sha,name
manifest={'version':'RC0-stable-catalog','image_id':source['image_id'],'runtime_sha256':source['sha256'],
 'baseline':{'designator_mode':'legacy','identity_transaction_mode':'none','ic_identity_mode':'none','side_pin_model':'none'},
 'candidate':{'designator_mode':'stable','identity_transaction_mode':'catalog','ic_identity_mode':'retrieval','side_pin_model':'models/candidates/side-strip-local/side-strip.onnx','side_name_anchors':True},
 'development_sets':['PCB55','10GT'],'default_promoted':False,
 'freeze_a':{'status':'not_ready','required_pages':25,'domain':'PCB/EDA','first_phase':['component_bbox','visible_designator','ic_name'],'required_independence':'design-level exclusion from all training, validation and prior rule selection','no_missing_gt_imputation':True},
 'freeze_b':{'source_manifest':'data/splits/amsnet-frozen.json','status':'frozen_protocol','selection':'first stored original per frozen normalized-SPICE group','groups':25,'scores':['class-aware bbox F1 at IoU0.3','class-aware bbox F1 at IoU0.5','parse failures','output validity'],'limitations':['normalized SPICE hash is not graph-isomorphism proof','AMSNet is cross-domain, not PCB blind validation','no pin/net pixel GT adapter; do not report PinF1/NetF1']},
 'promotion_gates':{'same_domain_independent_test':True,'default_and_candidate_50_case_pass':True,'official_evaluator_status_explicit':True},
 'no_tuning_after_first_freeze_b_score':True,'official_evaluator_verified':False}
out=ROOT/'data/splits/rc0-evaluation.json'
if out.exists(): assert load(out)==manifest,'refuse to overwrite frozen protocol'
else: save(out,manifest)
print('RC0 runtime and evaluation protocol frozen',flush=True)
