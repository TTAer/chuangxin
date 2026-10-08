"""Audit easy/hard pin association and abstention of both trained controls."""
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
import numpy as np
from experiments.pin_token_v1 import K

cache=ROOT/'data/processed/pin-token-v1'
data=np.load(cache/'tokens.npz'); records=json.loads((cache/'records.json').read_text(encoding='utf-8'))
indices=np.flatnonzero(data['split']==1); truth=data['association'][indices]; pins=truth<K; hard=pins&(truth!=0); easy=pins&(truth==0)
report={'training':json.loads((cache/'preparation.json').read_text(encoding='utf-8')),
    'nearest_endpoint_baseline':{'pin_accuracy':float((truth[pins]==0).mean()),'correct':int(easy.sum()),'pin_tokens':int(pins.sum())},
    'variants':{},'rc0_modified':False,'default_promoted':False,'end_to_end_pin_f1':None,'end_to_end_net_f1':None,
    'identity_recovery_rate':None,'freeze_a_scored':False,
    'decision':'STOP_V1_integration_pending_better_local_association_and_reviewed_labels',
    'reason':'trained affinity is below nearest endpoint control; six-role support and predicted-endpoint recovery not established'}
for variant in ('geometry-text','visual-joint'):
    path=ROOT/'output/training/pin-token-v1'/variant; pred=np.load(path/'validation.npz')
    assert np.array_equal(indices,pred['indices'])
    association=pred['association']; meta=json.loads((path/'best.json').read_text(encoding='utf-8'))
    report['variants'][variant]={'selected_checkpoint':meta,'checkpoint_sha256':hashlib.sha256((path/'best.pt').read_bytes()).hexdigest(),
        'hard_pin_tokens':int(hard.sum()),'hard_pin_correct':int((association[hard]==truth[hard]).sum()),
        'easy_pin_tokens':int(easy.sum()),'easy_pin_broken':int((association[easy]!=truth[easy]).sum()),
        'false_binding_of_null_tokens':int(((truth==K)&(association!=K)).sum()),
        'source_pages':sorted({records[i]['page'] for i in indices})}
out=ROOT/'output/ablation/pin-token-v1.json'; out.write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps({key:{k:v for k,v in value.items() if k not in ('selected_checkpoint','source_pages')} for key,value in report['variants'].items()},indent=2))
