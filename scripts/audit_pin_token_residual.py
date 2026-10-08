"""Include predicted-role rejection in the audit, without retuning a model."""
import json
from pathlib import Path
import re
import sys
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
import numpy as np
from experiments.pin_token_residual import preservation_metrics
cache=ROOT/'data/processed/pin-token-residual'; rows=json.loads((cache/'records.json').read_text(encoding='utf-8'))
report=json.loads((ROOT/'output/ablation/pin-token-residual.json').read_text(encoding='utf-8'))
for name,result in report['variants'].items():
    with np.load(ROOT/'output/training/pin-token-residual'/(name+'-validation.npz')) as stored: data={key:stored[key] for key in stored.files}
    truth=data['truth']; prediction=data['prediction']; joint=np.where(data['role']==2,8,prediction)
    result['predicted_role_plus_residual']=preservation_metrics(joint,truth)
    result['predicted_role_plus_residual']['non_pin_false_bindings']=int(((truth==8)&(joint!=8)).sum())
    result['predicted_role_plus_residual']['non_pin_tokens']=int((truth==8).sum())
    # Intended residual policy: rejection of a known anchor means keep, not delete.
    fallback=np.where(data['role']==2,0,prediction)
    result['role_abstention_keeps_existing_anchor']=preservation_metrics(fallback,truth)
    result['role_abstention_keeps_existing_anchor']['scope']='existing valid pin-token anchors only; no new binding authorization for non-pin/unknown tokens'
    result['changed_relations']=[]
    for i,actual_index in enumerate(data['indices']):
        if prediction[i]==0 or truth[i]==8: continue
        row=rows[actual_index]; risks=[]
        if row['role']=='pin_number' and not re.fullmatch(r'[A-Z]?[0-9]+',row['token']['text'],re.I): risks.append('number_format_requires_review')
        result['changed_relations'].append({'page':row['page'],'component':row['component'],'text':row['token']['text'],'role':row['role'],'target_rank':int(truth[i])+1,'selected_rank':int(prediction[i])+1,'correct':bool(prediction[i]==truth[i]),'label_risks':risks})
report['limitations'].append('association-only EPR does not include role-head rejection; combined predicted-role metrics separately reported')
path=ROOT/'output/ablation/pin-token-residual-audit.json'; path.write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps({k:v['predicted_role_plus_residual'] for k,v in report['variants'].items()},indent=2))
