"""Promote the measured multisource snapshot while retaining the previous baseline."""
import hashlib
import json
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parents[1]
ablation=json.loads((ROOT/'output/ablation/summary.json').read_text(encoding='utf-8'))
scores={name:run['mean_metrics']['Score_Accuracy_SingleCase_proxy'] for name,run in ablation['runs'].items()}
if scores['benchmark-multisource-55']<scores['benchmark-text-55']:
    raise ValueError('candidate did not outperform existing model with the same postprocessing')
baseline=ROOT/'models/baselines/pcb-v2'; baseline.mkdir(parents=True,exist_ok=True)
current=ROOT/'models/detector'; candidate=ROOT/'models/candidates/multisource-v1'
for name in ('pcb-yolo11n.onnx','pcb-yolo11n.pt','pcb-yolo11n.json'):
    if not (baseline/name).exists(): shutil.copyfile(current/name,baseline/name)
    shutil.copyfile(candidate/name,current/name)
config={'version':1,'pin_model':'models/pins/pin-heatmap.onnx','local_pin_ocr':False,'text_constraints':True,'selected_run':'output/benchmark-multisource-55','comparison':'output/ablation/summary.json','official_equivalence':False,'scores':scores,'model_sha256':{str(p.relative_to(ROOT)).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest() for p in (current/'pcb-yolo11n.onnx',ROOT/'models/pins/pin-heatmap.onnx')}}
(ROOT/'models/pipeline.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
print(json.dumps(config,indent=2))
