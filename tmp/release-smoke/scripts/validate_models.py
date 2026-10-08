"""Detector-only validation on clean PCB pages and separate external-domain pages."""
import json
import os
from pathlib import Path
import sys
import yaml

ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('YOLO_CONFIG_DIR',str(ROOT/'tmp/yolo'))
os.environ['YOLO_AUTOINSTALL']='False'
from ultralytics import YOLO

out=ROOT/'output/validation'; out.mkdir(parents=True,exist_ok=True)
experiments=[('pcb-baseline-clean55',ROOT/'models/detector/pcb-yolo11n.pt',ROOT/'data/processed/yolo/dataset.yaml',True),
             ('pcb-multisource-clean55',ROOT/'models/candidates/multisource-v1/pcb-yolo11n.pt',ROOT/'data/processed/multisource/dataset.yaml',True),
             ('aitee-multisource-166',ROOT/'models/candidates/multisource-v1/pcb-yolo11n.pt',ROOT/'data/processed/multisource/external-validation.yaml',False)]
results={}
for name,weights,data,clean in experiments:
    config=yaml.safe_load(data.read_text(encoding='utf-8'))
    if clean:
        base=Path(config['path']); val=base/config['val']
        files=val.read_text(encoding='utf-8').splitlines() if val.is_file() else [p.as_posix() for p in sorted(val.glob('*')) if p.suffix.lower() in ('.png','.jpg','.jpeg')]
        files=[p for p in files if Path(p).stem not in ('0237','pcbnet-0237')]
        listing=out/(name+'.txt'); listing.write_text('\n'.join(files)+'\n',encoding='utf-8'); config['val']=listing.resolve().as_posix()
        config_path=out/(name+'.yaml'); config_path.write_text(yaml.safe_dump(config,sort_keys=False),encoding='utf-8')
    else: config_path=data
    model=YOLO(str(weights)); m=model.val(data=str(config_path),imgsz=960,batch=8,device=0,workers=0,project=str(out),name=name,exist_ok=True,plots=False,verbose=False)
    results[name]={'metrics':{k:float(v) for k,v in m.results_dict.items()},'classes':model.names,'per_class_map50':m.box.ap50.tolist(),'per_class_map':m.box.maps.tolist(),'evaluated_class_ids':m.box.ap_class_index.tolist(),'weights':str(weights.relative_to(ROOT)),'official_equivalence':False}
    (out/'summary.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
print(json.dumps({k:v['metrics'] for k,v in results.items()},indent=2))
