import json
from pathlib import Path
import shutil
import sys
import os
import argparse

ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('YOLO_CONFIG_DIR',str(ROOT/'tmp/yolo'))
os.environ['YOLO_AUTOINSTALL']='False'


if __name__=='__main__':
    from ultralytics import YOLO
    parser=argparse.ArgumentParser(); parser.add_argument('--weights',default='output/training/pcb-yolo11n/weights/best.pt'); parser.add_argument('--output',default='models/detector'); args=parser.parse_args()
    p=ROOT/args.weights
    if not p.exists(): raise SystemExit('Training best.pt is missing')
    model=YOLO(str(p)); exported=model.export(format='onnx',imgsz=960,opset=17,simplify=False,dynamic=False,device='cpu')
    out=ROOT/args.output; out.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(exported,out/'pcb-yolo11n.onnx'); shutil.copyfile(p,out/'pcb-yolo11n.pt')
    metadata={'names':[model.names[i] for i in range(len(model.names))],'imgsz':960,'weights':str(p.relative_to(ROOT)),'source':'local fine-tuning; see training run and dataset preparation manifest','official_equivalence':False}
    (out/'pcb-yolo11n.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding='utf-8')
    print('Saved',out)
