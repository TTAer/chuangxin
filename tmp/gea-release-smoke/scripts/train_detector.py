"""Fine-tune YOLO; use local input weights to avoid implicit network downloads."""
import argparse
from pathlib import Path
import sys
import os

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
os.environ.setdefault('YOLO_CONFIG_DIR',str(ROOT/'tmp/yolo'))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--data',default='data/processed/yolo/dataset.yaml')
    p.add_argument('--weights',default='models/pretrained/yolo11n.pt')
    p.add_argument('--epochs',type=int,default=40)
    p.add_argument('--imgsz',type=int,default=960)
    p.add_argument('--batch',type=int,default=8)
    p.add_argument('--device',default='0')
    p.add_argument('--name',default='pcb-yolo11n')
    p.add_argument('--lr',type=float,default=.001)
    p.add_argument('--preserve-head-prefix',action='store_true',help='new classes must be appended after the old classes')
    args=p.parse_args()
    from ultralytics import YOLO
    if not Path(args.weights).is_file(): p.error('local pretrained weights required')
    model=YOLO(args.weights)
    if args.preserve_head_prefix:
        import torch
        old={k:v.detach().cpu().clone() for k,v in model.model.state_dict().items()}
        old_nc=len(model.names)
        def preserve_head(trainer):
            copied=[]
            with torch.no_grad():
                for key,value in trainer.model.state_dict().items():
                    source=old.get(key)
                    if source is not None and source.ndim in (1,4) and source.shape[0]==old_nc and value.shape[0]>old_nc and source.shape[1:]==value.shape[1:]:
                        value[:old_nc].copy_(source.to(value.device)); copied.append(key)
            print('Preserved existing class head rows:',copied,flush=True)
            import json
            (trainer.save_dir/'head-transfer.json').write_text(json.dumps({'old_classes':old_nc,'copied_rows':copied},indent=2),encoding='utf-8')
        model.add_callback('on_train_start',preserve_head)
    model.train(data=str(Path(args.data).resolve()),epochs=args.epochs,imgsz=args.imgsz,batch=args.batch,device=args.device,workers=0,project=str(ROOT/'output/training'),name=args.name,exist_ok=True,seed=20261003,deterministic=True,patience=18,amp=False,plots=True,save=True,cache=False,fliplr=0.,flipud=0.,degrees=0.,translate=.05,scale=.2,mosaic=.5,close_mosaic=8,optimizer='AdamW',lr0=args.lr,cos_lr=True)
