"""Evaluate image-only inference on the held-out public sample split."""
import argparse
import csv
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from pcbparse.__main__ import save
from pcbparse.core import evaluate
from pcbparse.dataset import convert_raw,index_dataset
from pcbparse.vision import ImageParser,overlay


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--data',default='data/raw/PCBnet/Dataset300'); p.add_argument('--output',default='output/benchmark'); p.add_argument('--limit',type=int,default=10); p.add_argument('--model',default='models/symbols'); p.add_argument('--detector',default='models/detector/pcb-yolo11n.onnx')
    p.add_argument('--pin-model',default=None)
    p.add_argument('--local-pin-ocr',action='store_true')
    p.add_argument('--text-constraints',action='store_true')
    p.add_argument('--graph-mode',choices=['legacy','association','full'],default='legacy')
    p.add_argument('--topology-model',default='auto')
    p.add_argument('--exclude-case',action='append',default=['0237'])
    args=p.parse_args(); parser=ImageParser(args.model,detector=args.detector,pin_model=args.pin_model,local_pin_ocr=args.local_pin_ocr,text_constraints=args.text_constraints,graph_mode=args.graph_mode,topology_model=args.topology_model)
    samples=[i for i in index_dataset(args.data) if i['split']=='validation' and i['id'] not in args.exclude_case]
    if args.limit>0: samples=samples[:args.limit]
    records=[]; out=Path(args.output); out.mkdir(parents=True,exist_ok=True); started=time.perf_counter()
    for i in samples:
        try:
            # Strict separation: parser sees only image path. Adapter reads truth later.
            prediction,obs,diag,texts=parser.parse(i['image'])
            raw=json.loads(Path(i['annotation']).read_text(encoding='utf-8'))
            golden,issues=convert_raw(raw)
            metrics=evaluate(prediction,golden)
            directory=out/i['id']; save(directory/'prediction.json',prediction); save(directory/'observations.json',obs); save(directory/'golden.json',golden); save(directory/'diagnostics.json',diag); save(directory/'adapter-issues.json',issues)
            if parser.last_graph_report:
                save(directory/'graph.json',parser.last_graph_report); save(directory/'review.json',parser.last_review_queue)
            overlay(i['image'],prediction,directory/'overlay.png')
            records.append({'id':i['id'],'split':'validation','style':i['style'],'status':'completed','seconds':diag['seconds'],'adapter_issues':len(issues),**metrics})
        except Exception as e:
            records.append({'id':i['id'],'status':'failed','error':str(e)})
        print(json.dumps(records[-1],ensure_ascii=False),flush=True)
        save(out/'progress.json',records)
    complete=[r for r in records if r['status']=='completed']; keys=['ComponentF1','PinF1','NetHypergraphF1','NetLineF1','Score_Accuracy_SingleCase_proxy']
    summary={'cases':len(records),'completed':len(complete),'failed':len(records)-len(complete),'wall_seconds':time.perf_counter()-started,'mean_metrics':{k:sum(r[k] for r in complete)/len(records) if records else 0 for k in keys},'mean_case_seconds':sum(r['seconds'] for r in complete)/len(complete) if complete else None,'official_equivalence':False,'ground_truth':'development conversion of CVJsonStd; incomplete text annotations reported','records':records}
    save(out/'summary.json',summary); print(json.dumps(summary,ensure_ascii=False,indent=2))
