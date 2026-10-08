"""Bounded batch inference: isolated case failures and explicit incomplete cases."""
import json
from pathlib import Path
import subprocess
import sys
import time


def run_batch(images,output,model,detector,case_timeout=115,total_timeout=6900,pin_model='auto',local_pin_ocr=False,text_constraints=False):
    if case_timeout<=0 or total_timeout<=0: raise ValueError('timeouts must be positive')
    root=Path(images); out=Path(output)
    files=sorted(x for x in root.rglob('*') if x.suffix.lower() in ('.png','.jpg','.jpeg'))
    if not files: raise ValueError('no PNG/JPG input images')
    ids=[x.stem for x in files]
    if len(ids)!=len(set(ids)): raise ValueError('duplicate image basenames; use a unique case filename')
    out.mkdir(parents=True,exist_ok=True); diagnostics=out/'_diagnostics'; diagnostics.mkdir(exist_ok=True)
    started=time.monotonic(); records=[]
    for p in files:
        remaining=total_timeout-(time.monotonic()-started)
        case=p.stem; target=out/case/(case+'.json')
        record={'case':case,'image':str(p.resolve())}
        if remaining<=0:
            records.append(dict(record,status='unfinished',reason='total deadline reached')); continue
        case_start=time.monotonic()
        command=[sys.executable,'-m','pcbparse','infer',str(p.resolve()),'--model',str(Path(model).resolve()),'--detector',str(Path(detector).resolve()),'--output',str(target.resolve()),'--diagnostics',str((diagnostics/(case+'.json')).resolve())]
        selector=pin_model if pin_model in ('auto','none') else str(Path(pin_model).resolve()) if pin_model else 'none'
        command.extend(['--pin-model',selector])
        if local_pin_ocr: command.append('--local-pin-ocr')
        if text_constraints: command.append('--text-constraints')
        try:
            process=subprocess.run(command,capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=min(case_timeout,remaining))
            record.update(status='completed' if process.returncode==0 else 'failed',seconds=time.monotonic()-case_start)
            if process.returncode: record['error']=process.stderr[-2500:]
        except subprocess.TimeoutExpired:
            record.update(status='failed',reason='case timeout',seconds=time.monotonic()-case_start)
        records.append(record)
        (diagnostics/'batch-progress.json').write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding='utf-8')
    result={'cases':len(files),'completed':sum(r['status']=='completed' for r in records),'failed':sum(r['status']=='failed' for r in records),'unfinished':sum(r['status']=='unfinished' for r in records),'seconds':time.monotonic()-started,'records':records}
    (diagnostics/'batch-summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    return result
