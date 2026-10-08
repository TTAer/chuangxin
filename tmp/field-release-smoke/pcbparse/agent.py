"""EDARace directory entry contract with isolated case failures."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from .batch import run_batch
from .competition import export_competition
from .__main__ import load,save

ROOT=Path(__file__).resolve().parents[1]

def correct_with_llm(log,output,max_calls,started,total_timeout,case_timeout,association_model='none',field_model='none'):
    """Explicit optional correction, charged to per-case and total deadlines."""
    if max_calls<=0: return
    output=Path(output); diag=output/'runs/local/_diagnostics'
    for record in log['cases']:
        if record['status']!='completed': continue
        remaining=min(total_timeout-(time.monotonic()-started),case_timeout-record['seconds'])
        if remaining<=1:
            record['llm_status']='skipped_deadline'; continue
        case=record['case']; work=output/'runs/llm'/case; work.mkdir(parents=True,exist_ok=True)
        response=work/'response.json'; queue=diag/(case+'-review.json')
        if not queue.exists(): record['llm_status']='no_review_queue'; continue
        correction_started=time.monotonic()
        try:
            subprocess.run([sys.executable,str(ROOT/'scripts/request_graph_corrections.py'),str(queue),record['image'],'--output',str(response),'--max-calls',str(min(max_calls,8))],capture_output=True,timeout=remaining,check=True)
            payload=load(response)
            if payload.get('status')=='not_configured': record['llm_status']='not_configured'; continue
            status=load(response.with_suffix('.status.json'))
            record.update(llm_calls=status['calls'],input_tokens=payload['usage']['input_tokens'],output_tokens=payload['usage']['output_tokens'])
            record['usage_complete']=status.get('usage_complete',False)
            remaining=min(total_timeout-(time.monotonic()-started),case_timeout-record['seconds']-(time.monotonic()-correction_started))
            if remaining<=1: record['llm_status']='proposal_saved_deadline'; continue
            graph=work/'graph.json'; result=work/'prediction.json'
            subprocess.run([sys.executable,'-m','pcbparse','graph-build',str(diag/(case+'-observations.json')),'--image',record['image'],'--output',str(result),'--graph-output',str(graph),'--corrections',str(response),'--association-model',association_model,'--field-model',field_model],capture_output=True,timeout=remaining,check=True)
            report=load(graph); record['llm_status']=report['llm']['status']
            if report['llm']['status']=='validated_proposal_replay':
                categories=load(diag/(case+'-observations.json')).get('component_categories')
                save(output/record['result_file'],export_competition(load(result),categories))
        except (OSError,ValueError,KeyError,subprocess.SubprocessError) as exc:
            # Preserve completed local prediction; do not log API bodies or secrets.
            record['llm_status']='local_fallback'; record['llm_error_type']=type(exc).__name__
        finally:
            record['seconds']+=time.monotonic()-correction_started
    log['llm_calls']=sum(r.get('llm_calls',0) for r in log['cases'])
    log['token_usage']={k:sum(r.get(k,0) for r in log['cases']) for k in ('input_tokens','output_tokens')}
    log['elapsed_seconds']=time.monotonic()-started
    log['usage_complete']=all(r.get('usage_complete',True) and not r.get('llm_error_type') for r in log['cases'])
    log['llm_call_count_complete']=not any(r.get('llm_error_type') for r in log['cases'])
    save(output/'decision_log.json',log)

def finalize(summary,output):
    output=Path(output); final=output/'final_results'; final.mkdir(parents=True,exist_ok=True)
    records=[]
    for item in summary['records']:
        record=dict(item)
        target=final/(item['case']+'_Pred.json')
        # A failed rerun must not silently leave its earlier answer for scoring.
        target.unlink(missing_ok=True)
        if item['status']=='completed':
            path=output/'runs/local'/item['case']/(item['case']+'.json')
            try:
                obs_path=output/'runs/local/_diagnostics'/(item['case']+'-observations.json')
                categories=load(obs_path).get('component_categories',{}) if obs_path.exists() else None
                prediction=export_competition(load(path),categories)
                save(target,prediction)
                record.update(result_file=target.relative_to(output).as_posix(),llm_calls=0,input_tokens=0,output_tokens=0)
            except (ValueError,OSError) as exc:
                record.update(status='failed',reason=str(exc)[:300])
        records.append(record)
    log={'framework':'GraphSchematic','status':'completed' if all(r['status']=='completed' for r in records) else 'incomplete','cases':records,'elapsed_seconds':summary['seconds'],'llm_calls':0,'token_usage':{'input_tokens':0,'output_tokens':0},'official_evaluator_verified':False,'decision_log_schema':'provisional audit schema; official EDARace detailed manual not supplied','decisions':['local image-only detection/OCR/pin perception','topology-gated dual graph association; preserve same-label flying nets','official library semantic constraints when enabled','canonical types from updated official type list']}
    save(output/'decision_log.json',log)
    return log

def main(argv=None):
    p=argparse.ArgumentParser(description='GraphSchematic EDARace agent')
    p.add_argument('-input','--input',required=True); p.add_argument('-output','--output',required=True)
    p.add_argument('--case-timeout',type=float,default=115); p.add_argument('--total-timeout',type=float,default=6900)
    p.add_argument('--llm-max-calls-per-case',type=int,default=0,help='optional explicit online correction budget; default local-only')
    p.add_argument('--pin-model',default='auto'); p.add_argument('--semantic-library',default='auto')
    p.add_argument('--pin-ensemble',default='auto')
    p.add_argument('--association-model',default='auto')
    p.add_argument('--field-model',default='auto')
    a=p.parse_args(argv); inputs=Path(a.input).resolve(); output=Path(a.output).resolve()
    if not inputs.is_dir(): p.error('input directory does not exist')
    if inputs==output or output.is_relative_to(inputs): p.error('output must be outside the read-only input directory')
    output.mkdir(parents=True,exist_ok=True)
    old=Path.cwd()
    started=time.monotonic()
    try:
        os.chdir(ROOT)
        summary=run_batch(inputs,output/'runs/local','models/symbols','models/detector/pcb-yolo11n.onnx',a.case_timeout,a.total_timeout,pin_model=a.pin_model,semantic_library=a.semantic_library,text_constraints=True,pin_ensemble=a.pin_ensemble,association_model=a.association_model,field_model=a.field_model)
        log=finalize(summary,output)
        correct_with_llm(log,output,a.llm_max_calls_per_case,started,a.total_timeout,a.case_timeout,association_model=a.association_model,field_model=a.field_model)
    finally: os.chdir(old)
    print(json.dumps({'status':log['status'],'cases':len(log['cases']),'output':str(output)},ensure_ascii=False))
    return 0 if log['status']=='completed' else 1
