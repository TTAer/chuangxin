"""Bounded, offline Docker acceptance on 10 images and 50 simulated cases.

No ground-truth JSON is mounted. Repeated images exercise runtime only, not
independent recognition accuracy. Memory is a sampled Docker working-set value.
"""
import hashlib
import argparse
import json
from pathlib import Path
import re
import shutil
import subprocess
import time
import uuid
import numpy as np
import sys

ROOT=Path(__file__).resolve().parents[1]
PREFIX=['wsl','-d','Ubuntu-22.04','-u','root','--exec','/usr/bin/docker']
TAG='graphschematic-recognition:20261004'
def command(args,check=True):
    return subprocess.run(PREFIX+args,capture_output=True,encoding='utf-8',errors='replace',check=check)
def linux(path):
    relative=path.resolve().relative_to(ROOT.resolve()).as_posix()
    return '/mnt/d/创芯EDA/gpt方案/'+relative
def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def load(path): return json.loads(path.read_text(encoding='utf-8'))
def memory_bytes(text):
    match=re.search(r'([\d.]+)\s*(GiB|MiB|KiB|GB|MB|kB|B)',text)
    units={'GiB':2**30,'MiB':2**20,'KiB':2**10,'GB':10**9,'MB':10**6,'kB':10**3,'B':1}
    return float(match[1])*units[match[2]] if match else None

def main():
    global TAG
    parser=argparse.ArgumentParser(); parser.add_argument('--tag',default=TAG); parser.add_argument('--report-prefix',default='container'); parser.add_argument('--profile',choices=['default','identity-side','rc0'],default='default'); parser.add_argument('--counts',nargs='+',type=int,default=[10,50]); a=parser.parse_args(); TAG=a.tag
    assert all(n>0 and n%10==0 for n in a.counts)
    reports=[]
    image_id=command(['image','inspect',TAG,'--format','{{.Id}}']).stdout.strip()
    for count in a.counts:
        inputs=ROOT/'tmp'/(a.report_prefix+'-input-'+str(count)); inputs.mkdir(parents=True,exist_ok=True)
        out=ROOT/'output'/(a.report_prefix+'-batch-'+str(count)); out.mkdir(parents=True,exist_ok=True)
        cases={}
        for repeat in range(count//10):
            for number in range(1,11 if a.profile=='default' else 9):
                source=ROOT/'data/raw/official/gt-0930/10GT'/str(number)/(str(number)+'.png')
                name='runtime_'+str(repeat)+'_'+str(number)
                destination=inputs/(name+'.png'); shutil.copyfile(source,destination); cases[name]=('gt10',str(number))
            if a.profile in ('identity-side','rc0'):
                sys.path.insert(0,str(ROOT))
                from pcbparse.dataset import index_dataset
                samples={s['id']:s for s in index_dataset(ROOT/'data/raw/PCBnet/Dataset300')}
                for number in (('0101','0197') if a.profile=='rc0' else ('0006','0184')):
                    name='runtime_'+str(repeat)+'_pcb'+number
                    shutil.copyfile(samples[number]['image'],inputs/(name+'.png')); cases[name]=('pcb55',number)
        assert len(list(inputs.iterdir()))==count and all(p.suffix=='.png' for p in inputs.iterdir())
        hashes={p.name:digest(p) for p in inputs.iterdir()}
        name='graphschematic-acceptance-'+uuid.uuid4().hex[:12]
        args=['run','--name',name,'--network','none','--cpus','2','--memory','2g','--pids-limit','256',
              '-v',linux(inputs)+':/input:ro','-v',linux(out)+':/output',image_id,'-input','/input','-output','/output']
        if a.profile in ('identity-side','rc0'): args+=['--ic-identity-mode','retrieval','--side-pin-model','models/candidates/side-strip-local/side-strip.onnx','--side-name-anchors']
        if a.profile=='rc0': args+=['--designator-mode','stable','--identity-transaction-mode','catalog']
        started=time.monotonic(); samples=[]
        with (out/'docker-console.log').open('w',encoding='utf-8') as stream:
            process=subprocess.Popen(PREFIX+args,stdout=stream,stderr=subprocess.STDOUT)
            try:
                while process.poll() is None:
                    sample=command(['stats','--no-stream','--format','{{.MemUsage}}',name],check=False)
                    value=memory_bytes(sample.stdout)
                    if value is not None: samples.append({'seconds':time.monotonic()-started,'memory_bytes':value})
                    if time.monotonic()-started>1500:
                        command(['stop','--time','5',name],check=False)
                        raise TimeoutError('bounded acceptance exceeded 1500 seconds')
                    time.sleep(5)
                state=command(['inspect','--format','{{json .State}}',name])
                # WSL startup notices go to stderr; stdout holds Docker JSON.
                state=json.loads(state.stdout.strip())
                assert process.returncode==0 and not state['OOMKilled'],state
                log=load(out/'decision_log.json'); assert log['status']=='completed' and len(log['cases'])==count
                assert log['llm_calls']==0
                exact=0; errors=[]
                def compare(actual,expected,path=''):
                    if isinstance(actual,dict):
                        assert actual.keys()==expected.keys(),path
                        for k in actual: compare(actual[k],expected[k],path+'/'+k)
                    elif isinstance(actual,list):
                        assert len(actual)==len(expected),path
                        for i,(x,y) in enumerate(zip(actual,expected)): compare(x,y,path+'/'+str(i))
                    elif actual!=expected:
                        assert isinstance(actual,(int,float)) and isinstance(expected,(int,float)) and (path.endswith(('/x','/y')) or '/bbox/' in path),path
                        difference=abs(actual-expected); assert difference<.001,(path,difference); errors.append(difference)
                for case in log['cases']:
                    assert case['status']=='completed'
                    actual=load(out/'final_results'/(case['case']+'_Pred.json'))
                    domain,key=cases[case['case']]
                    if a.profile=='default':
                        expected=load(ROOT/'output/benchmark-recognition-gt10'/key/'prediction.json'); assert actual==expected,case['case']
                    else:
                        expected_root=ROOT/'output/designator-stable-catalog'/domain if a.profile=='rc0' else ROOT/('output/ic-identity-'+domain+'-side')
                        expected=load(expected_root/key/'prediction.json')
                        if domain=='pcb55':
                            from pcbparse.competition import export_competition
                            expected=export_competition(expected,load(expected_root/key/'observations.json').get('component_categories'))
                        compare(actual,expected)
                    exact+=actual==expected
                assert hashes=={p.name:digest(p) for p in inputs.iterdir()}
                latency=[c['seconds'] for c in log['cases']]
                report={'image_id':image_id,'cases':count,'status':'passed','unique_source_images':10,'simulated_repeats':count>10,
                        'cpus':2,'memory_limit_bytes':2*2**30,'network':'none','input_read_only':True,'input_hashes_unchanged':True,
                        'failed':0,'timeout_rate':0.,'oom_killed':False,'profile':a.profile,'exact_prediction_matches':exact,
                        'exact_default_prediction_matches':count if a.profile=='default' else 0,'identity_names_nets_verified':True,'max_geometry_difference_pixels':max(errors,default=0),
                        'avg_case_seconds':float(np.mean(latency)),'p95_case_seconds':float(np.percentile(latency,95)),
                        'agent_elapsed_seconds':log['elapsed_seconds'],'host_wall_seconds':time.monotonic()-started,
                        'sampled_peak_working_set_bytes':max((s['memory_bytes'] for s in samples),default=None),
                        'memory_measurement':'sampled docker stats working set; may miss transient peak; not cgroup memory.peak',
                        'memory_samples':samples,'llm_calls':0,'input_tokens':0,'output_tokens':0,'official_runner_verified':False}
                (out/'acceptance.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
                reports.append(report); print(json.dumps({k:v for k,v in report.items() if k!='memory_samples'}),flush=True)
            finally:
                if process.poll() is None: command(['stop','--time','5',name],check=False); process.wait(timeout=30)
                command(['rm',name],check=False)
    result={'image_tag':TAG,'image_id':image_id,'profile':a.profile,'default_ic_pin_mode':'none','tests':reports,'official_runner_verified':False,
            'limits_are_local_test_choices':True,'fifty_case_run_is_not_independent_accuracy_test':True}
    (ROOT/'output'/(a.report_prefix+'-batch-acceptance.json')).write_text(json.dumps(result,indent=2),encoding='utf-8')
if __name__=='__main__': main()
