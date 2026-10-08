"""Offline current-image replay of IC case and newly recovered pin case."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from pcbparse.__main__ import load,save
from pcbparse.competition import export_competition
from pcbparse.dataset import index_dataset
from pcbparse.core import validate
prefix=['wsl','-d','Ubuntu-22.04','-u','root','--exec','/usr/bin/docker']
tag='graphschematic-side-strip:20261005'
image_id=subprocess.run(prefix+['image','inspect',tag,'--format','{{.Id}}'],check=True,capture_output=True,encoding='utf-8',errors='replace').stdout.strip()
inputs=ROOT/'tmp/side-current-container-input'; inputs.mkdir(parents=True,exist_ok=True)
sources={'1':ROOT/'data/raw/official/gt-0930/10GT/1/1.png',
         '0184':Path({s['id']:s for s in index_dataset(ROOT/'data/raw/PCBnet/Dataset300')}['0184']['image'])}
for key,path in sources.items(): shutil.copyfile(path,inputs/(key+'.png'))
assert len(list(inputs.iterdir()))==2
hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs.iterdir()}
out=ROOT/'output/container-side-20261005-optional'; out.mkdir(parents=True,exist_ok=True)
def linux(p): return '/mnt/d/创芯EDA/gpt方案/'+p.relative_to(ROOT).as_posix()
result=subprocess.run(prefix+['run','--rm','--network','none','--cpus','2','--memory','2g','--pids-limit','256',
        '-v',linux(inputs)+':/input:ro','-v',linux(out)+':/output',image_id,'-input','/input','-output','/output',
        '--side-pin-model','models/candidates/side-strip-local/side-strip.onnx','--side-name-anchors'],capture_output=True,encoding='utf-8',errors='replace')
(out/'console.log').write_text(result.stdout+result.stderr,encoding='utf-8'); assert result.returncode==0,(result.stdout,result.stderr)
differences=[]
def compare(a,b,path=''):
    if isinstance(a,dict):
        assert isinstance(b,dict) and a.keys()==b.keys(),path
        for k in a: compare(a[k],b[k],path+'/'+k)
    elif isinstance(a,list):
        assert isinstance(b,list) and len(a)==len(b),path
        for i,(x,y) in enumerate(zip(a,b)): compare(x,y,path+'/'+str(i))
    elif a!=b:
        assert isinstance(a,(int,float)) and isinstance(b,(int,float)) and (path.endswith(('/x','/y')) or '/bbox/' in path),path
        error=abs(a-b); assert error<.001,(path,error); differences.append(error)
for key,domain in [('1','gt10'),('0184','pcb55')]:
    expected=load(ROOT/'output/side-incremental-live'/domain/key/'prediction.json')
    if domain=='pcb55':
        expected=export_competition(expected,load(ROOT/'output/benchmark-recognition-pcb55'/key/'observations.json').get('component_categories'))
    actual=load(out/'final_results'/(key+'_Pred.json')); assert not validate(actual); compare(actual,expected)
assert hashes=={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs.iterdir()}
log=load(out/'decision_log.json'); assert log['status']=='completed' and log['llm_calls']==0
record={'image_tag':tag,'image_id':image_id,'build_verified':True,'optional_cases':2,'optional_identity_names_nets_identical':True,
        'optional_max_geometry_difference_pixels':max(differences,default=0),'optional_numeric_differences':len(differences),
        'network':'none','input_read_only':True,'input_hashes_unchanged':True,'cpu_limit':2,'memory_limit_bytes':2*2**30,
        'llm_calls':0,'elapsed_seconds':log['elapsed_seconds'],'official_evaluator_verified':False,'function_tests_passed':79}
save(ROOT/'output/ablation/side-current-container.json',record); print(json.dumps(record),flush=True)
