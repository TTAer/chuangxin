"""Small affected/control Docker replay; historical 50-case proof stays frozen."""
import json
from pathlib import Path
import shutil
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from pcbparse.__main__ import load,save
from pcbparse.dataset import index_dataset
from pcbparse.competition import export_competition

def main():
    prefix=['wsl','-d','Ubuntu-22.04','-u','root','--exec','/usr/bin/docker']
    image_id=subprocess.run(prefix+['image','inspect','graphschematic-association:20261005','--format','{{.Id}}'],capture_output=True,encoding='utf-8',errors='replace',check=True).stdout.strip()
    samples={s['id']:s for s in index_dataset(ROOT/'data/raw/PCBnet/Dataset300')}; reports=[]
    for mode in ('legacy','guarded'):
        inputs=ROOT/'tmp'/('designator-container-'+mode); inputs.mkdir(parents=True,exist_ok=True)
        if mode=='legacy':
            shutil.copyfile(ROOT/'data/examples/guide-schematic.png',inputs/'guide-schematic.png')
            expected={'guide-schematic':load(ROOT/'output/recognition-workspace-smoke/final_results/guide-schematic_Pred.json')}
        else:
            expected={}
            for case in ('0067','0101','0196'):
                shutil.copyfile(samples[case]['image'],inputs/(case+'.png'))
                base=ROOT/'output/designator-guarded/pcb55'/case
                expected[case]=export_competition(load(base/'prediction.json'),load(base/'observations.json')['component_categories'])
        assert len(list(inputs.iterdir()))==len(expected)
        out=ROOT/'output'/('designator-container-'+mode); out.mkdir(parents=True,exist_ok=True)
        def linux(p): return '/mnt/d/创芯EDA/gpt方案/'+p.relative_to(ROOT).as_posix()
        cmd=prefix+['run','--rm','--network','none','--cpus','2','--memory','2g','--pids-limit','256','-v',linux(inputs)+':/input:ro','-v',linux(out)+':/output',image_id,'-input','/input','-output','/output','--designator-mode',mode]
        if mode=='guarded': cmd+=['--ic-identity-mode','retrieval','--side-pin-model','models/candidates/side-strip-local/side-strip.onnx','--side-name-anchors']
        run=subprocess.run(cmd,capture_output=True,encoding='utf-8',errors='replace',timeout=400)
        (out/'console.log').write_text(run.stdout+run.stderr,encoding='utf-8'); assert run.returncode==0,run.stderr
        log=load(out/'decision_log.json'); assert log['status']=='completed' and log['llm_calls']==0
        errors=[]
        def compare(a,b,path=''):
            if isinstance(a,dict):
                assert a.keys()==b.keys(),path
                for k in a: compare(a[k],b[k],path+'/'+k)
            elif isinstance(a,list):
                assert len(a)==len(b),path
                for i,(x,y) in enumerate(zip(a,b)): compare(x,y,path+'/'+str(i))
            elif a!=b:
                assert isinstance(a,(float,int)) and isinstance(b,(float,int)) and (path.endswith(('/x','/y')) or '/bbox/' in path),path
                delta=abs(a-b); assert delta<.001,(path,delta); errors.append(delta)
        for case,pred in expected.items(): compare(load(out/'final_results'/(case+'_Pred.json')),pred)
        reports.append({'mode':mode,'cases':len(expected),'identity_names_nets_equal':True,'max_geometry_difference_pixels':max(errors,default=0),'elapsed_seconds':log['elapsed_seconds'],'llm_calls':0})
        print(reports[-1],flush=True)
    save(ROOT/'output/ablation/designator-container.json',{'image_id':image_id,'image_tag':'graphschematic-association:20261005','build_verified':True,'reports':reports,'current_50_case_verified':False,'network':'none','input_read_only':True,'cpus':2,'memory_limit_gib':2,'official_evaluator_verified':False})
if __name__=='__main__': main()
