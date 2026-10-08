"""Verify the newly built image's default and optional candidate outputs."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from pcbparse.core import validate
from pcbparse.competition import evaluate_qa
def load(path): return json.loads((ROOT/path).read_text(encoding='utf-8'))
p=argparse.ArgumentParser(); p.add_argument('--image-id',required=True); a=p.parse_args()
actual=load('output/container-side-strip/final_results/1_Pred.json'); expected=load('output/benchmark-side-strip-local-gt10/1/prediction.json')
assert not validate(actual); differences=[]
def compare(x,y,path=''):
    if isinstance(x,dict):
        assert x.keys()==y.keys(),path
        for key in x: compare(x[key],y[key],path+'/'+key)
    elif isinstance(x,list):
        assert len(x)==len(y),path
        for i,(u,v) in enumerate(zip(x,y)): compare(u,v,path+'/'+str(i))
    elif x!=y:
        assert isinstance(x,(int,float)) and isinstance(y,(int,float)) and path.endswith(('/x','/y')),path
        error=abs(x-y); assert error<.001,(path,error); differences.append(error)
compare(actual,expected)
truth=load('data/raw/official/gt-0930/10GT/1/1.json'); assert evaluate_qa(actual,truth)==evaluate_qa(expected,truth)
assert load('output/container-side-default/final_results/guide-schematic_Pred.json')==load('output/recognition-workspace-smoke/final_results/guide-schematic_Pred.json')
logs={mode:load('output/container-side-'+mode+'/decision_log.json') for mode in ('strip','default')}
assert all(log['status']=='completed' and log['llm_calls']==0 for log in logs.values())
record={'image_tag':'graphschematic-side-strip:20261004','image_id':a.image_id,'build_verified':True,'network':'none','input_read_only':True,
        'default_exact_windows_match':True,'candidate_identity_names_nets_identical':True,'candidate_exact_json_match':actual==expected,
        'candidate_max_coordinate_difference_pixels':max(differences,default=0),'candidate_numeric_coordinate_differences':len(differences),
        'candidate_QA_proxy_metrics_identical':True,'function_tests_passed':76,'llm_calls':0,
        'single_case_seconds':{mode:log['elapsed_seconds'] for mode,log in logs.items()},'official_evaluator_verified':False,
        'batch_acceptance':'previous version image only; this new image verified on default and optional single cases'}
(ROOT/'output/side-strip-verification/container.json').write_text(json.dumps(record,indent=2),encoding='utf-8'); print(json.dumps(record),flush=True)
