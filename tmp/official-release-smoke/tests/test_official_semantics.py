import copy
import json
from pathlib import Path
import tempfile
import subprocess
import time
import unittest
from unittest.mock import patch
from pcbparse.competition import evaluate_qa,export_competition
from pcbparse.semantic_library import SemanticLibrary
from pcbparse.agent import finalize,correct_with_llm
from pcbparse.__main__ import save

def fixture():
    return {'components':{'GND_1':{'type':'gnd','Name':None,'value':None,'bbox':[0,0,10,10]}},'pins':{'GND_1':{'pin_1':{'pinname':'','point':{'x':5,'y':10}}}},'nets':{'net_1':{'hyperGraph':'(GND_1.1)','edges':{}}}}

class OfficialSemanticsTests(unittest.TestCase):
    def test_random_power_component_and_pin_ids_match_by_coordinates(self):
        pred=fixture(); gt={'components':{'∅200':dict(pred['components']['GND_1'],Name='∅200')},'pins':{'∅200':{'pin_198':dict(pred['pins']['GND_1']['pin_1'],pinname='P_99')}},'nets':{'n':{'hyperGraph':'(∅200.198)','edges':{}}}}
        result=evaluate_qa(pred,gt)
        self.assertEqual(result['Score_Accuracy_SingleCase_proxy'],100)
        self.assertFalse(result['official_equivalence'])
        gt['pins']['∅200']['pin_198']['point']={'x':100,'y':100}
        self.assertEqual(evaluate_qa(pred,gt)['PinF1'],0)

    def test_box_numbers_and_names_still_matter(self):
        a=fixture(); a['components']['GND_1']['type']='box'; b=copy.deepcopy(a)
        b['pins']['GND_1']['pin_1']['pinname']='VIN'
        self.assertEqual(evaluate_qa(a,b)['PinF1'],0)

    def test_types_use_corrected_official_families_and_spelling(self):
        for source,target in [('resistor','r'),('ic','box'),('amp_5pin','amp'),('crystal_3pin','crystal'),('mosfet_pnp','mosfet'),('battery','battary')]:
            a=fixture(); a['components']['GND_1']['type']=source
            self.assertEqual(export_competition(a)['components']['GND_1']['type'],target)

    def library(self,root,pins):
        f=Path(root)/'library.json'; f.write_text(json.dumps({'entries':{'ABC1234':{'pins':pins}}}),encoding='utf-8')
        return SemanticLibrary(f)

    def obs(self):
        return {'components':{'U1':{'type':'ic','Name':'ABC1234','value':None,'bbox':[20,0,80,100]}},'pins':{'U1':{}},'texts':[{'text':'VIN','bbox':[22,47,40,53]}],'pin_hypotheses':[{'id':'U1@tip_1','component':'U1','point':{'x':10,'y':50},'score':.9,'status':'number_unresolved'}],'wires':[]}

    def test_unique_visual_name_resolves_number_without_moving_endpoint(self):
        with tempfile.TemporaryDirectory() as root:
            lib=self.library(root,{'1':{'names':['VIN'],'types':['IN']}}); obs=self.obs()
            result,report=lib.refine(obs)
            self.assertEqual(result['pins']['U1']['pin_1'],{'pinname':'VIN','point':{'x':10,'y':50}})
            self.assertEqual(result['wires'],obs['wires']); self.assertEqual(obs['pins']['U1'],{})

    def test_repeated_power_names_abstain_and_missing_text_abstains(self):
        with tempfile.TemporaryDirectory() as root:
            lib=self.library(root,{'1':{'names':['VIN']},'2':{'names':['VIN']}})
            result,_=lib.refine(self.obs()); self.assertEqual(result['pins']['U1'],{})
            obs=self.obs(); obs['texts']=[]
            result,_=lib.refine(obs); self.assertEqual(result['pins']['U1'],{})

    def test_agent_results_use_official_directory_contract(self):
        with tempfile.TemporaryDirectory() as root:
            out=Path(root); save(out/'runs/local/case one/case one.json',fixture())
            log=finalize({'seconds':1,'records':[{'case':'case one','status':'completed'}]},out)
            self.assertTrue((out/'final_results/case one_Pred.json').exists())
            self.assertTrue((out/'decision_log.json').exists()); self.assertEqual(log['status'],'completed')

    def test_category_provenance_preserves_bidirectional_ports_and_blocks(self):
        data=fixture(); data['components']['GND_1']['type']='vbus_in'
        self.assertEqual(export_competition(data,{'GND_1':'Net_bidirection'})['components']['GND_1']['type'],'net_bidirection')
        data['components']['GND_1']['type']='ic'
        self.assertEqual(export_competition(data,{'GND_1':'Block'})['components']['GND_1']['type'],'block')

    def test_failed_agent_rerun_does_not_leave_stale_prediction(self):
        with tempfile.TemporaryDirectory() as root:
            out=Path(root); save(out/'final_results/old_Pred.json',fixture())
            log=finalize({'seconds':1,'records':[{'case':'old','status':'failed'}]},out)
            self.assertFalse((out/'final_results/old_Pred.json').exists()); self.assertEqual(log['status'],'incomplete')

    def test_llm_timeout_preserves_local_answer_and_marks_usage_unknown(self):
        with tempfile.TemporaryDirectory() as root:
            out=Path(root); save(out/'runs/local/a/a.json',fixture())
            save(out/'runs/local/_diagnostics/a-review.json',{'tasks':[]})
            log=finalize({'seconds':1,'records':[{'case':'a','status':'completed','image':'a.png','seconds':1}]},out)
            answer=(out/'final_results/a_Pred.json').read_bytes()
            with patch('pcbparse.agent.subprocess.run',side_effect=subprocess.TimeoutExpired('adapter',1)):
                correct_with_llm(log,out,1,time.monotonic(),100,100)
            self.assertEqual((out/'final_results/a_Pred.json').read_bytes(),answer)
            self.assertFalse(log['usage_complete']); self.assertFalse(log['llm_call_count_complete'])
            self.assertEqual(log['cases'][0]['llm_status'],'local_fallback')

    def test_llm_is_skipped_after_total_deadline(self):
        with tempfile.TemporaryDirectory() as root:
            out=Path(root); save(out/'runs/local/a/a.json',fixture())
            log=finalize({'seconds':1,'records':[{'case':'a','status':'completed','image':'a.png','seconds':1}]},out)
            with patch('pcbparse.agent.subprocess.run') as run:
                correct_with_llm(log,out,1,time.monotonic()-10,1,100)
                run.assert_not_called()
            self.assertEqual(log['cases'][0]['llm_status'],'skipped_deadline')
