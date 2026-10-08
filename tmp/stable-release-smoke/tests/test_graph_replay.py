import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from pcbparse.__main__ import convert_coordinates, demo
from pcbparse.corrections import validate_proposals


class GraphReplayTests(unittest.TestCase):
    def test_extended_top_left_observations_transform_all_evidence_once(self):
        shared={'x':12,'y':20}
        obs={'coordinate_system':'top-left','image':{'height':100},'components':{},'pins':{},'wires':[], 'texts':[{'text':'RUN','bbox':[10,20,30,40]}], 'masked_symbol_boxes':[[10,20,30,40]], 'contact_decisions':[{'point':shared,'state':'uncertain'}], 'pin_hypotheses':[{'point':shared}]}
        original=copy.deepcopy(obs)
        result=convert_coordinates(obs)
        self.assertEqual(result['texts'][0]['bbox'],[10,60,30,80])
        self.assertEqual(result['masked_symbol_boxes'],[[10,60,30,80]])
        self.assertEqual(result['contact_decisions'][0]['point'],{'x':12,'y':80})
        self.assertEqual(result['pin_hypotheses'][0]['point'],{'x':12,'y':80})
        self.assertEqual(obs,original)
        self.assertEqual(convert_coordinates(result),result)

    def test_non_string_llm_candidate_becomes_validation_error(self):
        queue={'observation_sha256':'abc','tasks':[{'id':'candidate','allowed_decisions':['reject']}]}
        for candidate in ([],{},None):
            with self.assertRaises(ValueError):
                validate_proposals({'observation_sha256':'abc','corrections':[{'candidate_id':candidate,'decision':'reject'}]},queue)

    def test_invalid_correction_cli_preserves_local_prediction(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); demo(root)
            response=root/'bad.json'; response.write_text(json.dumps({'observation_sha256':'wrong','corrections':[]}),encoding='utf-8')
            command=[sys.executable,'-m','pcbparse','graph-build',str(root/'observations.json'),'--output',str(root/'result.json'),'--graph-output',str(root/'graph.json')]
            subprocess.run(command,check=True,capture_output=True)
            baseline=json.loads((root/'result.json').read_text())
            subprocess.run(command+['--corrections',str(response)],check=True,capture_output=True)
            self.assertEqual(json.loads((root/'result.json').read_text()),baseline)
            report=json.loads((root/'graph.json').read_text())
            self.assertEqual(report['llm']['status'],'proposal_rejected_local_fallback')
