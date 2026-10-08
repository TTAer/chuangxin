import unittest
import numpy as np
from scripts.audit_freeze_a1_revision_candidates import all_pin_segments,vector_error

class WholeSheetRevisionTests(unittest.TestCase):
    def test_hidden_design_pins_do_not_become_visible_geometry(self):
        pin=['pin','passive','line',['at','0','0','0'],['length','2.54'],['number','1'],['name','VDD']]
        hidden=['pin','passive','line','hide',['at','0','0','0'],['length','2.54'],['number','2'],['name','HIDDEN']]
        tree=['root',['lib_symbols',['symbol','X',['symbol','X_1_1',pin,hidden]]],
            ['symbol',['lib_id','X'],['property','Reference','U1'],['property','Value','M'],['at','10','20','90'],['unit','1'],['dnp','yes']]]
        row=all_pin_segments(tree)[0]
        self.assertEqual(len(row['pins']),1);self.assertEqual(row['hidden_pins_design_only'][0]['number'],'2');self.assertEqual(row['dnp'],'yes')
        self.assertAlmostEqual(row['pins'][0]['pin_length_px'],14.4)
        self.assertLess(row['pins'][0]['body_base_xy'][1],row['pins'][0]['outer_tip_xy'][1])

    def test_fixed_vector_registration_does_not_fit_away_error(self):
        self.assertEqual(vector_error([0,0,10,0],np.array([[0,0,10,0]])),0)
        self.assertEqual(vector_error([0,1,10,1],np.array([[0,0,10,0]])),1)
        self.assertIsNone(vector_error([0,0,10,0],np.array([])))

if __name__=='__main__':unittest.main()
