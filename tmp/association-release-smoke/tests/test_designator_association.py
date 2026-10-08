import unittest
from pcbparse.designator_association import associate

def detection(cat,bbox):
    return {'category':cat,'bbox':bbox}

class DesignatorAssociationTests(unittest.TestCase):
    def test_ground_cannot_consume_chip_designator(self):
        ds=[detection('Gnd',[0,0,10,10]),detection('Box',[15,0,40,20])]
        ts=[{'text':'U2','bbox':[0,0,10,10]}]
        _,old,_=associate(ds,ts,'legacy'); _,new,report=associate(ds,ts,'typed')
        self.assertEqual(old,{0:0}); self.assertEqual(new,{1:0})
        self.assertIsNone(report['records'][0]['designator'])

    def test_block_does_not_consume_resistor_designator(self):
        ds=[detection('Block',[0,0,40,40]),detection('R',[42,0,50,10])]
        _,actual,_=associate(ds,[{'text':'R1','bbox':[5,5,15,15]}],'typed')
        self.assertEqual(actual,{1:0})

    def test_unobserved_identity_stays_unassigned(self):
        _,actual,_=associate([detection('Box',[0,0,40,40])],[{'text':'ABC123','bbox':[5,5,15,15]}],'typed')
        self.assertEqual(actual,{})

    def test_boxed_connector_and_integrated_led_have_observed_designators(self):
        ds=[detection('Box',[0,0,40,40]),detection('Box',[60,0,100,40])]
        ts=[{'text':'USB1','bbox':[10,10,20,20]}, {'text':'LED2','bbox':[70,10,80,20]}]
        _,actual,_=associate(ds,ts,'boxed')
        self.assertEqual(actual,{0:0,1:1})
        _,legacy,_=associate(ds,ts,'legacy')
        self.assertEqual(legacy,{})

    def test_guarded_preserves_unknown_symbol_categories(self):
        ds=[detection('motor',[0,0,40,40]),detection('Gnd',[60,0,100,40])]
        ts=[{'text':'U18','bbox':[10,10,20,20]}, {'text':'U1','bbox':[70,10,80,20]}]
        _,actual,_=associate(ds,ts,'guarded')
        self.assertEqual(actual,{0:0})

    def test_one_text_cannot_be_reused_and_distant_text_abstains(self):
        ds=[detection('Box',[0,0,40,40]),detection('Box',[45,0,85,40])]
        _,actual,_=associate(ds,[{'text':'U1','bbox':[10,10,20,20]}, {'text':'U2','bbox':[500,500,510,510]}],'typed')
        self.assertEqual(actual,{0:0})
