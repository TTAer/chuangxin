import unittest
from pcbparse.ic_identity import ICIdentityResolver,observed_forms
from pcbparse.semantic_library import SemanticLibrary

class Library(SemanticLibrary):
    def __init__(self):
        self.entries={'ABC12345':{'pins':{'1':{'names':['RESET']},'2':{'names':['CLOCK']}}},
                      'XYZ99999':{'pins':{'1':{'names':['VIN']},'2':{'names':['GND']}}}}

def fixture():
    return {'components':{'U1':{'type':'ic','Name':'R12ABC12345','bbox':[20,20,100,100]}},
            'pins':{'U1':{}},'texts':[{'text':'R12ABC12345','bbox':[40,50,90,60],'confidence':.99},
                                   {'text':'RESET','bbox':[25,70,45,80],'confidence':.99},
                                   {'text':'CLOCK','bbox':[25,30,45,40],'confidence':.99}],
            'pin_hypotheses':[]}

class IdentityTests(unittest.TestCase):
    def test_glued_prefix_forms_are_constrained(self):
        self.assertIn(('AMS1117-3.3','prefix_split'),observed_forms('C7AMS1117-3.3'))
        self.assertEqual(observed_forms('FOOABC12345'),[('FOOABC12345','raw')])

    def test_glued_model_needs_two_local_pin_names(self):
        obs=fixture(); resolver=ICIdentityResolver(Library()); actual,report=resolver.refine(obs)
        self.assertEqual(actual['components']['U1']['Name'],'ABC12345')
        self.assertEqual(obs['components']['U1']['Name'],'R12ABC12345')
        self.assertTrue(report['edits'])
        obs['texts']=obs['texts'][:1]; actual,_=resolver.refine(obs)
        self.assertEqual(actual,obs)

    def test_duplicate_model_schema_abstains(self):
        lib=Library(); lib.entries['ABC12346']=lib.entries['ABC12345']
        obs=fixture(); obs['components']['U1']['Name']='ABC1234'; obs['texts'][0]['text']='ABC1234'
        actual,report=ICIdentityResolver(lib).refine(obs)
        self.assertEqual(actual,obs)
        self.assertFalse(report['edits'])

    def test_missing_model_not_replaced_by_similar_power_schema(self):
        obs=fixture(); obs['components']['U1']['Name']='RP2040'
        obs['texts'][0]['text']='RP2040'
        actual,_=ICIdentityResolver(Library()).refine(obs)
        self.assertEqual(actual,obs)
