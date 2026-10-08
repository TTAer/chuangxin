import copy
import itertools
import json
import random
import tempfile
import unittest
from pathlib import Path

from pcbparse.core import assignment, build, evaluate, references, validate
from pcbparse.__main__ import convert_coordinates, demo, load


def point(x,y): return {'x':x,'y':y}


def observation(wires, coords):
    components={}; pins={}
    for i,p in enumerate(coords):
        k='R'+str(i+1)
        components[k]={'type':'resistor','Name':None,'value':'10k','bbox':[p['x']-1,p['y']-1,p['x']+1,p['y']+1]}
        pins[k]={'pin_1':{'pinname':'1','point':p}}
    return {'components':components,'pins':pins,'wires':wires}


class CoreTests(unittest.TestCase):
    def test_T_junction(self):
        o=observation([[point(0,10),point(20,10)],[point(10,0),point(10,10)]],[point(0,10),point(10,0),point(20,10)])
        self.assertEqual(len(build(o,0)['nets']),1)

    def test_crossing_requires_dot(self):
        o=observation([[point(0,10),point(20,10)],[point(10,0),point(10,20)]],[point(0,10),point(10,0)])
        self.assertEqual(len(build(o,0)['nets']),2)
        o['junctions']=[point(10,10)]
        self.assertEqual(len(build(o,0)['nets']),1)

    def test_direct_pin_connection(self):
        o=observation([],[point(5,5),point(5,5)])
        net=next(iter(build(o,0)['nets'].values()))
        self.assertEqual(references(net['hyperGraph']),{'R1.1','R2.1'})
        self.assertEqual(net['edges'],{})

    def test_labels_merge_without_fake_wire(self):
        o=observation([[point(0,0),point(5,0)],[point(100,0),point(105,0)]],[point(0,0),point(105,0)])
        o['labels']=[{'name':'RUN','pin':'R1.1'},{'name':'RUN','pin':'R2.1'}]
        result=build(o,0)
        self.assertEqual(len(result['nets']),1)
        self.assertEqual(len(next(iter(result['nets'].values()))['edges']),2)

    def test_top_left_conversion(self):
        o=observation([[point(5,10),point(20,10)]],[point(5,10)])
        o.update(coordinate_system='top-left',image={'height':100})
        c=convert_coordinates(o)
        self.assertEqual(c['pins']['R1']['pin_1']['point'],point(5,90))
        self.assertEqual(c['components']['R1']['bbox'],[4,89,6,91])
        self.assertEqual(o['pins']['R1']['pin_1']['point'],point(5,10))

    def test_invalid_reference(self):
        r=build(observation([],[point(5,5)]))
        next(iter(r['nets'].values()))['hyperGraph']='(R9.1)'
        self.assertTrue(any('unknown pins' in e for e in validate(r)))

    def test_bad_geometry_and_nan(self):
        r=build(observation([],[point(5,5)]))
        next(iter(r['nets'].values()))['edges']={'edge_1':[point(0,0)]}
        r['pins']['R1']['pin_1']['point']['x']=float('nan')
        self.assertEqual(len(validate(r)),2)

    def test_duplicate_json_key(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'bad.json'; p.write_text('{"R1":1,"R1":2}')
            with self.assertRaises(ValueError): load(p)

    def test_hungarian_against_brute_force(self):
        rng=random.Random(7)
        for rows,cols in [(2,3),(3,2),(4,4),(1,4)]:
            for _ in range(10):
                w=[[rng.randrange(10) for _ in range(cols)] for _ in range(rows)]
                n=max(rows,cols)
                padded=[[w[i][j] if i<rows and j<cols else 0 for j in range(n)] for i in range(n)]
                expected=max(sum(padded[i][j] for i,j in enumerate(perm)) for perm in itertools.permutations(range(n)))
                self.assertEqual(sum(w[i][j] for i,j in assignment(w)),expected)

    def test_strict_text_and_coordinate_threshold(self):
        r=build(observation([],[point(5,5)])); p=copy.deepcopy(r)
        p['components']['R1']['value']='10K'
        self.assertEqual(evaluate(p,r)['ComponentF1'],0)
        p=copy.deepcopy(r); p['pins']['R1']['pin_1']['point']['x']+=5
        self.assertEqual(evaluate(p,r)['PinF1'],1)
        p['pins']['R1']['pin_1']['point']['x']+=.01
        self.assertEqual(evaluate(p,r)['PinF1'],0)

    def test_wrong_net_line_is_penalized(self):
        o=observation([[point(0,0),point(5,0)],[point(100,0),point(105,0)]],[point(0,0),point(105,0)])
        r=build(o,0); p=copy.deepcopy(r)
        a,b=list(p['nets'].values()); a['edges'],b['edges']=b['edges'],a['edges']
        self.assertEqual(evaluate(p,r)['NetHypergraphF1'],1)
        self.assertEqual(evaluate(p,r)['NetLineF1'],0)

    def test_reversed_line_endpoints(self):
        r=build(observation([[point(0,0),point(5,0)]],[point(0,0),point(5,0)]),0)
        p=copy.deepcopy(r); next(iter(p['nets'].values()))['edges']['edge_1'].reverse()
        self.assertEqual(evaluate(p,r)['NetLineF1'],1)

    def test_empty_prediction(self):
        r=build(observation([],[point(5,5)]))
        p={'components':{},'pins':{},'nets':{}}
        self.assertEqual(evaluate(p,r)['PinF1'],0)
        self.assertEqual(evaluate(p,r)['NetHypergraphF1'],0)

    def test_demo_independent_golden(self):
        with tempfile.TemporaryDirectory() as d:
            result=demo(d)
            self.assertEqual(result['metrics']['Score_Accuracy_SingleCase_proxy'],100)
            self.assertFalse(result['metrics']['official_equivalence'])
            self.assertFalse(validate(load(Path(d)/'prediction.json')))


if __name__=='__main__': unittest.main()
