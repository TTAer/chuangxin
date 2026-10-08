"""L1 observed-review placeholders separated from source L2 CAD evidence."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'output/freeze-a1-two-layer'
def load(p):return json.loads(p.read_text(encoding='utf-8'))
def save(p,d):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d,ensure_ascii=False,indent=2),encoding='utf-8')
def main():
    summary=load(ROOT/'output/freeze-a1-revision-audit/summary.json');ids=[]
    for page in summary['pages_detail']:
        source=load(ROOT/'output/freeze-a1-revision-audit'/(page['image_sha256'][:12]+'.json'))
        design=[]
        for row in source['instances']:
            design.append({'source_uuid':row['uuid'],'CAD_property_reference':row['reference'],'CAD_value':row['model'],
                'unit':row['unit'],'DNP':row['dnp'],'mirror':row['mirror'],'rotation':row['rotation'],
                'visible_nonzero_pin_geometry_candidate':row['pins'],'hidden_pin_design_metadata':row['hidden_pins_design_only'],
                'zero_length_pin_design_metadata':row['zero_length_pins'],
                'visible_identity_not_inferred_from_CAD':True})
        record={'schema':'freeze-a1-two-layer-evidence-v1','image_sha256':source['image_sha256'],
            'source_cad_sha256':source['source_cad_sha256'],'source_pdf_sha256':source['source_pdf_sha256'],
            'level_1_observed_truth':{'status':'pending_independent_pixel_review','coordinate_convention_frozen':False,
                'visible_components':[],'visible_pin_strokes':[],'visible_contacts':[],'verified_human_reviews':[]},
            'level_2_design_truth':{'status':'CAD source metadata; whole-page revision not yet certified','instances':design,
                'electrical_net_truth':None,'note':'No netlist fabricated from hidden pins or geometry registration.'},
            'visual_scoring_policy':'L1 only when reviewed; L2 hidden/unused properties never become visual GT automatically',
            'verified_design':False,'model_inference_runs':0}
        path=OUT/(source['image_sha256'][:12]+'.json')
        # Separate draft L1 can be filled later; do not overwrite prior observations.
        if path.exists():
            old=load(path)
            if old['image_sha256']!=record['image_sha256']:raise ValueError('Observed image changed')
            record['level_1_observed_truth']=old['level_1_observed_truth']
        save(path,record);ids.append(str(path.relative_to(ROOT)))
    save(OUT/'summary.json',{'pages':len(ids),'records':ids,'verified_designs':0,'certified_L1_components':0,
        'auto_CAD_to_visual_GT_enabled':False,'model_runs':0})
    print(json.dumps({'pages':len(ids),'verified_designs':0,'certified_L1_components':0}))
if __name__=='__main__':main()
