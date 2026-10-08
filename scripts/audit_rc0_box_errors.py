"""Post-first-pass geometric error taxonomy; consumes frozen predictions only.

Categories are observable annotation-relative errors, not image-reviewed causes.
Thresholds below are declared once and are never fitted to the 25 cases.
"""
from collections import Counter
import hashlib
import statistics
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pcbparse.__main__ import load, save
from pcbparse.competition import canonical_type
from pcbparse.core import assignment
from pcbparse.dataset import CATEGORIES
from pcbparse.vision import iou
from scripts.evaluation_metrics import annotation_boxes, ELIGIBLE

THRESHOLDS = {'strict_iou': .5, 'center_good': .25, 'center_near': .75,
              'area_large': 1.5, 'area_small': 2/3, 'wrong_type_iou': .3,
              'ambiguous_weight_gap': .05}

def measurements(p, g):
    pb, gb = p['bbox'], g['bbox']
    gw, gh = max(gb[2]-gb[0], 1e-9), max(gb[3]-gb[1], 1e-9)
    dx = ((pb[0]+pb[2])-(gb[0]+gb[2])) / (2*gw)
    dy = ((pb[1]+pb[3])-(gb[1]+gb[3])) / (2*gh)
    return {'iou': iou(pb, gb), 'center_dx_gt_width': dx,
            'center_dy_gt_height': dy, 'center_normalized_max': max(abs(dx), abs(dy)),
            'area_ratio': (pb[2]-pb[0])*(pb[3]-pb[1])/(gw*gh)}

def taxonomy(predictions, truth):
    """Staged deterministic Hungarian matching; each prediction and GT used once."""
    remaining_p = set(range(len(predictions))); remaining_g = set(range(len(truth)))
    rows = []; stages = ('strict_same_type', 'near_same_type', 'near_wrong_type')
    for stage in stages:
        pis, gis = sorted(remaining_p), sorted(remaining_g)
        weights = []
        for pi in pis:
            line = []
            for gi in gis:
                p, g = predictions[pi], truth[gi]; m = measurements(p,g)
                same = p['type'] == g['type']
                eligible = (same and m['iou'] >= .5) if stage == stages[0] else (
                    same and m['center_normalized_max'] <= .75 if stage == stages[1] else
                    not same and (m['iou'] >= .3 or m['center_normalized_max'] <= .25))
                line.append(1 + m['iou'] + .1/(1+m['center_normalized_max']) if eligible else 0)
            weights.append(line)
        for ip, ig in assignment(weights):
            if weights[ip][ig] <= 0: continue
            pi, gi = pis[ip], gis[ig]; p,g=predictions[pi],truth[gi]; m=measurements(p,g)
            if stage == stages[0]: category = 'strict_match'
            elif stage == stages[2]: category = 'type_mismatch_candidate'
            elif m['center_normalized_max'] > .25: category = 'center_shift'
            elif m['area_ratio'] >= 1.5: category = 'center_good_box_large'
            elif m['area_ratio'] <= 2/3: category = 'center_good_box_small'
            else: category = 'center_good_shape_or_extent_mismatch'
            # Association ambiguity is reported, not suppressed by this taxonomy.
            alternatives = [gis[j] for j, w in enumerate(weights[ip])
                            if j != ig and w > 0 and abs(w-weights[ip][ig]) <= .05]
            competing = [pis[j] for j,w in enumerate(weights) if j != ip
                         and w[ig] > 0 and abs(w[ig]-weights[ip][ig]) <= .05]
            rows.append({'truth_index':gi, 'prediction_index':pi, 'category':category,
                         'stage':stage, 'truth':g, 'prediction':p, **m,
                         'ambiguous_alternative_truth_indices':alternatives,
                         'ambiguous_competing_prediction_indices':competing})
            remaining_p.remove(pi); remaining_g.remove(gi)
    for gi in sorted(remaining_g):
        # Distinguish truly absent nearby proposals from proposals already used
        # elsewhere or an unaccepted geometry/type combination.
        near = [pi for pi,p in enumerate(predictions)
                if measurements(p,truth[gi])['center_normalized_max'] <= .75
                or measurements(p,truth[gi])['iou'] >= .3]
        rows.append({'truth_index':gi,'prediction_index':None,'truth':truth[gi],
                     'category':'unresolved_competition_or_type_geometry' if near else 'no_near_candidate',
                     'near_prediction_indices':near})
    return {'truth_rows':sorted(rows,key=lambda r:r['truth_index']),
            'counts':dict(Counter(r['category'] for r in rows)),
            'unmatched_predictions':[{'index':pi,**predictions[pi]} for pi in sorted(remaining_p)],
            'unmatched_prediction_count':len(remaining_p),
            'truth_count':len(truth),'prediction_count':len(predictions),
            'eligible_prediction_count':sum(p['type'] in ELIGIBLE for p in predictions)}

def main():
    pool = load(ROOT/'data/splits/amsnet-frozen.json'); rows=[]; inputs={}
    for group in pool['groups']:
        record=group['records'][0]; image=ROOT/record['image']; case=record['case']
        assert hashlib.sha256(image.read_bytes()).hexdigest() == record['image_sha256']
        obs_path=ROOT/'output/freeze-b-amsnet-rc0/rc0'/case/'observations.json'
        pred_path=obs_path.with_name('prediction.json'); obs,pred=load(obs_path),load(pred_path)
        for path in (obs_path,pred_path): inputs[str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
        h=obs['image']['height']; truth,excluded=annotation_boxes(load(image.with_name(case+'_bbox.json')))
        raw=[{'key':str(r['detection']),'type':canonical_type(CATEGORIES.get(r['category'],r['category'].lower())),
              'bbox':r['bbox'],'association_status':r['status']}
             for r in obs['designator_association']['records']]
        semantic=[{'key':k,'type':canonical_type(c['type']),
                   'bbox':[c['bbox'][0],h-c['bbox'][3],c['bbox'][2],h-c['bbox'][1]]}
                  for k,c in pred['components'].items()]
        rows.append({'case':case,'source_group':group['spice_text_group_sha256'],
                     'excluded_annotation_classes':excluded,
                     'raw_detector':taxonomy(raw,truth),'semantic_output':taxonomy(semantic,truth)})
    summary={}
    for mode in ('raw_detector','semantic_output'):
        counts=Counter()
        for r in rows: counts.update(r[mode]['counts'])
        summary[mode]={'counts':dict(counts), 'truth_count':sum(r[mode]['truth_count'] for r in rows),
                       'prediction_count':sum(r[mode]['prediction_count'] for r in rows),
                       'eligible_prediction_count':sum(r[mode]['eligible_prediction_count'] for r in rows),
                       'unmatched_prediction_count':sum(r[mode]['unmatched_prediction_count'] for r in rows)}
        paired=[x for r in rows for x in r[mode]['truth_rows'] if x.get('prediction')]
        summary[mode]['categories_by_truth_type']={category:dict(Counter(x['truth']['type'] for x in paired if x['category']==category))
                                                   for category in counts if category not in ('no_near_candidate','unresolved_competition_or_type_geometry')}
        summary[mode]['type_mismatch_pairs']={a+' -> '+b:n for (a,b),n in Counter(
            (x['truth']['type'],x['prediction']['type']) for x in paired if x['category']=='type_mismatch_candidate').items()}
        shifted=[x for x in paired if x['category']=='center_shift']
        summary[mode]['center_shift_medians']={k:statistics.median(x[k] for x in shifted) if shifted else None
                                               for k in ('center_dx_gt_width','center_dy_gt_height','area_ratio')}
        summary[mode]['ambiguous_paired_count']=sum(bool(x['ambiguous_alternative_truth_indices'] or x['ambiguous_competing_prediction_indices']) for x in paired)
    transitions=Counter()
    for r in rows:
        for a,b in zip(r['raw_detector']['truth_rows'],r['semantic_output']['truth_rows']):
            assert a['truth_index']==b['truth_index']
            transitions[a['category']+' -> '+b['category']]+=1
    result={'dataset':'Freeze-B AMSNet 25 groups; partial shared-class annotation',
            'post_first_pass_diagnostic_only':True,'runtime_changed':False,'inference_repeated':False,
            'thresholds':THRESHOLDS,'thresholds_fit_to_results':False,
            'annotation_adapter':'scripts/evaluation_metrics.py, yxyx to xyxy',
            'input_prediction_sha256':inputs,'summary':summary,'transitions':dict(transitions),'rows':rows,
            'limitations':['No near candidate is an observed proposal miss, not a visually adjudicated true miss.',
                           'Type mismatch is a geometric candidate association, not a proven semantic error.',
                           'Ambiguous Hungarian matches and unmatched predictions are retained.',
                           'Small/large boxes may reflect annotation convention; no visual ground-truth adjudication.',
                           'Normalized SPICE groups do not prove independent PCB designs; no official score.']}
    save(ROOT/'output/ablation/rc0-box-error-taxonomy.json',result)
    print(summary); print(dict(transitions))

if __name__ == '__main__': main()
