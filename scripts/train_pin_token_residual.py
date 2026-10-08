"""Fit tabular exceptions; calibration families choose gates, not validation."""
import hashlib
import json
from pathlib import Path
import pickle
import sys
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from experiments.pin_token_residual import residual_predictions,preservation_metrics,safe_anchors

def probability(model,x,label):
    classes=list(model.classes_)
    return model.predict_proba(x)[:,classes.index(label)] if label in classes else np.zeros(len(x))

def main():
    cache=ROOT/'data/processed/pin-token-residual'; source=cache/'residual-features.npz'; d=np.load(source)
    x=d['features']; y=d['target']; role=d['role']; rows=json.loads((cache/'records.json').read_text(encoding='utf-8'))
    validation=d['split']==1; calibration=np.array([r['split']=='train' and int(r['family'][:8],16)%5==0 for r in rows]); fit=(d['split']==0)&~calibration
    assert fit.any() and calibration.any() and validation.any()
    pins=y<8; out=ROOT/'output/training/pin-token-residual'; out.mkdir(parents=True,exist_ok=True)
    report={'protocol':{'seed':20261005,'threshold_grid':[.95,.975,.99,1.01],'alternative_probability_min':.9,'min_calibration_switches':10,
        'calibration_switch_precision_min':.95,'calibration_EPR_min_exclusive':.99,'validation_EPR_min_exclusive':.99,'validation_HRR_min_exclusive':.25,
        'calibration':'original train families whose signature hash modulo5=0; not original validation',
        'safe_anchor':'margin>=0.04 normalized body length and margin>=0.5*d1','source_sha256':hashlib.sha256(source.read_bytes()).hexdigest()},
        'split_tokens':{'fit':int(fit.sum()),'calibration':int(calibration.sum()),'validation':int(validation.sum())},'variants':{},
        'runtime_enabled':False,'neural_training_started':False,'official_score':None,'end_to_end_metrics':None}
    for variant in ('logistic','extra-trees'):
        wrong=(y!=0).astype(int)
        classifier=make_pipeline(StandardScaler(),LogisticRegression(C=1,class_weight='balanced',max_iter=1000,random_state=20261005)) if variant=='logistic' else ExtraTreesClassifier(n_estimators=200,max_depth=8,min_samples_leaf=3,class_weight='balanced',random_state=20261005,n_jobs=4)
        classifier.fit(x[fit&pins],wrong[fit&pins]); pwrong=probability(classifier,x,1)
        alternative=ExtraTreesClassifier(n_estimators=200,max_depth=6,min_samples_leaf=2,random_state=20261005,n_jobs=4)
        hard=fit&pins&(y!=0); alternatives=np.where(np.isin(y,[1,2]),y,3)
        assert hard.sum()>=5,'insufficient hard training relations for any tabular reranker'
        alternative.fit(x[hard],alternatives[hard]); alt=alternative.predict(x); confidence=alternative.predict_proba(x).max(axis=1)
        rolemodel=ExtraTreesClassifier(n_estimators=200,max_depth=8,min_samples_leaf=3,class_weight='balanced',random_state=20261005,n_jobs=4)
        rolemodel.fit(x[fit],role[fit]); predicted_role=rolemodel.predict(x)
        selection=[]; chosen=1.01; best=-1
        for threshold in report['protocol']['threshold_grid']:
            pred=residual_predictions(x,pwrong,alt,confidence,threshold); m=preservation_metrics(pred[calibration],y[calibration]); switched=calibration&pins&(pred!=0)
            precision=float((pred[switched]==y[switched]).mean()) if switched.any() else None
            admissible=m['switches']>=10 and m['EPR']>.99 and precision>=.95
            selection.append({'threshold':threshold,'metrics':m,'switch_precision':precision,'admissible':admissible})
            if admissible and m['hard_recovered']-m['easy_broken']>best: best=m['hard_recovered']-m['easy_broken']; chosen=threshold
        pred=residual_predictions(x,pwrong,alt,confidence,chosen); m=preservation_metrics(pred[validation],y[validation])
        gate=m['EPR']>.99 and m['HRR']>.25
        raw=residual_predictions(x,pwrong,alt,confidence,.95)
        result={'chosen_threshold':chosen,'calibration_selection':selection,'validation':m,
            'fixed_threshold_0_95_diagnostic_only':preservation_metrics(raw[validation],y[validation]),
            'role_macro_f1':float(f1_score(role[validation],predicted_role[validation],average='macro')),
            'role_classes':['pin_number','pin_name','non_pin_explicitly_labelled'],
            'safe_pin_tokens':int((validation&pins&safe_anchors(x)).sum()),
            'hard_protected_by_safe_anchor':int((validation&pins&(y!=0)&safe_anchors(x)).sum()),
            'gate_passed':gate,'decision':'eligible_for_development_end_to_end_trial_only' if gate else 'STOP_no_end_to_end_integration'}
        report['variants'][variant]=result
        with (out/(variant+'.pkl')).open('wb') as f: pickle.dump({'wrong':classifier,'alternative':alternative,'role':rolemodel,'threshold':chosen},f)
        np.savez_compressed(out/(variant+'-validation.npz'),indices=np.flatnonzero(validation),prediction=pred[validation],truth=y[validation],role=predicted_role[validation])
        print(json.dumps({'variant':variant,'threshold':chosen,'validation':m,'gate_passed':gate}),flush=True)
    report['limitations']=['weak source OCR references are not manually reviewed pin GT','oracle body/endpoints and exact aligned token subset','non-pin only from explicitly labelled designator/model; unknown or signal texts not assumed negatives','conditional association gate does not prove correct predicted role or electrical net','tabular probabilities uncalibrated; calibration is operating-point selection, not probability calibration','same-domain independent Freeze-A not used or scored']
    (ROOT/'output/ablation/pin-token-residual.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
if __name__=='__main__': main()
