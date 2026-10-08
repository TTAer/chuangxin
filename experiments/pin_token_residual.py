"""Geometry remains default; exceptions cannot modify safe anchors."""
import numpy as np

def safe_anchors(features):
    # d2-d1 relative to normalized d1; declared before fitting/validation.
    d1,d2,margin=features[:,-4],features[:,-3],features[:,-2]
    return (margin>=.04)&(margin>=.5*np.maximum(d1,.001))

def residual_predictions(features,wrong_probability,alternative,alternative_confidence,threshold):
    out=np.zeros(len(features),dtype=int)
    switch=(~safe_anchors(features))&(wrong_probability>=threshold)&(alternative_confidence>=.9)&np.isin(alternative,[1,2])
    out[switch]=alternative[switch]
    return out

def preservation_metrics(prediction,truth):
    pin=truth<8; easy=pin&(truth==0); hard=pin&(truth!=0)
    correct=int(((prediction==truth)&pin).sum()); broken=int((prediction[easy]!=0).sum()); recovered=int((prediction[hard]==truth[hard]).sum())
    return {'pin_tokens':int(pin.sum()),'nearest_correct':int(easy.sum()),'hard':int(hard.sum()),'easy_broken':broken,'hard_recovered':recovered,
        'EPR':1-broken/max(1,int(easy.sum())),'HRR':recovered/max(1,int(hard.sum())),
        'pin_association_accuracy':correct/max(1,int(pin.sum())),'switches':int(((prediction!=0)&pin).sum())}
