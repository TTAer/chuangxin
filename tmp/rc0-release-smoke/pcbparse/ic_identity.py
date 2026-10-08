"""Top-K observed model retrieval with explicit lexical and pin evidence gates.

Catalog order never supplies image-side order. Unknown chips remain unknown.
"""
import copy
import re
from .semantic_library import norm,similarity,center,box_gap

def observed_forms(text):
    """Split only recognizable glued designator/value prefixes; keep raw token."""
    result=[(text,'raw')]
    for pattern in (r'^(?:[RCUJ]\d+)([A-Z][A-Z0-9].*)$',r'^\d+(?:\.\d+)?(?:uF|nF|pF|u|n|p)([A-Z][A-Z0-9].*)$'):
        match=re.fullmatch(pattern,text)
        if match: result.append((match[1],'prefix_split'))
    return result

class ICIdentityResolver:
    def __init__(self,library): self.library=library

    def retrieve(self,component,texts,pins,k=5):
        box=component['bbox']; observed=[]
        for t in texts:
            if t.get('confidence',1)<.85 or box_gap(t['bbox'],box)>15: continue
            for value,source in observed_forms(t['text']):
                if len(norm(value))<5 or not re.search(r'\d',value): continue
                observed.append((value,t['text'],source))
        name=component.get('Name')
        if name:
            # Component Name is itself an OCR selection, not an independent vote.
            for value,source in observed_forms(name):
                if len(norm(value))>=5: observed.append((value,name,source))
        candidates=[]
        for model,entry in self.library.entries.items():
            if not observed: continue
            ranked=[]
            for value,raw,source in observed:
                lexical=similarity(value,model)
                # Confusion folding only recalls candidates, never confirms them alone.
                fold=lambda s:norm(s).translate(str.maketrans({'O':'0','I':'1','L':'1','S':'5','B':'8'}))
                from difflib import SequenceMatcher
                confusion=SequenceMatcher(None,fold(value),fold(model)).ratio()*.93
                family=.85 if len(norm(value))>=6 and norm(model).startswith(norm(value)) and len(norm(value))/len(norm(model))>=.6 else 0
                ranked.append((max(lexical,confusion,family),lexical,value,raw,source))
            score,raw_similarity,value,raw,source=max(ranked)
            if score<.80: continue
            supported=[]; contradictory=[]
            for key,pin in pins.items():
                number=key[4:]; allowed=entry['pins'].get(number,{}).get('names',[])
                choices=self.library.visual_names(pin['point'],box,texts,allowed)
                if choices and choices[0][0]>=.90: supported.append(number)
                elif pin.get('pinname') and allowed and max(similarity(pin['pinname'],n) for n in allowed)<.5:
                    contradictory.append(number)
            unique_names=set()
            for t in texts:
                tx,ty=center(t['bbox'])
                if t.get('confidence',1)<.9 or not (box[0]-8<=tx<=box[2]+8 and box[1]-8<=ty<=box[3]+8): continue
                if len(norm(t['text']))<3 or norm(t['text']).isdigit(): continue
                scores=[max((similarity(t['text'],s) for s in pin['names']),default=0) for pin in entry['pins'].values()]
                if scores and max(scores)>=.94: unique_names.add(norm(t['text']))
            evidence=min(1,(len(supported)+.5*len(unique_names))/4)
            joint=.8*score+.2*evidence-.03*min(3,len(contradictory))
            candidates.append({'model':model,'score':joint,'lexical_score':score,'raw_similarity':raw_similarity,
                'ocr_evidence':raw,'candidate_token':value,'token_source':source,'supported_numbered_pins':supported,
                'observed_names':sorted(unique_names),'contradictory_numbered_pins':contradictory,'expected_pins':len(entry['pins'])})
        candidates.sort(key=lambda r:(r['score'],r['lexical_score'],r['model']),reverse=True)
        top=candidates[:k]; accepted=None; status='unknown'
        if top:
            best=top[0]; margin=best['score']-top[1]['score'] if len(top)>1 else best['score']
            # Two observed pin names are required, and neither family nor folding alone confirms a model.
            direct=best['raw_similarity']>=.94
            schema_reranked=best['lexical_score']>=.85 and len(best['supported_numbered_pins'])>=3
            adequate=(direct or schema_reranked) and len(best['observed_names'])>=2 and not best['contradictory_numbered_pins']
            if adequate and margin>=(.035 if direct else .06): accepted=best['model']; status='evidence_confirmed'
            else: status='ambiguous' if adequate else 'insufficient_evidence'
        else: margin=0
        return accepted,{'status':status,'candidates':top,'second_best_margin':margin,'pin_order_available':False}

    def refine(self,observations):
        result=copy.deepcopy(observations); records=[]; edits=[]
        for key,c in result['components'].items():
            if c['type'] not in ('ic','box'): continue
            strict,old=self.library.identify(c.get('Name'),c['bbox'],result.get('texts',[]))
            if strict:
                records.append({'component':key,'status':'strict_identity_preserved','model':strict}); continue
            model,report=self.retrieve(c,result.get('texts',[]),result['pins'].get(key,{})); report['component']=key; report['previous_status']=old['status']; records.append(report)
            if model and c.get('Name')!=model:
                before=c.get('Name'); c['Name']=model; edits.append({'component':key,'before':before,'after':model})
                # Only the recovered component gets another semantic pass.
                isolated=copy.deepcopy(result); isolated['components']={key:c}; isolated['pins']={key:result['pins'][key]}
                isolated['pin_hypotheses']=[p for p in result.get('pin_hypotheses',[]) if p['component']==key]
                resolved,_=self.library.refine(isolated); result['pins'][key]=resolved['pins'][key]
                unresolved=[p for p in result.get('pin_hypotheses',[]) if p['component']!=key]
                result['pin_hypotheses']=unresolved+resolved.get('pin_hypotheses',[])
        return result,{'module':'ICIdentityRetrieval','components':records,'edits':edits,'ground_truth_used':False,
                       'confusion_and_family_are_retrieval_only':True,'library_side_order_assumed':False}
