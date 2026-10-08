"""Library-conditioned visual pin association with explicit ambiguity abstention."""
import copy
from difflib import SequenceMatcher
import json
import math
from pathlib import Path
import re
from .core import assignment

def norm(s): return re.sub(r'[^A-Z0-9+/-]','',str(s).upper())
def similarity(a,b): return SequenceMatcher(None,norm(a),norm(b)).ratio()
def center(box): return ((box[0]+box[2])/2,(box[1]+box[3])/2)
def box_gap(a,b): return math.hypot(max(a[0]-b[2],b[0]-a[2],0),max(a[1]-b[3],b[1]-a[3],0))

class SemanticLibrary:
    def __init__(self,path='models/semantics/component-library.json'):
        self.entries=json.loads(Path(path).read_text(encoding='utf-8'))['entries']

    def identify(self,name,box,texts):
        observed=[t['text'] for t in texts if box_gap(t['bbox'],box)<=45]
        if name: observed.append(name)
        observed=[s for s in observed if len(norm(s))>=5 and re.search(r'\d',s)]
        ranked=sorted([(max((similarity(s,m) for s in observed),default=0),m) for m in self.entries],reverse=True)
        if not ranked or ranked[0][0]<.94: return None,{'status':'no_supported_model'}
        score,model=ranked[0]
        # Variant suffixes can be near-identical but differ in pin numbering.
        if len(ranked)>1 and score-ranked[1][0]<.035: return None,{'status':'ambiguous_model','candidates':[m for s,m in ranked[:3]]}
        return model,{'status':'model_identified','model':model,'score':score,'ocr_evidence':max(observed,key=lambda s:similarity(s,model))}

    def refine(self,observations):
        result=copy.deepcopy(observations); evidence=[]; texts=result.get('texts',[])
        for component,c in result['components'].items():
            if c['type'] not in ('ic','box'): continue
            model,record=self.identify(c.get('Name'),c['bbox'],texts); record['component']=component
            evidence.append(record)
            if not model: continue
            # Correct model text only from strong local OCR evidence, never from page ID.
            c['Name']=model; catalog=self.entries[model]['pins']; pins=result['pins'][component]
            record['edits']=[]
            for key,p in pins.items():
                number=key[4:]
                if number not in catalog: continue
                allowed=catalog[number]['names']
                choices=self.visual_names(p['point'],c['bbox'],texts,allowed)
                if choices and (len(choices)==1 or choices[0][0]-choices[1][0]>=.08):
                    score,name,text=choices[0]
                    if p['pinname']!=name:
                        record['edits'].append({'kind':'pin_name','pin':key,'before':p['pinname'],'after':name,'ocr':text,'score':score})
                        p['pinname']=name
            # Unnumbered learned endpoints need a unique locally visible semantic name.
            tips=[p for p in result.get('pin_hypotheses',[]) if p['component']==component]
            numbers=[n for n in catalog if 'pin_'+n not in pins]
            weights=[]; matches={}
            for i,p in enumerate(tips):
                row=[]
                for j,n in enumerate(numbers):
                    choices=self.visual_names(p['point'],c['bbox'],texts,catalog[n]['names'])
                    best=choices[0] if choices and (len(choices)==1 or choices[0][0]-choices[1][0]>=.08) else None
                    row.append(best[0] if best else 0)
                    if best: matches[i,j]=best
                weights.append(row)
            for i,j in assignment(weights):
                score=weights[i][j]
                # Require the number to be locally unique, even after joint assignment.
                alternatives=sorted(weights[i],reverse=True)
                if score<=0 or len(alternatives)>1 and alternatives[0]-alternatives[1]<.08: continue
                p=tips[i]; n=numbers[j]; _,name,text=matches[i,j]
                pins['pin_'+n]={'pinname':name,'point':copy.deepcopy(p['point'])}
                p['status']='library_number_resolved'; p['resolved_number']=n
                record['edits'].append({'kind':'pin_number','tip_id':p['id'],'after':n,'name':name,'ocr':text,'score':score})
        result['pin_hypotheses']=[p for p in result.get('pin_hypotheses',[]) if p.get('status')!='library_number_resolved']
        return result,{'module':'LibraryConditionedSemanticAssociation','components':evidence,'edits':sum(len(r.get('edits',[])) for r in evidence),'uses_image_evidence':True,'copies_gt_geometry':False}

    @staticmethod
    def visual_names(point,box,texts,allowed):
        x,y=point['x'],point['y']; vertical=min(abs(y-box[1]),abs(y-box[3]))<min(abs(x-box[0]),abs(x-box[2]))
        best={}
        for t in texts:
            tx,ty=center(t['bbox']); gap=math.hypot(tx-x,ty-y); transverse=abs(tx-x) if vertical else abs(ty-y)
            if gap>48 or transverse>8 or box_gap(t['bbox'],box)>8: continue
            for name in allowed:
                if len(norm(name))<2 or norm(name).isdigit(): continue
                sim=similarity(t['text'],name)
                if sim<.88: continue
                score=.9*sim+.1*(1-gap/48)
                if name not in best or score>best[name][0]: best[name]=(score,name,t['text'])
        return sorted(best.values(),reverse=True)
