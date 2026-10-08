"""Evidence-bound LLM review queue and replayable, strictly scoped proposals."""
import hashlib
import json
import copy
import math
import re


def observation_hash(observations):
    return hashlib.sha256(json.dumps(observations,sort_keys=True,ensure_ascii=False,allow_nan=False,separators=(',',':')).encode('utf-8')).hexdigest()


def review_queue(observations,report,image_sha256=None,limit=8):
    accepted={o['id'] for o in report['association']['accepted']}
    rejected={o['candidate']:o['reason'] for o in report['association']['rejected']}
    options=[o for o in report['geometry_graph']['hypotheses'] if o['id'] not in accepted]
    options.sort(key=lambda o:(abs(o['score']-.62),o['id']))
    tasks=[]
    association_limit=min(limit,max(1,limit//2)) if limit>0 else 0
    for candidate in options[:association_limit]:
        pin=candidate['pin']; point=report['geometry_graph']['nodes'][pin]['point']
        tasks.append({'id':candidate['id'],'kind':'pin_wire_association','candidate':candidate,'pin_point':point,'wire':observations['wires'][candidate['wire']],'constraints':{'hard_rejection':rejected.get(candidate['id']),'requires_local_image_evidence':True,'allow_new_geometry':False},'allowed_decisions':['accept','reject']})
    for component,c in observations['components'].items():
        if len(tasks)>=limit: break
        field='Name' if c['type']=='ic' else 'value' if c['type'] in ('resistor','capacitor') else None
        if not field: continue
        box=c['bbox']; choices=[]
        for i,t in enumerate(observations.get('texts',[])):
            a=t['bbox']; gap=math.hypot(max(box[0]-a[2],a[0]-box[2],0),max(box[1]-a[3],a[1]-box[3],0))
            text=t['text']
            if gap>45: continue
            if field=='Name':
                if not re.search(r'[A-Za-z]',text) or not re.search(r'\d',text) or len(text)<4: continue
                if re.match(r'^(?:GPIO|IO|P[ABCD])\d',text,re.I) or re.fullmatch(r'\d+(?:\.\d+)?V',text,re.I): continue
            else:
                if not re.fullmatch(r'\d+(?:\.\d+)?[kKmMuUnNpPµμ]?(?:[FfΩQ]|[kKmM])',text): continue
                if c['type']=='resistor' and text.endswith(('F','f')): continue
                if c['type']=='capacitor' and text.endswith(('Ω','Q')): continue
            choices.append({'id':'text_'+str(i),'text':text,'bbox':a,'ocr_confidence':t.get('confidence'),'distance':gap})
        unique={choice['text']:choice for choice in sorted(choices,key=lambda x:-x['distance'])}
        choices=sorted(unique.values(),key=lambda x:x['distance'])[:4]
        if len(choices)<2 and c.get(field): continue
        if not choices: continue
        tasks.append({'id':component+'.'+field,'kind':'component_text_choice','component':component,'field':field,'current':c.get(field),'bbox':box,'pin_point':{'x':(box[0]+box[2])/2,'y':(box[1]+box[3])/2},'choices':choices,'allowed_decisions':['choose_text','reject'],'constraints':{'use_existing_ocr_only':True,'allow_new_geometry':False}})
    return {'framework':'GraphSchematic','observation_sha256':observation_hash(observations),'image_sha256':image_sha256,'tasks':tasks,'circuit_issues':report['circuit_issues'],'unresolved_pin_tips':observations.get('pin_hypotheses',[]),'topology_uncertainties':[d for d in observations.get('contact_decisions',[]) if d['state']=='uncertain'],'contract':'review existing associations/region-bound OCR choices only; preserve actual wires and semantic flying labels','llm_status':'queued_not_called'}


def validate_proposals(payload,queue):
    if not isinstance(payload,dict) or set(payload)-{'observation_sha256','corrections','usage'}: raise ValueError('invalid correction payload fields')
    if payload.get('observation_sha256')!=queue['observation_sha256']: raise ValueError('correction belongs to a different observation graph')
    corrections=payload.get('corrections')
    if not isinstance(corrections,list) or len(corrections)>len(queue['tasks']): raise ValueError('correction budget exceeded')
    known={task['id']:task for task in queue['tasks']}; seen=set(); accepted=[]; blocked=[]
    for proposal in corrections:
        if not isinstance(proposal,dict) or set(proposal)-{'candidate_id','decision','reason','text_option_id'}: raise ValueError('unsupported correction operation')
        candidate=proposal.get('candidate_id'); decision=proposal.get('decision')
        if not isinstance(candidate,str) or not isinstance(decision,str): raise ValueError('candidate_id and decision must be strings')
        if candidate not in known or candidate in seen or decision not in known[candidate]['allowed_decisions']: raise ValueError('unknown/duplicate correction candidate or decision')
        seen.add(candidate); task=known[candidate]
        if task['kind']=='component_text_choice':
            if decision=='choose_text' and (not isinstance(proposal.get('text_option_id'),str) or proposal['text_option_id'] not in {c['id'] for c in task['choices']}): raise ValueError('invented OCR option')
            continue
        option=task['candidate']
        if decision=='accept':
            if task['constraints']['hard_rejection'] or not option['topology_allowed'] or not option['image_evidence_available'] or option['image_support']<.55 or option['score']<.62:
                raise ValueError('LLM proposal violates image/topology/circuit evidence gate')
            accepted.append(candidate)
        else: blocked.append(candidate)
    return accepted,blocked


def apply_semantic_choices(observations,payload,queue):
    validate_proposals(payload,queue)
    result=copy.deepcopy(observations); tasks={t['id']:t for t in queue['tasks']}; edits=[]
    for proposal in payload['corrections']:
        task=tasks[proposal['candidate_id']]
        if task['kind']=='component_text_choice' and proposal['decision']=='choose_text':
            choice=next(c for c in task['choices'] if c['id']==proposal['text_option_id'])
            old=result['components'][task['component']][task['field']]
            result['components'][task['component']][task['field']]=choice['text']
            edits.append({'component':task['component'],'field':task['field'],'before':old,'after':choice['text'],'evidence':choice})
    return result,edits
