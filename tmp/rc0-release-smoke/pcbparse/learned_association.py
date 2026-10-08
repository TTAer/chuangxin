"""Optional learned association veto; image/topology/circuit gates remain hard."""
import json
from pathlib import Path
import numpy as np
from .association_features import candidate_graph

class LearnedAssociation:
    def __init__(self,path,threads=2):
        import onnxruntime as ort
        self.path = str(path)
        self.metadata = json.loads(Path(path).with_suffix('.json').read_text(encoding='utf-8'))
        options = ort.SessionOptions(); options.intra_op_num_threads = threads
        self.session = ort.InferenceSession(str(path),sess_options=options,providers=['CPUExecutionProvider'])
        self.input_names={i.name for i in self.session.get_inputs()}
    def filter(self,graph,candidates):
        report = {'module':'ElectricalAssociationNet','model':self.path,'policy':'learned/heuristic soft gain within hard visual/topology/circuit gates; supported high-precision negative veto only','blocked':[],'official_equivalence':False}
        if not candidates: report['status']='no_candidates'; return candidates,report
        try:
            tensors = candidate_graph(graph,candidates)
            logits = self.session.run(None,{k:v for k,v in tensors.items() if k in self.input_names})[0]
            if len(logits)!=len(candidates) or not np.isfinite(logits).all(): raise ValueError('invalid association probabilities')
        except Exception as error:
            report.update(status='local_heuristic_fallback',error_type=type(error).__name__)
            return candidates,report
        probabilities = 1/(1+np.exp(-np.clip(logits,-40,40)))
        threshold = self.metadata['negative_threshold']
        kept=[]
        for candidate,prob in zip(candidates,probabilities):
            if float(prob)<=threshold:
                report['blocked'].append({'candidate':candidate['id'],'probability':float(prob),'reason':'learned_nonconnection'})
            else:
                # Keep original score for the hard evidence gate. Only change
                # the soft beam objective; uncertain beam choices still abstain.
                option=dict(candidate)
                option['association_probability']=float(prob)
                option['association_gain']=.5*(candidate['score']-.62)+.5*(float(prob)-.5)
                kept.append(option)
        report.update(status='evaluated',candidate_probabilities={c['id']:float(p) for c,p in zip(candidates,probabilities)})
        return kept,report
