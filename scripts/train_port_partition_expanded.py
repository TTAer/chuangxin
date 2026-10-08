"""Actual training, fixed-threshold CNN control and structured graph model."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
import torch.nn.functional as F
from experiments.port_partition import RELATIONS, transitivity_violations
from experiments.port_partition_model import PortPartitionNet


def metrics(pred, truth, valid):
    def stats(mask):
        tp = int(((pred==1)&(truth==1)&mask).sum())
        fp = int(((pred==1)&(truth==0)&mask).sum())
        fn = int(((pred==0)&(truth==1)&mask).sum())
        return {'f1': 2*tp/max(1,2*tp+fp+fn), 'tp': tp, 'fp': fp, 'fn': fn,
                'relations': int(mask.sum())}
    mask = valid.astype(bool)
    cross = mask.copy(); cross[:, [0,5]] = False
    return {'all_pairs': stats(mask), 'cross_axis': stats(cross),
            'partial_partition_accuracy': float(np.all((pred==truth)|~mask,axis=1).mean()),
            'transitivity_violation_samples': sum(transitivity_violations(p)>0 for p in pred),
            'samples': len(pred)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=16)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--seed', type=int, default=20261006)
    parser.add_argument('--run-id', default='initial')
    parser.add_argument('--variants', default='cnn-pairs,port-graph')
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(args.seed); rng = np.random.default_rng(args.seed)
    source = ROOT/'data/processed/port-partition-expanded/relations.npz'
    data = np.load(source)
    x = torch.tensor(data['images'],dtype=torch.float32)/255
    y = torch.tensor(data['targets']); valid = torch.tensor(data['valid'])
    train = np.flatnonzero(data['split']==0); val = np.flatnonzero(data['split']==1)
    if not len(train) or not len(val):
        raise ValueError('Empty split')
    compatible = torch.tensor(np.all((RELATIONS[None]==data['targets'][:,None]) |
                                    ~data['valid'][:,None].astype(bool), axis=2))
    summaries = {}
    for variant in args.variants.split(','):
        if variant not in ('cnn-pairs','arm-pairs','partition-only','port-graph'):
            raise ValueError('Unknown variant')
        graph = variant != 'cnn-pairs'
        structured = variant in ('port-graph','partition-only')
        torch.manual_seed(args.seed)
        rng = np.random.default_rng(args.seed)
        out = ROOT/'output/training/port-partition-expanded'/args.run_id/variant
        dest = ROOT/'models/candidates/port-partition-expanded'/args.run_id/variant
        out.mkdir(parents=True,exist_ok=True); dest.mkdir(parents=True,exist_ok=True)
        if (out/'best.pt').exists():
            raise ValueError('Use a new run-id; do not overwrite an experiment')
        model = PortPartitionNet(graph,independent=variant=='arm-pairs',message_passing=variant!='partition-only').to(args.device)
        opt = torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
        history=[]; best=-1; started=time.perf_counter()
        for epoch in range(args.epochs):
            model.train(); losses=[]
            for ids in np.array_split(rng.permutation(train),max(1,int(np.ceil(len(train)/64)))):
                image=x[ids].to(args.device)
                # Grayscale train augmentation: color is a feature, never a label.
                if rng.random()<.5:
                    image=image.mean(1,keepdim=True).expand(-1,3,-1,-1)
                logits=model(image)
                if structured:
                    logp=logits.log_softmax(1)
                    loss=-torch.logsumexp(logp.masked_fill(~compatible[ids].to(args.device),-1e9),dim=1).mean()
                else:
                    raw=F.binary_cross_entropy_with_logits(logits,y[ids].to(args.device),reduction='none')
                    loss=(raw*valid[ids].to(args.device)).sum()/valid[ids].sum().to(args.device)
                opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),2);opt.step()
                losses.append(loss.item())
            model.eval()
            with torch.no_grad():
                logits=torch.cat([model(x[ids].to(args.device)).cpu() for ids in np.array_split(val,max(1,int(np.ceil(len(val)/128))))])
            pred=RELATIONS[logits.argmax(1).numpy()] if structured else (logits.numpy()>=0).astype(np.float32)
            result=metrics(pred,data['targets'][val],data['valid'][val])
            row={'epoch':epoch+1,'loss':float(np.mean(losses)), 'seconds':time.perf_counter()-started,**result}
            history.append(row)
            score=result['cross_axis']['f1']
            if score>best:
                best=score;torch.save(model.state_dict(),out/'best.pt')
                np.savez_compressed(out/'validation.npz',indices=val,prediction=pred,logits=logits.numpy())
                meta={'variant':variant,'best_epoch':row,'seed':args.seed,'epochs_planned':args.epochs,
                      'parameters':sum(p.numel() for p in model.parameters()),'device':args.device,
                      'message_passing':variant=='port-graph', 'structured_partition':structured,
                      'training_samples':len(train),'development_samples':len(val),
                      'data_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
                      'selection':'development cross-axis F1, first maximum; no threshold scan',
                      'architecture':'shared canonical arm encoder + two message passing rounds + 15-partition energy MAP' if structured else ('shared canonical arm encoder + independent pair heads' if graph else 'whole patch CNN + six unconstrained pair logits'),
                      'limitations':['weak wire-group labels','original validation fixed, training expanded with pixel-only candidates','no independent design-family test','no competition or end-to-end score'],
                      'runtime_modified':False,'Freeze_A_used':False}
                (out/'best.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
            (out/'history.json').write_text(json.dumps(history,indent=2),encoding='utf-8')
            print(json.dumps({'variant':variant,**row}),flush=True)
        model.load_state_dict(torch.load(out/'best.pt',weights_only=True));model.cpu().eval()
        torch.onnx.export(model,x[val[:1]],str(dest/'model.onnx'),input_names=['image'],
                          output_names=['logits'],opset_version=17,
                          dynamic_axes={'image':{0:'batch'},'logits':{0:'batch'}},dynamo=False)
        (dest/'model.json').write_bytes((out/'best.json').read_bytes())
        with torch.no_grad(): reference=model(x[val[:8]]).numpy()
        np.savez_compressed(out/'export-reference.npz',image=x[val[:8]].numpy(),logits=reference)
        summaries[variant]=json.loads((out/'best.json').read_text(encoding='utf-8'))
    (ROOT/'output/training/port-partition-expanded'/args.run_id/'comparison.json').write_text(json.dumps(summaries,indent=2),encoding='utf-8')


if __name__=='__main__':
    main()
