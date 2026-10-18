"""Frozen target/aspect features and sentiment-only residual correction."""
from dataclasses import replace

import torch
from torch import nn
from torch.utils.data import DataLoader

from .calibrated_fusion import fuse_labels, aspect_polarity_logits
from .data import SequenceCollator
from .labels import tags_to_aspects
from .runtime import model_inputs
from .target_context import TargetDataset, TargetCollator
from .train import make_dataset


def anchor_scores(base_logits, target_logits):
    if (base_logits.ndim!=2 or base_logits.shape[1]!=3 or base_logits.shape!=target_logits.shape
            or not torch.isfinite(base_logits).all() or not torch.isfinite(target_logits).all()):
        raise ValueError('finite matching [N,3] expert scores required')
    probability=.5*base_logits.detach().double().softmax(-1)+.5*target_logits.detach().double().softmax(-1)
    return probability.clamp_min(1e-300).log()


def corrected_labels(scores, fixed_labels):
    if (scores.ndim!=2 or scores.shape[1]!=3 or fixed_labels.shape!=(len(scores),)
            or fixed_labels.dtype!=torch.long or not torch.isfinite(scores).all()
            or ((fixed_labels<0)|(fixed_labels>2)).any()):
        raise ValueError('invalid corrected scores or fallback labels')
    if not len(scores):return fixed_labels.clone()
    labels=scores.argmax(-1)
    ties=scores[torch.arange(len(scores),device=scores.device),fixed_labels]==scores.max(-1).values
    labels[ties]=fixed_labels[ties]
    return labels


class CorrectionHead(nn.Module):
    def __init__(self,base_dim,target_dim,variant,hidden=128,dropout=.1):
        super().__init__()
        if variant not in ('plain','bounded') or min(base_dim,target_dim,hidden)<1 or not 0<=dropout<1:
            raise ValueError('invalid correction configuration')
        self.variant=variant
        self.base_norm=nn.LayerNorm(base_dim)
        self.target_norm=nn.LayerNorm(target_dim)
        self.hidden=nn.Sequential(nn.Linear(base_dim+target_dim,hidden),nn.GELU(),nn.Dropout(dropout))
        self.output=nn.Linear(hidden,3)
        nn.init.zeros_(self.output.weight);nn.init.zeros_(self.output.bias)

    def forward(self,base_features,target_features,base_logits,target_logits):
        if (base_features.shape!=(len(base_logits),self.base_norm.normalized_shape[0])
                or target_features.shape!=(len(base_logits),self.target_norm.normalized_shape[0])
                or not torch.isfinite(base_features).all() or not torch.isfinite(target_features).all()):
            raise ValueError('invalid feature matrices')
        anchor=anchor_scores(base_logits,target_logits)
        features=torch.cat([self.base_norm(base_features.detach()),self.target_norm(target_features.detach())],-1)
        raw=self.output(self.hidden(features))
        delta=2*torch.tanh(raw/2) if self.variant=='bounded' else raw
        return dict(scores=anchor+delta.double(),delta=delta,anchor=anchor)


@torch.no_grad()
def extract_features(base_model,target_model,examples,base_tokenizer,target_tokenizer,base_config,features,device):
    if (base_model.config.mode!='fusion' or base_model.config.label_scheme!='five'
            or base_model.config.word_context or base_model.config.visual_gate):
        raise ValueError('feature extraction requires original five-tag fusion MACT')
    if len({e.id for e in examples})!=len(examples):raise ValueError('duplicate example IDs')
    gold={e.id:e.aspects for e in examples}
    clean=[replace(e,aspects=[],ignored_spans=[]) for e in examples]
    base_model.eval();target_model.eval()
    for model in [base_model,target_model]:model.requires_grad_(False)
    loader=DataLoader(make_dataset(clean,base_tokenizer,base_config,features),batch_size=1,
                      collate_fn=SequenceCollator(base_tokenizer.pad_token_id))
    captured=[]
    hook=base_model.visual_head[1].register_forward_hook(lambda module,args,out:captured.append(out.detach()))
    base_features,base_logits,keys,baseline=[],[],[],{}
    try:
        for batch in loader:
            captured.clear()
            inputs=model_inputs(batch,device)
            if 'tags' in inputs:raise ValueError('gold tags reached inference')
            emissions=base_model(**inputs)['emissions']
            if len(captured)!=1:raise ValueError('ambiguous visual feature hook')
            decoded=base_model.crf.decode(emissions,inputs['word_mask'])
            for i,(ex,tags) in enumerate(zip(batch['examples'],decoded)):
                spans=tags_to_aspects(tags);baseline[ex.id]=spans
                base_logits.extend(aspect_polarity_logits(base_model.crf,emissions[i,:len(ex.words)],spans).cpu())
                for start,end,_ in spans:
                    keys.append((ex.id,start,end))
                    base_features.append(captured[0][i,start:end].mean(0).cpu())
    finally:
        hook.remove()
    ds=TargetDataset(clean,baseline,target_tokenizer,False,258)
    if ds.keys!=keys:raise ValueError('target aspect ordering changed')
    loader=DataLoader(ds,batch_size=8,collate_fn=TargetCollator(target_tokenizer.pad_token_id))
    target_features,target_logits=[],[]
    for batch in loader:
        if 'labels' in batch:raise ValueError('gold labels reached inference')
        batch={k:v.to(device) for k,v in batch.items()}
        hidden=target_model.encoder(input_ids=batch['input_ids'],attention_mask=batch['attention_mask']).last_hidden_state
        vector=hidden[torch.arange(len(hidden),device=hidden.device),batch['target_index']]
        logits=target_model.classifier(target_model.dropout(vector))
        target_features.extend(vector.cpu());target_logits.extend(logits.cpu())
    def stack(values,dim):return torch.stack(values) if values else torch.empty(0,dim)
    b,t=stack(base_logits,3),stack(target_logits,3)
    old=torch.tensor([p for spans in baseline.values() for _,_,p in spans],dtype=torch.long)
    new=t.argmax(-1)
    fixed=fuse_labels(b,t,old,new,1.,1.,.5)
    matched={(sid,s,e):p for sid,spans in gold.items() for s,e,p in spans}
    return dict(keys=keys,gold=gold,baseline=baseline,
        base_features=stack(base_features,base_model.config.hidden_size),
        target_features=stack(target_features,target_model.encoder.config.hidden_size),
        base_logits=b,target_logits=t,base_labels=old,target_labels=new,fixed_labels=fixed,
        labels=torch.tensor([matched.get(k,-100) for k in keys],dtype=torch.long))
