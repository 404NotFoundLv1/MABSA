"""Independent OCR emissions over an immutable five-tag MACT checkpoint."""
import itertools
import math
from pathlib import Path

import torch
from torch import nn

from .common.cache import file_hash
from .data import SequenceCollator
from .runtime import load_checkpoint, model_inputs


class OCRDataset:
    """Keep original inputs intact; OCR is encoded in a separate sequence."""
    def __init__(self, base, tokenizer, texts, max_tokens=64):
        if type(max_tokens) is not int or not 1 <= max_tokens <= 510:
            raise ValueError('OCR content budget must be in [1,510]')
        self.base, self.ocr = base, []
        for i in range(len(base)):
            example = base[i]['example']
            if example.image_id not in texts or not isinstance(texts[example.image_id],str):
                raise ValueError('missing or invalid OCR text')
            content = tokenizer(texts[example.image_id],add_special_tokens=False)['input_ids'][:max_tokens]
            ids = tokenizer.build_inputs_with_special_tokens(content)
            layout = tokenizer.build_inputs_with_special_tokens([-1]*len(content))
            self.ocr.append(dict(ocr_input_ids=torch.tensor(ids,dtype=torch.long),
                ocr_attention_mask=torch.ones(len(ids),dtype=torch.long),
                ocr_content_mask=torch.tensor([i==-1 for i in layout],dtype=torch.bool)))

    def __len__(self):
        return len(self.base)

    def __getitem__(self,index):
        return dict(self.base[index],**{k:v.clone() for k,v in self.ocr[index].items()})


class OCRCollator(SequenceCollator):
    def __call__(self,items):
        batch = super().__call__(items)
        length = max(len(x['ocr_input_ids']) for x in items)
        for key in ('ocr_input_ids','ocr_attention_mask','ocr_content_mask'):
            fill = self.pad_token_id if key=='ocr_input_ids' else 0
            batch[key] = torch.full((len(items),length),fill,dtype=items[0][key].dtype)
            for i,item in enumerate(items):
                batch[key][i,:len(item[key])] = item[key]
        return batch


class OCRBranch(nn.Module):
    def __init__(self,text_size,hidden_size,heads):
        super().__init__()
        self.projection = nn.Sequential(nn.Linear(text_size,hidden_size),nn.LayerNorm(hidden_size))
        self.attention = nn.MultiheadAttention(hidden_size,heads,dropout=0.,batch_first=True)
        self.output = nn.Linear(hidden_size,5,bias=False)
        nn.init.zeros_(self.output.weight)

    def forward(self,words,ocr,content_mask):
        memory = self.projection(ocr)
        context = self.attention(words,memory,memory,key_padding_mask=~content_mask,need_weights=False)[0]
        return self.output(context)


class OCRResidualModel(nn.Module):
    def __init__(self,baseline,heads=4,memory_source='ocr',standalone_weight=0.,memory_max_tokens=64):
        super().__init__()
        if memory_source not in ('ocr','text'):
            raise ValueError('memory_source must be ocr or text')
        if (type(standalone_weight) not in (int,float) or not math.isfinite(standalone_weight)
                or standalone_weight<0):
            raise ValueError('standalone_weight must be finite and nonnegative')
        if type(memory_max_tokens) is not int or not 1<=memory_max_tokens<=510:
            raise ValueError('memory_max_tokens must be in [1,510]')
        cfg = baseline.config
        if (cfg.mode!='fusion' or cfg.label_scheme!='five' or cfg.visual_gate
                or cfg.word_context or cfg.aspect_sentiment_weight):
            raise ValueError('OCR residual requires an unmodified five-tag fusion baseline')
        self.baseline = baseline.requires_grad_(False).eval()
        self.config, self.heads = cfg, heads
        self.memory_source,self.standalone_weight,self.memory_max_tokens = memory_source,standalone_weight,memory_max_tokens
        self.branch = OCRBranch(baseline.text_encoder.config.hidden_size,cfg.hidden_size,heads)

    @property
    def crf(self):
        return self.baseline.crf

    def train(self,mode=True):
        super().train(mode)
        self.baseline.eval()
        return self

    def forward(self,input_ids,attention_mask,word_ids,word_mask,visual_features,
                ocr_input_ids,ocr_attention_mask,ocr_content_mask,tags=None,force_null=False):
        with torch.no_grad():
            base = self.baseline(input_ids,attention_mask,word_ids,word_mask,
                                 visual_features=visual_features,force_null=force_null,return_words=True)
        active = ocr_content_mask.any(dim=1)
        emissions = base['emissions'].clone()
        if active.any():
            with torch.no_grad():
                if self.memory_source=='ocr':
                    memory = self.baseline.text_encoder(input_ids=ocr_input_ids[active],
                        attention_mask=ocr_attention_mask[active]).last_hidden_state
                    content_mask = ocr_content_mask[active]
                else:
                    # Match capacity/availability; never read OCR token content in this control.
                    memory = self.baseline.text_encoder(input_ids=input_ids[active],
                        attention_mask=attention_mask[active]).last_hidden_state
                    content_mask = (word_ids[active]>=0)&attention_mask[active].bool()
                    content_mask = content_mask & (content_mask.long().cumsum(1)<=self.memory_max_tokens)
            residual = self.branch(base['words'][active],memory,content_mask)
            emissions[active] = emissions[active] + residual * word_mask[active].unsqueeze(-1)
        output = dict(emissions=emissions,base_emissions=base['emissions'],has_ocr=active)
        if tags is not None:
            joint = self.crf(emissions,tags,word_mask)
            standalone = joint.new_zeros(joint.shape)
            if self.standalone_weight and active.any():
                standalone[active] = self.crf(residual,tags[active],word_mask[active])
            output.update(joint_losses=joint,standalone_losses=standalone,
                losses=joint+self.standalone_weight*standalone if self.standalone_weight else joint)
        return output


def residual_train_epoch(model,loader,optimizer,device,accumulation=8,gradient_clip=1.,scheduler=None):
    if accumulation<1:
        raise ValueError('accumulation must be positive')
    model.train()
    iterator = iter(loader)
    examples = updates = windows = active_examples = 0
    loss_sum = joint_sum = standalone_sum = 0.
    while window := list(itertools.islice(iterator,accumulation)):
        count = sum(len(b['input_ids']) for b in window)
        optimizer.zero_grad(set_to_none=True)
        for batch in window:
            output = model(**model_inputs(batch,device,include_tags=True))
            losses = output['losses']
            if not torch.isfinite(losses).all():
                raise FloatingPointError('nonfinite OCR residual loss')
            if losses.requires_grad:
                (losses.sum()/count).backward()
            loss_sum += float(losses.detach().sum())
            joint_sum += float(output['joint_losses'].detach().sum())
            standalone_sum += float(output['standalone_losses'].detach().sum())
            active_examples += int(output['has_ocr'].sum())
        parameters = [p for p in model.branch.parameters() if p.grad is not None]
        if parameters:
            nn.utils.clip_grad_norm_(parameters,gradient_clip,error_if_nonfinite=True)
            optimizer.step()
            updates += 1
        # Schedule is explicitly defined over all data windows, including empty ones.
        if scheduler is not None:
            scheduler.step()
        examples += count
        windows += 1
    return dict(sentence_nll=joint_sum/examples if examples else 0.,
                standalone_nll=standalone_sum/examples if examples else 0.,
                objective_loss=loss_sum/examples if examples else 0.,examples=examples,
                has_ocr_examples=active_examples,updates=updates,windows=windows)


def save_residual(path,model,baseline_path,run_config,epoch,metrics):
    content = dict(format='mact_ocr_residual_v1',baseline_path=str(Path(baseline_path).resolve()),
        baseline_sha256=file_hash(baseline_path),heads=model.heads,run_config=run_config,
        memory_source=model.memory_source,standalone_weight=model.standalone_weight,
        memory_max_tokens=model.memory_max_tokens,
        epoch=epoch,dev_metrics=metrics,
        branch_state={k:v.detach().cpu() for k,v in model.branch.state_dict().items()})
    path=Path(path)
    temporary=path.with_suffix('.tmp')
    torch.save(content,temporary)
    temporary.replace(path)


def load_residual(path,device='cpu'):
    content = torch.load(path,map_location='cpu',weights_only=True)
    if content.get('format')!='mact_ocr_residual_v1':
        raise ValueError('unsupported OCR residual checkpoint')
    if file_hash(content['baseline_path'])!=content['baseline_sha256']:
        raise ValueError('baseline checkpoint hash mismatch')
    baseline,metadata = load_checkpoint(content['baseline_path'])
    if metadata['run_config'].get('ocr_cache'):
        raise ValueError('concatenated OCR checkpoint is not a baseline')
    model = OCRResidualModel(baseline,heads=content['heads'],memory_source=content.get('memory_source','ocr'),
        standalone_weight=content.get('standalone_weight',0.),memory_max_tokens=content.get('memory_max_tokens',64))
    model.branch.load_state_dict(content.pop('branch_state'),strict=True)
    return model.to(device).eval(),content
