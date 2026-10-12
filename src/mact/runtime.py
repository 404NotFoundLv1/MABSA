"""Sentence-mean training, exact tuple evaluation, and portable checkpoints."""

import itertools
import json
from dataclasses import asdict
from pathlib import Path

import torch

from .common.backbones import text_from_config
from .common.metrics import evaluate_predictions
from .labels import TAGS, tags_to_aspects
from .model import SequenceModel, ModelConfig


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    temporary.replace(path)


def model_inputs(batch, device, include_tags=False):
    keys = ['input_ids', 'attention_mask', 'word_ids', 'word_mask', 'visual_features']
    if include_tags:
        keys.append('tags')
    return {key: batch[key].to(device) for key in keys if key in batch}


def train_epoch(model, loader, optimizer, device, accumulation=8, gradient_clip=1.,
                scheduler=None, max_updates=None, log_every=50, logger=None, epoch=1, epochs=1):
    if accumulation < 1:
        raise ValueError('accumulation must be positive')
    model.train()
    iterator = iter(loader)
    updates, examples, total_loss = 0, 0, 0.
    total_nll, total_aspect_ce, auxiliary = 0., 0., False
    batch_step = 0
    while window := list(itertools.islice(iterator, accumulation)):
        count = sum(len(b['input_ids']) for b in window)
        optimizer.zero_grad(set_to_none=True)
        for batch in window:
            output = model(**model_inputs(batch, device, include_tags=True))
            losses = output['losses']
            if not torch.isfinite(losses).all():
                raise FloatingPointError('nonfinite training loss; no update performed')
            batch_step += 1
            if logger is not None and log_every and (batch_step % log_every == 0 or batch_step == len(loader)):
                logger.info(f'Epoch [{epoch}/{epochs}], Step [{batch_step}/{len(loader)}], '
                            f'Loss: {float(losses.detach().mean()):.4f}')
            (losses.sum() / count).backward()
            total_loss += float(losses.detach().sum())
            if 'aspect_sentiment_losses' in output:
                auxiliary = True
                total_nll += float(output['crf_losses'].detach().sum())
                total_aspect_ce += float(output['aspect_sentiment_losses'].detach().sum())
        parameters = [p for p in model.parameters() if p.grad is not None]
        torch.nn.utils.clip_grad_norm_(parameters, gradient_clip or float('inf'), error_if_nonfinite=True)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        updates += 1
        examples += count
        if logger is None and log_every and updates % log_every == 0:
            print(json.dumps({'event': 'train_progress', 'updates': updates, 'examples': examples,
                              ('mean_sentence_loss' if auxiliary else 'mean_sentence_nll'): total_loss / examples}), flush=True)
        if max_updates is not None and updates >= max_updates:
            break
    result = {'sentence_nll': total_loss / examples if examples else 0., 'examples': examples, 'updates': updates}
    if auxiliary:
        result.update(sentence_loss=total_loss / examples, sentence_nll=total_nll / examples,
                      aspect_sentiment_loss=total_aspect_ce / examples)
    return result


@torch.no_grad()
def evaluate_model(model, loader, device, force_null=False):
    model.eval()
    gold, predictions = {}, {}
    for batch in loader:
        inputs = model_inputs(batch, device)
        output = model(**inputs, force_null=force_null)
        paths = model.crf.decode(output['emissions'], inputs['word_mask'])
        for example, tags in zip(batch['examples'], paths):
            if example.id in gold:
                raise ValueError('duplicate example ID in evaluation')
            gold[example.id] = example.aspects
            predictions[example.id] = tags_to_aspects(tags)
    return evaluate_predictions(gold, predictions), predictions


def save_checkpoint(path, model, run_config, epoch, metrics):
    content = {
        'format': 'mabsa_sequence_crf_v1', 'tags': list(TAGS),
        'model_config': asdict(model.config), 'run_config': run_config,
        'text_config': model.text_encoder.config.to_dict(),
        'text_attention': model.text_encoder.config._attn_implementation,
        'state_dict': {k: v.detach().cpu() for k, v in model.state_dict().items()},
        'epoch': epoch, 'dev_metrics': metrics,
    }
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    torch.save(content, temporary)
    temporary.replace(path)


def load_checkpoint(path, device='cpu'):
    content = torch.load(path, map_location='cpu', weights_only=True)
    if content.get('format') != 'mabsa_sequence_crf_v1' or content.get('tags') != list(TAGS):
        raise ValueError('unsupported checkpoint or tag schema')
    text = text_from_config(content['text_config'], content.get('text_attention', 'eager'))
    model = SequenceModel(text, ModelConfig(**content['model_config']))
    model.load_state_dict(content.pop('state_dict'), strict=True)
    return model.to(device).eval(), content
