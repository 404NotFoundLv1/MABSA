import copy
import json
import unittest
from pathlib import Path

import torch
from transformers import DebertaV2Config, DebertaV2Model

import test_data_model as fixtures
import mact.train as training
from mact.model import SequenceModel, ModelConfig
from mact.runtime import load_checkpoint


def tiny_deberta():
    return DebertaV2Model(DebertaV2Config(vocab_size=12, hidden_size=8,
        num_hidden_layers=2, num_attention_heads=2, intermediate_size=16,
        max_position_embeddings=32, relative_attention=True, position_buckets=8,
        norm_rel_ebd='layer_norm', pos_att_type=['p2c','c2p'],
        hidden_dropout_prob=.1, attention_probs_dropout_prob=.1))


class PartialFreezeTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_freeze_config_defaults_and_rejects_invalid_or_combined_treatments(self):
        self.assertEqual(training.resolve_config({}).get('text_freeze_layers'), 0)
        for value in (-1, True, 1.5, '6'):
            with self.assertRaises(ValueError):
                training.resolve_config(dict(text_freeze_layers=value))
        self.assertEqual(training.resolve_config(dict(text_freeze_layers=1))['text_freeze_layers'], 1)
        for change in (dict(rdrop_weight=.1), dict(model=dict(aspect_sentiment_weight=.1)),
                       dict(model=dict(label_scheme='seven'))):
            with self.assertRaises(ValueError):
                training.resolve_config(dict(text_freeze_layers=1, **change))

    def test_frozen_bottom_and_shared_position_weights_survive_real_optimizer_step(self):
        self.assertTrue(callable(getattr(training, 'freeze_text_layers', None)))
        model=SequenceModel(tiny_deberta(), ModelConfig(mode='text',hidden_size=8,attention_heads=2))
        before={n:p.detach().clone() for n,p in model.named_parameters()}
        rng=torch.get_rng_state().clone()
        policy=training.freeze_text_layers(model,1)
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        model.train()
        self.assertTrue(model.text_encoder.embeddings.dropout.training)
        self.assertTrue(model.text_encoder.encoder.layer[0].training)
        trainable=[p for p in model.parameters() if p.requires_grad]
        optimizer=torch.optim.AdamW(trainable,lr=.01,weight_decay=.1)
        x=dict(input_ids=torch.tensor([[2,5,6,3]]),attention_mask=torch.ones(1,4,dtype=torch.long),
            word_ids=torch.tensor([[-1,0,1,-1]]),word_mask=torch.ones(1,2,dtype=torch.bool),
            tags=torch.tensor([[0,1]]))
        model(**x)['losses'].sum().backward()
        optimizer.step()
        for module in (model.text_encoder.embeddings,model.text_encoder.encoder.layer[0],
                       model.text_encoder.encoder.rel_embeddings,model.text_encoder.encoder.LayerNorm):
            self.assertTrue(all(not p.requires_grad and p.grad is None for p in module.parameters()))
        for name in policy['frozen_names']:
            self.assertTrue(torch.equal(before[name],dict(model.named_parameters())[name]))
        for name in ('text_encoder.encoder.layer.1.attention.self.query_proj.weight','text_head.weight'):
            self.assertFalse(torch.equal(before[name],dict(model.named_parameters())[name]))

    def test_zero_is_noop_and_out_of_range_fails_without_mutation(self):
        self.assertTrue(callable(getattr(training, 'freeze_text_layers', None)))
        model=SequenceModel(tiny_deberta(), ModelConfig(mode='text',hidden_size=8,attention_heads=2)).eval()
        before=copy.deepcopy(model.state_dict());rng=torch.get_rng_state().clone()
        self.assertIsNone(training.freeze_text_layers(model,0))
        self.assertTrue(all(p.requires_grad for p in model.parameters()))
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        for n,p in model.state_dict().items():self.assertTrue(torch.equal(p,before[n]))
        for bad in (2,3,-1,True):
            with self.assertRaises(ValueError):training.freeze_text_layers(model,bad)
        self.assertTrue(all(p.requires_grad for p in model.parameters()))

    def test_training_entrypoint_applies_freeze_and_records_integrity(self):
        fixture=fixtures.DataModelTests();fixture.setUp();self.addCleanup(fixture.doCleanups)
        root=Path(fixture.tmp.name);backbone=root/'backbone'
        original=tiny_deberta();original.save_pretrained(backbone);fixture.tokenizer.save_pretrained(backbone)
        for split in ('train','dev'):
            (root/f'{split}.json').write_text(json.dumps([dict(id=split,words=['good','washington'],
                image_id='unused.jpg',aspects=[dict(**{'from':1,'to':2},polarity='positive')])]))
        cfg=dict(text_model=str(backbone),train_data=str(root/'train.json'),dev_data=str(root/'dev.json'),
            text_freeze_layers=1,device='cpu',num_threads=1,epochs=1,log_every=0,
            model=dict(mode='text',hidden_size=8,attention_heads=2))
        summary=training.run_training(cfg,root/'run')
        self.assertTrue(summary['text_freeze']['frozen_exact'])
        loaded,meta=load_checkpoint(root/'run/last.pt')
        self.assertEqual(meta['run_config']['text_freeze_layers'],1)
        for prefix in ('embeddings.','encoder.layer.0.','encoder.rel_embeddings.','encoder.LayerNorm.'):
            for n,p in original.state_dict().items():
                if n.startswith(prefix):self.assertTrue(torch.equal(p,loaded.text_encoder.state_dict()[n]))
        self.assertFalse(torch.equal(original.encoder.layer[1].attention.self.query_proj.weight,
                                     loaded.text_encoder.encoder.layer[1].attention.self.query_proj.weight))


if __name__=='__main__':unittest.main()
