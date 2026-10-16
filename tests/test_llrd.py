import copy
import math
import unittest
import torch
from transformers import DebertaV2Config, DebertaV2Model
from mact.model import SequenceModel, ModelConfig
from mact import finetuning
from mact.train import resolve_config
from test_partial_freeze import tiny_deberta


class LLRDTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def groups(self, model, decay):
        self.assertTrue(callable(getattr(finetuning, 'optimizer_parameter_groups', None)))
        return finetuning.optimizer_parameter_groups(model, 2e-5, 1e-4, .01, decay)

    def test_llrd_covers_every_parameter_once_and_assigns_shared_weights(self):
        config = tiny_deberta().config
        config.num_hidden_layers = 12
        config.conv_kernel_size = 3
        model = SequenceModel(DebertaV2Model(config), ModelConfig(mode='text', hidden_size=8, attention_heads=2))
        groups, records = self.groups(model, .9)
        ids = [id(p) for g in groups for p in g['params']]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(set(ids), {id(p) for p in model.parameters()})
        by_name = {n:r for r in records for n in r['names']}
        for name, expected in [
            ('text_encoder.encoder.layer.11.attention.self.query_proj.weight', 2e-5),
            ('text_encoder.encoder.layer.0.attention.self.query_proj.weight', 6.276211921800002e-6),
            ('text_encoder.embeddings.word_embeddings.weight', 5.648590729620002e-6),
            ('text_encoder.encoder.rel_embeddings.weight', 5.648590729620002e-6),
            ('text_encoder.encoder.LayerNorm.weight', 5.648590729620002e-6),
            ('text_encoder.encoder.conv.conv.weight', 6.276211921800002e-6),
            ('text_head.weight', 1e-4), ('crf.transitions', 1e-4)]:
            self.assertAlmostEqual(by_name[name]['lr'], expected, places=15)
        for n,p in model.named_parameters():
            self.assertEqual(by_name[n]['weight_decay'], .01 if p.ndim >= 2 else 0.)
        self.assertEqual(sum(r['numel'] for r in records), sum(p.numel() for p in model.parameters()))

    def test_default_groups_and_optimizer_updates_match_original(self):
        model = SequenceModel(tiny_deberta(), ModelConfig(mode='text', hidden_size=8, attention_heads=2))
        other = copy.deepcopy(model)
        expected = {}
        for n,p in other.named_parameters():
            expected.setdefault((n.startswith('text_encoder.'), p.ndim >= 2), []).append(p)
        original = [{'params':ps,'lr':2e-5 if text else 1e-4,'weight_decay':.01 if wd else 0.}
                    for (text,wd),ps in expected.items()]
        actual, records = self.groups(model, 1.)
        self.assertEqual(len(actual), len(original))
        for a,b in zip(actual,original):
            self.assertEqual((a['lr'],a['weight_decay']), (b['lr'],b['weight_decay']))
            for p,q in zip(a['params'],b['params']): self.assertTrue(torch.equal(p,q))
        for m,gs in [(model,actual),(other,original)]:
            opt=torch.optim.AdamW(gs)
            sum(p.square().sum() for p in m.parameters()).backward(); opt.step()
        for a,b in zip(model.parameters(),other.parameters()): self.assertTrue(torch.equal(a,b))

    def test_unknown_text_parameter_and_invalid_config_fail(self):
        cfg=resolve_config({})
        self.assertEqual((cfg.get('layer_lr_decay'),cfg.get('ema_decay')), (1.,0.))
        for key,values in [('layer_lr_decay',[0,-1,1.1,True,float('nan'),float('inf')]),
                           ('ema_decay',[-1,1,True,float('nan'),float('inf')])]:
            for v in values:
                with self.assertRaises(ValueError): resolve_config({key:v})
        for extra in [dict(rdrop_weight=1),dict(text_freeze_layers=1),dict(ocr_cache='x'),
                      dict(model={'visual_gate':True}),dict(model={'label_scheme':'seven'})]:
            with self.assertRaises(ValueError):resolve_config(dict(ema_decay=.99,**extra))
        model=SequenceModel(tiny_deberta(), ModelConfig(mode='text',hidden_size=8,attention_heads=2))
        model.text_encoder.register_parameter('unknown', torch.nn.Parameter(torch.ones(1)))
        with self.assertRaisesRegex(ValueError,'unknown|unrecognized'):
            self.groups(model,.9)
