import copy
import tempfile
import unittest
from pathlib import Path

import torch

import test_data_model as fixtures
from mact.model import ModelConfig, SequenceModel
from mact.runtime import model_inputs, train_epoch, save_checkpoint, load_checkpoint
from mact.train import resolve_config


class VisualGateTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.fixture = fixtures.DataModelTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)

    def config(self, **overrides):
        self.assertIn('visual_gate', ModelConfig.__dataclass_fields__, 'C requires an opt-in gate')
        return ModelConfig(**dict(dict(hidden_size=8, attention_heads=2, visual_size=8,
                                       dropout=0., visual_dropout=0., visual_gate=True), **overrides))

    def model(self, **overrides):
        return SequenceModel(fixtures.tiny_text(), self.config(**overrides))

    def inputs(self, tags=False):
        return model_inputs(self.fixture.batch(), 'cpu', include_tags=tags)

    def baseline_for(self, model):
        baseline = SequenceModel(copy.deepcopy(model.text_encoder), self.config(visual_gate=False))
        baseline.load_state_dict({k:v for k,v in model.state_dict().items() if not k.startswith('visual_gate.')})
        return baseline.eval()

    def test_gate_endpoints_preserve_null_text_branch_and_original_fusion(self):
        model = self.model().eval()
        baseline = self.baseline_for(model)
        inputs = self.inputs()
        normal = baseline(**inputs)['emissions']
        null = baseline(**inputs, force_null=True)['emissions']
        self.assertGreater(float((normal-null).abs().max()), 0.)
        with torch.no_grad():
            model.visual_gate.weight.zero_()
            model.visual_gate.bias.fill_(-torch.inf)
        zero = model(**inputs)
        torch.testing.assert_close(zero['emissions'], null, rtol=0, atol=0)
        # Null must retain the existing text-dependent branch, not reduce to text_head.
        self.assertGreater(float((null-zero['text_emissions']).abs().max()), 0.)
        with torch.no_grad():
            model.visual_gate.bias.fill_(torch.inf)
        one = model(**inputs)
        torch.testing.assert_close(one['emissions'], normal, rtol=1e-6, atol=1e-6)
        self.assertEqual(model.crf.decode(one['emissions'],inputs['word_mask']),
                         baseline.crf.decode(normal,inputs['word_mask']))

    def test_initial_gate_interpolates_each_word_and_uses_only_crf_loss(self):
        model = self.model(visual_gate_init=.9).eval()
        inputs = self.inputs(tags=True)
        baseline = self.baseline_for(model)
        no_tags = {k:v for k,v in inputs.items() if k!='tags'}
        normal = baseline(**no_tags)['emissions']
        null = baseline(**no_tags,force_null=True)['emissions']
        output = model(**inputs)
        self.assertEqual(output['visual_gate'].shape, (2,2,1))
        torch.testing.assert_close(output['visual_gate'], torch.full((2,2,1),.9))
        torch.testing.assert_close(output['emissions'], .1*null+.9*normal, atol=1e-6,rtol=1e-6)
        torch.testing.assert_close(output['losses'],model.crf(output['emissions'],inputs['tags'],inputs['word_mask']))
        self.assertFalse(hasattr(model,'aspect_sentiment_head'))
        self.assertNotIn('aspect_sentiment_losses',output)

    def test_gate_and_both_modalities_receive_crf_gradients(self):
        model = self.model().train()
        output = model(**self.inputs(tags=True))
        output['losses'].mean().backward()
        for parameter in (model.visual_gate.weight, model.visual_gate.bias,
                          model.text_encoder.embeddings.word_embeddings.weight,
                          model.visual_projection[0].weight,model.null_evidence):
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(float(parameter.grad.abs().sum()),0.)

    def test_forced_null_and_visual_dropout_cancel_gate_effect(self):
        model = self.model().eval()
        inputs = self.inputs(tags=True)
        output = model(**inputs,force_null=True)
        torch.testing.assert_close(output['emissions'],output['null_emissions'],rtol=0,atol=0)
        output['losses'].sum().backward()
        self.assertEqual(float(model.visual_gate.weight.grad.abs().sum()),0.)
        self.assertEqual(float(model.visual_gate.bias.grad.abs().sum()),0.)
        model.train();model.config.visual_dropout=1.
        output = model(**inputs)
        torch.testing.assert_close(output['emissions'],output['null_emissions'],rtol=0,atol=0)

    def test_gold_changes_loss_but_not_gate_or_emissions(self):
        model = self.model().eval()
        inputs = self.inputs(tags=True)
        a = model(**inputs)
        inputs['tags'] = torch.zeros_like(inputs['tags'])
        b = model(**inputs)
        for key in ('visual_gate','emissions'):
            torch.testing.assert_close(a[key],b[key],rtol=0,atol=0)
        self.assertFalse(torch.equal(a['losses'],b['losses']))
        inputs.pop('tags')
        inference = model(**inputs)
        torch.testing.assert_close(a['emissions'],inference['emissions'],rtol=0,atol=0)

    def test_disabled_and_old_checkpoints_have_no_gate_parameters(self):
        self.config()
        cfg=dict(hidden_size=8,attention_heads=2,visual_size=8,dropout=0.,visual_dropout=0.)
        torch.manual_seed(42)
        default=SequenceModel(fixtures.tiny_text(),ModelConfig(**cfg))
        rng=torch.get_rng_state().clone()
        torch.manual_seed(42)
        disabled=self.model(visual_gate=False)
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        self.assertFalse(hasattr(disabled,'visual_gate'))
        inputs=self.inputs(tags=True)
        for key,value in default(**inputs).items():
            torch.testing.assert_close(value,disabled(**inputs)[key],rtol=0,atol=0)
        path=Path(self.fixture.tmp.name)/'old.pt'
        save_checkpoint(path,default,{},1,{})
        data=torch.load(path,weights_only=True)
        for key in ('visual_gate','visual_gate_init'):
            data['model_config'].pop(key)
        torch.save(data,path)
        loaded,_=load_checkpoint(path)
        for key,value in default(**inputs).items():
            torch.testing.assert_close(value,loaded(**inputs)[key],rtol=0,atol=0)

    def test_gate_checkpoint_roundtrip(self):
        model=self.model().eval()
        with torch.no_grad():model.visual_gate.weight.normal_()
        path=Path(self.fixture.tmp.name)/'c.pt'
        save_checkpoint(path,model,{},1,{})
        loaded,_=load_checkpoint(path)
        inputs=self.inputs(tags=True)
        for key,value in model(**inputs).items():
            torch.testing.assert_close(value,loaded(**inputs)[key],rtol=0,atol=0)

    def test_epoch_gate_statistics_exclude_padding(self):
        model=self.model(visual_gate_init=.8)
        batch=self.fixture.batch()
        optimizer=torch.optim.SGD(model.parameters(),lr=0.)
        result=train_epoch(model,[batch],optimizer,'cpu',log_every=0)
        stats=result['visual_gate']
        self.assertEqual(stats['words'],3)
        self.assertAlmostEqual(stats['mean'],.8,places=6)
        self.assertAlmostEqual(stats['std'],0.,places=6)
        self.assertEqual(stats['fraction_below_0_1'],0.)
        self.assertEqual(stats['fraction_above_0_9'],0.)

    def test_invalid_gate_configuration_rejected(self):
        self.config()
        for cfg in ({'visual_gate':1},{'visual_gate':'true'},
                    {'visual_gate':True,'mode':'text'},{'visual_gate':True,'mode':'evidence'},
                    *({'visual_gate_init':x} for x in (0.,1.,float('nan'),float('inf'),True,'0.9'))):
            with self.subTest(cfg=cfg),self.assertRaises(ValueError):
                resolve_config({'model':cfg})


if __name__=='__main__':unittest.main()
