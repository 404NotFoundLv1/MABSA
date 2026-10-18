import importlib.util
import unittest
from dataclasses import replace

import torch
import test_data_model as fixtures
from mact.common.data import Example
from mact.model import SequenceModel, ModelConfig
from mact.target_context import TargetClassifier, add_markers


class SentimentCorrectionTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('mact.sentiment_correction'), 'correction implementation missing')
        from mact import sentiment_correction
        self.c=sentiment_correction
        torch.set_num_threads(1)

    def test_zero_identity_ties_and_no_encoder_gradients(self):
        for variant in ['plain','bounded']:
            head=self.c.CorrectionHead(4,6,variant,hidden=8,dropout=0.)
            base=torch.randn(3,4,requires_grad=True);target=torch.randn(3,6,requires_grad=True)
            z=torch.tensor([[0.,0.,0.],[1.,2.,3.],[3.,2.,1.]])
            out=head(base,target,z,z)
            self.assertTrue(torch.equal(out['delta'],torch.zeros(3,3)))
            self.assertEqual(self.c.corrected_labels(out['scores'],torch.tensor([2,2,0])).tolist(),[2,2,0])
            torch.nn.functional.cross_entropy(out['scores'],torch.tensor([0,1,2])).backward()
            self.assertIsNone(base.grad);self.assertIsNone(target.grad)
            self.assertGreater(float(head.output.weight.grad.abs().sum()),0)

    def test_bounded_correction_can_override_shared_wrong_label(self):
        head=self.c.CorrectionHead(4,6,'bounded',hidden=8,dropout=0.).eval()
        with torch.no_grad():head.output.bias.copy_(torch.tensor([0.,0.,3.]))
        z=torch.tensor([[.45,.40,.15]]).log()
        out=head(torch.zeros(1,4),torch.zeros(1,6),z,z)
        self.assertLessEqual(float(out['delta'].abs().max()),2.)
        self.assertEqual(self.c.corrected_labels(out['scores'],torch.tensor([0])).tolist(),[2])
        with self.assertRaises(ValueError):head(torch.zeros(1,4),torch.zeros(1,6),z*float('nan'),z)

    def test_empty_aspects_and_invalid_variant(self):
        head=self.c.CorrectionHead(4,6,'plain',hidden=8)
        out=head(torch.empty(0,4),torch.empty(0,6),torch.empty(0,3),torch.empty(0,3))
        self.assertEqual(self.c.corrected_labels(out['scores'],torch.empty(0,dtype=torch.long)).shape,(0,))
        with self.assertRaises(ValueError):self.c.CorrectionHead(4,6,'unknown')

    def test_real_multimodal_extraction_is_gold_independent_and_cleans_hooks(self):
        helper=fixtures.DataModelTests();helper.setUp();self.addCleanup(helper.doCleanups)
        base=SequenceModel(fixtures.tiny_text(),ModelConfig(mode='fusion',hidden_size=8,
            attention_heads=2,visual_size=8,dropout=0.,visual_dropout=0.)).eval()
        with torch.no_grad():
            base.text_head.weight.zero_();base.text_head.bias.copy_(torch.tensor([-5.,5.,-5.,-5.,-5.]))
            base.visual_head[-1].weight.zero_();base.visual_head[-1].bias.zero_()
        add_markers(helper.tokenizer)
        enc=fixtures.tiny_text();enc.resize_token_embeddings(len(helper.tokenizer),mean_resizing=False)
        target=TargetClassifier(enc,0.).eval()
        features={'a.jpg':torch.randn(3,8),'b.jpg':torch.randn(3,8)}
        args=dict(base_model=base,target_model=target,base_tokenizer=helper.tokenizer,
            target_tokenizer=helper.tokenizer,base_config=dict(max_length=256),features=features,device='cpu')
        first=self.c.extract_features(examples=helper.examples,**args)
        changed=[replace(e,aspects=[(0,len(e.words),0)]) for e in helper.examples]
        second=self.c.extract_features(examples=changed,**args)
        self.assertEqual(first['keys'],[('a',0,1),('a',1,2),('b',0,1)])
        self.assertEqual(first['labels'].tolist(),[-100,2,-100])
        for name in ['base_features','target_features','base_logits','target_logits','fixed_labels']:
            torch.testing.assert_close(first[name],second[name],rtol=0,atol=0)
        self.assertEqual(first['base_features'].shape,(3,8))
        self.assertEqual(first['target_features'].shape,(3,16))
        self.assertEqual(len(base.visual_head[1]._forward_hooks),0)
        self.assertTrue(all(p.grad is None for m in [base,target] for p in m.parameters()))


if __name__=='__main__':unittest.main()
