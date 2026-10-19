import copy
import importlib.util
from pathlib import Path
import unittest

import torch
import test_data_model as fixtures
from mact.common.data import Example
from mact.target_context import add_markers, TargetClassifier, TargetDataset, TargetCollator


class RepresentationTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('mact.target_representation'),
                             'representation model is missing')
        from mact import target_representation as rp
        self.rp = rp
        self.fixture = fixtures.DataModelTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.tok = self.fixture.tokenizer
        add_markers(self.tok)

    def dataset(self, words, span):
        ex = Example('x', words, 'x.jpg', [span])
        return self.rp.RepresentationDataset([ex], {'x': [span]}, self.tok, True)

    def test_repeated_target_equal_word_weights_unknown_context_and_padding(self):
        ds = self.dataset(['washington', 'mystery', 'washington', 'good', 'bad'], (2, 4, 2))
        # CLS wash ##ington UNK TGT wash ##ington good /TGT bad SEP
        self.assertEqual(ds[0]['span_weights'], [0, 0, 0, 0, 0, .25, .25, .5, 0, 0, 0])
        self.assertEqual(ds[0]['context_mask'], [False, True, True, True, False,
                                               False, False, False, False, True, False])
        short = self.dataset(['good'], (0, 1, 2))
        batch = self.rp.RepresentationCollator(self.tok.pad_token_id)([ds[0], short[0]])
        self.assertEqual(batch['context_mask'].dtype, torch.bool)
        self.assertFalse(batch['context_mask'][1].any())
        self.assertEqual(batch['span_weights'][1, 5:].sum().item(), 0.)
        self.assertEqual(batch['span_weights'].sum(-1).tolist(), [1., 1.])
        legacy = TargetDataset([Example('x', ['good'], 'x.jpg', [])], {'x': [(0, 1, 0)]}, self.tok)
        self.assertEqual(short[0]['input_ids'], legacy[0]['input_ids'])

    def test_gold_and_input_polarity_do_not_change_inference_features(self):
        a = Example('a', ['good', 'washington', 'bad'], 'a.jpg', [(0, 1, 2)])
        b = copy.deepcopy(a); b.aspects = [(2, 3, 0)]
        first = self.rp.RepresentationDataset([a], {'a': [(1, 3, 0)]}, self.tok)
        second = self.rp.RepresentationDataset([b], {'a': [(1, 3, 2)]}, self.tok)
        self.assertEqual(first.items, second.items)
        self.assertNotIn('labels', first[0])

    def test_marker_control_matches_legacy_initialization_logits_and_gradients(self):
        ds = self.dataset(['good', 'washington', 'bad'], (1, 2, 2))
        batch = self.rp.RepresentationCollator(self.tok.pad_token_id)(ds.items)
        torch.manual_seed(42)
        old = TargetClassifier(fixtures.tiny_text(), 0.)
        torch.manual_seed(42)
        new = self.rp.RepresentationClassifier(fixtures.tiny_text(), 'marker', 0.)
        old_batch = {k: v for k, v in batch.items() if k not in ('span_weights', 'context_mask')}
        torch.testing.assert_close(old(**old_batch)['logits'], new(**batch)['logits'], atol=0, rtol=0)
        old(**old_batch)['loss'].backward(); new(**batch)['loss'].backward()
        for name, p in old.named_parameters():
            torch.testing.assert_close(p.grad, dict(new.named_parameters())[name].grad, atol=0, rtol=0)

    def test_context_mask_empty_rows_and_masked_values_do_not_influence_attention(self):
        model = self.rp.RepresentationClassifier(fixtures.tiny_text(), 'context', 0., 8)
        hidden = torch.randn(2, 4, 16)
        aspect = torch.randn(2, 16)
        mask = torch.tensor([[False, True, False, True], [False, False, False, False]])
        context, weights = model.attend_context(hidden, aspect, mask)
        self.assertEqual(weights[0, [0, 2]].tolist(), [0., 0.])
        self.assertAlmostEqual(weights[0].sum().item(), 1., places=6)
        self.assertEqual(context[1].abs().sum().item(), 0.)
        changed = hidden.clone(); changed[~mask] = 1e6
        other, _ = model.attend_context(changed, aspect, mask)
        torch.testing.assert_close(context, other, atol=0, rtol=0)
        self.assertTrue(torch.isfinite(context).all())

    def test_variants_receive_gradients_and_checkpoint_replays_exactly(self):
        ds = self.dataset(['good', 'washington', 'bad'], (1, 2, 2))
        batch = self.rp.RepresentationCollator(self.tok.pad_token_id)(ds.items)
        for variant in ('marker', 'span', 'context'):
            with self.subTest(variant=variant):
                model = self.rp.RepresentationClassifier(fixtures.tiny_text(), variant, 0., 8)
                loss = model(**batch)['loss']; loss.backward()
                self.assertGreater(model.encoder.embeddings.word_embeddings.weight.grad.abs().sum().item(), 0.)
                self.assertGreater(model.classifier.weight.grad.abs().sum().item(), 0.)
                if variant == 'context':
                    self.assertGreater(model.query.weight.grad.abs().sum().item(), 0.)
                    self.assertGreater(model.key.weight.grad.abs().sum().item(), 0.)
                model.eval()
                path = Path(self.fixture.tmp.name) / (variant + '.pt')
                self.rp.save_representation(path, model, {'epoch': 1})
                restored, meta = self.rp.load_representation(path)
                self.assertEqual(meta, {'epoch': 1})
                self.assertEqual(restored.variant, variant)
                torch.testing.assert_close(restored(**batch)['logits'], model(**batch)['logits'], atol=0, rtol=0)
        with self.assertRaises(ValueError):
            self.rp.RepresentationClassifier(fixtures.tiny_text(), 'invalid')


if __name__ == '__main__':
    unittest.main()
