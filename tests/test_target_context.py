import copy
import importlib.util
import unittest

import torch

import test_data_model as fixtures
from mact.common.data import Example


class TargetContextTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('mact.target_context'),
                             'target context feature is missing')
        from mact import target_context as tc
        self.tc = tc
        self.fixture = fixtures.DataModelTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.tokenizer = self.fixture.tokenizer
        tc.add_markers(self.tokenizer)

    def test_repeated_multisubword_target_marks_requested_occurrence(self):
        encoded = self.tc.encode_target(self.tokenizer,
            ['washington', 'good', 'washington', 'bad'], 2, 3, 258)
        tokens = self.tokenizer.convert_ids_to_tokens(encoded['input_ids'])
        self.assertEqual(tokens, ['[CLS]', 'wash', '##ington', 'good',
            '[TGT]', 'wash', '##ington', '[/TGT]', 'bad', '[SEP]'])
        self.assertEqual(encoded['target_index'], 4)
        with self.assertRaises(ValueError):
            self.tc.encode_target(self.tokenizer, ['washington'] * 10, 0, 1, 8)
        with self.assertRaises(ValueError):
            self.tc.encode_target(self.tokenizer, ['[TGT]', 'bad'], 1, 2, 258)
        with self.assertRaises(ValueError):
            self.tc.encode_target(self.tokenizer, ['bad'], 1, 2, 258)

    def test_inference_ignores_gold_and_original_polarities_and_keeps_empty_ids(self):
        examples = self.fixture.examples
        spans = {'a': [(0, 2, 0)], 'b': []}  # deliberately wrong boundary
        first = self.tc.TargetDataset(examples, spans, self.tokenizer, supervised=False)
        changed = copy.deepcopy(examples)
        changed[0].aspects = [(0, 1, 1)]
        second = self.tc.TargetDataset(changed, {'a': [(0, 2, 2)], 'b': []},
                                      self.tokenizer, supervised=False)
        self.assertEqual(first[0], second[0])
        self.assertNotIn('labels', first[0])
        self.assertEqual(self.tc.relabel(spans, first.keys, [2]), {'a': [(0, 2, 2)], 'b': []})
        self.assertEqual(spans['a'], [(0, 2, 0)])
        for invalid in ({'a': []}, {'a': [(0, 2, 0), (0, 2, 1)], 'b': []},
                        {'a': [(0, 3, 0)], 'b': []}):
            with self.assertRaises(ValueError):
                self.tc.TargetDataset(examples, invalid, self.tokenizer, supervised=False)
        with self.assertRaises(ValueError):
            self.tc.relabel(spans, first.keys, [])

    def test_trainable_encoder_and_portable_checkpoint(self):
        ds = self.tc.TargetDataset(self.fixture.examples,
            {x.id: x.aspects for x in self.fixture.examples}, self.tokenizer, supervised=True)
        self.assertEqual(len(ds), 1)
        batch = self.tc.TargetCollator(self.tokenizer.pad_token_id)([ds[0]])
        self.assertEqual(batch['labels'].tolist(), [2])
        model = self.tc.TargetClassifier(fixtures.tiny_text(), dropout=0.)
        output = model(**batch)
        self.assertEqual(tuple(output['logits'].shape), (1, 3))
        output['loss'].backward()
        self.assertGreater(model.encoder.embeddings.word_embeddings.weight.grad.abs().sum().item(), 0.)
        model.eval()
        expected = model(**batch)['logits']
        path = self.fixture.tmp.name + '/target.pt'
        self.tc.save_target(path, model, {'epoch': 3})
        restored, meta = self.tc.load_target(path)
        self.assertEqual(meta, {'epoch': 3})
        torch.testing.assert_close(restored(**batch)['logits'], expected, atol=0, rtol=0)


if __name__ == '__main__':
    unittest.main()
