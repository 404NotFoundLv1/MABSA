import tempfile
import unittest
from pathlib import Path

import torch
from transformers import BertConfig, BertModel, BertTokenizerFast

from mact.common.data import Example
from mact.data import SequenceDataset, SequenceCollator
from mact.model import SequenceModel, ModelConfig, pool_words


def tiny_text():
    return BertModel(BertConfig(vocab_size=20, hidden_size=16, num_hidden_layers=1,
                               num_attention_heads=2, intermediate_size=24,
                               hidden_dropout_prob=0., attention_probs_dropout_prob=0.))


class DataModelTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        vocab = Path(self.tmp.name) / 'vocab.txt'
        vocab.write_text('[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\ngood\nwash\n##ington\nbad\n')
        self.tokenizer = BertTokenizerFast(vocab_file=str(vocab))
        self.examples = [Example('a', ['good', 'washington'], 'a.jpg', [(1, 2, 2)]),
                         Example('b', ['bad'], 'b.jpg', [])]

    def batch(self):
        ds = SequenceDataset(self.examples, self.tokenizer)
        items = [ds[i] for i in range(len(ds))]
        for item in items:
            item['visual_features'] = torch.randn(3, 8)
        return SequenceCollator(self.tokenizer.pad_token_id)(items)

    def test_subwords_special_tokens_and_padding_preserve_word_labels(self):
        batch = self.batch()
        self.assertEqual(batch['word_ids'].tolist(), [[-1, 0, 1, 1, -1], [-1, 0, -1, -1, -1]])
        self.assertEqual(batch['tags'].tolist(), [[0, 1], [0, -100]])
        self.assertEqual(batch['word_mask'].tolist(), [[True, True], [True, False]])
        self.assertNotIn('spans', batch)
        tokens = torch.tensor([[[100.], [2.], [4.], [8.], [100.]], [[100.], [9.], [100.], [100.], [100.]]])
        self.assertEqual(pool_words(tokens, batch['word_ids'], batch['word_mask']).squeeze(-1).tolist(), [[2., 6.], [9., 0.]])
        with self.assertRaises(ValueError):
            SequenceDataset(self.examples, self.tokenizer, max_length=4)
        with self.assertRaises(ValueError):
            SequenceDataset([Example('c', ['bad'], 'c.jpg', [], [(0, 1)])], self.tokenizer)

    def test_real_transformer_visual_attention_and_crf_receive_gradients(self):
        for mode in ('fusion', 'evidence', 'text'):
            with self.subTest(mode=mode):
                model = SequenceModel(tiny_text(), ModelConfig(mode=mode, hidden_size=8, attention_heads=2,
                                      visual_size=8, dropout=0., visual_dropout=0.))
                batch = self.batch()
                output = model(**{k: v for k, v in batch.items() if k != 'examples'})
                self.assertEqual(list(output['emissions'].shape), [2, 2, 5])
                output['losses'].mean().backward()
                self.assertGreater(float(model.text_encoder.embeddings.word_embeddings.weight.grad.abs().sum()), 0.)
                self.assertGreater(float(model.crf.transitions.grad.abs().sum()), 0.)
                if mode != 'text':
                    self.assertGreater(float(model.visual_projection[0].weight.grad.abs().sum()), 0.)
                self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))

    def test_null_evidence_exactly_cancels_visual_residual(self):
        model = SequenceModel(tiny_text(), ModelConfig(mode='evidence', hidden_size=8, attention_heads=2,
                              visual_size=8, dropout=.2, visual_dropout=.15)).eval()
        batch = {k: v for k, v in self.batch().items() if k != 'examples'}
        output = model(**batch, force_null=True)
        torch.testing.assert_close(output['emissions'], output['text_emissions'], rtol=0, atol=0)
        model.train()
        model.config.visual_dropout = 1.
        output = model(**batch)
        torch.testing.assert_close(output['emissions'], output['text_emissions'], rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
