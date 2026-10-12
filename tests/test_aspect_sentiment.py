"""Auxiliary supervision must not change the decoding contract or baseline."""
import copy
import contextlib
import io
import json
import math
import unittest
from pathlib import Path

import torch

import test_data_model as fixtures
from mact.model import ModelConfig, SequenceModel
from mact.runtime import (model_inputs, save_checkpoint, load_checkpoint,
                          train_epoch, evaluate_model)
from mact.train import run_training, resolve_config


class AspectSentimentTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.DataModelTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.root = Path(self.fixture.tmp.name)

    def config(self, weight=.1):
        self.assertIn('aspect_sentiment_weight', ModelConfig.__dataclass_fields__,
                      'B requires an opt-in auxiliary sentiment objective')
        return ModelConfig(hidden_size=8, attention_heads=2, visual_size=8,
                           dropout=0., visual_dropout=0., aspect_sentiment_weight=weight)

    def model(self, weight=.1):
        return SequenceModel(fixtures.tiny_text(), self.config(weight))

    def test_span_pooling_polarity_mapping_padding_and_sentence_normalization(self):
        # Catches first-word-only pooling, swapped POS/NEG, counting padding,
        # and summing rather than averaging multiple aspects in a sentence.
        head = self.model().aspect_sentiment_head
        with torch.no_grad():
            head.classifier.weight.zero_()
            head.classifier.weight[:, :3].copy_(torch.eye(3))
            head.classifier.bias.zero_()
        words = torch.zeros(3, 5, 8)
        words[0, 0, 2] = 2.  # POS aspect spans words 0 and 1: mean logit = 1.
        words[0, 3, 0] = 2.  # Adjacent NEG aspect is independent.
        words[1, 1, 1] = 3.  # NEU singleton.
        words[:, 4] = 1000.  # Padding cannot affect a loss or receive gradients.
        words.requires_grad_()
        tags = torch.tensor([[1, 4, 0, 3, 4], [0, 2, -100, -100, -100],
                             [0, 0, -100, -100, -100]])
        mask = torch.tensor([[1, 1, 1, 1, 0], [1, 1, 0, 0, 0], [1, 1, 0, 0, 0]], dtype=torch.bool)
        loss = head(words, tags, mask)
        expected = torch.tensor([(math.log(math.exp(1)+2)-1 + math.log(math.exp(2)+2)-2)/2,
                                 math.log(math.exp(3)+2)-3, 0.])
        torch.testing.assert_close(loss, expected)
        loss.sum().backward()
        self.assertGreater(float(words.grad[0, 1].abs().sum()), 0.)
        torch.testing.assert_close(words.grad[0, 0], words.grad[0, 1])
        self.assertEqual(float(words.grad[~mask].abs().sum()), 0.)
        self.assertEqual(float(words.grad[2].abs().sum()), 0.)
        empty_loss = head(words, torch.zeros_like(tags), mask)
        self.assertTrue(torch.isfinite(empty_loss).all())
        self.assertEqual(empty_loss.tolist(), [0., 0., 0.])

    def test_adjacent_begins_are_distinct_aspects(self):
        head = self.model().aspect_sentiment_head
        with torch.no_grad():
            head.classifier.weight.zero_()
            head.classifier.bias.copy_(torch.tensor([2., 0., -1.]))
        loss = head(torch.zeros(1, 2, 8), torch.tensor([[3, 1]]), torch.ones(1, 2, dtype=torch.bool))
        expected = math.log(math.exp(2)+1+math.exp(-1)) - .5
        self.assertAlmostEqual(loss.item(), expected, places=6)

    def test_auxiliary_loss_reaches_encoder_but_never_changes_emissions(self):
        model = self.model(.2).train()
        inputs = model_inputs(self.fixture.batch(), 'cpu', include_tags=True)
        output = model(**inputs)
        self.assertEqual(output['losses'].shape, (2,))
        torch.testing.assert_close(output['losses'], output['crf_losses'] + .2 * output['aspect_sentiment_losses'])
        self.assertEqual(output['aspect_sentiment_losses'][1].item(), 0.)
        output['aspect_sentiment_losses'].sum().backward()
        self.assertGreater(float(model.aspect_sentiment_head.classifier.weight.grad.abs().sum()), 0.)
        self.assertGreater(float(model.text_encoder.embeddings.word_embeddings.weight.grad.abs().sum()), 0.)
        model.eval()
        inference = model(**{k: v for k, v in inputs.items() if k != 'tags'})
        inputs['tags'] = torch.zeros_like(inputs['tags'])
        altered_gold = model(**inputs)
        torch.testing.assert_close(inference['emissions'], altered_gold['emissions'], rtol=0, atol=0)
        self.assertNotIn('aspect_sentiment_losses', inference)
        self.assertEqual(model.crf.decode(inference['emissions'], inputs['word_mask']),
                         model.crf.decode(altered_gold['emissions'], inputs['word_mask']))

    def test_disabled_branch_and_old_checkpoint_keep_baseline_exact(self):
        torch.manual_seed(42)
        baseline = SequenceModel(fixtures.tiny_text(), ModelConfig(hidden_size=8, attention_heads=2,
                                 visual_size=8, dropout=0., visual_dropout=0.))
        torch.manual_seed(42)
        disabled = self.model(0.)
        self.assertFalse(hasattr(disabled, 'aspect_sentiment_head'))
        self.assertEqual(set(baseline.state_dict()), set(disabled.state_dict()))
        inputs = model_inputs(self.fixture.batch(), 'cpu', include_tags=True)
        for key, value in baseline(**inputs).items():
            torch.testing.assert_close(value, disabled(**inputs)[key], rtol=0, atol=0)
        checkpoint = self.root / 'old.pt'
        save_checkpoint(checkpoint, baseline, {}, 1, {})
        content = torch.load(checkpoint, weights_only=True)
        content['model_config'].pop('aspect_sentiment_weight')
        torch.save(content, checkpoint)
        old, _ = load_checkpoint(checkpoint)
        for name, value in baseline.state_dict().items():
            torch.testing.assert_close(value, old.state_dict()[name], rtol=0, atol=0)

    def test_enabled_checkpoint_roundtrip(self):
        model = self.model(.1).eval()
        checkpoint = self.root / 'b.pt'
        save_checkpoint(checkpoint, model, {}, 2, {})
        restored, _ = load_checkpoint(checkpoint)
        self.assertEqual(restored.config.aspect_sentiment_weight, .1)
        inputs = model_inputs(self.fixture.batch(), 'cpu', include_tags=True)
        for key, value in model(**inputs).items():
            torch.testing.assert_close(value, restored(**inputs)[key], rtol=0, atol=0)

    def test_combined_objective_preserves_sentence_mean_accumulation(self):
        actual = self.model(.1)
        expected = copy.deepcopy(actual)
        first = self.fixture.batch()
        second = {k: (v[:1] if isinstance(v, torch.Tensor) else v[:1]) for k, v in first.items()}
        batches = [first, second]
        opt_a = torch.optim.SGD(actual.parameters(), lr=.03)
        opt_b = torch.optim.SGD(expected.parameters(), lr=.03)
        result = train_epoch(actual, batches, opt_a, 'cpu', accumulation=2, gradient_clip=0, log_every=0)
        outputs = [expected(**model_inputs(b, 'cpu', include_tags=True)) for b in batches]
        loss = sum(o['losses'].sum() for o in outputs) / 3
        loss.backward()
        opt_b.step()
        for a, b in zip(actual.parameters(), expected.parameters()):
            torch.testing.assert_close(a, b)
        for key in ('sentence_loss', 'sentence_nll', 'aspect_sentiment_loss'):
            self.assertTrue(math.isfinite(result[key]))
        self.assertAlmostEqual(result['sentence_loss'], result['sentence_nll'] + .1 * result['aspect_sentiment_loss'], places=6)

    def test_invalid_auxiliary_weight_rejected(self):
        self.config()
        for value in (-.1, float('nan'), float('inf'), True, '0.1'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                resolve_config({'model': {'aspect_sentiment_weight': value}})

    def test_training_logs_components_and_reload_dev_without_step_noise(self):
        config = self.config(.1)
        backbone = self.root / 'backbone'
        fixtures.tiny_text().save_pretrained(backbone)
        self.fixture.tokenizer.save_pretrained(backbone)
        rows = [{'id': str(i), 'words': ['good', 'washington'], 'image_id': 'unused.jpg',
                 'aspects': [{'from': 0, 'to': 2, 'polarity': 'positive'}] if i == 0 else []}
                for i in range(2)]
        for split, data in [('train', rows), ('dev', [dict(rows[0], id='dev')])]:
            (self.root / f'{split}.json').write_text(json.dumps(data))
        cfg = dict(text_model=str(backbone), train_data=str(self.root/'train.json'),
                   dev_data=str(self.root/'dev.json'), device='cpu', num_threads=1,
                   accumulation=2, epochs=1, log_every=0,
                   model=dict(mode='text', hidden_size=8, attention_heads=2, dropout=0.,
                              aspect_sentiment_weight=config.aspect_sentiment_weight))
        output = self.root/'run'
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            summary = run_training(cfg, output)
        log = buffer.getvalue()
        self.assertNotIn('Step [', log)
        self.assertIn('CRF NLL:', log)
        self.assertIn('Aspect CE:', log)
        self.assertIn('DEV  aesc_p:', log)
        history = json.loads((output/'history.json').read_text())[0]['train']
        self.assertAlmostEqual(history['sentence_loss'], history['sentence_nll'] + .1*history['aspect_sentiment_loss'], places=6)
        provenance = json.loads((output/'provenance.json').read_text())
        self.assertEqual(provenance['objective'], 'sentence_mean_crf_nll_plus_aspect_sentiment_ce')
        model, _ = load_checkpoint(output/'best.pt')
        self.assertEqual(model.config.aspect_sentiment_weight, .1)
        self.assertEqual(summary['completed_epochs'], 1)


if __name__ == '__main__':
    unittest.main()
