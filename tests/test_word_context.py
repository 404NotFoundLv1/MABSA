"""Protect padding, neighbor interaction, baseline parity and inference contracts."""
import copy
import contextlib
import io
import json
import math
import unittest
from pathlib import Path

import torch

import test_data_model as fixtures
import mact.model as modeling
from mact.model import ModelConfig, SequenceModel
from mact.runtime import model_inputs, save_checkpoint, load_checkpoint, evaluate_model
from mact.train import run_training
from mact.evaluate import run_evaluation
from mact.common.cache import file_hash, model_hashes


class WordContextTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.DataModelTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)

    def model(self, enabled=True, **overrides):
        self.assertIn('word_context', ModelConfig.__dataclass_fields__,
                      'The optional word-context residual is not implemented')
        args = dict(hidden_size=8, attention_heads=2, visual_size=8, dropout=0.,
                    visual_dropout=0., word_context=enabled, word_context_size=4,
                    word_context_heads=2, word_context_max_distance=2)
        args.update(overrides)
        return SequenceModel(fixtures.tiny_text(), ModelConfig(**args))

    def block(self):
        self.assertTrue(hasattr(modeling, 'WordContextResidual'),
                        'A mask-aware word-context residual is required')
        block = modeling.WordContextResidual(2, 2, 1, 1)
        with torch.no_grad():
            block.q.weight.zero_()
            block.k.weight.zero_()
            block.v.weight.copy_(torch.eye(2))
            block.relative_bias.zero_()
            block.output.weight.zero_()
            block.output.weight[:2].copy_(torch.eye(2))
            block.output.bias.zero_()
        return block

    def test_initial_residual_preserves_shared_parameters_rng_and_training_scores(self):
        torch.manual_seed(42)
        baseline = self.model(False, dropout=.1, visual_dropout=.15)
        rng = torch.get_rng_state().clone()
        torch.manual_seed(42)
        actual = self.model(True, dropout=.1, visual_dropout=.15)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertFalse(hasattr(baseline, 'word_context'))
        for name, value in baseline.state_dict().items():
            torch.testing.assert_close(actual.state_dict()[name], value, rtol=0, atol=0)
        inputs = model_inputs(self.fixture.batch(), 'cpu', include_tags=True)
        results = []
        for model in (baseline, actual):
            torch.manual_seed(97)
            result = model(**inputs)
            result['losses'].mean().backward()
            results.append(result)
        for name in ('emissions', 'text_emissions', 'losses'):
            torch.testing.assert_close(results[0][name], results[1][name], rtol=0, atol=0)
        for name, parameter in baseline.named_parameters():
            other = dict(actual.named_parameters())[name]
            if parameter.grad is None:
                self.assertIsNone(other.grad)
            else:
                torch.testing.assert_close(parameter.grad, other.grad, rtol=0, atol=0)

    def test_context_reads_other_valid_words_and_ignores_padding(self):
        block = self.block()
        words = torch.tensor([[[2., 0.], [0., 6.], [1000., -2000.]]])
        mask = torch.tensor([[True, True, False]])
        expected = torch.tensor([[[1., 3., 0., 0., 0.], [1., 3., 0., 0., 0.],
                                  [0., 0., 0., 0., 0.]]])
        torch.testing.assert_close(block(words, mask), expected)
        torch.testing.assert_close(block(words[:, :2], mask[:, :2]), expected[:, :2])
        words[0, 1, 1] = 10.
        self.assertAlmostEqual(float(block(words, mask)[0, 0, 1]), 5.)
        words[0, 2] = torch.tensor([-1e6, 1e6])
        self.assertAlmostEqual(float(block(words, mask)[0, 0, 1]), 5.)

    def test_relative_key_minus_query_distance_is_signed_and_clipped(self):
        block = self.block()
        with torch.no_grad():
            # For query 0, keys 1 and 2 share the clipped +1 bucket.
            block.relative_bias[0, 2] = math.log(3.)
        words = torch.tensor([[[2., 0.], [0., 6.], [4., 0.]]])
        result = block(words, torch.ones(1, 3, dtype=torch.bool))
        torch.testing.assert_close(result[0, 0, :2], torch.tensor([2., 18./7]))
        # Last query sees [-1, -1, 0], all unbiased.
        torch.testing.assert_close(result[0, 2, :2], torch.tensor([2., 2.]))

    def test_each_attention_head_uses_its_own_distance_bias(self):
        self.assertTrue(hasattr(modeling, 'WordContextResidual'))
        block = modeling.WordContextResidual(2, 2, 2, 1)
        with torch.no_grad():
            block.q.weight.zero_()
            block.k.weight.zero_()
            block.v.weight.copy_(torch.eye(2))
            block.relative_bias.zero_()
            block.relative_bias[0, 2] = math.log(3.)
            block.output.weight.zero_()
            block.output.weight[:2].copy_(torch.eye(2))
            block.output.bias.zero_()
        result = block(torch.tensor([[[2., 0.], [0., 6.]]]), torch.ones(1, 2, dtype=torch.bool))
        torch.testing.assert_close(result[0, 0, :2], torch.tensor([.5, 3.]))

    def test_all_padding_is_rejected_instead_of_producing_nan(self):
        block = self.block()
        with self.assertRaises(ValueError):
            block(torch.zeros(1, 2, 2), torch.zeros(1, 2, dtype=torch.bool))

    def test_crf_training_activates_zero_head_then_context_parameters(self):
        torch.manual_seed(42)
        model = self.model()
        inputs = model_inputs(self.fixture.batch(), 'cpu', include_tags=True)
        optimizer = torch.optim.SGD(model.parameters(), lr=.03)
        first = model(**inputs)
        first['losses'].mean().backward()
        self.assertGreater(float(model.word_context.output.weight.grad.abs().sum()), 0.)
        optimizer.step()
        optimizer.zero_grad()
        second = model(**inputs)
        second['losses'].mean().backward()
        for name in ('q.weight', 'k.weight', 'v.weight', 'relative_bias', 'output.weight'):
            grad = dict(model.word_context.named_parameters())[name].grad
            self.assertTrue(torch.isfinite(grad).all(), name)
            self.assertGreater(float(grad.abs().sum()), 0., name)
        self.assertGreater(float(model.text_encoder.embeddings.word_embeddings.weight.grad.abs().sum()), 0.)
        self.assertGreater(float(model.visual_projection[0].weight.grad.abs().sum()), 0.)
        self.assertGreater(float(model.crf.transitions.grad.abs().sum()), 0.)

    def test_learned_residual_survives_reload_and_predictions_do_not_read_gold(self):
        model = self.model().eval()
        with torch.no_grad():
            model.word_context.output.weight.normal_(std=.1)
        batch = self.fixture.batch()
        inputs = model_inputs(batch, 'cpu')
        expected = model(**inputs)['emissions']
        checkpoint = Path(self.fixture.tmp.name) / 'model.pt'
        save_checkpoint(checkpoint, model, {'seed': 42}, 1, {})
        loaded, metadata = load_checkpoint(checkpoint)
        self.assertEqual(metadata['format'], 'mabsa_sequence_crf_v1')
        self.assertEqual(metadata['tags'], ['O', 'B-POS', 'B-NEU', 'B-NEG', 'I'])
        self.assertTrue(loaded.config.word_context)
        torch.testing.assert_close(loaded(**inputs)['emissions'], expected, rtol=0, atol=0)
        predictions = evaluate_model(loaded, [batch], 'cpu')[1]
        batch['tags'].fill_(-999)
        self.assertEqual(evaluate_model(loaded, [batch], 'cpu')[1], predictions)
        for example in batch['examples']:
            example.aspects = []
        self.assertEqual(evaluate_model(loaded, [batch], 'cpu')[1], predictions)
        trained = copy.deepcopy(loaded)
        with torch.no_grad():
            trained.word_context.output.weight.zero_()
            trained.word_context.output.bias.zero_()
        self.assertFalse(torch.equal(trained(**inputs)['emissions'], expected))

    def test_force_null_masks_real_images_for_the_context_branch_too(self):
        model = self.model().eval()
        with torch.no_grad():
            model.word_context.output.weight.normal_(std=.1)
        inputs = model_inputs(self.fixture.batch(), 'cpu')
        expected = model(**inputs, force_null=True)['emissions']
        inputs['visual_features'] = torch.randn_like(inputs['visual_features']) * 1000
        torch.testing.assert_close(model(**inputs, force_null=True)['emissions'], expected, rtol=0, atol=0)

    def test_invalid_sizes_and_unstudied_combinations_are_rejected(self):
        self.assertIn('word_context', ModelConfig.__dataclass_fields__)
        invalid = [{'word_context': 1}, {'word_context_size': 0}, {'word_context_heads': True},
                   {'word_context_size': 7, 'word_context_heads': 2}, {'word_context_max_distance': -1},
                   {'word_context_max_distance': 1.5}, {'mode': 'text'}, {'mode': 'evidence'},
                   {'label_scheme': 'seven'}, {'visual_gate': True}, {'aspect_sentiment_weight': .1}]
        for values in invalid:
            with self.subTest(values=values), self.assertRaises(ValueError):
                ModelConfig(**{'word_context': True, **values})

    def test_old_checkpoint_without_context_fields_keeps_baseline_predictions(self):
        model = self.model(False).eval()
        inputs = model_inputs(self.fixture.batch(), 'cpu')
        checkpoint = Path(self.fixture.tmp.name) / 'old.pt'
        save_checkpoint(checkpoint, model, {}, 1, {})
        content = torch.load(checkpoint, weights_only=True)
        for key in list(content['model_config']):
            if key.startswith('word_context'):
                del content['model_config'][key]
        torch.save(content, checkpoint)
        loaded, _ = load_checkpoint(checkpoint)
        self.assertFalse(loaded.config.word_context)
        torch.testing.assert_close(loaded(**inputs)['emissions'], model(**inputs)['emissions'], rtol=0, atol=0)

    def test_full_training_records_module_and_reloads_dev_without_step_logs(self):
        root = Path(self.fixture.tmp.name)
        backbone = root / 'backbone'
        fixtures.tiny_text().save_pretrained(backbone)
        self.fixture.tokenizer.save_pretrained(backbone)
        vision, images, cache = (root / name for name in ('vision', 'images', 'cache'))
        for directory in (vision, images, cache):
            directory.mkdir()
        (vision / 'config.json').write_text('{}')
        (images / 'a.jpg').write_bytes(b'fixture-image')
        torch.save({'a.jpg': torch.arange(24, dtype=torch.float32).reshape(3, 8)/24}, cache / 'features.pt')
        (cache / 'manifest.json').write_text(json.dumps(dict(format_version=1,
            vision_model='../vision', image_root='../images', model_sha256=model_hashes(vision),
            features_sha256=file_hash(cache / 'features.pt'), token_shape=[3, 8],
            image_sha256={'a.jpg': file_hash(images / 'a.jpg')})))
        for split in ('train', 'dev'):
            (root / f'{split}.json').write_text(json.dumps([dict(id=split,
                words=['good', 'washington'], image_id='a.jpg',
                aspects=[{'from': 1, 'to': 2, 'polarity': 'positive'}])]))
        cfg = dict(text_model=str(backbone), vision_model=str(vision), image_root=str(images),
            visual_cache=str(cache), train_data=str(root / 'train.json'), dev_data=str(root / 'dev.json'),
            epochs=2, accumulation=1, log_every=0, device='cpu', num_threads=1,
            model=dict(hidden_size=8, attention_heads=2, visual_size=8, word_context=True,
                       word_context_size=4, word_context_heads=2, word_context_max_distance=2))
        stdout = io.StringIO()
        output = root / 'training'
        with contextlib.redirect_stdout(stdout):
            summary = run_training(cfg, output)
        self.assertEqual(summary['completed_epochs'], 2)
        provenance = json.loads((output / 'provenance.json').read_text())
        self.assertEqual(provenance['objective'], 'sentence_mean_crf_nll')
        self.assertIn('word_context', provenance, 'Training must identify the active inference module')
        self.assertEqual(provenance['word_context']['feature_source'], 'visual_head_hidden')
        self.assertFalse(provenance['word_context']['uses_gold_inputs'])
        self.assertIn('Word context:', stdout.getvalue())
        self.assertNotIn('Step [', stdout.getvalue())
        metrics, predictions = run_evaluation(output / 'best.pt', cfg['dev_data'], root / 'reload')
        self.assertEqual(metrics, summary['best_dev'])
        self.assertEqual(json.loads(json.dumps(predictions)),
                         json.loads((output / 'best_dev_predictions.json').read_text()))
        model, _ = load_checkpoint(output / 'last.pt')
        self.assertGreater(float(model.word_context.output.weight.norm()), 0.)


if __name__ == '__main__':
    unittest.main()
