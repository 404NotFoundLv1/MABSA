"""R-Drop sequence-distribution math and paired stochastic training."""
import copy
import itertools
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import torch

import test_data_model as fixtures
from mact.crf import ConstrainedCRF
from mact.model import ModelConfig, SequenceModel
from mact import runtime
from mact.train import resolve_config, run_training


def exhaustive(crf, emissions):
    length, tags = emissions.shape
    paths = [p for p in itertools.product(range(tags), repeat=length)
             if crf.allowed_start[p[0]] and all(crf.allowed[a, b] for a, b in zip(p, p[1:]))]
    scores = torch.stack([crf.start_transitions[p[0]] + crf.end_transitions[p[-1]]
                         + sum(emissions[i, k] for i, k in enumerate(p))
                         + sum((crf.transitions[a, b] for a, b in zip(p, p[1:])), emissions.new_zeros(()))
                         for p in paths])
    return paths, scores.log_softmax(0)


class CRFConsistencyTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_exact_sequence_kl_values_and_all_gradients_with_padding(self):
        for scheme in ('five', 'seven'):
            torch.manual_seed(9)
            crf = ConstrainedCRF(scheme).double()
            for param in crf.parameters():
                torch.nn.init.normal_(param, std=.3)
            a = torch.randn(3, 3, crf.num_tags, dtype=torch.double, requires_grad=True)
            b = torch.randn_like(a, requires_grad=True)
            mask = torch.tensor([[True, True, True], [True, True, False], [True, False, False]])
            expected, marginals = [], torch.zeros_like(a)
            for row, length in enumerate(mask.sum(1).tolist()):
                paths, p = exhaustive(crf, a[row, :length])
                _, q = exhaustive(crf, b[row, :length])
                expected.append(.5 * ((p.exp() - q.exp()) * (p - q)).sum())
                for path, probability in zip(paths, p.exp()):
                    for t, tag in enumerate(path):
                        marginals[row, t, tag] += probability
            expected = torch.stack(expected)
            actual = crf.symmetric_kl(a, b, mask)
            torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-12)
            torch.testing.assert_close(crf.marginals(a, mask), marginals, rtol=1e-10, atol=1e-12)
            parameters = (a, b, *crf.parameters())
            ga = torch.autograd.grad(actual.sum(), parameters, retain_graph=True)
            ge = torch.autograd.grad(expected.sum(), parameters)
            for left, right in zip(ga, ge):
                self.assertTrue(torch.isfinite(left).all())
                torch.testing.assert_close(left, right, rtol=1e-9, atol=1e-11)
            self.assertEqual(torch.count_nonzero(ga[0][~mask]), 0)
            self.assertEqual(torch.count_nonzero(ga[1][~mask]), 0)

    def test_identical_distributions_zero_and_reduced_precision_safe(self):
        crf = ConstrainedCRF()
        a = torch.randn(2, 3, 5, requires_grad=True)
        mask = torch.tensor([[True, True, True], [True, False, False]])
        loss = crf.symmetric_kl(a, a, mask)
        torch.testing.assert_close(loss, torch.zeros(2), rtol=0, atol=0)
        loss.sum().backward()
        torch.testing.assert_close(a.grad, torch.zeros_like(a), rtol=0, atol=0)
        b = (a.detach() + 1).half()
        self.assertEqual(crf.symmetric_kl(a.half(), b, mask).dtype, torch.float32)
        self.assertTrue(torch.isfinite(crf.symmetric_kl(a.half(), b, mask)).all())


class PairedTrainingTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.DataModelTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        torch.manual_seed(7)
        self.model = SequenceModel(fixtures.tiny_text(), ModelConfig(hidden_size=8,
            attention_heads=2, visual_size=8, dropout=.3, visual_dropout=.5))
        self.inputs = runtime.model_inputs(self.fixture.batch(), 'cpu', include_tags=True)

    def test_two_passes_share_visual_mask_average_nll_and_use_sequence_kl(self):
        seen = []
        handle = self.model.register_forward_hook(lambda model, args, kwargs, out:
            seen.append((kwargs, out)), with_kwargs=True)
        try:
            output = runtime.training_output(self.model, self.inputs, rdrop_weight=.7)
        finally:
            handle.remove()
        self.assertEqual(len(seen), 2)
        first, second = seen[0][1], seen[1][1]
        self.assertIs(seen[1][0]['visual_drop_mask'], first['visual_drop_mask'])
        torch.testing.assert_close(second['visual_drop_mask'], first['visual_drop_mask'])
        self.assertFalse(torch.equal(first['emissions'], second['emissions']))
        expected_nll = .5 * (first['losses'] + second['losses'])
        expected_kl = self.model.crf.symmetric_kl(first['emissions'], second['emissions'], self.inputs['word_mask'])
        torch.testing.assert_close(output['crf_losses'], expected_nll)
        torch.testing.assert_close(output['rdrop_losses'], expected_kl)
        torch.testing.assert_close(output['losses'], expected_nll + .7 * expected_kl)
        output['losses'].mean().backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in self.model.parameters() if p.grad is not None))

    def test_disabled_path_exact_outputs_rng_and_gradients(self):
        other = copy.deepcopy(self.model)
        torch.manual_seed(18)
        expected = other(**self.inputs)
        expected['losses'].sum().backward()
        rng = torch.get_rng_state()
        torch.manual_seed(18)
        with patch.object(self.model, 'forward', wraps=self.model.forward) as forward:
            actual = runtime.training_output(self.model, self.inputs, rdrop_weight=0)
        self.assertEqual(forward.call_count, 1)
        self.assertEqual(actual.keys(), expected.keys())
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        actual['losses'].sum().backward()
        for name in actual:
            torch.testing.assert_close(actual[name], expected[name], rtol=0, atol=0)
        for left, right in zip(self.model.parameters(), other.parameters()):
            if right.grad is not None:
                torch.testing.assert_close(left.grad, right.grad, rtol=0, atol=0)

    def test_partial_accumulation_and_epoch_stats(self):
        other = copy.deepcopy(self.model)
        batch = self.fixture.batch()
        # Three batches of two sentences: final window contains two, not four.
        batches = [batch] * 3
        opt = torch.optim.SGD(self.model.parameters(), lr=.01)
        ref_opt = torch.optim.SGD(other.parameters(), lr=.01)
        torch.manual_seed(38)
        stats = runtime.train_epoch(self.model, batches, opt, 'cpu', accumulation=2,
            gradient_clip=0, log_every=0, rdrop_weight=1.)
        torch.manual_seed(38)
        totals = dict(losses=0., crf_losses=0., rdrop_losses=0.)
        for window in (batches[:2], batches[2:]):
            ref_opt.zero_grad()
            for item in window:
                result = runtime.training_output(other, runtime.model_inputs(item, 'cpu', True), 1.)
                (result['losses'].sum() / (2 * len(window))).backward()
                for key in totals:
                    totals[key] += float(result[key].detach().sum())
            ref_opt.step()
        for left, right in zip(self.model.parameters(), other.parameters()):
            torch.testing.assert_close(left, right, rtol=0, atol=0)
        self.assertEqual((stats['examples'], stats['updates']), (6, 2))
        for key, field in [('losses', 'sentence_loss'), ('crf_losses', 'sentence_nll'), ('rdrop_losses', 'rdrop_symmetric_kl')]:
            self.assertAlmostEqual(stats[field], totals[key] / 6)

    def test_visual_override_validation_and_all_null_identity(self):
        self.model.eval()
        inputs = {k: v for k, v in self.inputs.items() if k != 'tags'}
        a = self.model(**inputs, force_null=True)['emissions']
        b = self.model(**inputs, visual_drop_mask=torch.ones(2, dtype=torch.bool))['emissions']
        torch.testing.assert_close(a, b, rtol=0, atol=0)
        for mask in (torch.ones(2), torch.ones(3, dtype=torch.bool)):
            with self.assertRaises(ValueError):
                self.model(**inputs, visual_drop_mask=mask)

    def test_config_rejects_invalid_or_combined_experiments(self):
        self.assertEqual(resolve_config({})['rdrop_weight'], 0.)
        self.assertEqual(resolve_config({'rdrop_weight': 1.})['rdrop_weight'], 1.)
        for value in (-1, float('nan'), float('inf'), True):
            with self.assertRaises(ValueError):
                resolve_config({'rdrop_weight': value})
        for model in ({'word_context': True}, {'visual_gate': True},
                      {'aspect_sentiment_weight': .1}, {'label_scheme': 'seven'}, {'mode': 'evidence'}):
            with self.assertRaises(ValueError):
                resolve_config({'rdrop_weight': 1., 'model': model})

    def test_real_training_logs_checkpoint_and_single_pass_evaluation(self):
        root = Path(self.fixture.tmp.name)
        backbone = root / 'backbone'
        fixtures.tiny_text().save_pretrained(backbone)
        self.fixture.tokenizer.save_pretrained(backbone)
        row = dict(words=['good', 'washington'], image_id='unused.jpg')
        row['aspects'] = [{'from': 1, 'to': 2, 'polarity': 'positive'}]
        train, dev = root / 'train.json', root / 'dev.json'
        train.write_text(json.dumps([dict(row, id=str(i)) for i in range(3)]))
        dev.write_text(json.dumps([dict(row, id='dev')]))
        cfg = dict(text_model=str(backbone), train_data=str(train), dev_data=str(dev),
                   device='cpu', num_threads=1, accumulation=2, epochs=1, log_every=0,
                   rdrop_weight=1., model=dict(mode='text', hidden_size=8, attention_heads=2, dropout=.3))
        output = root / 'run'
        summary = run_training(cfg, output)
        history = json.loads((output / 'history.json').read_text())[0]['train']
        self.assertGreater(history['rdrop_symmetric_kl'], 0)
        self.assertAlmostEqual(history['sentence_loss'], history['sentence_nll'] + history['rdrop_symmetric_kl'], places=6)
        model, meta = runtime.load_checkpoint(output / 'best.pt')
        self.assertEqual(meta['run_config']['rdrop_weight'], 1.)
        from mact.evaluate import run_evaluation
        metrics, _ = run_evaluation(output / 'best.pt', dev, root / 'evaluation')
        self.assertEqual(metrics, summary['best_dev'])
        with patch.object(model, 'forward', wraps=model.forward) as forward:
            runtime.evaluate_model(model, [self.fixture.batch()], 'cpu')
        self.assertEqual(forward.call_count, 1)
        self.assertNotIn('tags', forward.call_args.kwargs)
        log = (output / 'log.txt').read_text()
        self.assertNotIn('Step [', log)
        self.assertIn('Symmetric KL:', log)
        provenance = json.loads((output / 'provenance.json').read_text())
        self.assertEqual(provenance['rdrop']['distribution'], 'constrained_crf_sequences')
