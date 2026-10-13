"""Seven-label models must start as the same scorer, then untie I polarities."""
import unittest

import torch
from torch.nn import functional as F

from test_data_model import tiny_text
from mact.model import ModelConfig, SequenceModel


class SevenModelTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def model(self, scheme='five', **options):
        config = dict(hidden_size=8, attention_heads=2, visual_size=8,
                      dropout=0., visual_dropout=0., **options)
        if scheme == 'seven':
            self.assertIn('label_scheme', ModelConfig.__dataclass_fields__,
                          'ModelConfig must expose the optional seven-label model')
            config['label_scheme'] = scheme
        return SequenceModel(tiny_text(), ModelConfig(**config))

    def inputs(self, scheme='seven'):
        # Three different two-word aspects exercise all three continuation states.
        return dict(input_ids=torch.tensor([[2, 5, 6, 3], [2, 7, 8, 3], [2, 9, 10, 3]]),
                    attention_mask=torch.ones(3, 4, dtype=torch.long),
                    word_ids=torch.tensor([[-1, 0, 1, -1]] * 3),
                    word_mask=torch.ones(3, 2, dtype=torch.bool),
                    visual_features=torch.arange(72, dtype=torch.float32).reshape(3, 3, 8) / 72,
                    tags=torch.tensor([[1, 4], [2, 5 if scheme == 'seven' else 4],
                                       [3, 6 if scheme == 'seven' else 4]]))

    def test_expansion_preserves_common_initial_parameters_rng_and_sequence_scores(self):
        mapping = torch.tensor([0, 1, 2, 3, 4, 4, 4])
        for mode in ('text', 'fusion', 'evidence'):
            with self.subTest(mode=mode):
                torch.manual_seed(42)
                five = self.model(mode=mode).eval()
                rng = torch.get_rng_state().clone()
                torch.manual_seed(42)
                seven = self.model('seven', mode=mode).eval()
                self.assertTrue(torch.equal(rng, torch.get_rng_state()))
                for name, value in five.named_parameters():
                    other = dict(seven.named_parameters())[name]
                    if name in ('text_head.weight', 'text_head.bias', 'visual_head.2.weight',
                                'visual_head.2.bias', 'crf.start_transitions', 'crf.end_transitions'):
                        expected = value.index_select(0, mapping)
                    elif name == 'crf.transitions':
                        expected = value.index_select(0, mapping).index_select(1, mapping)
                    else:
                        expected = value
                    torch.testing.assert_close(other, expected, rtol=0, atol=0)
                a, b = five(**self.inputs('five')), seven(**self.inputs())
                self.assertEqual(b['emissions'].shape[-1], 7)
                torch.testing.assert_close(b['emissions'], a['emissions'].index_select(-1, mapping))
                torch.testing.assert_close(b['losses'], a['losses'])
                five_paths = five.crf.decode(a['emissions'], self.inputs()['word_mask'])
                seven_paths = seven.crf.decode(b['emissions'], self.inputs()['word_mask'])
                self.assertEqual([[min(t, 4) for t in path] for path in seven_paths], five_paths)

    def test_inside_polarities_receive_distinct_finite_gradients_and_learn_independently(self):
        model = self.model('seven')
        optimizer = torch.optim.SGD(model.parameters(), lr=.01)
        output = model(**self.inputs())
        self.assertEqual(set(output), {'emissions', 'text_emissions', 'losses'})
        output['losses'].mean().backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
        for name in ('text_encoder.embeddings.word_embeddings.weight', 'visual_projection.0.weight',
                     'text_head.weight', 'visual_head.2.weight', 'crf.transitions'):
            self.assertGreater(dict(model.named_parameters())[name].grad.abs().sum().item(), 0.)
        grad = model.text_head.weight.grad[4:]
        self.assertFalse(torch.equal(grad[0], grad[1]))
        self.assertFalse(torch.equal(grad[1], grad[2]))
        optimizer.step()
        self.assertFalse(torch.equal(model.text_head.weight[4], model.text_head.weight[5]))
        self.assertFalse(torch.equal(model.text_head.weight[5], model.text_head.weight[6]))

    def test_existing_optional_branches_can_consume_seven_tags(self):
        model = self.model('seven', aspect_sentiment_weight=.1, visual_gate=True)
        captured = []
        hook = model.word_projection.register_forward_hook(lambda module, args, output: captured.append(output))
        try:
            output = model(**self.inputs())
        finally:
            hook.remove()
        # All gold aspects include BOTH words, including I-NEU and I-NEG.
        logits = model.aspect_sentiment_head.classifier(captured[0].mean(1))
        expected = F.cross_entropy(logits, torch.tensor([2, 1, 0]), reduction='none')
        torch.testing.assert_close(output['aspect_sentiment_losses'], expected)
        torch.testing.assert_close(output['losses'], output['crf_losses'] + .1 * expected)
        self.assertEqual(tuple(output['emissions'].shape), (3, 2, 7))
        self.assertEqual(tuple(output['visual_gate'].shape), (3, 2, 1))
        output['losses'].mean().backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))

    def test_model_rejects_invalid_label_schemes(self):
        self.assertIn('label_scheme', ModelConfig.__dataclass_fields__)
        for value in (None, 7, True, 'SEVEN', 'bio', []):
            with self.subTest(value=value), self.assertRaises(ValueError):
                ModelConfig(label_scheme=value)


if __name__ == '__main__':
    unittest.main()
