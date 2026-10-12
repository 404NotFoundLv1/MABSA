import itertools
import math
import unittest

import torch

from mact.labels import aspects_to_tags, tags_to_aspects
from mact.crf import ConstrainedCRF


class LabelTests(unittest.TestCase):
    def test_adjacent_aspects_and_last_word_roundtrip(self):
        aspects = [(0, 2, 2), (2, 3, 0), (4, 6, 1)]
        tags = [1, 4, 3, 0, 2, 4]
        self.assertEqual(aspects_to_tags(6, aspects), tags)
        self.assertEqual(tags_to_aspects(tags), aspects)
        self.assertEqual(tags_to_aspects([0, 0]), [])

    def test_invalid_spans_and_orphan_inside_are_rejected(self):
        for spans in ([(0, 2, 2), (1, 3, 1)], [(1, 4, 0)], [(0, 1, 7)], [(-1, 1, 0)]):
            with self.assertRaises(ValueError):
                aspects_to_tags(3, spans)
        for tags in ([4], [0, 4], [1, 0, 4], [8]):
            with self.assertRaises(ValueError):
                tags_to_aspects(tags)


class CRFTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.crf = ConstrainedCRF().double()

    def test_partition_and_viterbi_agree_with_exhaustive_legal_paths(self):
        emissions = torch.randn(1, 3, 5, dtype=torch.double, requires_grad=True)
        mask = torch.ones(1, 3, dtype=torch.bool)
        scores, paths = [], []
        for path in itertools.product(range(5), repeat=3):
            if path[0] == 4 or any(a == 0 and b == 4 for a, b in zip(path, path[1:])):
                continue
            score = self.crf.start_transitions[path[0]] + self.crf.end_transitions[path[-1]]
            score = score + sum(emissions[0, i, tag] for i, tag in enumerate(path))
            score = score + sum(self.crf.transitions[a, b] for a, b in zip(path, path[1:]))
            scores.append(score)
            paths.append(list(path))
        scores = torch.stack(scores)
        gold = [1, 4, 3]
        expected = scores.logsumexp(0) - scores[paths.index(gold)]
        loss = self.crf(emissions, torch.tensor([gold]), mask)
        torch.testing.assert_close(loss[0], expected)
        self.assertEqual(self.crf.decode(emissions, mask), [paths[int(scores.argmax())]])
        loss.sum().backward()
        self.assertTrue(torch.isfinite(emissions.grad).all())
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in self.crf.parameters()))

    def test_padding_has_no_loss_gradient_or_decoding_effect(self):
        emissions = torch.randn(2, 4, 5, dtype=torch.double, requires_grad=True)
        tags = torch.tensor([[2, 4, 0, 1], [3, -100, -100, -100]])
        mask = torch.tensor([[1, 1, 1, 1], [1, 0, 0, 0]], dtype=torch.bool)
        loss = self.crf(emissions, tags, mask)
        single = self.crf(emissions[1:2, :1], tags[1:2, :1], mask[1:2, :1])
        torch.testing.assert_close(loss[1], single[0])
        self.assertEqual(self.crf.decode(emissions, mask)[1], self.crf.decode(emissions[1:2, :1], mask[1:2, :1])[0])
        loss.sum().backward()
        self.assertEqual(float(emissions.grad[1, 1:].abs().sum()), 0.)

    def test_illegal_inside_is_impossible_even_with_extreme_emissions(self):
        emissions = torch.zeros(1, 2, 5, dtype=torch.double)
        emissions[0, 0, 0] = 1e8
        emissions[0, :, 4] = 1e6
        mask = torch.ones(1, 2, dtype=torch.bool)
        self.assertEqual(self.crf.decode(emissions, mask), [[0, 0]])
        for tags in ([[4, 4]], [[0, 4]]):
            with self.assertRaises(ValueError):
                self.crf(emissions, torch.tensor(tags), mask)

    def test_uniform_single_word_partition_excludes_inside(self):
        with torch.no_grad():
            for p in self.crf.parameters():
                p.zero_()
        loss = self.crf(torch.zeros(1, 1, 5, dtype=torch.double), torch.tensor([[0]]), torch.ones(1, 1, dtype=torch.bool))
        self.assertAlmostEqual(loss.item(), math.log(4), places=10)
        for mask in (torch.tensor([[False, False]]), torch.tensor([[True, False, True]])):
            with self.assertRaises(ValueError):
                self.crf.decode(torch.zeros(1, mask.shape[1], 5), mask)


if __name__ == '__main__':
    unittest.main()
