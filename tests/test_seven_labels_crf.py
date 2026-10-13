import itertools
import math
import unittest

import torch

from mact import labels
from mact.crf import ConstrainedCRF


def legal_paths(num_tags, length):
    """Enumerate the specification independently of CRF constraint buffers."""
    for path in itertools.product(range(num_tags), repeat=length):
        if path[0] >= 4:
            continue
        if num_tags == 5:
            valid = all(previous != 0 or current != 4
                        for previous, current in zip(path, path[1:]))
        else:
            valid = all(current < 4 or previous in (current - 3, current)
                        for previous, current in zip(path, path[1:]))
        if valid:
            yield path


def path_score(crf, emissions, path):
    score = crf.start_transitions[path[0]] + crf.end_transitions[path[-1]]
    score = score + sum(emissions[i, tag] for i, tag in enumerate(path))
    return score + sum(crf.transitions[a, b] for a, b in zip(path, path[1:]))


def fill_parameters(crf):
    generator = torch.Generator().manual_seed(137)
    with torch.no_grad():
        for parameter in crf.parameters():
            parameter.copy_(torch.randn(parameter.shape, generator=generator,
                                        dtype=parameter.dtype))


class SevenLabelTests(unittest.TestCase):
    def test_scheme_vocabulary_keeps_default_five_and_adds_seven(self):
        get_tags = getattr(labels, "get_tags", None)
        self.assertTrue(callable(get_tags), "label scheme vocabulary API is missing")
        self.assertEqual(get_tags(), ("O", "B-POS", "B-NEU", "B-NEG", "I"))
        self.assertEqual(get_tags("five"), labels.TAGS)
        self.assertEqual(get_tags("seven"),
                         ("O", "B-POS", "B-NEU", "B-NEG", "I-POS", "I-NEU", "I-NEG"))
        self.assertEqual(get_tags("seven"), labels.TAGS_SEVEN)

    def test_literal_roundtrip_preserves_polarity_and_adjacent_aspects(self):
        aspects = [(0, 2, 2), (2, 4, 0), (4, 6, 1), (7, 8, 2)]
        self.assertEqual(labels.aspects_to_tags(8, aspects, label_scheme="seven"),
                         [1, 4, 3, 6, 2, 5, 0, 1])
        self.assertEqual(labels.tags_to_aspects([1, 4, 3, 6, 2, 5, 0, 1], "seven"),
                         aspects)
        self.assertEqual(labels.tags_to_aspects([1, 1, 4, 0, 0], "seven"),
                         [(0, 1, 2), (1, 3, 2)])
        self.assertEqual(labels.tags_to_aspects([0, 0], "seven"), [])
        self.assertEqual(labels.aspects_to_tags(8, aspects), [1, 4, 3, 4, 2, 4, 0, 1])

    def test_seven_rejects_orphan_wrong_polarity_and_invalid_inside_tags(self):
        for tags in ([4], [5], [6], [0, 4], [1, 5], [2, 6], [3, 4],
                     [1, 4, 5], [1, 4, 0, 4], [-1], [7], [True]):
            with self.subTest(tags=tags), self.assertRaises(ValueError):
                labels.tags_to_aspects(tags, "seven")

    def test_seven_keeps_span_validation(self):
        for aspects in ([(0, 2, 2), (1, 3, 1)], [(1, 4, 0)], [(0, 1, 7)],
                        [(-1, 1, 0)], [(0, 0, 1)], [(False, 1, 0)]):
            with self.subTest(aspects=aspects), self.assertRaises(ValueError):
                labels.aspects_to_tags(3, aspects, "seven")

    def test_every_entrypoint_rejects_unknown_schemes(self):
        get_tags = getattr(labels, "get_tags", None)
        self.assertTrue(callable(get_tags), "label scheme vocabulary API is missing")
        for scheme in ("", "Five", "SEVEN", "seven ", "bio", 5, 7, None, [], {}):
            for operation in (lambda: get_tags(scheme),
                              lambda: labels.aspects_to_tags(1, [], scheme),
                              lambda: labels.tags_to_aspects([], scheme),
                              lambda: ConstrainedCRF(label_scheme=scheme)):
                with self.subTest(scheme=scheme), self.assertRaises(ValueError):
                    operation()


class SevenCRFTests(unittest.TestCase):
    def test_nonzero_partition_viterbi_and_gradients_match_exhaustive_paths(self):
        crf = ConstrainedCRF(label_scheme="seven").double()
        self.assertEqual(crf.num_tags, 7)
        fill_parameters(crf)
        emissions = torch.tensor([[[0.2, -0.4, 0.8, 0.1, 2.0, -0.1, 0.5],
                                   [-0.3, 0.9, 0.1, -0.8, 0.7, 1.1, 0.6],
                                   [0.4, 0.2, -0.6, 0.8, 1.0, 0.3, -0.1]]],
                                 dtype=torch.double, requires_grad=True)
        mask = torch.ones(1, 3, dtype=torch.bool)
        paths = list(legal_paths(7, 3))
        scores = torch.stack([path_score(crf, emissions[0], path) for path in paths])
        gold = (2, 5, 3)
        expected = scores.logsumexp(0) - scores[paths.index(gold)]
        loss = crf(emissions, torch.tensor([gold]), mask)
        torch.testing.assert_close(loss[0], expected)
        self.assertEqual(crf.decode(emissions, mask), [list(paths[int(scores.argmax())])])
        loss.sum().backward()
        self.assertTrue(torch.isfinite(emissions.grad).all())
        for parameter in crf.parameters():
            self.assertTrue(torch.isfinite(parameter.grad).all())
        self.assertEqual(float(crf.start_transitions.grad[4:].abs().sum()), 0.)
        self.assertEqual(float(crf.transitions.grad[0, 4:].abs().sum()), 0.)

    def test_padding_and_single_word_have_finite_gradients(self):
        crf = ConstrainedCRF("seven").double()
        fill_parameters(crf)
        emissions = torch.randn(3, 4, 7, dtype=torch.double, requires_grad=True)
        tags = torch.tensor([[2, 5, 0, 1], [3, -100, -100, -100],
                             [1, 4, -100, -100]])
        mask = torch.tensor([[1, 1, 1, 1], [1, 0, 0, 0], [1, 1, 0, 0]], dtype=torch.bool)
        loss = crf(emissions, tags, mask)
        decoded = crf.decode(emissions, mask)
        for i, length in enumerate((4, 1, 2)):
            single_loss = crf(emissions[i:i + 1, :length], tags[i:i + 1, :length],
                              mask[i:i + 1, :length])
            torch.testing.assert_close(loss[i], single_loss[0])
            self.assertEqual(decoded[i], crf.decode(emissions[i:i + 1, :length],
                                                    mask[i:i + 1, :length])[0])
        loss.sum().backward()
        self.assertTrue(torch.isfinite(emissions.grad).all())
        self.assertEqual(float(emissions.grad[~mask].abs().sum()), 0.)
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in crf.parameters()))

    def test_uniform_single_word_partition_has_only_four_start_states(self):
        crf = ConstrainedCRF("seven").double()
        emissions = torch.zeros(1, 1, 7, dtype=torch.double, requires_grad=True)
        loss = crf(emissions, torch.tensor([[0]]), torch.ones(1, 1, dtype=torch.bool))
        self.assertAlmostEqual(loss.item(), math.log(4), places=10)
        loss.sum().backward()
        self.assertTrue(torch.isfinite(emissions.grad).all())
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all()
                            for p in crf.parameters()))

    def test_gold_constraints_cover_every_transition_and_ignore_padding(self):
        crf = ConstrainedCRF("seven")
        emissions = torch.zeros(1, 3, 7)
        mask = torch.tensor([[True, True, False]])
        permitted = set(legal_paths(7, 2))
        for first, second in itertools.product(range(7), repeat=2):
            tags = torch.tensor([[first, second, -100]])
            with self.subTest(first=first, second=second):
                if (first, second) in permitted:
                    self.assertTrue(torch.isfinite(crf(emissions, tags, mask)).all())
                else:
                    with self.assertRaises(ValueError):
                        crf(emissions, tags, mask)
        for tags in ([[1, 4, 5]], [[2, 5, 6]], [[3, 6, 4]], [[1, -100, 0]], [[1, 7, 0]]):
            with self.subTest(tags=tags), self.assertRaises(ValueError):
                crf(emissions, torch.tensor(tags), torch.ones(1, 3, dtype=torch.bool))

    def test_wrong_polarity_is_impossible_even_with_extreme_emissions(self):
        crf = ConstrainedCRF("seven").double()
        emissions = torch.zeros(1, 2, 7, dtype=torch.double)
        emissions[0, 0, 1] = 1e8
        emissions[0, 0, 4:] = 1e6
        emissions[0, 1, 5:] = 1e6
        emissions[0, 1, 4] = 1.
        self.assertEqual(crf.decode(emissions, torch.ones(1, 2, dtype=torch.bool)), [[1, 4]])

    def test_emission_dimension_must_match_scheme(self):
        for scheme, wrong_dimension in (("five", 7), ("seven", 5)):
            with self.subTest(scheme=scheme), self.assertRaises(ValueError):
                ConstrainedCRF(scheme).decode(torch.zeros(1, 1, wrong_dimension),
                                             torch.ones(1, 1, dtype=torch.bool))

    def test_expanded_five_parameters_preserve_every_legal_path_and_padded_likelihood(self):
        five = ConstrainedCRF().double()
        seven = ConstrainedCRF("seven").double()
        self.assertEqual(five.num_tags, 5)
        fill_parameters(five)
        mapping = torch.tensor([0, 1, 2, 3, 4, 4, 4])
        with torch.no_grad():
            seven.start_transitions.copy_(five.start_transitions[mapping])
            seven.end_transitions.copy_(five.end_transitions[mapping])
            seven.transitions.copy_(five.transitions[mapping[:, None], mapping[None, :]])
        generator = torch.Generator().manual_seed(931)
        emissions_five = torch.randn(3, 3, 5, generator=generator, dtype=torch.double)
        emissions_seven = emissions_five[:, :, mapping]
        five_paths = set(legal_paths(5, 3))
        collapsed = set()
        for path in legal_paths(7, 3):
            collapsed_path = tuple(int(mapping[tag]) for tag in path)
            self.assertNotIn(collapsed_path, collapsed)
            collapsed.add(collapsed_path)
            torch.testing.assert_close(path_score(seven, emissions_seven[0], path),
                                       path_score(five, emissions_five[0], collapsed_path))
        self.assertEqual(collapsed, five_paths)
        five_scores = torch.stack([path_score(five, emissions_five[0], p) for p in five_paths])
        seven_scores = torch.stack([path_score(seven, emissions_seven[0], p)
                                    for p in legal_paths(7, 3)])
        torch.testing.assert_close(five_scores.logsumexp(0), seven_scores.logsumexp(0))
        tags_five = torch.tensor([[2, 4, 4], [3, 4, -100], [0, -100, -100]])
        tags_seven = torch.tensor([[2, 5, 5], [3, 6, -100], [0, -100, -100]])
        mask = torch.tensor([[1, 1, 1], [1, 1, 0], [1, 0, 0]], dtype=torch.bool)
        torch.testing.assert_close(five(emissions_five, tags_five, mask),
                                   seven(emissions_seven, tags_seven, mask))
        decoded_five = five.decode(emissions_five, mask)
        decoded_seven = seven.decode(emissions_seven, mask)
        self.assertEqual(decoded_five, [[int(mapping[tag]) for tag in path]
                                       for path in decoded_seven])


if __name__ == "__main__":
    unittest.main()
