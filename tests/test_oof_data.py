import copy
import importlib
import unittest

from mact.common.data import Example


class OOFSplitTests(unittest.TestCase):
    def setUp(self):
        self.m = importlib.import_module('mact.oof_data')
        self.examples = [Example(str(i), ['sentence', str(i)], str(i)+'.jpg', [(0, 1, i % 3)]) for i in range(24)]
        self.hashes = {e.image_id: e.image_id for e in self.examples}
        self.examples[1].words = ['SENTENCE', '0']
        self.hashes['2.jpg'] = self.hashes['1.jpg']

    def test_transitive_duplicates_never_cross_fit_inner_or_outer(self):
        p = self.m.make_partition(self.examples, self.hashes)
        self.m.verify_partition(self.examples, p, self.hashes)
        self.assertEqual(p['groups']['0'], p['groups']['1'])
        self.assertEqual(p['groups']['1'], p['groups']['2'])
        held = []
        for fold in p['folds']:
            roles = [set(fold[k]) for k in ('fit_ids', 'inner_ids', 'held_ids')]
            self.assertTrue(all(roles))
            self.assertEqual(set.union(*roles), set(p['groups']))
            for i in range(3):
                for j in range(i):
                    self.assertFalse(roles[i] & roles[j])
                    self.assertFalse({p['groups'][s] for s in roles[i]} & {p['groups'][s] for s in roles[j]})
            held.extend(fold['held_ids'])
        self.assertEqual(sorted(held), sorted(e.id for e in self.examples))

    def test_assignment_ignores_labels_and_input_order(self):
        p = self.m.make_partition(self.examples, self.hashes)
        changed = copy.deepcopy(self.examples)
        for e in changed: e.aspects = []
        self.assertEqual(p, self.m.make_partition(list(reversed(changed)), self.hashes))
        wrong = copy.deepcopy(p)
        wrong['folds'][0]['fit_ids'].append(wrong['folds'][0]['held_ids'][0])
        with self.assertRaises(ValueError): self.m.verify_partition(self.examples, wrong, self.hashes)
        with self.assertRaises(ValueError): self.m.make_partition(self.examples[:2], self.hashes)
        with self.assertRaises(ValueError): self.m.make_partition(self.examples+self.examples[:1], self.hashes)


if __name__ == '__main__': unittest.main()
