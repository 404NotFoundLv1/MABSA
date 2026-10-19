import copy
import importlib.util
import json
from pathlib import Path
import sys
import unittest

import torch
import test_target_context_training as fixtures
from mact.common.cache import file_hash
from mact.common.data import Example
from mact.data import load_labeled
from mact.experiment import code_identity


class RepresentationExperimentTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('mact.representation_experiment'),
                             'fusion-selected representation runner missing')
        from mact import representation_experiment as exp
        self.exp = exp
        self.fx = fixtures.TargetTrainingTests(); self.fx.setUp()
        self.addCleanup(self.fx.doCleanups)
        self.root = self.fx.root
        cfg = dict(self.fx.cfg, dataset='twitter2015', variant='context', attention_dim=8,
                   score_cache=str(self.root / 'scores.json'))
        self.cfg = cfg
        self.cache = dict(format='mact_fusion_scores_v1', dataset='twitter2015', seed=42,
            gold={'a': [[1, 2, 2]], 'b': []}, baseline=self.fx.pred, target=self.fx.pred,
            keys=[['a', 1, 2]], base_logits=[[2., 0., -1.]], target_logits=[[2., 0., -1.]],
            provenance=dict(input_sha256={cfg[k]: file_hash(cfg[k]) for k in
                ('dev_data', 'baseline_predictions', 'baseline_summary', 'baseline_checkpoint')},
                code=code_identity(), baseline_replay_exact=True, target_replay_exact=True, test_used=False))
        self.save_cache()

    def save_cache(self):
        path = Path(self.cfg['score_cache'])
        path.write_text(json.dumps(self.cache))
        path.with_suffix('.sha256').write_text(file_hash(path))

    def test_tiny_training_fusion_selection_full_reload_no_test_and_resume_identity(self):
        guard = {'active': True, 'attempts': 0}
        def deny(event, args):
            if event == 'open' and guard['active'] and isinstance(args[0], (str, bytes)):
                if Path(args[0]).name == 'test.json':
                    guard['attempts'] += 1
                    raise AssertionError('Test accessed')
        sys.addaudithook(deny)
        try:
            out = self.root / 'new'
            result = self.exp.run_training(self.cfg, out)
            self.assertEqual(guard['attempts'], 0)
        finally:
            guard['active'] = False
        self.assertEqual(result['selection_metric'], 'fixed_fusion_predicted_boundary_dev_joint_f1')
        hist = json.loads((out / 'history.json').read_text())
        best = max(hist, key=lambda row: row['fusion']['joint']['f1'])
        self.assertEqual(result['best_epoch'], best['epoch'])
        self.assertEqual(result['best_fusion'], best['fusion'])
        pred = json.loads((out / 'best_fusion_predictions.json').read_text())
        self.assertEqual(pred['b'], [])
        self.assertEqual(pred['a'][0][:2], [1, 2])
        self.assertTrue(json.loads((out / 'reload-verification.json').read_text())['predictions_exact'])
        self.assertEqual(self.exp.verify_completed(out, self.cfg), result)
        with self.assertRaises(FileExistsError):
            self.exp.run_training(self.cfg, out)
        with self.assertRaisesRegex(ValueError, 'config'):
            self.exp.verify_completed(out, dict(self.cfg, variant='span'))
        (out / 'best_fusion_predictions.json').write_text('{}')
        with self.assertRaisesRegex(ValueError, 'artifact|changed'):
            self.exp.verify_completed(out, self.cfg)

    def test_wrong_dataset_dev_identity_and_test_path_rejected(self):
        with self.assertRaisesRegex(ValueError, 'dataset'):
            self.exp.run_training(dict(self.cfg, dataset='twitter2017'), self.root / 'wrong')
        changed = self.root / 'other-dev.json'; changed.write_text(Path(self.cfg['dev_data']).read_text())
        with self.assertRaisesRegex(ValueError, 'identity|bound|cache'):
            self.exp.run_training(dict(self.cfg, dev_data=str(changed)), self.root / 'wrongdev')
        with self.assertRaisesRegex(ValueError, 'Test|test'):
            self.exp.run_training(dict(self.cfg, train_data=str(self.root / 'test.json')), self.root / 'testbad')

    def test_fusion_scoring_counts_all_boundaries_and_empty_prediction_sets(self):
        cache = dict(baseline={'a': [[0, 1, 0], [2, 3, 0]], 'b': []},
            keys=[['a', 0, 1], ['a', 2, 3]], base_logits=[[.1, 0, 0], [.1, 0, 0]])
        gold = {'a': [(0, 1, 2)], 'b': [(0, 1, 0)]}
        row, preds = self.exp.evaluate_scores(cache, [[-5, -5, 5], [-5, -5, 5]], gold)
        self.assertEqual(row['fusion']['joint']['tp'], 1)
        self.assertEqual(row['fusion']['joint']['predicted'], 2)
        self.assertEqual(row['fusion']['joint']['gold'], 2)
        self.assertEqual(row['fusion']['joint']['f1'], .5)
        self.assertEqual(preds['fusion']['b'], [])
        empty = dict(baseline={'a': [], 'b': []}, keys=[], base_logits=[])
        row, preds = self.exp.evaluate_scores(empty, [], gold)
        self.assertEqual(row['fusion']['joint']['f1'], 0.)
        self.assertEqual(preds['target'], {'a': [], 'b': []})

    def test_selection_ignores_classifier_metric_and_preserves_earliest_tie(self):
        self.assertFalse(self.exp.is_better({'fusion': {'joint': {'f1': .5}},
                                           'target': {'joint': {'f1': 1.}}}, .5))
        self.assertTrue(self.exp.is_better({'fusion': {'joint': {'f1': .51}},
                                          'target': {'joint': {'f1': 0.}}}, .5))

    def test_diagnostic_counts_shared_errors_without_treating_cues_as_causes(self):
        from mact.representation_diagnostic import diagnose, error_transitions
        examples = [Example('a', ['good', 'coffee', 'but', 'not', 'service'], 'a.jpg',
                            [(1, 2, 2), (4, 5, 0)]), Example('b', ['fine'], 'b.jpg', [])]
        cache = dict(baseline={'a': [[1, 2, 1], [4, 5, 2]], 'b': []},
            target={'a': [[1, 2, 0], [4, 5, 0]], 'b': []},
            keys=[['a', 1, 2], ['a', 4, 5]], base_logits=[[0, 4, 0], [0, 0, 4]],
            target_logits=[[4, 0, 0], [4, 0, 0]])
        result = diagnose(examples, cache)
        self.assertEqual(result['shared_errors'], 1)
        self.assertEqual(result['slices']['mixed_polarity']['matched'], 2)
        self.assertEqual(result['slices']['negation_cue']['matched'], 2)
        gold = {e.id: e.aspects for e in examples}
        changes = error_transitions(gold, cache['baseline'], cache['target'])
        self.assertEqual(changes['corrected'], 1)
        self.assertEqual(changes['regressed'], 0)


if __name__ == '__main__':
    unittest.main()
