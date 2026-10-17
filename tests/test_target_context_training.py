import importlib.util
import json
from pathlib import Path
import sys
import unittest

import test_data_model as fixtures
from mact.common.cache import file_hash
from mact.common.metrics import evaluate_predictions


class TargetTrainingTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('mact.target_context_train'),
                             'target Train/Dev runner is missing')
        from mact import target_context_train as tt
        self.tt = tt
        self.fixture = fixtures.DataModelTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.root = Path(self.fixture.tmp.name)
        model_dir = self.root / 'text'
        fixtures.tiny_text().save_pretrained(model_dir)
        self.fixture.tokenizer.save_pretrained(model_dir)
        for split, word in [('train', 'good'), ('dev', 'bad')]:
            (self.root / (split + '.json')).write_text(json.dumps([
                dict(id='a', words=[word, 'washington'], image_id='a.jpg',
                     aspects=[{'from': 1, 'to': 2, 'polarity': 'positive'}]),
                dict(id='b', words=['bad'], image_id='b.jpg', aspects=[])]))
        self.pred = {'a': [[1, 2, 0]], 'b': []}
        (self.root / 'pred.json').write_text(json.dumps(self.pred))
        (self.root / 'baseline.pt').write_bytes(b'unchanged archived baseline')
        metrics = evaluate_predictions({'a': [(1, 2, 2)], 'b': []}, self.pred)
        (self.root / 'summary.json').write_text(json.dumps(dict(best_dev=metrics,
            checkpoint_sha256=file_hash(self.root / 'baseline.pt'))))
        self.cfg = dict(text_model=str(model_dir), train_data=str(self.root / 'train.json'),
            dev_data=str(self.root / 'dev.json'), baseline_predictions=str(self.root / 'pred.json'),
            baseline_summary=str(self.root / 'summary.json'), baseline_checkpoint=str(self.root / 'baseline.pt'),
            seed=42, epochs=2, batch_size=2, eval_batch_size=2, device='cpu', num_threads=1,
            learning_rate=2e-5, head_learning_rate=1e-4, weight_decay=.01, warmup_ratio=.1,
            gradient_clip=1., dropout=.1, max_length=258, train_limit=None)

    def test_real_training_full_dev_reload_and_fresh_output(self):
        before = file_hash(self.root / 'baseline.pt')
        guard = {'active': True, 'attempts': 0}
        def deny_test(event, args):
            if event == 'open' and guard['active'] and isinstance(args[0], (str, bytes)):
                if Path(args[0]).name == 'test.json':
                    guard['attempts'] += 1
                    raise AssertionError('Test file opened during Train/Dev pilot')
        sys.addaudithook(deny_test)
        try:
            with self.assertRaisesRegex(AssertionError, 'Test file'):
                (self.root / 'test.json').read_text()
            guard['attempts'] = 0
            result = self.tt.run_training(self.cfg, self.root / 'run')
            self.assertEqual(guard['attempts'], 0)
        finally:
            guard['active'] = False
        self.assertEqual(result['completed_epochs'], 2)
        self.assertEqual(result['selection_metric'], 'predicted_boundary_dev_joint_f1')
        verification = json.loads((self.root / 'run/reload-verification.json').read_text())
        self.assertTrue(verification['predictions_exact'])
        self.assertTrue(verification['full_dev'])
        pred = json.loads((self.root / 'run/best_dev_predictions.json').read_text())
        self.assertEqual(list(pred), ['a', 'b'])
        self.assertEqual(pred['a'][0][:2], [1, 2])
        self.assertEqual(pred['b'], [])
        self.assertEqual(file_hash(self.root / 'baseline.pt'), before)
        self.assertEqual(json.loads((self.root / 'run.status.json').read_text())['state'], 'completed')
        with self.assertRaises(FileExistsError):
            self.tt.run_training(self.cfg, self.root / 'run')

    def test_stale_baseline_and_test_inputs_are_rejected(self):
        cfg = dict(self.cfg, test_data=str(self.root / 'test.json'))
        with self.assertRaises(ValueError):
            self.tt.run_training(cfg, self.root / 'forbidden')
        bad = json.loads((self.root / 'summary.json').read_text())
        bad['best_dev']['joint']['f1'] = .8
        (self.root / 'summary.json').write_text(json.dumps(bad))
        with self.assertRaisesRegex(ValueError, 'baseline'):
            self.tt.run_training(self.cfg, self.root / 'bad')
        self.assertEqual(json.loads((self.root / 'bad.status.json').read_text())['state'], 'failed')

    def test_selection_uses_primary_metric_and_earliest_tie(self):
        self.assertFalse(self.tt.is_better({'joint': {'f1': .4}, 'gold_accuracy': .99}, .5))
        self.assertFalse(self.tt.is_better({'joint': {'f1': .5}}, .5))
        self.assertTrue(self.tt.is_better({'joint': {'f1': .6}, 'gold_accuracy': 0.}, .5))


if __name__ == '__main__':
    unittest.main()
