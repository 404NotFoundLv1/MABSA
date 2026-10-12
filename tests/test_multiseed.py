import contextlib
import io
import json
import sys
import unittest
from pathlib import Path

import test_data_model as fixtures
from mact.train import run_training

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mact.experiment import summarize_rows


class MultiseedTests(unittest.TestCase):
    def test_statistics_require_all_seeds_and_use_sample_standard_deviation(self):
        rows = [{'dataset': 'twitter2015', 'seed': seed, 'test_joint_f1': score}
                for seed, score in [(42, .6), (43, .7), (44, .8)]]
        summary = summarize_rows(rows, expected_seeds=(42, 43, 44))['twitter2015']
        self.assertEqual(summary['state'], 'completed')
        self.assertAlmostEqual(summary['test_joint_f1_mean'], .7)
        self.assertAlmostEqual(summary['test_joint_f1_sample_std'], .1)
        partial = summarize_rows(rows[:2], expected_seeds=(42, 43, 44))['twitter2015']
        self.assertEqual(partial['missing_seeds'], [44])
        self.assertIsNone(partial['test_joint_f1_mean'])
        with self.assertRaises(ValueError):
            summarize_rows(rows + [rows[0]], expected_seeds=(42, 43, 44))
        self.assertEqual(summarize_rows([], expected_seeds=(24,25,26,27))['twitter2015']['missing_seeds'], [24, 25, 26, 27])

    def test_log_every_zero_suppresses_all_steps_but_keeps_each_epoch_result(self):
        fixture = fixtures.DataModelTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        root = Path(fixture.tmp.name)
        backbone = root / 'backbone'
        fixtures.tiny_text().save_pretrained(backbone)
        fixture.tokenizer.save_pretrained(backbone)
        row = {'id': 'train', 'words': ['good', 'washington'], 'image_id': 'unused.jpg',
               'aspects': [{'from': 1, 'to': 2, 'polarity': 'positive'}]}
        (root / 'train.json').write_text(json.dumps([row]))
        (root / 'dev.json').write_text(json.dumps([dict(row, id='dev')]))
        output = root / 'run'
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            result = run_training(dict(seed=43, text_model=str(backbone), train_data=str(root / 'train.json'),
                dev_data=str(root / 'dev.json'), device='cpu', num_threads=1, epochs=2, log_every=0,
                model=dict(mode='text', hidden_size=8, attention_heads=2, dropout=0.)), output)
        for text in (buffer.getvalue(), (output / 'log.txt').read_text()):
            self.assertNotIn('Step [', text)
            self.assertIn('Epoch [1/2], Mean Loss:', text)
            self.assertIn('Epoch [2/2], Mean Loss:', text)
            self.assertEqual(sum(line.startswith('DEV  aesc_p:') for line in buffer.getvalue().splitlines()), 2)
            self.assertIn('BEST DEV  aesc_p:', text)
        self.assertEqual(result['completed_epochs'], 2)


if __name__ == '__main__':
    unittest.main()
