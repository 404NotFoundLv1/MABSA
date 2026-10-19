import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


class RepresentationBatchTests(unittest.TestCase):
    def setUp(self):
        path = Path(__file__).resolve().parents[1] / 'scripts/run_target_representation.py'
        self.assertTrue(path.is_file(), 'sequential representation batch is missing')
        spec = importlib.util.spec_from_file_location('representation_cli', path)
        self.cli = importlib.util.module_from_spec(spec); spec.loader.exec_module(self.cli)

    def test_schedule_is_six_sequential_paired_jobs(self):
        self.assertEqual(self.cli.schedule(), [
            ('twitter2015', 'marker'), ('twitter2015', 'span'), ('twitter2015', 'context'),
            ('twitter2017', 'marker'), ('twitter2017', 'span'), ('twitter2017', 'context')])

    def test_read_guard_blocks_test_data_cache_and_results_but_allows_train_dev(self):
        root = self.cli.ROOT
        for name in ['assets/data/twitter2015/test.json', 'assets/cache/twitter2015/test/features.pt',
                     'outputs/correction_test_s42_20261018/comparison.json',
                     'reference/runs/twitter2017/seed44/test/predictions.json']:
            with self.subTest(path=name), self.assertRaises(RuntimeError):
                self.cli.deny_test('open', (str(root / name), 'r', 0))
        for name in ['assets/data/twitter2015/train.json', 'assets/data/twitter2017/dev.json',
                     'outputs/new/diagnostic.json', 'assets/cache/twitter2015/train_dev/features.pt']:
            self.cli.deny_test('open', (str(root / name), 'r', 0))

    def test_decision_rejects_changed_source_and_partial_existing_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / 'pilot'
            out.mkdir(); (out / 'random.txt').write_text('partial')
            with self.assertRaisesRegex(ValueError, 'fresh|partial'):
                self.cli.lock_decision(out, {'format': 'test', 'code': {'x': 'a'}})
            out = Path(tmp) / 'fresh'
            plan = {'format': 'test', 'code': {'x': 'a'}}
            self.cli.lock_decision(out, plan)
            self.cli.lock_decision(out, plan)
            with self.assertRaisesRegex(ValueError, 'decision'):
                self.cli.lock_decision(out, {'format': 'test', 'code': {'x': 'b'}})

    def test_paired_comparison_uses_same_dataset_a_and_marks_smoke(self):
        rows = {}
        for dataset, variant in self.cli.schedule():
            f1 = {'marker': .60, 'span': .61, 'context': .59}[variant]
            rows[(dataset, variant)] = dict(best_fusion={'joint': {'f1': f1}},
                historical_reference={'fusion': {'joint': {'f1': .58}}}, is_smoke=True)
        report = self.cli.paired_rows(rows)
        self.assertAlmostEqual(report['twitter2015']['span']['delta_vs_A_pp'], 1.)
        self.assertAlmostEqual(report['twitter2017']['context']['delta_vs_A_pp'], -1.)
        with self.assertRaisesRegex(ValueError, 'incomplete'):
            self.cli.paired_rows({k:v for k,v in rows.items() if k != ('twitter2017','marker')})


if __name__ == '__main__':
    unittest.main()
