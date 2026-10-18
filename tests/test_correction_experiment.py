import copy
import importlib
import json
import tempfile
import unittest
from pathlib import Path

import torch

from mact.common.cache import file_hash
from mact.sentiment_correction import CorrectionHead


class CorrectionExperimentTests(unittest.TestCase):
    def setUp(self):
        self.m = importlib.import_module('mact.correction_experiment')
        torch.set_num_threads(1)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.inputs = {}
        for name in ('base.pt', 'target.pt', 'train.json', 'dev.json'):
            p = self.root/name
            p.write_text(name)
            self.inputs[str(p)] = file_hash(p)

    def cache(self, split):
        ids = [f'{split}:{i}' for i in range(3)]
        keys = [(ids[0], 0, 1), (ids[0], 2, 3), (ids[1], 0, 1)]
        gold = {ids[0]: [(0, 1, 0)], ids[1]: [(0, 1, 2)], ids[2]: []}
        baseline = {ids[0]: [(0, 1, 0), (2, 3, 1)], ids[1]: [(0, 1, 2)], ids[2]: []}
        scores = torch.tensor([[5., 0., 0.], [0., 5., 0.], [0., 0., 5.]])
        return dict(keys=keys, gold=gold, baseline=baseline,
            base_features=torch.arange(12).float().reshape(3, 4),
            target_features=torch.arange(18).float().reshape(3, 6),
            base_logits=scores, target_logits=scores.clone(),
            base_labels=torch.tensor([0, 1, 2]), target_labels=torch.tensor([0, 1, 2]),
            fixed_labels=torch.tensor([0, 1, 2]), labels=torch.tensor([0, -100, 2]),
            meta=dict(format='sentiment_correction_v1', dataset='twitter2015', seed=42,
                split=split, data_path=str(self.root/f'{split}.json'), sentence_ids=ids,
                input_sha256=self.inputs.copy(), source_sha256=self.m.source_hashes(),
                checkpoints={str(self.root/n): self.inputs[str(self.root/n)] for n in ('base.pt', 'target.pt')},
                dev_replay_exact=True, test_used=False))

    def save(self, cache, name):
        p = self.root/name
        self.m.save_cache(p, cache)
        return p

    def test_train_reload_and_epoch_zero_fallback_preserve_all_boundaries(self):
        train = self.save(self.cache('train'), 'train.pt')
        dev = self.save(self.cache('dev'), 'dev.pt')
        result = self.m.train_pair(train, dev, self.root/'fit', dict(epochs=2, batch_size=2))
        for variant in ('plain', 'bounded'):
            row = result['variants'][variant]
            self.assertEqual(row['selected_epoch'], 0)
            self.assertGreaterEqual(row['best_trained_epoch'], 1)
            self.assertTrue(row['reload_exact'])
            self.assertEqual(len(row['history']), 3)
            pred = json.loads((self.root/'fit'/variant/'predictions.json').read_text())
            self.assertEqual(pred['dev:2'], [])
            self.assertEqual(pred['dev:0'], [[0, 1, 0], [2, 3, 1]])
            self.assertEqual(row['selected_metrics']['joint']['predicted'], 3)
        self.assertEqual(result['variants']['plain']['initial_sha256'], result['variants']['bounded']['initial_sha256'])
        with self.assertRaises(FileExistsError):
            self.m.train_pair(train, dev, self.root/'fit', dict(epochs=1))

    def test_reject_split_overlap_seed_identity_and_corruption(self):
        for change in ('split', 'overlap', 'seed', 'checkpoints', 'labels', 'nonfinite', 'missing_sentence'):
            with self.subTest(change=change):
                train, dev = self.cache('train'), self.cache('dev')
                if change == 'split': dev['meta']['split'] = 'test'
                if change == 'overlap':
                    dev = copy.deepcopy(train)
                    dev['meta']['split'] = 'dev'
                    dev['meta']['data_path'] = str(self.root/'dev.json')
                if change == 'seed': dev['meta']['seed'] = 43
                if change == 'checkpoints': dev['meta']['checkpoints'][str(self.root/'base.pt')] = '0'*64
                if change == 'labels': dev['labels'][1] = 2
                if change == 'nonfinite': dev['base_features'][0, 0] = float('nan')
                if change == 'missing_sentence': del dev['baseline']['dev:2']
                a = self.save(train, f'{change}-train.pt')
                b = self.save(dev, f'{change}-dev.pt')
                with self.assertRaises(ValueError): self.m.train_pair(a, b, self.root/change, dict(epochs=1))
        p = self.save(self.cache('train'), 'corrupt.pt')
        with p.open('ab') as f: f.write(b'corrupt')
        with self.assertRaises(ValueError): self.m.load_cache(p)
        q = self.save(self.cache('train'), 'stale.pt')
        (self.root/'base.pt').write_text('changed')
        with self.assertRaises(ValueError): self.m.load_cache(q)

    def test_inference_never_uses_gold_and_zero_residual_preserves_consensus(self):
        cache = self.cache('dev')
        # Historical fixed consensus can differ from rounded conditional argmax.
        cache['fixed_labels'][0] = 1
        model = CorrectionHead(4, 6, 'plain')
        labels = self.m.predict(model, cache)
        self.assertEqual(labels.tolist(), [1, 1, 2])
        cache['labels'][:] = 2
        cache['gold'] = {}
        self.assertTrue(torch.equal(labels, self.m.predict(model, cache)))
        with torch.no_grad(): model.output.bias[2] = 20
        self.assertEqual(self.m.predict(model, cache).tolist(), [2, 2, 2])


if __name__ == '__main__': unittest.main()
