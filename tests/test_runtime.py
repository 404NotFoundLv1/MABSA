import copy
import json
import tempfile
import unittest
from pathlib import Path

import torch
from torch.utils.data import DataLoader

import test_data_model as fixtures
from mact.data import SequenceDataset, SequenceCollator, load_labeled
from mact.model import ModelConfig, SequenceModel
from mact.runtime import train_epoch, evaluate_model, save_checkpoint, load_checkpoint, model_inputs
from mact.train import resolve_config, run_training
from mact.evaluate import run_evaluation
from mact.common.cache import file_hash


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.DataModelTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.tmp = Path(self.fixture.tmp.name)

    def model(self):
        return SequenceModel(fixtures.tiny_text(), ModelConfig(mode='text', hidden_size=8, attention_heads=2, dropout=0.))

    def loader(self):
        examples = self.fixture.examples + [copy.deepcopy(self.fixture.examples[0])]
        examples[2].id = 'c'
        return DataLoader(SequenceDataset(examples, self.fixture.tokenizer), batch_size=1,
                          collate_fn=SequenceCollator(0), shuffle=False)

    def test_partial_accumulation_matches_sentence_mean_updates(self):
        torch.manual_seed(8)
        actual, expected = self.model(), None
        expected = copy.deepcopy(actual)
        opt_a = torch.optim.SGD(actual.parameters(), lr=.03)
        opt_b = torch.optim.SGD(expected.parameters(), lr=.03)
        batches = list(self.loader())
        result = train_epoch(actual, batches, opt_a, 'cpu', accumulation=2, gradient_clip=0, log_every=0)
        for window in (batches[:2], batches[2:]):
            opt_b.zero_grad()
            loss = sum(expected(**model_inputs(b, 'cpu', include_tags=True))['losses'].sum() for b in window) / len(window)
            loss.backward()
            opt_b.step()
        self.assertEqual(result['updates'], 2)
        self.assertEqual(result['examples'], 3)
        for a, b in zip(actual.parameters(), expected.parameters()):
            torch.testing.assert_close(a, b)

    def test_checkpoint_reload_reproduces_emissions_and_predictions(self):
        model = self.model().eval()
        loader = self.loader()
        metrics, predictions = evaluate_model(model, loader, 'cpu')
        checkpoint = self.tmp / 'model.pt'
        save_checkpoint(checkpoint, model, {'seed': 42}, 1, metrics)
        loaded, meta = load_checkpoint(checkpoint)
        self.assertEqual(meta['epoch'], 1)
        self.assertEqual(evaluate_model(loaded, loader, 'cpu'), (metrics, predictions))
        batch = next(iter(loader))
        torch.testing.assert_close(model(**model_inputs(batch, 'cpu'))['emissions'],
                                   loaded(**model_inputs(batch, 'cpu'))['emissions'], rtol=0, atol=0)
        # Predictions must never read gold tags.
        batch['tags'].fill_(4)
        self.assertNotIn('tags', model_inputs(batch, 'cpu'))
        loaded(**model_inputs(batch, 'cpu'))

    def test_missing_gold_is_rejected_but_empty_gold_is_valid(self):
        path = self.tmp / 'data.json'
        row = {'words': ['bad'], 'image_id': 'a.jpg'}
        path.write_text(json.dumps([row]))
        with self.assertRaises(ValueError):
            load_labeled(path)
        row['aspects'] = []
        path.write_text(json.dumps([row]))
        self.assertEqual(load_labeled(path)[0][0].aspects, [])

    def test_unknown_or_invalid_config_fails_before_training(self):
        for cfg in ({'bogus': 1}, {'accumulation': 0}, {'model': {'mode': 'span'}}, {'train_limit': 0}):
            with self.assertRaises(ValueError):
                resolve_config(cfg)

    def test_limited_training_counts_partial_epoch_and_evaluation_records_identity(self):
        backbone = self.tmp / 'backbone'
        fixtures.tiny_text().save_pretrained(backbone)
        self.fixture.tokenizer.save_pretrained(backbone)
        rows = [{'id': str(i), 'words': ['good', 'washington'], 'image_id': 'unused.jpg',
                 'aspects': [{'from': 1, 'to': 2, 'polarity': 'positive'}]} for i in range(3)]
        train_path, dev_path = self.tmp / 'train.json', self.tmp / 'dev.json'
        train_path.write_text(json.dumps(rows))
        dev_path.write_text(json.dumps([dict(rows[0], id='dev')]))
        output = self.tmp / 'run'
        cfg = dict(text_model=str(backbone), train_data=str(train_path), dev_data=str(dev_path),
                   device='cpu', num_threads=1, accumulation=2, epochs=3, max_updates=1,
                   log_every=0, model=dict(mode='text', hidden_size=8, attention_heads=2, dropout=0.))
        summary = run_training(cfg, output)
        self.assertEqual(summary['completed_epochs'], 0)
        self.assertEqual(summary['evaluated_epochs'], 1)
        metrics, _ = run_evaluation(output / 'best.pt', dev_path, self.tmp / 'eval')
        self.assertEqual(metrics, summary['best_dev'])
        manifest = json.loads((self.tmp / 'eval' / 'provenance.json').read_text())
        self.assertEqual(manifest['checkpoint_sha256'], file_hash(output / 'best.pt'))
        self.assertEqual(manifest['data_sha256'], file_hash(dev_path))
        self.assertFalse(manifest['force_null'])
        self.assertFalse(manifest['predict_only'])


if __name__ == '__main__':
    unittest.main()
