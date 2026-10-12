import contextlib
import io
import json
import unittest
from pathlib import Path

import test_data_model as fixtures
from mact.train import run_training


class LoggingTests(unittest.TestCase):
    def test_training_prints_vlp_steps_and_percent_metrics_without_json_noise(self):
        fixture = fixtures.DataModelTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        root = Path(fixture.tmp.name)
        backbone = root / 'backbone'
        fixtures.tiny_text().save_pretrained(backbone)
        fixture.tokenizer.save_pretrained(backbone)
        rows = [{'id': str(i), 'words': ['good', 'washington'], 'image_id': 'unused.jpg',
                 'aspects': [{'from': 1, 'to': 2, 'polarity': 'positive'}]} for i in range(3)]
        for split, values in [('train', rows), ('dev', [dict(rows[0], id='dev')])]:
            (root / f'{split}.json').write_text(json.dumps(values))
        output = root / 'run'
        cfg = dict(text_model=str(backbone), train_data=str(root / 'train.json'),
            dev_data=str(root / 'dev.json'), device='cpu', num_threads=1,
            accumulation=2, epochs=2, log_every=1,
            model=dict(mode='text', hidden_size=8, attention_heads=2, dropout=0.))
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            summary = run_training(cfg, output)
        stdout = buffer.getvalue()
        self.assertIn('Epoch [1/2], Step [1/3], Loss:', stdout)
        self.assertIn('Epoch [1/2], Step [3/3], Loss:', stdout)
        self.assertIn('Epoch [2/2], Step [3/3], Loss:', stdout)
        self.assertIn('BEST DEV:-----', stdout)
        self.assertNotIn('"event":', stdout)
        self.assertNotIn('TEST  aesc_p:', stdout)
        self.assertEqual(summary['optimizer_updates'], 4)
        self.assertEqual(summary['completed_epochs'], 2)
        for split, metric in [('DEV', summary['last_dev']['joint']), ('BEST DEV', summary['best_dev']['joint'])]:
            expected = f"{split}  aesc_p:{metric['precision']*100:.2f} aesc_r:{metric['recall']*100:.2f} aesc_f:{metric['f1']*100:.2f}"
            self.assertIn(expected, stdout)
            self.assertIn(expected, (output / 'log.txt').read_text())
        events = [json.loads(line) for line in (output / 'events.jsonl').read_text().splitlines()]
        self.assertEqual([x['epoch'] for x in events if x['event'] == 'epoch'], [1, 2])
        self.assertEqual(summary['selection_metric'], 'dev_joint_f1')


if __name__ == '__main__':
    unittest.main()
