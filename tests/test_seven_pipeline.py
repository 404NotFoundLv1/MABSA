import contextlib
import copy
import io
import json
import unittest
from pathlib import Path

import torch
from torch.utils.data import DataLoader

import test_data_model as fixtures
from mact.audit import audit_config
from mact.common.data import Example
from mact.data import SequenceCollator, SequenceDataset
from mact.evaluate import run_evaluation
from mact.model import ModelConfig, SequenceModel
from mact.runtime import evaluate_model, load_checkpoint, model_inputs, save_checkpoint
from mact.train import make_dataset, run_training


FIVE_TAGS = ['O', 'B-POS', 'B-NEU', 'B-NEG', 'I']
SEVEN_TAGS = ['O', 'B-POS', 'B-NEU', 'B-NEG', 'I-POS', 'I-NEU', 'I-NEG']


class SevenPipelineTests(unittest.TestCase):
    def setUp(self):
        fixture = fixtures.DataModelTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        self.tokenizer = fixture.tokenizer
        self.root = Path(fixture.tmp.name)
        self.examples = [
            Example('multi', ['good', 'washington', 'good', 'washington', 'bad', 'washington', 'good'],
                    'multi.jpg', [(0, 2, 2), (2, 4, 1), (4, 6, 0)]),
            Example('empty', ['bad'], 'empty.jpg', []),
        ]

    def model(self, label_scheme='seven'):
        return SequenceModel(fixtures.tiny_text(), ModelConfig(
            mode='text', hidden_size=8, attention_heads=2, dropout=0., label_scheme=label_scheme))

    def resources(self):
        backbone = self.root / 'backbone'
        fixtures.tiny_text().save_pretrained(backbone)
        self.tokenizer.save_pretrained(backbone)
        rows = [dict(id=ex.id, words=ex.words, image_id=ex.image_id,
                     aspects=[{'from': start, 'to': end, 'polarity': polarity}
                              for start, end, polarity in ex.aspects]) for ex in self.examples]
        for split, values in [('train', rows), ('dev', [dict(rows[0], id='dev')])]:
            (self.root / f'{split}.json').write_text(json.dumps(values))
        return dict(text_model=str(backbone), train_data=str(self.root / 'train.json'),
                    dev_data=str(self.root / 'dev.json'), device='cpu', num_threads=1,
                    accumulation=2, epochs=1, log_every=0,
                    model=dict(mode='text', hidden_size=8, attention_heads=2, dropout=0.,
                               label_scheme='seven'))

    def neutral_span_model(self):
        model = self.model().eval()
        with torch.no_grad():
            model.text_head.weight.zero_()
            model.text_head.bias.copy_(torch.tensor([0., 0., 5., 0., 0., 10., 0.]))
            for parameter in model.crf.parameters():
                parameter.zero_()
        return model

    def test_multitoken_polarities_subwords_and_padding_use_selected_schema(self):
        dataset = SequenceDataset(self.examples, self.tokenizer, label_scheme='seven')
        batch = SequenceCollator(self.tokenizer.pad_token_id)(list(dataset))
        self.assertEqual(batch['tags'].tolist(), [[1, 4, 2, 5, 3, 6, 0], [0, -100, -100, -100, -100, -100, -100]])
        self.assertEqual(batch['word_ids'][0].tolist(), [-1, 0, 1, 1, 2, 3, 3, 4, 5, 5, 6, -1])
        self.assertEqual(batch['word_mask'][1].tolist(), [True, False, False, False, False, False, False])
        self.assertNotIn('spans', batch)
        legacy = make_dataset(self.examples, self.tokenizer, {'max_length': 256}, None)
        self.assertEqual(legacy[0]['tags'].tolist(), [1, 4, 2, 4, 3, 4, 0])
        configured = make_dataset(self.examples, self.tokenizer,
                                  {'max_length': 256, 'model': {'label_scheme': 'seven'}}, None)
        self.assertEqual(configured[0]['tags'].tolist(), [1, 4, 2, 5, 3, 6, 0])

    def test_invalid_schema_is_rejected_even_for_an_empty_dataset(self):
        with self.assertRaises(ValueError):
            SequenceDataset([], self.tokenizer, label_scheme='unknown')

    def test_evaluation_decodes_inside_polarity_without_gold_inputs(self):
        example = Example('neutral', ['good', 'washington'], 'a.jpg', [(0, 2, 1)])
        loader = DataLoader(SequenceDataset([example], self.tokenizer, label_scheme='seven'),
                            collate_fn=SequenceCollator(self.tokenizer.pad_token_id))
        batch = next(iter(loader))
        model = self.neutral_span_model()
        expected = {'neutral': [(0, 2, 1)]}
        metrics, predictions = evaluate_model(model, [batch], 'cpu')
        self.assertEqual(predictions, expected)
        self.assertEqual(metrics['joint']['f1'], 1.)
        batch['tags'].fill_(-888)
        self.assertNotIn('tags', model_inputs(batch, 'cpu'))
        self.assertEqual(evaluate_model(model, [batch], 'cpu')[1], expected)
        del batch['tags']
        example.aspects = []
        metrics, predictions = evaluate_model(model, [batch], 'cpu')
        self.assertEqual(predictions, expected)
        self.assertEqual(metrics['joint']['gold'], 0)

    def test_seven_checkpoint_export_uses_model_schema_over_stale_run_config(self):
        run = self.root / 'run'
        run.mkdir()
        self.tokenizer.save_pretrained(run / 'tokenizer')
        data = self.root / 'data.json'
        data.write_text(json.dumps([dict(id='neutral', words=['good', 'washington'], image_id='a.jpg',
                                        aspects=[{'from': 0, 'to': 2, 'polarity': 'neutral'}])]))
        model = self.neutral_span_model()
        checkpoint = run / 'model.pt'
        save_checkpoint(checkpoint, model,
                        {'num_threads': 1, 'max_length': 256, 'model': {'label_scheme': 'five'}}, 1, {})
        content = torch.load(checkpoint, weights_only=True)
        self.assertEqual(content['format'], 'mabsa_sequence_crf_v2')
        self.assertEqual(content['tags'], SEVEN_TAGS)
        loaded, metadata = load_checkpoint(checkpoint)
        self.assertEqual(loaded.config.label_scheme, 'seven')
        self.assertEqual(metadata['run_config']['model']['label_scheme'], 'five')
        output = self.root / 'eval'
        metrics, predictions = run_evaluation(checkpoint, data, output)
        self.assertEqual(metrics['joint']['f1'], 1.)
        self.assertEqual(predictions, {'neutral': [(0, 2, 1)]})
        self.assertEqual(json.loads((output / 'word_tags.json').read_text())[0]['tags'], ['B-NEU', 'I-NEU'])
        provenance = json.loads((output / 'provenance.json').read_text())
        self.assertEqual(provenance['tags'], SEVEN_TAGS)
        self.assertEqual(provenance['label_scheme'], 'seven')
        self.assertEqual(json.loads((output / 'metrics.json').read_text()), metrics)
        data.write_text(json.dumps([dict(id='neutral', words=['good', 'washington'], image_id='a.jpg')]))
        prediction_output = self.root / 'predict'
        self.assertEqual(run_evaluation(checkpoint, data, prediction_output, predict_only=True), (None, predictions))
        self.assertFalse((prediction_output / 'metrics.json').exists())

    def test_legacy_v1_without_schema_still_loads_without_rewriting_metadata(self):
        model = SequenceModel(fixtures.tiny_text(), ModelConfig(mode='text', hidden_size=8,
                                                               attention_heads=2, dropout=0.)).eval()
        checkpoint = self.root / 'legacy.pt'
        save_checkpoint(checkpoint, model, {'seed': 42}, 1, {})
        content = torch.load(checkpoint, weights_only=True)
        self.assertEqual(content['format'], 'mabsa_sequence_crf_v1')
        self.assertEqual(content['tags'], FIVE_TAGS)
        content['model_config'].pop('label_scheme', None)
        torch.save(content, checkpoint)
        loaded, metadata = load_checkpoint(checkpoint)
        self.assertEqual(getattr(loaded.config, 'label_scheme', 'five'), 'five')
        self.assertNotIn('label_scheme', metadata['model_config'])
        self.assertEqual(metadata['run_config'], {'seed': 42})
        batch = SequenceCollator(self.tokenizer.pad_token_id)(list(SequenceDataset(self.examples, self.tokenizer)))
        torch.testing.assert_close(model(**model_inputs(batch, 'cpu'))['emissions'],
                                   loaded(**model_inputs(batch, 'cpu'))['emissions'], rtol=0, atol=0)

    def test_checkpoint_format_tags_and_schema_must_agree(self):
        model = SequenceModel(fixtures.tiny_text(), ModelConfig(mode='text', hidden_size=8,
                                                               attention_heads=2, dropout=0.))
        checkpoint = self.root / 'bad.pt'
        save_checkpoint(checkpoint, model, {}, 1, {})
        original = torch.load(checkpoint, weights_only=True)
        for version, tags, scheme in [
            ('v1', FIVE_TAGS, 'seven'), ('v1', SEVEN_TAGS, 'seven'),
            ('v2', FIVE_TAGS, 'seven'), ('v2', SEVEN_TAGS, 'five'),
            ('v2', SEVEN_TAGS, None), ('v1', FIVE_TAGS, 'invalid'),
        ]:
            with self.subTest(version=version, tags=tags, scheme=scheme):
                content = copy.deepcopy(original)
                content.update(format='mabsa_sequence_crf_' + version, tags=tags)
                content['model_config'].pop('label_scheme', None)
                if scheme is not None:
                    content['model_config']['label_scheme'] = scheme
                torch.save(content, checkpoint)
                with self.assertRaises(ValueError):
                    load_checkpoint(checkpoint)

    def test_audit_roundtrip_counts_inside_tags_for_each_polarity(self):
        cfg = self.resources()
        config = self.root / 'config.json'
        config.write_text(json.dumps(cfg))
        audit = audit_config(config)
        self.assertTrue(audit['train']['roundtrip_exact'])
        self.assertEqual(audit['train']['tag_counts'],
                         {'O': 2, 'B-POS': 1, 'B-NEU': 1, 'B-NEG': 1, 'I-POS': 1, 'I-NEU': 1, 'I-NEG': 1})

    def test_real_seven_training_reload_metrics_and_auxiliary_provenance(self):
        cfg = self.resources()
        for weight in (0., .2):
            with self.subTest(aspect_sentiment_weight=weight):
                cfg['model']['aspect_sentiment_weight'] = weight
                output = self.root / f'run-{weight}'
                stdout = io.StringIO()
                with contextlib.redirect_stdout(stdout):
                    summary = run_training(cfg, output)
                self.assertEqual(summary['completed_epochs'], 1)
                self.assertEqual(summary['optimizer_updates'], 1)
                self.assertGreater(summary['emission_head_delta_l2'], 0.)
                self.assertIn('MABSA seven-tag CRF', stdout.getvalue())
                model, metadata = load_checkpoint(output / 'best.pt')
                self.assertEqual(model.config.label_scheme, 'seven')
                self.assertEqual(metadata['tags'], SEVEN_TAGS)
                metrics, _ = run_evaluation(output / 'best.pt', cfg['dev_data'], self.root / f'eval-{weight}')
                self.assertEqual(metrics, summary['best_dev'])
                provenance = json.loads((output / 'provenance.json').read_text())
                self.assertEqual(provenance['tags'], SEVEN_TAGS)
                self.assertEqual(provenance['label_scheme'], 'seven')
                self.assertEqual(provenance['initialization_strategy'], 'five_tag_inside_parameter_expansion')
                if weight:
                    self.assertEqual(provenance['decoding'], 'seven_tag_crf')


if __name__ == '__main__':
    unittest.main()
