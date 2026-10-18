import importlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from transformers import BertConfig, BertModel, BertTokenizerFast

from mact.common.cache import file_hash, model_hashes
from mact.train import resolve_config
from mact.runtime import write_json


class OOFFoldIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.m = importlib.import_module('mact.oof_correction')
        self.data = importlib.import_module('mact.oof_data')
        torch.set_num_threads(1)
        torch.manual_seed(42)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for module in (self.m, self.data):
            p = patch.object(module, 'ROOT', self.root)
            p.start(); self.addCleanup(p.stop)
        text = self.root/'assets/models/deberta-v3-base'
        text.mkdir(parents=True)
        vocab = text/'vocab.txt'
        vocab.write_text('[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\ngood\nbad\n')
        BertTokenizerFast(vocab_file=str(vocab)).save_pretrained(text)
        BertModel(BertConfig(vocab_size=7, hidden_size=16, num_hidden_layers=1,
                            num_attention_heads=2, intermediate_size=24)).save_pretrained(text)
        vision = self.root/'assets/models/siglip2'
        vision.mkdir(); write_json(vision/'config.json', {})
        images = self.root/'assets/images/twitter2015'
        images.mkdir(parents=True)
        records, features, hashes = [], {}, {}
        for i in range(18):
            name = f'{i}.jpg'
            (images/name).write_bytes(str(i).encode())
            hashes[name] = file_hash(images/name)
            records.append(dict(id=f'train:{i}', words=['good', str(i)], image_id=name,
                                aspects=[dict(**{'from': 0}, to=1, polarity='POS')]))
            features[name] = torch.zeros(2, 8)
        source = self.root/'assets/data/twitter2015'
        source.mkdir(parents=True); write_json(source/'train.json', records)
        cache = self.root/'assets/cache/twitter2015/train_dev'
        cache.mkdir(parents=True); torch.save(features, cache/'features.pt')
        write_json(cache/'manifest.json', dict(format_version=1, vision_model=str(vision), image_root=str(images),
            model_sha256=model_hashes(vision), image_sha256=hashes, token_shape=[2, 8], features_sha256=file_hash(cache/'features.pt')))
        splits = self.data.prepare_dataset('twitter2015', self.root/'out')
        base = resolve_config(dict(seed=42, epochs=1, batch_size=2, accumulation=1, eval_batch_size=1,
            learning_rate=.001, head_learning_rate=.01,
            text_model=str(text), vision_model=str(vision), visual_cache=str(cache), image_root=str(images),
            num_threads=1, device='cpu', log_every=0,
            model=dict(mode='fusion', hidden_size=8, visual_size=8, attention_heads=2, dropout=0., visual_dropout=0.)))
        target = dict(seed=42, epochs=1, batch_size=2, eval_batch_size=8, learning_rate=2e-5,
            head_learning_rate=1e-4, weight_decay=.01, warmup_ratio=.1, gradient_clip=1., dropout=.1,
            max_length=258, num_threads=1, device='cpu', train_limit=None, text_model=str(text))
        self.job = self.m.make_job('twitter2015', 0, splits, base, target)

    def test_real_fold_training_inner_replay_outer_features_and_stale_refusal(self):
        # Real tiny pretrained models exercise both trainers and checkpoint paths.
        torch.use_deterministic_algorithms(True)  # previous CPU head fitting changed global state
        result = self.m.run_fold(self.job)
        self.assertFalse(torch.are_deterministic_algorithms_enabled())
        path = Path(result['cache'])
        c = self.m.load_fold(path, self.job)
        expected = self.job['membership']['held_ids']
        self.assertEqual(set(c['gold']), set(expected))
        self.assertFalse(set(c['gold']) & set(self.job['membership']['fit_ids']))
        self.assertTrue(result['inner_replay_exact'])
        self.assertEqual(c['base_features'].shape[1], 8)
        self.assertEqual(c['target_features'].shape[1], 16)
        self.assertEqual(self.m.run_fold(self.job), result)
        bad = dict(self.job)
        bad['base_config'] = dict(self.job['base_config'], train_data=self.job['held_data'])
        with self.assertRaises(ValueError): self.m.validate_job(bad)
        Path(self.job['base_config']['train_data']).write_text('[]')
        with self.assertRaises(ValueError): self.m.load_fold(path, self.job)

    def test_three_real_folds_aggregate_fit_audit_and_reject_old_sources(self):
        from mact.correction_experiment import load_cache, save_cache, source_hashes
        import copy
        jobs, paths = [], []
        for f in range(3):
            j = self.m.make_job('twitter2015', f, self.job['splits'], self.job['base_config'], self.job['target_config'])
            jobs.append(j); paths.append(self.m.run_fold(j)['cache'])
        aggregate = self.root/'out/twitter2015/oof.pt'
        merged = self.m.aggregate_oof(jobs, paths, aggregate)
        self.assertEqual(len(merged['gold']), 18)
        self.assertGreater(int((merged['labels'] >= 0).sum()), 0)
        with self.assertRaises(ValueError): load_cache(aggregate)
        dev = copy.deepcopy(self.m.load_fold(paths[0], jobs[0]))
        rename = {sid: 'dev:'+sid for sid in dev['gold']}
        for key in ('gold', 'baseline'): dev[key] = {rename[sid]: v for sid, v in dev[key].items()}
        dev['keys'] = [(rename[sid], s, e) for sid, s, e in dev['keys']]
        data = self.root/'dev.json'; data.write_text('dev identity fixture')
        dev['meta'] = dict(format='sentiment_correction_v1', dataset='twitter2015', seed=42, split='dev',
            data_path=str(data), sentence_ids=list(dev['gold']), input_sha256={**dev['meta']['input_sha256'], str(data): file_hash(data)},
            source_sha256=source_hashes(), checkpoints=dev['meta']['checkpoints'], dev_replay_exact=True, test_used=False)
        dev_path = self.root/'dev.pt'; save_cache(dev_path, dev)
        with patch.object(self.m, 'target_run', return_value=Path(jobs[0]['output'])/'target'):
            result = self.m.fit_oof_pair(aggregate, dev_path, self.root/'fit', dict(epochs=2, batch_size=2))
        self.assertTrue(result['variants']['plain']['reload_exact'])
        self.m.audit_fit(self.root/'fit', dev_path)
        bad = torch.load(aggregate, weights_only=True)
        bad['base_features'][0, 0] += 1
        corrupt = self.root/'corrupt.pt'; save_cache(corrupt, bad)
        with self.assertRaises(ValueError): self.m.load_oof(corrupt)
        p = Path(jobs[0]['output'])/'base/provenance.json'
        meta = json.loads(p.read_text()); meta['source_sha256']['train.py'] = '0'*64
        write_json(p, meta)
        with self.assertRaises(ValueError): self.m.verify_expert(jobs[0], 'base')

    def test_incomplete_fold_set_cannot_be_aggregated(self):
        with self.assertRaises(ValueError): self.m.aggregate_oof([], [], self.root/'oof.pt')
        with self.assertRaises(ValueError): self.m.aggregate_oof([self.job]*3, [], self.root/'oof.pt')


if __name__ == '__main__': unittest.main()
