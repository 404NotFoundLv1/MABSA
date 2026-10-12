import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from mact.experiment import resolve_experiment, summarize_rows
from mact.common.cache import open_cache, file_hash, model_hashes
from mact.common.data import Example
import mact.experiment as experiment
import test_data_model as fixtures


class ProjectTests(unittest.TestCase):
    def test_shared_entrypoint_trains_reloads_and_evaluates_test_with_local_assets(self):
        fixture=fixtures.DataModelTests();fixture.setUp();self.addCleanup(fixture.doCleanups)
        root=Path(fixture.tmp.name)
        model=root/'model';fixtures.tiny_text().save_pretrained(model);fixture.tokenizer.save_pretrained(model)
        vision=root/'vision';vision.mkdir();(vision/'config.json').write_text('{}')
        images=root/'images';images.mkdir();(images/'a.jpg').write_bytes(b'fixture-image')
        cache=root/'cache';cache.mkdir();torch.save({'a.jpg':torch.ones(2,8)},cache/'features.pt')
        meta=dict(format_version=1,vision_model='../vision',image_root='../images',model_sha256=model_hashes(vision),
                  features_sha256=file_hash(cache/'features.pt'),image_sha256={'a.jpg':file_hash(images/'a.jpg')},token_shape=[2,8])
        (cache/'manifest.json').write_text(json.dumps(meta))
        for split in ('train','dev','test'):
            (root/f'{split}.json').write_text(json.dumps([dict(id=split,words=['good','washington'],image_id='a.jpg',
                aspects=[dict(**{'from':1,'to':2},polarity='positive')])]))
        (root/'configs/datasets').mkdir(parents=True)
        (root/'configs/base.json').write_text(json.dumps(dict(epochs=2,log_every=0,device='cpu',num_threads=1,
            text_model='model',vision_model='vision',model=dict(mode='fusion',hidden_size=8,attention_heads=2,visual_size=8))))
        (root/'configs/datasets/twitter2015.json').write_text(json.dumps(dict(train_data='train.json',dev_data='dev.json',
            test_data='test.json',image_root='images',visual_cache='cache',test_visual_cache='cache')))
        with patch.object(experiment,'ROOT',root):
            out=root/'outputs/run'
            summary=experiment.run_experiment('configs/base.json','twitter2015',seed=25,output=out)
        self.assertEqual(summary['completed_epochs'],2)
        self.assertEqual(json.loads(out.with_name('run.status.json').read_text())['state'],'completed')
        selection=json.loads((out/'test/selection.json').read_text())
        self.assertEqual(selection['selected_on'],'dev_joint_f1')
        self.assertFalse(selection['test_used_for_selection'])
        self.assertNotIn('test_data',json.loads((out/'config.json').read_text()))
        self.assertTrue(json.loads((out/'reload-verification.json').read_text())['predictions_exact'])

    def test_resolved_config_uses_one_baseline_and_only_local_resources(self):
        root = Path(__file__).resolve().parents[1]
        cfg, test = resolve_experiment(root / 'configs/base.json', 'twitter2015', seed=25, device='cpu')
        self.assertEqual((cfg['seed'], cfg['epochs'], cfg['log_every']), (25, 35, 0))
        self.assertEqual(cfg['model']['mode'], 'fusion')
        for key in ('text_model','vision_model','visual_cache','train_data','dev_data','image_root'):
            self.assertTrue(Path(cfg[key]).is_relative_to(root))
        self.assertTrue(Path(test['test_data']).is_relative_to(root))
        self.assertNotIn('test_data', cfg)
        self.assertNotIn('test_visual_cache', cfg)

    def test_unknown_dataset_and_external_resources_are_rejected(self):
        root = Path(__file__).resolve().parents[1]
        with self.assertRaises(ValueError):
            resolve_experiment(root / 'configs/base.json', '../old-project')
        with tempfile.TemporaryDirectory() as tmp:
            config = Path(tmp) / 'bad.json'
            cfg = json.loads((root/'configs/base.json').read_text())
            cfg['text_model'] = '/tmp/unrelated-model'
            config.write_text(json.dumps(cfg))
            with self.assertRaises(ValueError):
                resolve_experiment(config, 'twitter2015')

    def test_cache_identity_paths_are_relative_to_manifest_not_working_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); cache=root/'cache'; images=root/'images'; vision=root/'vision'
            for p in (cache,images,vision):p.mkdir()
            (images/'a.jpg').write_bytes(b'fixture-image')
            (vision/'config.json').write_text('{}')
            (vision/'model.safetensors').write_bytes(b'fixture-weights')
            torch.save({'a.jpg':torch.ones(2,3)},cache/'features.pt')
            meta=dict(format_version=1,vision_model='../vision',image_root='../images',
                      model_sha256=model_hashes(vision),features_sha256=file_hash(cache/'features.pt'),
                      image_sha256={'a.jpg':file_hash(images/'a.jpg')},token_shape=[2,3])
            (cache/'manifest.json').write_text(json.dumps(meta))
            features,_=open_cache(cache,vision,images,[Example('a',['word'],'a.jpg',[])])
            self.assertEqual(features['a.jpg'].shape,(2,3))
            (images/'a.jpg').write_bytes(b'changed')
            with self.assertRaises(ValueError):
                open_cache(cache,vision,images,[Example('a',['word'],'a.jpg',[])])

    def test_missing_seeds_never_produce_a_complete_mean(self):
        rows=[dict(dataset='twitter2015',seed=24,test_joint_f1=.6),
              dict(dataset='twitter2015',seed=25,test_joint_f1=.8)]
        result=summarize_rows(rows,expected_seeds=(24,25))['twitter2015']
        self.assertAlmostEqual(result['test_joint_f1_mean'],.7)
        self.assertAlmostEqual(result['test_joint_f1_sample_std'],2**.5*.1)
        self.assertIsNone(summarize_rows(rows[:1],expected_seeds=(24,25))['twitter2015']['test_joint_f1_mean'])
        with self.assertRaises(ValueError):summarize_rows(rows+rows,expected_seeds=(24,25))
