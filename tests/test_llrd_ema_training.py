import builtins
import copy
import json
from pathlib import Path
import unittest
from unittest.mock import patch
import torch
import test_data_model as fixtures
from test_partial_freeze import tiny_deberta
import mact.train as training
import mact.experiment as experiment
from mact.runtime import load_checkpoint
from mact.common.cache import file_hash, model_hashes


class EMATrainingTests(unittest.TestCase):
    def setUp(self):
        self.fixture=fixtures.DataModelTests();self.fixture.setUp();self.addCleanup(self.fixture.doCleanups)
        self.root=Path(self.fixture.tmp.name)
        backbone=self.root/'model'; tiny_deberta().save_pretrained(backbone); self.fixture.tokenizer.save_pretrained(backbone)
        for split in ('train','dev'):
            rows=[dict(id=f'{split}{i}',words=['good','washington'],image_id='a.jpg',
                       aspects=[{'from':1,'to':2,'polarity':'positive'}]) for i in range(3)]
            (self.root/f'{split}.json').write_text(json.dumps(rows))
        self.cfg=dict(text_model=str(backbone),train_data=str(self.root/'train.json'),dev_data=str(self.root/'dev.json'),
                      device='cpu',num_threads=1,epochs=2,accumulation=2,log_every=0,ema_decay=.99,
                      model=dict(mode='text',hidden_size=8,attention_heads=2,dropout=.2))

    def test_online_and_ema_choose_independent_earliest_maxima(self):
        cfg=dict(self.cfg,epochs=3)
        curve=iter([.4,.3,.6,.5,.6,.7])
        original=training.evaluate_model
        def evaluate(*args,**kwargs):
            metrics,predictions=original(*args,**kwargs)
            metrics['joint']['f1']=next(curve)
            return metrics,predictions
        out=self.root/'selection'
        with patch.object(training,'evaluate_model',side_effect=evaluate):summary=training.run_training(cfg,out)
        self.assertIn('ema',summary)
        es=json.loads((out/'ema/summary.json').read_text())
        self.assertEqual((summary['best_epoch'],es['best_epoch']),(2,3))
        self.assertEqual(es['ema_updates'],6)
        _,meta=load_checkpoint(out/'ema/best.pt');self.assertEqual(meta['weight_metadata']['source'],'ema')

    def test_full_training_ema_does_not_change_online_rng_or_weights(self):
        enabled=training.run_training(self.cfg,self.root/'enabled'); rng=torch.get_rng_state()
        disabled=training.run_training(dict(self.cfg,ema_decay=0),self.root/'disabled');rng2=torch.get_rng_state()
        self.assertIn('ema',enabled);self.assertNotIn('ema',disabled)
        a,_=load_checkpoint(self.root/'enabled/last.pt');b,_=load_checkpoint(self.root/'disabled/last.pt')
        for n,p in a.state_dict().items():self.assertTrue(torch.equal(p,b.state_dict()[n]),n)
        self.assertTrue(torch.equal(rng,rng2))
        ha=json.loads((self.root/'enabled/history.json').read_text());hb=json.loads((self.root/'disabled/history.json').read_text())
        for a,b in zip(ha,hb):self.assertEqual((a['train'],a['dev']),(b['train'],b['dev']))

    def project(self):
        root=self.root
        vision=root/'vision';vision.mkdir();(vision/'config.json').write_text('{}')
        images=root/'images';images.mkdir();(images/'a.jpg').write_bytes(b'image')
        cache=root/'cache';cache.mkdir();torch.save({'a.jpg':torch.ones(2,8)},cache/'features.pt')
        (cache/'manifest.json').write_text(json.dumps(dict(format_version=1,vision_model='../vision',image_root='../images',
            model_sha256=model_hashes(vision),features_sha256=file_hash(cache/'features.pt'),
            image_sha256={'a.jpg':file_hash(images/'a.jpg')},token_shape=[2,8])))
        (root/'configs/datasets').mkdir(parents=True)
        cfg=dict(self.cfg,text_model='model',vision_model='vision',layer_lr_decay=.9,
                 model=dict(mode='fusion',hidden_size=8,attention_heads=2,visual_size=8))
        cfg.pop('train_data');cfg.pop('dev_data')
        (root/'configs/base.json').write_text(json.dumps(cfg))
        (root/'configs/datasets/twitter2015.json').write_text(json.dumps(dict(train_data='train.json',dev_data='dev.json',
            test_data='forbidden_test.json',test_visual_cache='forbidden_cache',image_root='images',visual_cache='cache')))

    def test_both_checkpoint_streams_reload_exactly_and_dev_only_never_opens_or_hashes_test(self):
        self.project();original_open=builtins.open;original_path_open=Path.open;original_hash=experiment.file_hash
        def guard(path):
            if 'forbidden' in str(path):raise AssertionError('Test resource accessed')
        def checked_open(path,*a,**kw):guard(path);return original_open(path,*a,**kw)
        def checked_path_open(path,*a,**kw):guard(path);return original_path_open(path,*a,**kw)
        def checked_hash(path):guard(path);return original_hash(path)
        with patch.object(experiment,'ROOT',self.root),patch('builtins.open',checked_open),patch.object(Path,'open',checked_path_open),patch.object(experiment,'file_hash',checked_hash):
            summary=experiment.run_experiment('configs/base.json','twitter2015',seed=42,output=self.root/'run',evaluate_test=False)
        self.assertIn('ema',summary)
        for directory in [self.root/'run',self.root/'run/ema']:
            check=json.loads((directory/'reload-verification.json').read_text())
            self.assertTrue(check['predictions_exact'] and check['metrics_exact'])
        self.assertIsNone(json.loads((self.root/'run.status.json').read_text())['test'])

    def test_configs_only_differ_in_layer_decay(self):
        root=Path(__file__).resolve().parents[1]
        a=root/'configs/ema_uniform.json';b=root/'configs/ema_llrd.json'
        self.assertTrue(a.exists() and b.exists())
        x,y=json.loads(a.read_text()),json.loads(b.read_text())
        self.assertEqual((x.pop('layer_lr_decay'),y.pop('layer_lr_decay')),(1.,.9))
        self.assertEqual(x,y);self.assertEqual(x['ema_decay'],.99)
