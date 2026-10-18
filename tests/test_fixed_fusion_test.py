import importlib.util
import json
import unittest
from pathlib import Path

import torch
import test_data_model as fixtures
from mact.common.cache import file_hash
from mact.common.metrics import evaluate_predictions
from mact.model import SequenceModel, ModelConfig
from mact.runtime import save_checkpoint, write_json
from mact.target_context import TargetClassifier, add_markers, save_target


class FixedFusionTestTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('mact.fusion_test'), 'fixed Test runner missing')
        from mact import fusion_test
        self.f=fusion_test
        helper=fixtures.DataModelTests();helper.setUp();self.addCleanup(helper.doCleanups)
        self.root=Path(helper.tmp.name)
        base=self.root/'base';run=self.root/'target';base.mkdir();run.mkdir()
        self.run=run
        self.data=self.root/'twitter2015'/'test.json';self.data.parent.mkdir()
        self.records=[dict(id='test-a',words=['good','washington'],image_id='a.jpg',
            aspects=[{'from':1,'to':2,'polarity':'positive'}]),
            dict(id='test-b',words=['bad'],image_id='b.jpg',aspects=[])]
        write_json(self.data,self.records)
        self.base_pred={'test-a':[[0,1,2],[1,2,2]],'test-b':[[0,1,2]]}
        self.target_pred={'test-a':[[0,1,0],[1,2,0]],'test-b':[[0,1,0]]}
        gold={'test-a':[(1,2,2)],'test-b':[]}
        bm=evaluate_predictions(gold,self.base_pred)
        tm=evaluate_predictions(gold,self.target_pred)
        model=SequenceModel(fixtures.tiny_text(),ModelConfig(mode='text',hidden_size=8,
            attention_heads=2,visual_size=8,dropout=0.,visual_dropout=0.)).eval()
        with torch.no_grad():
            # Strict margin avoids a permutation-dependent floating-point tie;
            # this test covers label isolation, not the separately tested tie rule.
            model.text_head.weight.zero_();model.text_head.bias.copy_(torch.tensor([0.,4.,0.,0.,-4.]))
        helper.tokenizer.save_pretrained(base/'tokenizer')
        save_checkpoint(base/'best.pt',model,dict(seed=42,num_threads=1,max_length=256,model={'mode':'text'}),1,bm)
        write_json(base/'summary.json',dict(best_dev=bm,checkpoint_sha256=file_hash(base/'best.pt')))
        cfg=dict(seed=42,num_threads=1,max_length=258,eval_batch_size=8,
                 baseline_summary=str(base/'summary.json'),baseline_checkpoint=str(base/'best.pt'))
        write_json(run/'config.json',cfg)
        add_markers(helper.tokenizer);encoder=fixtures.tiny_text()
        encoder.resize_token_embeddings(len(helper.tokenizer),mean_resizing=False)
        target=TargetClassifier(encoder,0.).eval()
        with torch.no_grad():
            target.classifier.weight.zero_();target.classifier.bias.copy_(torch.tensor([3.,0.,0.]))
        helper.tokenizer.save_pretrained(run/'tokenizer')
        save_target(run/'best.pt',target,dict(epoch=1,config=cfg,dev_metrics=tm))
        write_json(run/'summary.json',dict(best_epoch=1,best_dev=tm,checkpoint_sha256=file_hash(run/'best.pt')))
        self.paths=dict(test_data=str(self.data))
        for role,pred,metric in [('baseline',self.base_pred,bm),('target',self.target_pred,tm)]:
            pp=self.root/(role+'_predictions.json');mp=self.root/(role+'_metrics.json')
            write_json(pp,pred);write_json(mp,metric)
            self.paths[role+'_predictions']=str(pp);self.paths[role+'_metrics']=str(mp)
        self.lock=self.root/'decision.json'
        lock=dict(format='mact_fixed_equal_test_v1',parameters=dict(temperature_base=1.,temperature_target=1.,alpha=.5),
            datasets=['twitter2015'],seeds=[42],models=[dict(dataset='twitter2015',seed=42,run=str(run),
                baseline_checkpoint=str(base/'best.pt'),target_checkpoint=str(run/'best.pt'),
                baseline_sha256=file_hash(base/'best.pt'),target_sha256=file_hash(run/'best.pt'),config_sha256=file_hash(run/'config.json'))],
            fusion_source_sha256=file_hash(Path(fusion_test.__file__).with_name('calibrated_fusion.py')),
            selection_uses_test=False)
        write_json(self.lock,lock);self.lock.with_suffix('.sha256').write_text(file_hash(self.lock)+'\n')

    def test_real_test_replay_label_independence_and_fresh_output(self):
        a=self.f.run_pair(self.lock,'twitter2015',42,'cpu',self.root/'first',self.paths)
        self.assertEqual(a['predictions']['baseline'],self.base_pred)
        self.assertEqual(a['predictions']['target'],self.target_pred)
        self.assertEqual(a['predictions']['fusion'],self.base_pred)
        self.assertTrue(a['provenance']['gold_removed_from_inference'])
        self.assertEqual(a['metrics']['fusion']['mate'],a['metrics']['baseline']['mate'])
        self.records[0]['aspects'][0]['polarity']='negative';write_json(self.data,self.records)
        gold={'test-a':[(1,2,0)],'test-b':[]}
        for role,pred in [('baseline',self.base_pred),('target',self.target_pred)]:
            write_json(self.paths[role+'_metrics'],evaluate_predictions(gold,pred))
        b=self.f.run_pair(self.lock,'twitter2015',42,'cpu',self.root/'second',self.paths)
        self.assertEqual(a['predictions'],b['predictions'])
        self.assertNotEqual(a['metrics'],b['metrics'])
        with self.assertRaises(FileExistsError):
            self.f.run_pair(self.lock,'twitter2015',42,'cpu',self.root/'first',self.paths)

    def test_locked_parameters_and_requested_identity_reject_changes(self):
        lock=json.loads(self.lock.read_text());lock['parameters']['alpha']=.75
        write_json(self.lock,lock);self.lock.with_suffix('.sha256').write_text(file_hash(self.lock)+'\n')
        with self.assertRaisesRegex(ValueError,'fixed|parameters'):
            self.f.load_decision(self.lock)
        lock['parameters']['alpha']=.5;write_json(self.lock,lock)
        self.lock.with_suffix('.sha256').write_text(file_hash(self.lock)+'\n')
        with self.assertRaisesRegex(ValueError,'cohort|identity'):
            self.f.run_pair(self.lock,'twitter2017',42,'cpu',self.root/'wrong',self.paths)

    def test_archive_mismatch_fails_without_success_result(self):
        write_json(self.paths['baseline_predictions'],{'test-a':[],'test-b':[]})
        with self.assertRaisesRegex(ValueError,'baseline.*replay'):
            self.f.run_pair(self.lock,'twitter2015',42,'cpu',self.root/'bad',self.paths)
        self.assertFalse((self.root/'bad/result.json').exists())

    def test_aggregation_refuses_missing_and_wrong_identity_results(self):
        with self.assertRaises((FileNotFoundError,ValueError)):
            self.f.summarize(self.lock,self.root/'empty')
        self.f.run_pair(self.lock,'twitter2015',42,'cpu',self.root/'complete/twitter2015/seed42',self.paths)
        path=self.root/'complete/twitter2015/seed42/result.json'
        original=json.loads(path.read_text());wrong=dict(original,seed=43)
        write_json(path,wrong);path.with_suffix('.sha256').write_text(file_hash(path)+'\n')
        with self.assertRaisesRegex(ValueError,'identity'):
            self.f.summarize(self.lock,self.root/'complete')
        write_json(path,original);path.with_suffix('.sha256').write_text(file_hash(path)+'\n')
        report=self.f.summarize(self.lock,self.root/'complete')
        self.assertEqual(report['completed_pairs'],1)
        self.assertAlmostEqual(report['datasets']['twitter2015']['all']['fusion']['mean'],50.)
