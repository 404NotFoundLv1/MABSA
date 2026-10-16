import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import torch
from unittest.mock import patch


class ResultsTests(unittest.TestCase):
    def module(self):
        self.assertIsNotNone(importlib.util.find_spec('mact.llrd_ema_results'))
        from mact import llrd_ema_results
        return llrd_ema_results

    def rows(self):
        rows=[]
        for ds in ('twitter2015','twitter2017'):
            for seed,i in [(42,0),(43,1),(44,2)]:
                for arm,score in [('A',.60),('B',[.61,.62,.59][i]),('C',.60),('D',.59)]:
                    rows.append(dict(dataset=ds,seed=seed,arm=arm,joint_f1=score))
        return rows

    def test_paired_deltas_sample_sd_and_common_boundary_metrics(self):
        m=self.module();result=m.summarize(self.rows(),[42,43,44])
        b=result['aggregate']['twitter2015']['B']
        self.assertEqual((b['wins'],b['ties'],b['losses']),(2,0,1))
        self.assertAlmostEqual(b['mean_delta_pp'],2/3)
        self.assertAlmostEqual(b['delta_sample_std_pp'],1.5275252316519468)
        self.assertEqual(result['selected_candidate'],'B')
        gold={'a':[(0,1,2),(2,3,1)]};a={'a':[(0,1,2)]};b={'a':[(0,1,0),(2,3,1)]}
        self.assertEqual(m.common_boundary_sentiment(gold,a,b),dict(count=1,baseline_correct=1,treatment_correct=0,
                          baseline_accuracy=1.,treatment_accuracy=0.))

    def test_incomplete_duplicate_nonfinite_or_mismatched_results_are_rejected(self):
        m=self.module();rows=self.rows()
        partial=m.summarize(rows[:-1],[42,43,44]);self.assertEqual(partial['state'],'incomplete');self.assertIsNone(partial['aggregate'])
        for bad in [rows+[rows[0]], [dict(rows[0],joint_f1=float('nan'))]+rows[1:],
                    [dict(rows[0],seed=99)]+rows[1:], [dict(rows[0],arm='E')]+rows[1:]]:
            with self.assertRaises(ValueError):m.summarize(bad,[42,43,44])
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);(root/'launch.json').write_text(json.dumps(dict(seeds=[42,43,44],jobs=[])))
            with self.assertRaises(ValueError):m.collect(root)

    def test_collect_recomputes_metrics_and_rejects_empty_source_provenance(self):
        m=self.module()
        from mact.common.cache import file_hash
        from mact.common.metrics import evaluate_predictions
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);data=root/'dev.json'
            root_patch=patch.object(m,'ROOT',root,create=True);root_patch.start();self.addCleanup(root_patch.stop)
            data.write_text(json.dumps([dict(id='x',words=['good'],image_id='x',aspects=[{'from':0,'to':1,'polarity':'POS'}])]))
            predictions={'x':[[0,1,2]]};metrics=evaluate_predictions(predictions,predictions)
            jobs=[];resources={}
            for ds in ('twitter2015','twitter2017'):
                paths={k:str(root/'assets/data'/ds/f'{split}.json') for k,split in [('train_data','train'),('dev_data','dev')]}
                for path in paths.values():
                    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(data.read_text())
                    resources[path]=file_hash(p)
                for seed in (42,43,44):
                    for t in ('uniform','llrd'):
                        run=root/t/ds/f'seed{seed}';run.mkdir(parents=True)
                        cfg=dict(seed=seed,epochs=35,ema_decay=.99,layer_lr_decay=1. if t=='uniform' else .9,
                                 train_limit=None,dev_limit=None,max_updates=None,**paths)
                        jobs.append(dict(dataset=ds,seed=seed,trajectory=t,run_config=cfg))
                        (run/'config.json').write_text(json.dumps(cfg))
                        (run/'provenance.json').write_text(json.dumps({'source_sha256':{'train.py':'abc'}}))
                        run.with_name(run.name+'.status.json').write_text(json.dumps({'state':'completed'}))
                        for source in ('online','ema'):
                            d=run if source=='online' else run/'ema';d.mkdir(exist_ok=True)
                            checkpoint=dict(run_config=cfg,epoch=1,dev_metrics=metrics,
                                state_dict={'head':torch.ones(1)},weight_metadata=dict(source=source,ema_decay=.99,
                                ema_updates=2,trajectory=t,selected_epoch=1))
                            torch.save(checkpoint,d/'best.pt')
                            summary=dict(is_smoke=False,completed_epochs=35,evaluated_epochs=35,weight_source=source,
                                trajectory=t,best_epoch=1,checkpoint_sha256=file_hash(d/'best.pt'),best_dev=metrics,
                                optimizer_updates=70,ema_updates=70,best_ema_updates=2)
                            (d/'summary.json').write_text(json.dumps(summary))
                            (d/'best_dev_predictions.json').write_text(json.dumps(predictions))
                            (d/'reload-verification.json').write_text(json.dumps(dict(full_dev=True,predictions_exact=True,metrics_exact=True)))
            launch=dict(seeds=[42,43,44],jobs=jobs,resource_sha256=resources,
                        code={'source_sha256':{'src/mact/train.py':'abc'}})
            (root/'launch.json').write_text(json.dumps(launch))
            self.assertEqual(m.collect(root)['state'],'completed')
            self.assertEqual(len(json.loads((root/'results.json').read_text())['rows']),24)
            original=copy.deepcopy(checkpoint)
            for section,key,value in [('run_config','seed',42),('weight_metadata','source','online'),
                    ('weight_metadata','trajectory','uniform'),('weight_metadata','ema_decay',.9),
                    ('weight_metadata','selected_epoch',2),('weight_metadata','ema_updates',3),
                    (None,'epoch',2),(None,'dev_metrics',{})]:
                corrupted=copy.deepcopy(original)
                (corrupted[section] if section else corrupted)[key]=value
                torch.save(corrupted,d/'best.pt')
                summary['checkpoint_sha256']=file_hash(d/'best.pt')
                (d/'summary.json').write_text(json.dumps(summary))
                with self.subTest(section=section,key=key),self.assertRaisesRegex(ValueError,'checkpoint'):
                    m.collect(root)
            torch.save(original,d/'best.pt')
            summary['checkpoint_sha256']=file_hash(d/'best.pt')
            (d/'summary.json').write_text(json.dumps(summary))
            forbidden=root/'assets/data/twitter2017/test.json';forbidden.write_text(data.read_text())
            launch['resource_sha256'][str(forbidden)]=file_hash(forbidden)
            cfg['dev_data']=str(forbidden)
            (run/'config.json').write_text(json.dumps(cfg));(root/'launch.json').write_text(json.dumps(launch))
            original_open=Path.open
            def guarded_hash(path):
                self.assertNotEqual(Path(path),forbidden,'collector hashed Test')
                return file_hash(path)
            def guarded_open(path,*args,**kwargs):
                self.assertNotEqual(path,forbidden,'collector opened Test')
                return original_open(path,*args,**kwargs)
            with patch.object(m,'file_hash',side_effect=guarded_hash),patch.object(Path,'open',guarded_open):
                with self.assertRaisesRegex(ValueError,'Train/Dev'):m.collect(root)
            cfg['dev_data']=paths['dev_data']
            (run/'config.json').write_text(json.dumps(cfg));(root/'launch.json').write_text(json.dumps(launch))
            (run/'provenance.json').write_text(json.dumps({'source_sha256':{}}))
            with self.assertRaisesRegex(ValueError,'source'):m.collect(root)
