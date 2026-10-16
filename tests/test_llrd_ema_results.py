import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


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
            data.write_text(json.dumps([dict(id='x',words=['good'],image_id='x',aspects=[{'from':0,'to':1,'polarity':'POS'}])]))
            predictions={'x':[[0,1,2]]};metrics=evaluate_predictions(predictions,predictions)
            jobs=[]
            for ds in ('twitter2015','twitter2017'):
                for seed in (42,43,44):
                    for t in ('uniform','llrd'):
                        run=root/t/ds/f'seed{seed}';run.mkdir(parents=True)
                        cfg=dict(seed=seed,epochs=35,ema_decay=.99,layer_lr_decay=1. if t=='uniform' else .9,
                                 train_limit=None,dev_limit=None,max_updates=None,train_data=str(data),dev_data=str(data))
                        jobs.append(dict(dataset=ds,seed=seed,trajectory=t,run_config=cfg))
                        (run/'config.json').write_text(json.dumps(cfg))
                        (run/'provenance.json').write_text(json.dumps({'source_sha256':{'train.py':'abc'}}))
                        run.with_name(run.name+'.status.json').write_text(json.dumps({'state':'completed'}))
                        for source in ('online','ema'):
                            d=run if source=='online' else run/'ema';d.mkdir(exist_ok=True)
                            (d/'best.pt').write_bytes(b'checkpoint-fixture')
                            summary=dict(is_smoke=False,completed_epochs=35,evaluated_epochs=35,weight_source=source,
                                trajectory=t,best_epoch=1,checkpoint_sha256=file_hash(d/'best.pt'),best_dev=metrics,
                                optimizer_updates=70,ema_updates=70)
                            (d/'summary.json').write_text(json.dumps(summary))
                            (d/'best_dev_predictions.json').write_text(json.dumps(predictions))
                            (d/'reload-verification.json').write_text(json.dumps(dict(full_dev=True,predictions_exact=True,metrics_exact=True)))
            launch=dict(seeds=[42,43,44],jobs=jobs,resource_sha256={str(data):file_hash(data)},
                        code={'source_sha256':{'src/mact/train.py':'abc'}})
            (root/'launch.json').write_text(json.dumps(launch))
            self.assertEqual(m.collect(root)['state'],'completed')
            self.assertEqual(len(json.loads((root/'results.json').read_text())['rows']),24)
            (run/'provenance.json').write_text(json.dumps({'source_sha256':{}}))
            with self.assertRaisesRegex(ValueError,'source'):m.collect(root)
