import copy
import importlib.util
import json
import unittest
from pathlib import Path

import torch
from torch.utils.data import DataLoader

import test_data_model as fixtures
from mact.common.cache import file_hash
from mact.common.metrics import evaluate_predictions
from mact.data import SequenceDataset, SequenceCollator
from mact.model import SequenceModel, ModelConfig
from mact.runtime import save_checkpoint, evaluate_model
from mact.target_context import (TargetClassifier, TargetDataset, TargetCollator,
                                 add_markers, save_target, predict_targets, relabel)


class FusionExperimentTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('mact.fusion_experiment'),
                             'fusion experiment runner missing')
        from mact import fusion_experiment
        self.f = fusion_experiment
        torch.set_num_threads(1)

    def runs(self):
        runs=[]
        for seed in [42,43]:
            gold={f'i{k}': [[0,1,k%2],[1,2,k%2]] for k in range(30)}
            base={key:[[0,1,0],[1,2,0]] for key in gold}
            target={key:[[0,1,1],[1,2,1]] for key in gold}
            keys=[[key,s,e] for key,v in base.items() for s,e,_ in v]
            runs.append(dict(dataset='twitter2015',seed=seed,gold=gold,baseline=base,
                target=target,keys=keys,base_logits=[[2.,0.,-1.]]*len(keys),
                target_logits=[[0.,3.,-1.]]*len(keys)))
        return runs

    def test_held_out_labels_never_affect_that_folds_parameters(self):
        runs=self.runs()
        before=self.f.crossfit(runs,expected_seeds=[42,43])
        changed=copy.deepcopy(runs)
        for run in changed:
            for key in run['gold']:
                if self.f.sentence_fold(key)==0:
                    run['gold'][key]=[[0,1,2],[1,2,2]]
        after=self.f.crossfit(changed,expected_seeds=[42,43])
        self.assertEqual(before['fold_parameters'][0],after['fold_parameters'][0])
        for row in before['rows']:
            self.assertEqual(set(row['crossfit_predictions']),set(runs[0]['gold']))
            self.assertEqual(row['baseline']['mate'],row['calibrated']['mate'])

    def test_full_joint_denominator_counts_false_positives_and_missed_gold(self):
        run=dict(dataset='twitter2015',seed=42,gold={'a':[[0,1,0],[1,2,0]],'b':[[0,1,1]]},
            baseline={'a':[[0,1,0],[1,2,0],[2,3,0]],'b':[]},
            target={'a':[[0,1,1],[1,2,1],[2,3,1]],'b':[]},
            keys=[['a',0,1],['a',1,2],['a',2,3]],
            base_logits=[[2.,0.,0.]]*3,target_logits=[[0.,2.,0.]]*3)
        metrics,_=self.f.score_run(run,dict(temperature_base=1.,temperature_target=1.,alpha=0.))
        self.assertAlmostEqual(metrics['joint']['f1'],2/3)
        self.assertEqual(metrics['joint']['gold'],3)
        self.assertEqual(metrics['joint']['predicted'],3)

    def test_incomplete_duplicate_or_misaligned_runs_rejected(self):
        runs=self.runs()
        for bad in [runs[:1],runs+ [runs[0]]]:
            with self.assertRaises(ValueError):
                self.f.crossfit(bad,expected_seeds=[42,43])
        bad=copy.deepcopy(runs);bad[0]['keys'][1]=bad[0]['keys'][0]
        with self.assertRaises(ValueError):
            self.f.crossfit(bad,expected_seeds=[42,43])
        bad=copy.deepcopy(runs);bad[0]['target']['i0'][0][1]=9
        with self.assertRaises(ValueError):
            self.f.crossfit(bad,expected_seeds=[42,43])

    def test_real_model_extraction_exact_replay_and_stale_cache_rejection(self):
        f=fixtures.DataModelTests(); f.setUp();self.addCleanup(f.doCleanups)
        root=Path(f.tmp.name);base_dir=root/'base';target_dir=root/'target'
        base_dir.mkdir();target_dir.mkdir()
        data=root/'dev.json'
        data.write_text(json.dumps([dict(id=e.id,words=e.words,image_id=e.image_id,
            aspects=[{'from':s,'to':e_,'polarity':['negative','neutral','positive'][p]}
                     for s,e_,p in e.aspects]) for e in f.examples]))
        model=SequenceModel(fixtures.tiny_text(),ModelConfig(mode='text',hidden_size=8,
            attention_heads=2,visual_size=8,dropout=0.,visual_dropout=0.)).eval()
        with torch.no_grad():
            model.text_head.weight.zero_()
            model.text_head.bias.copy_(torch.tensor([0.,3.,0.,0.,-4.]))
        f.tokenizer.save_pretrained(base_dir/'tokenizer')
        loader=DataLoader(SequenceDataset(f.examples,f.tokenizer),batch_size=1,
                         collate_fn=SequenceCollator(f.tokenizer.pad_token_id))
        bm,bp=evaluate_model(model,loader,'cpu')
        bc=dict(seed=42,num_threads=1,max_length=256,model={'mode':'text'})
        save_checkpoint(base_dir/'best.pt',model,bc,1,bm)
        (base_dir/'best_dev_predictions.json').write_text(json.dumps(bp))
        (base_dir/'summary.json').write_text(json.dumps(dict(best_dev=bm,
            checkpoint_sha256=file_hash(base_dir/'best.pt'))))
        add_markers(f.tokenizer);encoder=fixtures.tiny_text()
        encoder.resize_token_embeddings(len(f.tokenizer),mean_resizing=False)
        target=TargetClassifier(encoder,0.).eval()
        f.tokenizer.save_pretrained(target_dir/'tokenizer')
        ds=TargetDataset(f.examples,bp,f.tokenizer)
        tp=relabel(bp,ds.keys,predict_targets(target,DataLoader(ds,batch_size=8,
            collate_fn=TargetCollator(f.tokenizer.pad_token_id)),'cpu'))
        tm=evaluate_predictions({e.id:e.aspects for e in f.examples},tp)
        cfg=dict(seed=42,dev_data=str(data),baseline_predictions=str(base_dir/'best_dev_predictions.json'),
                 baseline_summary=str(base_dir/'summary.json'),baseline_checkpoint=str(base_dir/'best.pt'),
                 max_length=258,eval_batch_size=8,num_threads=1)
        save_target(target_dir/'best.pt',target,dict(epoch=1,dev_metrics=tm,config=cfg))
        (target_dir/'config.json').write_text(json.dumps(cfg))
        (target_dir/'summary.json').write_text(json.dumps(dict(best_epoch=1,best_dev=tm,
            baseline_dev=bm,checkpoint_sha256=file_hash(target_dir/'best.pt'))))
        (target_dir/'best_dev_predictions.json').write_text(json.dumps(tp))
        out=root/'scores.json'
        cache=self.f.extract_run(target_dir,'twitter2015',42,'cpu',out,{})
        self.assertTrue(cache['provenance']['baseline_replay_exact'])
        self.assertTrue(cache['provenance']['target_replay_exact'])
        loaded=self.f.load_verified(out)
        self.assertEqual(loaded['baseline'],json.loads(json.dumps(bp)))
        with self.assertRaises(FileExistsError):
            self.f.extract_run(target_dir,'twitter2015',42,'cpu',out,{})
        data.write_text(data.read_text()+' ')
        with self.assertRaisesRegex(ValueError,'changed|stale'):
            self.f.load_verified(out)
