import copy
import json
import tempfile
import unittest
from pathlib import Path

from mact.ocr_residual_train import read_config
from mact.ocr_residual_results import paired_summary, same_treatment
from scripts.run_ocr_residual_pilot import make_jobs

ROOT=Path(__file__).resolve().parents[1]


class ResidualMultiseedTests(unittest.TestCase):
    def test_four_jobs_use_correct_seed_output_and_two_gpus(self):
        jobs=make_jobs((43,44),Path('/tmp/paired'))
        self.assertEqual(len(jobs),4)
        self.assertEqual(len({str(j['output']) for j in jobs}),4)
        self.assertEqual([(j['dataset'],j['seed'],j['gpu']) for j in jobs],
            [('twitter2015',43,0),('twitter2015',44,0),('twitter2017',43,1),('twitter2017',44,1)])
        for job in jobs:self.assertEqual(job['output'].name,f"seed{job['seed']}")
        for seeds in ([],[43,43],[101],[True]):
            with self.assertRaises(ValueError):make_jobs(seeds,Path('/tmp/paired'))

    def test_seed_override_selects_matching_baseline_without_changing_treatment(self):
        with tempfile.TemporaryDirectory() as tmp:
            original=json.loads((ROOT/'configs/ocr_residual.json').read_text())
            original['baseline']='assets/checkpoints/baseline_s{seed}/{dataset}.pt'
            path=Path(tmp)/'config.json';path.write_text(json.dumps(original))
            for seed in (42,43,44):
                cfg=read_config(path,'twitter2015','cpu',seed=seed)
                self.assertEqual(cfg['seed'],seed)
                self.assertEqual(cfg['baseline_run']['seed'],seed)
                self.assertEqual(Path(cfg['baseline']).parent.name,f'baseline_s{seed}')
                self.assertEqual(cfg['epochs'],35)
                self.assertEqual(cfg['learning_rate'],.0001)
            for invalid in (0,101,-1,True):
                with self.assertRaises(ValueError):read_config(path,'twitter2015','cpu',seed=invalid)

    def test_seed42_default_configuration_unchanged(self):
        a=read_config(ROOT/'configs/ocr_residual.json','twitter2017','cpu')
        b=read_config(ROOT/'configs/ocr_residual.json','twitter2017','cpu',seed=42)
        self.assertEqual(a,b)
        self.assertIn('baseline_s42',a['baseline'])

    def test_treatment_checks_ignore_seed_and_paths_but_reject_hyperparameter_changes(self):
        a=read_config(ROOT/'configs/ocr_residual.json','twitter2015','cpu')
        b=copy.deepcopy(a);b['seed']=43;b['baseline']='another.pt';b['baseline_run']['seed']=43
        self.assertTrue(same_treatment(a,b))
        b['learning_rate']=.002
        self.assertFalse(same_treatment(a,b))
        b=copy.deepcopy(a);b['baseline_run']['model']['dropout']=.4
        self.assertFalse(same_treatment(a,b))

    def rows(self):
        return [dict(dataset=ds,seed=s,baseline_f1=.6,model_f1=.6+(s-43)/100)
                for ds in ('twitter2015','twitter2017') for s in (42,43,44)]

    def test_paired_aggregate_and_sample_standard_deviation(self):
        result=paired_summary(self.rows(),[42,43,44])
        self.assertEqual(result['state'],'completed')
        for dataset,values in result['datasets'].items():
            self.assertAlmostEqual(values['delta_points']['mean'],0)
            self.assertAlmostEqual(values['delta_points']['sample_std'],1)
            self.assertAlmostEqual(values['model_f1_percent']['sample_std'],1)
            self.assertEqual((values['wins'],values['ties'],values['losses']),(1,1,1))

    def test_aggregation_rejects_missing_duplicate_and_nonfinite_results(self):
        for rows in (self.rows()[:-1],self.rows()+[self.rows()[0]]):
            with self.assertRaises(ValueError):paired_summary(rows,[42,43,44])
        rows=self.rows();rows[0]['model_f1']=float('nan')
        with self.assertRaises(ValueError):paired_summary(rows,[42,43,44])


if __name__=='__main__':unittest.main()
