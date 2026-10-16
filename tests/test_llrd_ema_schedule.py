import importlib.util
from pathlib import Path
import tempfile
import unittest
import sys
from unittest.mock import patch


class CohortScheduleTests(unittest.TestCase):
    def module(self):
        path=Path(__file__).resolve().parents[1]/'scripts/run_llrd_ema.py'
        self.assertTrue(path.exists(),'cohort launcher missing')
        spec=importlib.util.spec_from_file_location('llrd_launcher',path)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        return module

    def test_twelve_jobs_have_isolated_logs_and_train_only(self):
        m=self.module()
        with tempfile.TemporaryDirectory() as td:
            groups=m.cohort_groups(Path(td))
            jobs=[j for arm in groups.values() for group in arm for j in group]
            self.assertEqual(len(jobs),12)
            self.assertEqual(len({(j['trajectory'],j['dataset'],j['seed']) for j in jobs}),12)
            paths=set()
            for j in jobs:
                argv=m.training_command(j)
                self.assertIn('--train-only',argv)
                self.assertEqual(argv[argv.index('--device')+1],'cuda:0')
                self.assertEqual(j['gpu'],0 if j['dataset']=='twitter2015' else 1)
                paths.add(str(Path(j['output']).parents[1]/'logs'/f"{j['dataset']}-seed{j['seed']}.log"))
            self.assertEqual(len(paths),12)
            self.assertTrue(all(len(g)==2 for arm in groups.values() for g in arm))

    def test_hash_change_or_failed_smoke_blocks_launch(self):
        m=self.module()
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises((ValueError,FileNotFoundError)):m.validate_smokes(Path(td))
            path=Path(td)/'code.py';path.write_text('x')
            with self.assertRaises(ValueError):m.verify_files({str(path):'wrong'})

    def test_detached_launcher_records_failure_without_orphans(self):
        m=self.module()
        with tempfile.TemporaryDirectory() as td:
            out=Path(td)/'run'
            with patch.object(m,'validate_smokes',return_value={}),patch.object(m,'resource_hashes',return_value={}),\
                 patch.object(m,'run_schedule',side_effect=RuntimeError('worker failed')):
                with self.assertRaisesRegex(RuntimeError,'worker failed'):m.run_cohort(out,Path(td)/'smoke')
            import json
            self.assertEqual(json.loads((out/'status.json').read_text())['state'],'failed')

    def test_resource_drift_during_last_group_blocks_completion(self):
        m=self.module()
        with tempfile.TemporaryDirectory() as td:
            out=Path(td)/'run';resource=Path(td)/'weights';resource.write_text('original')
            hashes={str(resource):m.file_hash(resource)}
            calls=[]
            def schedule(groups,output,command,before_group,after_group):
                before_group();calls.append(output)
                if len(calls)==2:resource.write_text('changed during final group')
                return {'state':'completed'}
            with patch.object(m,'validate_smokes',return_value={'resource_sha256':hashes}),\
                 patch.object(m,'resource_hashes',return_value=hashes),patch.object(m,'run_schedule',side_effect=schedule),\
                 patch.object(m,'collect',return_value={'state':'completed'}):
                with self.assertRaisesRegex(ValueError,'frozen file'):m.run_cohort(out,Path(td)/'smoke')
