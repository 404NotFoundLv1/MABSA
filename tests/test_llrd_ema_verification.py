import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import test_llrd_ema_schedule as fixtures


class AdmissionTests(unittest.TestCase):
    def test_smoke_requires_cuda_updates_both_streams_and_current_hashes(self):
        launcher=fixtures.CohortScheduleTests().module()
        identity=launcher.code_identity()
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            (root/'manifest.json').write_text(json.dumps(dict(source_sha256=identity['source_sha256'],resource_sha256={})))
            checks=[]
            for arm in ('uniform','llrd'):
                for gpu,ds in enumerate(('twitter2015','twitter2017')):
                    d=root/arm/ds;d.mkdir(parents=True)
                    check=dict(status='passed',device='cuda:0',physical_gpu=gpu,peak_cuda_allocated_bytes=100,
                        optimizer_updates=2,ema_updates=2,head_updated=True,finite_gradients=True,lr_groups_match=True,
                        source_sha256=identity['source_sha256'],config_sha256=launcher.file_hash(launcher.ROOT/f'configs/ema_{arm}.json'),
                        online_reload_exact=True,ema_reload_exact=True)
                    p=d/'integration_check.json';p.write_text(json.dumps(check));checks.append((p,check))
            launcher.validate_smokes(root)
            p,original=checks[0]
            for change in [dict(device='cpu'),dict(optimizer_updates=0),dict(ema_updates=1),
                           dict(peak_cuda_allocated_bytes=0),dict(source_sha256={}),dict(ema_reload_exact=False),
                           dict(head_updated=False),dict(finite_gradients=False),dict(lr_groups_match=False)]:
                p.write_text(json.dumps(dict(original,**change)))
                with self.subTest(change=change),self.assertRaises(ValueError):launcher.validate_smokes(root)
                p.write_text(json.dumps(original))

    def test_verifier_refuses_cpu_before_loading_resources(self):
        path=Path(__file__).resolve().parents[1]/'scripts/verify_llrd_ema.py'
        self.assertTrue(path.exists(),'verification entrypoint missing')
        spec=importlib.util.spec_from_file_location('llrd_verifier',path)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as td,patch.object(module.torch.cuda,'is_available',return_value=False):
            with self.assertRaisesRegex(RuntimeError,'CUDA'):module.run_worker(Path(td)/'run','twitter2015','uniform',0)
