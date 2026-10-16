import copy
import importlib.util
import random
import unittest
import numpy as np
import torch
from mact.runtime import train_epoch, evaluate_model
from test_runtime import RuntimeTests


class AveragingTests(unittest.TestCase):
    def setUp(self):
        self.fixture=RuntimeTests(); self.fixture.setUp(); self.addCleanup(self.fixture.doCleanups)
        torch.set_num_threads(1)

    def ema(self, model):
        self.assertIsNotNone(importlib.util.find_spec('mact.averaging'), 'EMA module missing')
        from mact.averaging import WeightEMA
        return WeightEMA(model,.99)

    def test_ema_formula_no_grad_and_partial_accumulation(self):
        model=self.fixture.model(); ema=self.ema(model)
        name='text_head.weight'; before=model.text_head.weight.detach().clone()
        with torch.no_grad():model.text_head.weight.add_(2)
        ema.update(model)
        torch.testing.assert_close(ema.shadow[name], .99*before+.01*model.text_head.weight, rtol=0, atol=0)
        self.assertEqual(ema.updates,1)
        self.assertTrue(all(not p.requires_grad and p.dtype==torch.float32 for p in ema.shadow.values()))
        ema=self.ema(model);opt=torch.optim.AdamW(model.parameters())
        stats=train_epoch(model,self.fixture.loader(),opt,'cpu',accumulation=2,log_every=0,
                          after_optimizer_step=lambda:ema.update(model))
        self.assertEqual((stats['updates'],ema.updates),(2,2))

    def test_context_restores_parameters_buffers_modes_rng_on_error(self):
        model=self.fixture.model();ema=self.ema(model)
        model.train();model.text_head.eval()
        model.register_buffer('counter',torch.tensor(7))
        ids=[id(p) for p in model.parameters()]; before=copy.deepcopy(model.state_dict())
        modes=[m.training for m in model.modules()]
        random.seed(7);np.random.seed(7);torch.manual_seed(7)
        rng=(random.getstate(),np.random.get_state(),torch.get_rng_state().clone())
        with self.assertRaisesRegex(RuntimeError,'deliberate'):
            with ema.average_parameters(model):
                model.eval();model.counter.add_(5)
                random.random();np.random.rand();torch.rand(3)
                with torch.no_grad():model.text_head.weight.add_(100)
                raise RuntimeError('deliberate')
        self.assertEqual(ids,[id(p) for p in model.parameters()])
        self.assertEqual(modes,[m.training for m in model.modules()])
        for n,p in model.state_dict().items():self.assertTrue(torch.equal(p,before[n]))
        self.assertEqual(random.getstate(),rng[0]);self.assertTrue(np.array_equal(np.random.get_state()[1],rng[1][1]))
        self.assertTrue(torch.equal(torch.get_rng_state(),rng[2]))

    def test_online_two_epoch_trajectory_is_identical_with_ema_evaluation(self):
        torch.manual_seed(9); original=self.fixture.model()
        for module in original.modules():
            if isinstance(module,torch.nn.Dropout):module.p=.3
        results=[]
        for enabled in (False,True):
            model=copy.deepcopy(original);opt=torch.optim.AdamW(model.parameters(),lr=.003)
            ema=self.ema(model) if enabled else None
            torch.manual_seed(123)
            losses=[]
            for epoch in range(2):
                losses.append(train_epoch(model,self.fixture.loader(),opt,'cpu',accumulation=2,log_every=0,
                    after_optimizer_step=(lambda:ema.update(model)) if enabled else None))
                evaluate_model(model,self.fixture.loader(),'cpu')
                if enabled:
                    with ema.average_parameters(model):evaluate_model(model,self.fixture.loader(),'cpu')
            results.append((copy.deepcopy(model.state_dict()),copy.deepcopy(opt.state_dict()),torch.rand(5),losses))
        for n,p in results[0][0].items():self.assertTrue(torch.equal(p,results[1][0][n]),n)
        for k,s in results[0][1]['state'].items():
            for n,v in s.items():self.assertTrue(torch.equal(v,results[1][1]['state'][k][n]))
        self.assertTrue(torch.equal(results[0][2],results[1][2]));self.assertEqual(results[0][3],results[1][3])
