"""EMA weights with exception-safe evaluation isolated from online training."""
from contextlib import contextmanager
import math
import random

import numpy as np
import torch


class WeightEMA:
    def __init__(self, model, decay):
        if type(decay) not in (int, float) or not math.isfinite(decay) or not 0 < decay < 1:
            raise ValueError('EMA decay must be finite and strictly between zero and one')
        if any(p.dtype != torch.float32 for p in model.parameters()):
            raise ValueError('EMA requires FP32 model parameters')
        self.decay, self.updates = decay, 0
        self.shadow = {n: p.detach().clone() for n, p in model.named_parameters()}

    @torch.no_grad()
    def update(self, model):
        parameters = dict(model.named_parameters())
        if parameters.keys() != self.shadow.keys():
            raise ValueError('EMA parameter set changed')
        for name, parameter in parameters.items():
            self.shadow[name].copy_(self.decay * self.shadow[name] + (1 - self.decay) * parameter)
        self.updates += 1

    @contextmanager
    @torch.no_grad()
    def average_parameters(self, model):
        parameters = dict(model.named_parameters())
        if parameters.keys() != self.shadow.keys():
            raise ValueError('EMA parameter set changed')
        online = {n: p.detach().clone() for n, p in parameters.items()}
        buffers = {n: b.detach().clone() for n, b in model.named_buffers()}
        modes = [(module, module.training) for module in model.modules()]
        py_rng, np_rng, cpu_rng = random.getstate(), np.random.get_state(), torch.get_rng_state()
        devices = {p.device.index for p in parameters.values() if p.is_cuda}
        cuda_rng = {d: torch.cuda.get_rng_state(d) for d in devices}
        try:
            for name, parameter in parameters.items():
                parameter.copy_(self.shadow[name])
            yield
        finally:
            for name, parameter in parameters.items():
                parameter.copy_(online[name])
            for name, buffer in model.named_buffers():
                buffer.copy_(buffers[name])
            for module, mode in modes:
                module.training = mode
            random.setstate(py_rng)
            np.random.set_state(np_rng)
            torch.set_rng_state(cpu_rng)
            for device, state in cuda_rng.items():
                torch.cuda.set_rng_state(state, device)
