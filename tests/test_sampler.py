"""The sampler finding, tested on the authors' own FlowLoss code with an exact predictor (no checkpoint needed).

Toy problem with a known answer: data y1 ~ N(0, 1), noise y0 ~ N(0, 1), path y_t = t*y1 + (1 - t)*y0. The best
possible end-point predictor is E[y1 | y_t = y] = t / (t^2 + (1 - t)^2) * y, so a correct sampler should
return samples with standard deviation close to 1.
"""
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from aurora_model import import_aurora, use_sampler  # noqa: E402

try:
    import_aurora()
    from aurora.flow_loss import FlowLoss
except FileNotFoundError:
    pytest.skip('vendor/aurora missing: run `python setup_aurora.py`', allow_module_level=True)

STEPS = 50


class Predictor(torch.nn.Module):
    """Stands in for the network: an end-point predictor that records every call."""

    def __init__(self, fn):
        super().__init__()
        self.fn, self.calls = fn, []

    def forward(self, x, t_scaled, z):
        pred = self.fn(x, t_scaled[:, None] / 1000)    # FlowLoss passes t * 1000
        self.calls.append(pred)
        return pred


def flow_head(fn):
    head = FlowLoss(target_channels=1, z_channels=4, depth=1, width=8, num_sampling_steps=STEPS)
    head.net = Predictor(fn)
    return head.train()                                 # training mode returns raw samples (no patch resampling)


def exact_predictor(x, t):
    return t / (t ** 2 + (1 - t) ** 2) * x


def sample_sd(sampler, n=200_000):
    head = flow_head(exact_predictor)
    use_sampler(head, sampler)
    torch.manual_seed(0)
    with torch.no_grad():
        return head.sample(torch.zeros(1, 4), num_samples=n).std().item()


def test_released_sample_is_the_average_of_the_predictions():
    head = flow_head(lambda x, t: 0.7 * x + 0.3 * t)   # any predictor shows the identity
    use_sampler(head, 'released')
    torch.manual_seed(0)
    with torch.no_grad():
        out = head.sample(torch.zeros(1, 4), num_samples=1000).reshape(-1)
    average = torch.stack(head.net.calls).mean(0).reshape(-1)
    assert len(head.net.calls) == STEPS
    assert torch.allclose(out, average, atol=1e-5)


def test_released_sampler_halves_the_spread():
    assert 0.42 < sample_sd('released') < 0.54           # about 0.48 against a true value of 1


def test_corrected_sampler_recovers_the_spread():
    assert 0.95 < sample_sd('corrected') < 1.01          # about 0.98: Euler error only
