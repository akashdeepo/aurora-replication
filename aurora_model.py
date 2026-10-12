"""Load the released Aurora checkpoint and switch its flow-matching sampler.

The model code is the authors' `aurora` package (github.com/decisionintelligence/Aurora, commit a247760),
fetched into vendor/ by setup_aurora.py; the weights are DecisionIntelligence/Aurora on Hugging Face. The
package is imported directly because transformers' remote-code loader leaves out its bert_config folder.

Two samplers, selected with `use_sampler(model, name)`:
  released   the update as shipped, x <- x + (prediction - x0) * dt. Summed over the steps, the final
             sample is the average of the network's end-point predictions along the path.
  corrected  the Euler step on the velocity implied by an end-point prediction, (prediction - x) / (1 - t),
             which is the update the paper describes. Everything else is unchanged.
"""
import os
import sys
import types

import torch
import torch.nn.functional as F

VENDOR = os.environ.get('AURORA_REPO', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'vendor'))
CHECKPOINT = 'DecisionIntelligence/Aurora'
SAMPLERS = ('released', 'corrected')


def import_aurora():
    """Make the vendored `aurora` package importable and return it."""
    if not os.path.isdir(os.path.join(VENDOR, 'aurora')):
        raise FileNotFoundError(f'{VENDOR}/aurora not found: run `python setup_aurora.py` first')
    if VENDOR not in sys.path:
        sys.path.insert(0, VENDOR)
    import aurora
    return aurora


def load_model(device='cuda'):
    """Download (once) and load the released checkpoint, in eval mode."""
    import_aurora()
    from aurora.modeling_aurora import AuroraForPrediction
    from huggingface_hub import snapshot_download
    return AuroraForPrediction.from_pretrained(snapshot_download(CHECKPOINT)).to(device).eval()


def flow_head(model):
    """The model's flow-matching head (the `FlowLoss` module that owns `sample`)."""
    heads = [m for m in model.modules() if type(m).__name__ == 'FlowLoss']
    if len(heads) != 1:
        raise RuntimeError(f'expected one FlowLoss module, found {len(heads)}')
    return heads[0]


def corrected_sample(self, z, prototype=None, num_samples=1, inference_token_len=48):
    """The shipped `FlowLoss.sample`, with only the update rule changed (marked below)."""
    from aurora.util_functions import resample
    z = z.repeat(num_samples, 1)
    x = torch.randn(z.shape[0], self.in_channels).to(z.device)
    if prototype is not None:
        x = x + prototype.repeat(num_samples, 1)
    n = self.num_sampling_steps
    for i in range(n):
        t = (torch.ones(x.shape[0]) * i / n).to(x.device)
        pred = self.net(x, t * 1000, z)                 # the network's estimate of the end point
        x = x + (pred - x) / (n - i)                    # CHANGED: velocity (pred - x) / (1 - t), times dt = 1/n
    if not self.training:
        new_weight = resample(torch.eye(self.in_channels).to(x.device), inference_token_len).T
        x = F.linear(x, new_weight)
        return x.reshape(num_samples, -1, inference_token_len).transpose(0, 1)
    return x.reshape(num_samples, -1, self.in_channels).transpose(0, 1)


def use_sampler(model_or_head, name):
    """Switch a model (or a FlowLoss module) to the 'released' or the 'corrected' sampler."""
    head = model_or_head if type(model_or_head).__name__ == 'FlowLoss' else flow_head(model_or_head)
    if name == 'released':
        head.__dict__.pop('sample', None)               # fall back to the class method, as shipped
    elif name == 'corrected':
        head.sample = types.MethodType(corrected_sample, head)
    else:
        raise ValueError(f'unknown sampler {name!r}; choose from {SAMPLERS}')
