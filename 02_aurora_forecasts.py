"""Step 2: zero-shot next-day return forecasts from the released Aurora checkpoint.

Two variants, run on identical random noise (same seed per forecast):
  released  : the sampler exactly as shipped (each Euler step moves by prediction - START point,
              so the final sample is the average of the 50 endpoint predictions)
  corrected : Euler on the velocity implied by the endpoint prediction, (prediction - x) / (1 - t),
              which is what the paper's Eq. 25 / Algorithm 1 describe for a velocity model

Each forecast: last 528 daily returns in -> 1,000 sampled 48-day paths out; we keep day 1.
Batch size 1 on purpose: Aurora picks the image's cycle length from the batch-averaged spectrum,
so batching several windows together would make each forecast depend on its neighbours.

Usage: python 02_aurora_forecasts.py [released|corrected|both] [--limit N]
Output: results/aurora_<variant>_<asset>.npz
"""
import os, sys, time, types
import numpy as np
import torch
import torch.nn.functional as F

from common import ASSETS, LOOKBACK, N_SAMPLES, HORIZON, TAUS, RESULTS, load_returns, origins

variants = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith('--') else 'both'
variants = ['released', 'corrected'] if variants == 'both' else [variants]
limit = int(sys.argv[sys.argv.index('--limit') + 1]) if '--limit' in sys.argv else None

# Model code: the released `aurora` package from github.com/decisionintelligence/Aurora (commit a247760).
# Its .py files are identical to those in the Hugging Face repo (checked with diff, ignoring line endings);
# importing it directly avoids a loader bug where transformers' remote-code cache omits the bert_config folder.
# Weights: model.safetensors from the Hugging Face repo DecisionIntelligence/Aurora.
AURORA_REPO = os.environ.get('AURORA_REPO', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'vendor'))
sys.path.insert(0, AURORA_REPO)
from aurora.modeling_aurora import AuroraForPrediction
from huggingface_hub import snapshot_download

t0 = time.time()
ckpt = snapshot_download('DecisionIntelligence/Aurora')
model = AuroraForPrediction.from_pretrained(ckpt)
model = model.cuda().eval()
print(f'model loaded in {time.time() - t0:.1f}s; parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M')

flow = [m for m in model.modules() if type(m).__name__ == 'FlowLoss']
assert len(flow) == 1, f'expected one FlowLoss module, found {len(flow)}'
flow = flow[0]
released_sample = flow.sample                                    # bound method as shipped
resample = sys.modules[type(flow).__module__].resample          # helper used by the shipped sampler


def corrected_sample(self, z, prototype=None, num_samples=1, inference_token_len=48):
    """Identical to the shipped sampler except for the update rule (marked)."""
    z = z.repeat(num_samples, 1)
    noise = torch.randn(z.shape[0], self.in_channels).to(z.device)
    if prototype is not None:
        prototype = prototype.repeat(num_samples, 1)
        x = noise + prototype
    else:
        x = noise
    n = self.num_sampling_steps
    for i in range(n):
        t = (torch.ones((x.shape[0])) * i / n).to(x.device)
        pred = self.net(x, t * 1000, z)            # network's estimate of the clean end point y1
        x = x + (pred - x) / (n - i)               # CHANGED: velocity (pred - x)/(1 - t) times dt = 1/n
    if not self.training:
        old_weight = torch.eye(self.in_channels).to(x.device)
        new_weight = resample(old_weight, inference_token_len).T
        x = F.linear(x, new_weight)
        x = x.reshape(num_samples, -1, inference_token_len).transpose(0, 1)
        return x
    x = x.reshape(num_samples, -1, self.in_channels).transpose(0, 1)
    return x


returns = load_returns()

for variant in variants:
    flow.sample = released_sample if variant == 'released' else types.MethodType(corrected_sample, flow)
    for asset in ASSETS:
        suffix = '_smoke' if limit else ''                 # test runs never overwrite or block the real files
        out_path = os.path.join(RESULTS, f'aurora_{variant}_{asset}{suffix}.npz')
        if os.path.exists(out_path) and limit is None:
            print(f'{variant} {asset}: exists, skipping')
            continue
        r = returns[asset]
        vals = r.values.astype(np.float32)
        idx = origins(r)
        if limit:
            idx = idx[:limit]
        Q = np.empty((len(idx), len(TAUS)), dtype=np.float32)
        pit = np.empty(len(idx), dtype=np.float32)
        mean = np.empty(len(idx), dtype=np.float32)
        sd = np.empty(len(idx), dtype=np.float32)
        tic = time.time()
        for j, t in enumerate(idx):
            window = torch.from_numpy(vals[t - LOOKBACK + 1: t + 1]).unsqueeze(0).cuda()   # (1, 528)
            torch.manual_seed(int(t) * 7919 + ASSETS.index(asset))                        # same noise for both variants
            with torch.no_grad():
                out = model.generate(inputs=window, max_output_length=HORIZON,
                                     num_samples=N_SAMPLES, inference_token_len=48)        # (1, 1000, 48)
            day1 = out[0, :, 0].float().cpu().numpy()                                     # next-day samples
            Q[j] = np.quantile(day1, TAUS)
            pit[j] = np.mean(day1 <= vals[t + 1])
            mean[j], sd[j] = day1.mean(), day1.std()
            if j == 19:
                per = (time.time() - tic) / 20
                print(f'{variant} {asset}: {per * 1000:.0f} ms per forecast -> ~{per * len(idx) / 60:.1f} min for {len(idx)} forecasts')
        np.savez(out_path, origin=r.index[idx].values.astype('datetime64[D]'),
                 target=r.index[idx + 1].values.astype('datetime64[D]'), y=vals[idx + 1],
                 q=Q, pit=pit, mean=mean, sd=sd, taus=TAUS)
        print(f'{variant} {asset}: {len(idx)} forecasts in {(time.time() - tic) / 60:.1f} min -> {os.path.basename(out_path)}')

print(f'total {(time.time() - t0) / 60:.1f} min')
