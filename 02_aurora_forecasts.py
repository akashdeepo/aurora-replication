"""Step 2: zero-shot next-day return forecasts from the released Aurora checkpoint.

For every forecast origin, the last 528 daily returns go in and 1,000 sampled 48-day paths come out; day 1
is kept. Both samplers (see aurora_model.py) run on identical random noise: the seed depends only on the
origin and the asset. Batch size is 1 on purpose: Aurora picks its image period from the batch-averaged
spectrum, so batching windows together would make each forecast depend on its neighbours.

Usage:  python 02_aurora_forecasts.py [released|corrected|both] [--limit N]
Output: results/aurora_<sampler>_<asset>.npz (with --limit: *_smoke.npz, so test runs never touch real results)
"""
import argparse
import os
import time

import numpy as np
import torch

from aurora_model import SAMPLERS, load_model, use_sampler
from common import ASSETS, HORIZON, LOOKBACK, N_SAMPLES, RESULTS, TAUS, load_returns, origins, save_forecasts


def forecast_asset(model, ret, asset, limit=None):
    """Next-day forecast quantiles, PIT, mean and sd for every origin of one asset."""
    vals = ret.values.astype(np.float32)
    idx = origins(ret)[:limit] if limit else origins(ret)
    q = np.empty((len(idx), len(TAUS)), np.float32)
    pit, mean, sd = (np.empty(len(idx), np.float32) for _ in range(3))
    tic = time.time()
    for j, t in enumerate(idx):
        window = torch.from_numpy(vals[t - LOOKBACK + 1: t + 1]).unsqueeze(0).cuda()     # (1, 528)
        torch.manual_seed(int(t) * 7919 + ASSETS.index(asset))                          # same noise for both samplers
        with torch.no_grad():
            paths = model.generate(inputs=window, max_output_length=HORIZON,
                                   num_samples=N_SAMPLES, inference_token_len=48)        # (1, 1000, 48)
        day1 = paths[0, :, 0].float().cpu().numpy()
        q[j] = np.quantile(day1, TAUS)
        pit[j] = np.mean(day1 <= vals[t + 1])
        mean[j], sd[j] = day1.mean(), day1.std()
        if j == 19:
            per = (time.time() - tic) / 20
            print(f'  {per * 1000:.0f} ms per forecast -> about {per * len(idx) / 60:.1f} min for {len(idx)} forecasts')
    return idx, q, pit, mean, sd


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('sampler', nargs='?', default='both', choices=SAMPLERS + ('both',))
    parser.add_argument('--limit', type=int, default=None, help='only the first N origins per asset (smoke test)')
    args = parser.parse_args()
    samplers = SAMPLERS if args.sampler == 'both' else (args.sampler,)

    t0 = time.time()
    model = load_model('cuda')
    print(f'model loaded in {time.time() - t0:.1f}s; {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M parameters')
    returns = load_returns()

    for sampler in samplers:
        use_sampler(model, sampler)
        for asset in ASSETS:
            suffix = '_smoke' if args.limit else ''
            path = os.path.join(RESULTS, f'aurora_{sampler}_{asset}{suffix}.npz')
            if os.path.exists(path) and not args.limit:
                print(f'{sampler} {asset}: exists, skipping')
                continue
            tic = time.time()
            idx, q, pit, mean, sd = forecast_asset(model, returns[asset], asset, args.limit)
            save_forecasts(path, returns[asset], idx, q, pit, mean, sd)
            print(f'{sampler} {asset}: {len(idx)} forecasts in {(time.time() - tic) / 60:.1f} min -> {os.path.basename(path)}')
    print(f'total {(time.time() - t0) / 60:.1f} min')


if __name__ == '__main__':
    main()
