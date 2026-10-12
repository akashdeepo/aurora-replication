"""Diagnostic: look at the raw sample distribution of a single Aurora forecast (S&P 500, origin 2020-01-02),
for both samplers, before running the full evaluation. Prints quantiles and saves results/diag_samples.npz."""
import os, sys, types
import numpy as np
import torch

sys.argv = [sys.argv[0], 'released', '--limit', '0']          # reuse the model setup from step 2 without running it
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), '02_aurora_forecasts.py'), encoding='utf-8').read()
setup = src.split('returns = load_returns()')[0]                # model loading + corrected_sample definition only
exec(compile(setup, '02_aurora_forecasts.py', 'exec'))

from common import load_returns, LOOKBACK
r = load_returns()['SPX']
t = int(np.searchsorted(r.index.values, np.datetime64('2020-01-02')))
window = torch.from_numpy(r.values[t - LOOKBACK + 1: t + 1].astype(np.float32)).unsqueeze(0).cuda()
print(f'origin {r.index[t].date()}  lookback mean {window.mean().item():+.3f}%  sd {window.std().item():.3f}%')

out = {}
for variant in ('released', 'corrected'):
    flow.sample = released_sample if variant == 'released' else types.MethodType(corrected_sample, flow)
    torch.manual_seed(12345)
    with torch.no_grad():
        o = model.generate(inputs=window, max_output_length=48, num_samples=1000, inference_token_len=48)
    s = o[0].float().cpu().numpy()          # (1000, 48)
    out[variant] = s
    d1 = s[:, 0]
    qs = np.quantile(d1, [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
    print(f'{variant:9s} day-1: mean {d1.mean():+.3f} sd {d1.std():.3f} | q01 {qs[0]:+.2f} q05 {qs[1]:+.2f} q25 {qs[2]:+.2f} '
          f'q50 {qs[3]:+.2f} q75 {qs[4]:+.2f} q95 {qs[5]:+.2f} q99 {qs[6]:+.2f} | min {d1.min():+.2f} max {d1.max():+.2f}')
    print(f'{"":9s} sd across all 48 days (avg per day): {s.std(axis=0).mean():.3f};  per-day sd day1..5: {np.round(s.std(axis=0)[:5], 3)}')
np.savez(os.path.join(RESULTS, 'diag_samples.npz'), **out)
