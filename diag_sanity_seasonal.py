"""Sanity check of the pipeline: on a series Aurora was built for (daily-style seasonality + trend + noise),
does the same code produce sensible forecasts? If yes, poor results on returns are about the model and the
data, not about how we call it. Runs both samplers."""
import os, sys
import numpy as np
import torch

sys.argv = [sys.argv[0], 'released', '--limit', '0']
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), '02_aurora_forecasts.py'), encoding='utf-8').read()
exec(compile(src.split('returns = load_returns()')[0], '02_aurora_forecasts.py', 'exec'))

import types
for variant in ('released', 'corrected'):
  flow.sample = released_sample if variant == 'released' else types.MethodType(corrected_sample, flow)
  rng = np.random.default_rng(0)
  n_series, L, H = 50, 528, 48
  cover90, corr, mase_aur, mase_naive = [], [], [], []
  for i in range(n_series):
      k = np.arange(L + H)
      period = rng.choice([24, 48, 12])
      y = (2 * np.sin(2 * np.pi * k / period + rng.uniform(0, 6.28)) + 0.004 * k * rng.uniform(-1, 1)
           + rng.normal(0, 0.5, L + H)).astype(np.float32)
      x = torch.from_numpy(y[:L]).unsqueeze(0).cuda()
      torch.manual_seed(i)
      with torch.no_grad():
          s = model.generate(inputs=x, max_output_length=H, num_samples=1000, inference_token_len=48)[0].cpu().numpy()
      truth = y[L:]
      med = np.median(s, axis=0)
      lo, hi = np.quantile(s, 0.05, axis=0), np.quantile(s, 0.95, axis=0)
      cover90.append(np.mean((truth >= lo) & (truth <= hi)))
      corr.append(np.corrcoef(med, truth)[0, 1])
      seasonal_naive = y[L - period: L][np.arange(H) % period]          # repeat the last cycle
      scale = np.mean(np.abs(np.diff(y[:L])))
      mase_aur.append(np.mean(np.abs(med - truth)) / scale)
      mase_naive.append(np.mean(np.abs(seasonal_naive - truth)) / scale)

  print(f'{n_series} noisy seasonal series, 48-step forecasts ({variant} sampler):')
  print(f'  corr(median forecast, truth): mean {np.mean(corr):.2f}')
  print(f'  90% interval coverage:        mean {100 * np.mean(cover90):.1f}%')
  print(f'  MASE Aurora median {np.mean(mase_aur):.3f}  vs seasonal-naive {np.mean(mase_naive):.3f}  (lower is better)')
