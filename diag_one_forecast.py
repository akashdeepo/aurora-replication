"""Diagnostic: the raw sample distribution of one Aurora forecast (S&P 500, origin 2 January 2020), with both
samplers on the same noise. Run it before the full evaluation. Prints quantiles; saves results/diag_samples.npz.
"""
import os

import numpy as np
import torch

from aurora_model import SAMPLERS, load_model, use_sampler
from common import LOOKBACK, RESULTS, load_returns

ORIGIN = '2020-01-02'


def main():
    model = load_model('cuda')
    ret = load_returns()['SPX']
    t = int(np.searchsorted(ret.index.values, np.datetime64(ORIGIN)))
    window = torch.from_numpy(ret.values[t - LOOKBACK + 1: t + 1].astype(np.float32)).unsqueeze(0).cuda()
    print(f'origin {ret.index[t].date()}  lookback mean {window.mean().item():+.3f}%  sd {window.std().item():.3f}%')

    out = {}
    for sampler in SAMPLERS:
        use_sampler(model, sampler)
        torch.manual_seed(12345)
        with torch.no_grad():
            paths = model.generate(inputs=window, max_output_length=48, num_samples=1000, inference_token_len=48)
        s = paths[0].float().cpu().numpy()                   # (1000, 48)
        out[sampler] = s
        d1 = s[:, 0]
        qs = np.quantile(d1, [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
        print(f'{sampler:9s} day 1: mean {d1.mean():+.3f} sd {d1.std():.3f} | q01 {qs[0]:+.2f} q05 {qs[1]:+.2f} '
              f'q25 {qs[2]:+.2f} q50 {qs[3]:+.2f} q75 {qs[4]:+.2f} q95 {qs[5]:+.2f} q99 {qs[6]:+.2f}')
    np.savez(os.path.join(RESULTS, 'diag_samples.npz'), **out)


if __name__ == '__main__':
    main()
