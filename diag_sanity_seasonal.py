"""Sanity check of the pipeline on data Aurora is built for: 50 noisy seasonal series with a trend.

If the same calling code gives sensible forecasts here, poor results on returns are about the model and the
data, not about how the model is called. Runs both samplers.
"""
import numpy as np
import torch

from aurora_model import SAMPLERS, load_model, use_sampler

N_SERIES, CONTEXT, HORIZON = 50, 528, 48


def seasonal_series(rng):
    k = np.arange(CONTEXT + HORIZON)
    period = rng.choice([24, 48, 12])
    y = (2 * np.sin(2 * np.pi * k / period + rng.uniform(0, 6.28)) + 0.004 * k * rng.uniform(-1, 1)
         + rng.normal(0, 0.5, CONTEXT + HORIZON))
    return y.astype(np.float32), period


def evaluate(model):
    rng = np.random.default_rng(0)
    cover90, corr, mase_model, mase_naive = [], [], [], []
    for i in range(N_SERIES):
        y, period = seasonal_series(rng)
        x = torch.from_numpy(y[:CONTEXT]).unsqueeze(0).cuda()
        torch.manual_seed(i)
        with torch.no_grad():
            s = model.generate(inputs=x, max_output_length=HORIZON, num_samples=1000, inference_token_len=48)[0].cpu().numpy()
        truth, median = y[CONTEXT:], np.median(s, axis=0)
        lo, hi = np.quantile(s, 0.05, axis=0), np.quantile(s, 0.95, axis=0)
        naive = y[CONTEXT - period: CONTEXT][np.arange(HORIZON) % period]     # repeat the last cycle
        scale = np.mean(np.abs(np.diff(y[:CONTEXT])))
        cover90.append(np.mean((truth >= lo) & (truth <= hi)))
        corr.append(np.corrcoef(median, truth)[0, 1])
        mase_model.append(np.mean(np.abs(median - truth)) / scale)
        mase_naive.append(np.mean(np.abs(naive - truth)) / scale)
    return np.mean(corr), 100 * np.mean(cover90), np.mean(mase_model), np.mean(mase_naive)


def main():
    model = load_model('cuda')
    for sampler in SAMPLERS:
        use_sampler(model, sampler)
        corr, cov, mase, naive = evaluate(model)
        print(f'{N_SERIES} noisy seasonal series, {HORIZON}-step forecasts ({sampler} sampler):')
        print(f'  corr(median forecast, truth): {corr:.2f}')
        print(f'  90% interval coverage:        {cov:.1f}%')
        print(f'  MASE: Aurora median {mase:.3f}, seasonal-naive {naive:.3f} (lower is better)')


if __name__ == '__main__':
    main()
