"""Step 3: classical baselines on exactly the same forecast origins as Aurora.

  gauss  normal distribution with the mean and sd of the last 528 returns (what Aurora's own
         normalization would give if the network added nothing)
  hs     historical simulation: empirical quantiles of the last 528 returns (the data Aurora sees)
  garch  GARCH(1,1) with Student-t errors, refit every day on the last 1,000 returns

Output: results/<method>_<asset>.npz, with the same fields as the Aurora files.
"""
import os
import time
import warnings

import numpy as np
from arch import arch_model
from scipy import stats

from common import ASSETS, GARCH_WINDOW, LOOKBACK, RESULTS, TAUS, load_returns, origins, save_forecasts

METHODS = ('gauss', 'hs', 'garch')


def gauss_forecast(window, y):
    mu, s = window.mean(), window.std(ddof=1)
    return mu + s * stats.norm.ppf(TAUS), stats.norm.cdf((y - mu) / s), mu, s


def hs_forecast(window, y):
    return np.quantile(window, TAUS), np.mean(window <= y), window.mean(), window.std(ddof=1)


class GarchT:
    """Daily-refit GARCH(1,1)-t; each fit starts from the previous day's estimates."""

    def __init__(self):
        self.start = None

    def forecast(self, window, y):
        model = arch_model(window, mean='Constant', vol='GARCH', p=1, q=1, dist='t', rescale=False)
        res = model.fit(disp='off', starting_values=self.start, options={'maxiter': 300})
        self.start = res.params.values
        f = res.forecast(horizon=1, reindex=False)
        m1, sig = float(f.mean.values[-1, 0]), float(np.sqrt(f.variance.values[-1, 0]))
        nu = float(res.params['nu'])
        z = model.distribution.ppf(TAUS, [nu])                 # unit-variance Student-t quantiles
        pit = model.distribution.cdf(np.array([(y - m1) / sig]), [nu])[0]
        return m1 + sig * z, pit, m1, sig


def run_asset(ret):
    v, idx = ret.values, origins(ret)
    n = len(idx)
    out = {m: dict(q=np.empty((n, len(TAUS))), pit=np.empty(n), mean=np.empty(n), sd=np.empty(n)) for m in METHODS}
    garch = GarchT()
    tic = time.time()
    for j, t in enumerate(idx):
        y = v[t + 1]
        w = v[t - LOOKBACK + 1: t + 1]
        fits = {'gauss': gauss_forecast(w, y), 'hs': hs_forecast(w, y),
                'garch': garch.forecast(v[t - GARCH_WINDOW + 1: t + 1], y)}
        for m, (q, pit, mean, sd) in fits.items():
            out[m]['q'][j], out[m]['pit'][j], out[m]['mean'][j], out[m]['sd'][j] = q, pit, mean, sd
        if j == 49:
            per = (time.time() - tic) / 50
            print(f'  {per * 1000:.0f} ms per origin -> about {per * n / 60:.1f} min')
    return idx, out


def main():
    warnings.filterwarnings('ignore')                          # arch's convergence chatter on short windows
    returns = load_returns()
    for asset in ASSETS:
        tic = time.time()
        idx, out = run_asset(returns[asset])
        for m, d in out.items():
            save_forecasts(os.path.join(RESULTS, f'{m}_{asset}.npz'), returns[asset], idx, **d)
        print(f'{asset}: {len(idx)} origins in {(time.time() - tic) / 60:.1f} min')


if __name__ == '__main__':
    main()
