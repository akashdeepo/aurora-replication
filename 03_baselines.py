"""Step 3: classical baselines on exactly the same forecast origins as Aurora.

  gauss : normal distribution with the mean and sd of the last 528 returns (what Aurora's own
          normalization step would give if the network added nothing)
  hs    : historical simulation, empirical quantiles of the last 528 returns (same data Aurora sees)
  garch : GARCH(1,1) with Student-t errors, refit every day on the last 1,000 returns
          (the standard risk-desk model for next-day VaR)

Output: results/<method>_<asset>.npz with the same fields as the Aurora files.
"""
import os, time, warnings
import numpy as np
from scipy import stats
from arch import arch_model

from common import ASSETS, LOOKBACK, GARCH_WINDOW, TAUS, RESULTS, load_returns, origins

warnings.filterwarnings('ignore')
returns = load_returns()


def save(name, asset, r, idx, Q, pit, mean, sd):
    np.savez(os.path.join(RESULTS, f'{name}_{asset}.npz'),
             origin=r.index[idx].values.astype('datetime64[D]'),
             target=r.index[idx + 1].values.astype('datetime64[D]'), y=r.values[idx + 1].astype(np.float32),
             q=Q.astype(np.float32), pit=pit.astype(np.float32), mean=mean.astype(np.float32),
             sd=sd.astype(np.float32), taus=TAUS)


for asset in ASSETS:
    r = returns[asset]
    v = r.values
    idx = origins(r)
    n = len(idx)
    out = {m: dict(Q=np.empty((n, len(TAUS))), pit=np.empty(n), mean=np.empty(n), sd=np.empty(n))
           for m in ('gauss', 'hs', 'garch')}
    start_vals = None
    tic = time.time()
    for j, t in enumerate(idx):
        y = v[t + 1]
        w = v[t - LOOKBACK + 1: t + 1]
        mu, s = w.mean(), w.std(ddof=1)
        out['gauss']['Q'][j] = mu + s * stats.norm.ppf(TAUS)
        out['gauss']['pit'][j] = stats.norm.cdf((y - mu) / s)
        out['gauss']['mean'][j], out['gauss']['sd'][j] = mu, s

        out['hs']['Q'][j] = np.quantile(w, TAUS)
        out['hs']['pit'][j] = np.mean(w <= y)
        out['hs']['mean'][j], out['hs']['sd'][j] = mu, s

        gw = v[t - GARCH_WINDOW + 1: t + 1]
        am = arch_model(gw, mean='Constant', vol='GARCH', p=1, q=1, dist='t', rescale=False)
        res = am.fit(disp='off', starting_values=start_vals, options={'maxiter': 300})
        start_vals = res.params.values
        f = res.forecast(horizon=1, reindex=False)
        m1, var1 = float(f.mean.values[-1, 0]), float(f.variance.values[-1, 0])
        nu = float(res.params['nu'])
        sig = np.sqrt(var1)
        z = am.distribution.ppf(TAUS, [nu])                    # standardized (unit-variance) Student-t quantiles
        out['garch']['Q'][j] = m1 + sig * z
        out['garch']['pit'][j] = am.distribution.cdf(np.array([(y - m1) / sig]), [nu])[0]
        out['garch']['mean'][j], out['garch']['sd'][j] = m1, sig
        if j == 49:
            print(f'{asset}: {(time.time() - tic) / 50 * 1000:.0f} ms per origin -> ~{(time.time() - tic) / 50 * n / 60:.1f} min')
    for m, d in out.items():
        save(m, asset, r, idx, d['Q'], d['pit'], d['mean'], d['sd'])
    print(f'{asset}: {n} origins done in {(time.time() - tic) / 60:.1f} min')
