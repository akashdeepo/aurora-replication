"""Scoring rules and value-at-risk backtests used by 04_evaluate.py."""
import numpy as np
from scipy import stats


def quantile_column(forecast, tau):
    """Column of the forecast's quantile matrix closest to level tau."""
    return forecast['q'][:, int(np.argmin(np.abs(forecast['taus'] - tau)))]


def pinball(y, q, tau):
    """Mean quantile (pinball) loss of quantile forecasts q at level tau."""
    u = y - q
    return np.mean(np.maximum(tau * u, (tau - 1) * u))


def crps_from_quantiles(y, forecast, taus=None):
    """CRPS approximated as twice the mean pinball loss over a grid of levels (1% to 99% by default)."""
    taus = np.round(np.arange(0.01, 1.0, 0.01), 2) if taus is None else taus
    return 2 * np.mean([pinball(y, quantile_column(forecast, t), t) for t in taus])


def kupiec(hits, p):
    """Kupiec's proportion-of-failures test: p-value for 'the breach rate equals p'."""
    n, x = len(hits), int(hits.sum())
    if n == 0:
        return np.nan
    phat = x / n
    ll0 = (n - x) * np.log(1 - p) + x * np.log(p)
    ll1 = (n - x) * np.log(1 - phat) + x * np.log(phat) if 0 < x < n else 0.0
    return float(stats.chi2.sf(-2 * (ll0 - ll1), 1))


def christoffersen(hits):
    """Christoffersen's independence test: p-value for 'a breach today does not change tomorrow's odds'."""
    h = hits.astype(int)
    a, b = h[:-1], h[1:]
    n00, n01 = np.sum((a == 0) & (b == 0)), np.sum((a == 0) & (b == 1))
    n10, n11 = np.sum((a == 1) & (b == 0)), np.sum((a == 1) & (b == 1))

    def xlogy(x, y):
        return x * np.log(y) if x > 0 else 0.0

    pi01 = n01 / (n00 + n01) if n00 + n01 else 0.0
    pi11 = n11 / (n10 + n11) if n10 + n11 else 0.0
    pi = (n01 + n11) / max(n00 + n01 + n10 + n11, 1)
    l0 = xlogy(n00 + n10, 1 - pi) + xlogy(n01 + n11, pi)
    l1 = xlogy(n00, 1 - pi01) + xlogy(n01, pi01) + xlogy(n10, 1 - pi11) + xlogy(n11, pi11)
    return float(stats.chi2.sf(-2 * (l0 - l1), 1))
