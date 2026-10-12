"""Scoring rules and VaR backtests on cases with known answers."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from metrics import christoffersen, crps_from_quantiles, kupiec, pinball, quantile_column  # noqa: E402
from scipy import stats  # noqa: E402


def test_pinball_weights_misses_by_level():
    assert np.isclose(pinball(np.array([1.0]), np.array([0.0]), 0.9), 0.9)    # outcome above the quantile
    assert np.isclose(pinball(np.array([-1.0]), np.array([0.0]), 0.9), 0.1)   # outcome below the quantile


def test_kupiec_accepts_the_right_rate_and_rejects_the_wrong_one():
    hits = np.zeros(1000, bool)
    hits[:10] = True                                      # exactly 1% breaches
    assert kupiec(hits, 0.01) > 0.99
    hits[:100] = True                                     # 10% breaches against a 1% target
    assert kupiec(hits, 0.01) < 1e-6


def test_christoffersen_flags_clustered_breaches():
    rng = np.random.default_rng(0)
    scattered = rng.random(2000) < 0.05
    clustered = np.zeros(2000, bool)
    for start in range(0, 2000, 200):
        clustered[start:start + 10] = True                # same 5% rate, in runs of ten
    assert christoffersen(scattered) > 0.01
    assert christoffersen(clustered) < 1e-6


def test_crps_matches_the_closed_form_for_a_normal_forecast():
    taus = np.round(np.arange(0.01, 1.0, 0.01), 2)
    y = np.random.default_rng(1).normal(size=5000)
    forecast = {'q': np.tile(stats.norm.ppf(taus), (len(y), 1)), 'taus': taus}
    exact = np.mean(y * (2 * stats.norm.cdf(y) - 1) + 2 * stats.norm.pdf(y) - 1 / np.sqrt(np.pi))
    assert abs(crps_from_quantiles(y, forecast, taus) - exact) / exact < 0.03


def test_quantile_column_picks_the_nearest_level():
    forecast = {'q': np.array([[1.0, 2.0, 3.0]]), 'taus': np.array([0.01, 0.5, 0.99])}
    assert quantile_column(forecast, 0.5)[0] == 2.0
