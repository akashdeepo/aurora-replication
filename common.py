"""Settings and data access shared by every step."""
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, 'data')
RESULTS = os.path.join(HERE, 'results')
FIGS = os.path.join(HERE, 'figures')
for _d in (DATA, RESULTS, FIGS):
    os.makedirs(_d, exist_ok=True)

ASSETS = ['SPX', 'AAPL', 'JPM', 'BTC']
LOOKBACK = 528            # Aurora's default context: 11 patches of 48 points
GARCH_WINDOW = 1000       # rolling estimation window for the GARCH baseline
TEST_START = '2020-01-01' # first forecast origin on or after this date (includes the COVID crash)
N_SAMPLES = 1000          # Aurora samples per forecast, so the 1% quantile rests on about 10 samples
HORIZON = 48              # Aurora generates one 48-point patch; step 1 (the next day) is scored

# Quantile grid used for every method (pinball loss, CRPS approximation, VaR, intervals)
TAUS = np.round(np.concatenate([[0.005], np.arange(0.01, 1.0, 0.01), [0.995]]), 3)

PERIODS = {
    'full (2020-01 to 2026-10)': ('2020-01-01', '2026-12-31'),
    'COVID crash (2020-02-15 to 2020-04-30)': ('2020-02-15', '2020-04-30'),
    'unseen by Aurora (2025-10 to 2026-10)': ('2025-10-01', '2026-12-31'),
}


def load_returns():
    """Daily log returns in percent, one Series per asset, indexed by date."""
    path = os.path.join(DATA, 'returns.csv')
    if not os.path.exists(path):
        raise FileNotFoundError(f'{path} not found: run `python 01_get_data.py` first')
    df = pd.read_csv(path, parse_dates=['date'])
    return {a: df[df.asset == a].set_index('date')['ret'].sort_index() for a in ASSETS}


def origins(ret):
    """Forecast origins: indices t (last observed day) whose next-day return exists, with enough history."""
    idx = np.arange(max(LOOKBACK, GARCH_WINDOW) - 1, len(ret) - 1)
    return idx[ret.index[idx] >= pd.Timestamp(TEST_START)]


def save_forecasts(path, ret, idx, q, pit, mean, sd):
    """Save one method's forecasts for one asset; every method uses the same fields."""
    np.savez(path, origin=ret.index[idx].values.astype('datetime64[D]'),
             target=ret.index[idx + 1].values.astype('datetime64[D]'),
             y=ret.values[idx + 1].astype(np.float32), q=np.asarray(q, np.float32),
             pit=np.asarray(pit, np.float32), mean=np.asarray(mean, np.float32),
             sd=np.asarray(sd, np.float32), taus=TAUS)


def load_forecasts(method, asset):
    """Load the forecasts saved by save_forecasts as a dict of arrays."""
    with np.load(os.path.join(RESULTS, f'{method}_{asset}.npz')) as d:
        return {k: d[k] for k in d.files}
