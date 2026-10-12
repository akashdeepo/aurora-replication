"""Settings shared by every step of the pilot."""
import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, 'data')
RESULTS = os.path.join(HERE, 'results')
FIGS = os.path.join(HERE, 'figures')
for d in (DATA, RESULTS, FIGS):
    os.makedirs(d, exist_ok=True)

ASSETS = ['SPX', 'AAPL', 'JPM', 'BTC']
LOOKBACK = 528            # Aurora's default context: 11 patches of 48 points
GARCH_WINDOW = 1000       # standard rolling estimation window for the GARCH baseline
TEST_START = '2020-01-01' # first forecast origin on or after this date (includes the COVID crash)
N_SAMPLES = 1000          # Aurora samples per forecast (1,000 so the 1% quantile is not just the minimum of 100)
HORIZON = 48              # Aurora forecasts one 48-point patch; we score step 1 (next day)

# Quantile grid used for every method (pinball loss, CRPS approximation, VaR, intervals)
TAUS = np.round(np.concatenate([[0.005], np.arange(0.01, 1.0, 0.01), [0.995]]), 3)

PERIODS = {
    'full (2020-01 to 2026-10)': ('2020-01-01', '2026-12-31'),
    'COVID crash (2020-02-15 to 2020-04-30)': ('2020-02-15', '2020-04-30'),
    'unseen by Aurora (2025-10 to 2026-10)': ('2025-10-01', '2026-12-31'),
}


def load_returns():
    df = pd.read_csv(os.path.join(DATA, 'returns.csv'), parse_dates=['date'])
    return {a: df[df.asset == a].set_index('date')['ret'].sort_index() for a in ASSETS}


def origins(ret):
    """Indices t (last observed day) for which the next-day return t+1 exists and enough history precedes t."""
    idx = np.arange(max(LOOKBACK, GARCH_WINDOW) - 1, len(ret) - 1)
    keep = ret.index[idx] >= pd.Timestamp(TEST_START)
    return idx[keep]
