"""Step 1: download daily adjusted prices from Yahoo Finance and build log returns.

Output: data/returns.csv (columns date, asset, ret; ret = 100 * log(P_t / P_{t-1}), in percent),
        data/prices_<asset>.csv and data/data_summary.txt
"""
import json
import os
import time
import urllib.request

import numpy as np
import pandas as pd

from common import DATA

SYMBOLS = {'^GSPC': 'SPX', 'AAPL': 'AAPL', 'JPM': 'JPM', 'BTC-USD': 'BTC'}   # Yahoo symbol -> short name
START, END = '2005-01-01', '2026-10-08'
USER_AGENT = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36'


def fetch_adjusted_close(symbol):
    """Daily split- and dividend-adjusted closes for one symbol, indexed by date."""
    p1, p2 = (int(pd.Timestamp(d, tz='UTC').timestamp()) for d in (START, END))
    url = (f'https://query1.finance.yahoo.com/v8/finance/chart/{urllib.request.quote(symbol)}'
           f'?period1={p1}&period2={p2}&interval=1d&events=div%2Csplit')
    request = urllib.request.Request(url, headers={'User-Agent': USER_AGENT})
    with urllib.request.urlopen(request, timeout=60) as response:
        result = json.load(response)['chart']['result'][0]
    dates = pd.to_datetime(result['timestamp'], unit='s').normalize()
    prices = pd.Series(result['indicators']['adjclose'][0]['adjclose'], index=dates, dtype='float64')
    return prices[~prices.index.duplicated(keep='last')].dropna()


def summarize(name, symbol, ret):
    return (f'{name:5s} {symbol:8s} {len(ret):5d} returns  {ret.index[0].date()} -> {ret.index[-1].date()}  '
            f'mean {ret.mean():+.3f}%  sd {ret.std():.3f}%  min {ret.min():+.2f}%  max {ret.max():+.2f}%  '
            f'excess kurtosis {ret.kurt():.1f}  |r|>25%: {int((ret.abs() > 25).sum())}')


def main():
    frames, lines = [], []
    for symbol, name in SYMBOLS.items():
        prices = fetch_adjusted_close(symbol)
        prices.to_csv(os.path.join(DATA, f'prices_{name}.csv'), header=['adjclose'], index_label='date')
        ret = (100 * np.log(prices / prices.shift(1))).dropna()
        frames.append(pd.DataFrame({'date': ret.index, 'asset': name, 'ret': ret.values}))
        lines.append(summarize(name, symbol, ret))
        time.sleep(1.0)                                    # be polite to the endpoint
    pd.concat(frames, ignore_index=True).to_csv(os.path.join(DATA, 'returns.csv'), index=False)
    summary = '\n'.join(lines)
    with open(os.path.join(DATA, 'data_summary.txt'), 'w', encoding='utf-8') as f:
        f.write(summary + '\n')
    print(summary)


if __name__ == '__main__':
    main()
