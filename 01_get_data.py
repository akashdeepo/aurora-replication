"""Step 1: download daily adjusted prices from Yahoo Finance and build log returns.

Output: data/returns.csv  (columns: date, asset, ret)   ret = 100 * log(P_t / P_{t-1}), in percent
        data/data_summary.txt
"""
import json, os, time, urllib.request
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'data')
os.makedirs(OUT, exist_ok=True)

ASSETS = {            # Yahoo symbol -> short name used everywhere else
    '^GSPC': 'SPX',
    'AAPL': 'AAPL',
    'JPM': 'JPM',
    'BTC-USD': 'BTC',
}
P1, P2 = 1104537600, 1791417600   # 2005-01-01 .. 2026-10-08 (unix seconds)
UA = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36'


def fetch(symbol):
    url = (f'https://query1.finance.yahoo.com/v8/finance/chart/{urllib.request.quote(symbol)}'
           f'?period1={P1}&period2={P2}&interval=1d&events=div%2Csplit')
    req = urllib.request.Request(url, headers={'User-Agent': UA})
    with urllib.request.urlopen(req, timeout=60) as r:
        res = json.load(r)['chart']['result'][0]
    adj = res['indicators']['adjclose'][0]['adjclose']   # split- and dividend-adjusted close
    dates = pd.to_datetime(res['timestamp'], unit='s').normalize()
    s = pd.Series(adj, index=dates, name=symbol, dtype='float64')
    s = s[~s.index.duplicated(keep='last')].dropna()
    return s


frames, lines = [], []
for sym, name in ASSETS.items():
    px = fetch(sym)
    px.to_csv(os.path.join(OUT, f'prices_{name}.csv'), header=['adjclose'], index_label='date')
    ret = 100 * np.log(px / px.shift(1))
    ret = ret.dropna()
    big = ret[ret.abs() > 25]
    frames.append(pd.DataFrame({'date': ret.index, 'asset': name, 'ret': ret.values}))
    lines.append(f'{name:5s} {sym:8s} {len(ret):5d} returns  {ret.index[0].date()} -> {ret.index[-1].date()}  '
                 f'mean {ret.mean():+.3f}%  sd {ret.std():.3f}%  min {ret.min():+.2f}%  max {ret.max():+.2f}%  '
                 f'excess kurtosis {ret.kurt():.1f}  |r|>25%: {len(big)}')
    time.sleep(1.0)

df = pd.concat(frames, ignore_index=True)
df.to_csv(os.path.join(OUT, 'returns.csv'), index=False)
summary = '\n'.join(lines)
open(os.path.join(OUT, 'data_summary.txt'), 'w').write(summary + '\n')
print(summary)
