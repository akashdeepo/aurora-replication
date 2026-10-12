"""Show the 'endogenous image' Aurora builds from a time series, using Aurora's own UnifiedImageProcessor.

Three 528-point windows, each normalized the way Aurora normalizes its input (subtract mean, divide by sd):
  1. a seasonal series (period 24, slow trend, noise): the case the image lane is designed for
  2. S&P 500 daily returns, window ending 2 Jan 2020 (calm market)
  3. S&P 500 daily returns, window ending 20 Mar 2020 (contains the COVID crash)
Output: figures/fig5_endogenous_images.png
"""
import os, sys
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, 'vendor'))
from aurora.configuration_aurora import AuroraConfig
from aurora.modality_connector import UnifiedImageProcessor
from common import load_returns, LOOKBACK, FIGS

cfg = AuroraConfig.from_json_file(os.path.join(HERE, 'vendor', 'aurora', 'config.json'))
proc = UnifiedImageProcessor(cfg)

rng = np.random.default_rng(3)
k = np.arange(LOOKBACK)
seasonal = 2 * np.sin(2 * np.pi * k / 24) + 0.006 * k + rng.normal(0, 0.5, LOOKBACK)
spx = load_returns()['SPX']
def window(end):
    t = int(np.searchsorted(spx.index.values, np.datetime64(end)))
    return spx.values[t - LOOKBACK + 1: t + 1]
cases = [('Seasonal series (period 24)', seasonal),
         ('S&P 500 returns, window ending 2 Jan 2020', window('2020-01-02')),
         ('S&P 500 returns, window ending 20 Mar 2020', window('2020-03-20'))]

INK, INK2, GRID, SURF, LINE = '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb', '#2a78d6'
plt.rcParams.update({'font.size': 9.5, 'axes.edgecolor': GRID, 'axes.labelcolor': INK2, 'xtick.color': INK2,
                     'ytick.color': INK2, 'figure.facecolor': SURF, 'axes.facecolor': SURF, 'savefig.facecolor': SURF})
fig, axes = plt.subplots(2, 3, figsize=(13, 7.2), gridspec_kw={'height_ratios': [1, 2.1]})
for j, (title, series) in enumerate(cases):
    x = (series - series.mean()) / series.std()                    # Aurora's per-window normalization
    xt = torch.tensor(x, dtype=torch.float32).unsqueeze(0)
    period = int(proc._period_search(xt)[0])
    period = period if 0 < period < LOOKBACK else cfg.token_len
    img = proc.process_pseudo_image(xt)[0, 0].numpy()             # one of the 3 identical channels, 224 x 224
    ncycles = int(np.ceil(LOOKBACK / period))

    ax = axes[0, j]
    ax.plot(x, color=LINE, lw=0.8)
    ax.set_title(title, fontsize=10, loc='left', color=INK)
    ax.set_xlim(0, LOOKBACK)
    ax.set_xlabel('time step (oldest to newest)')
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)

    ax = axes[1, j]
    lim = np.percentile(np.abs(img), 99)
    ax.imshow(img, cmap='gray', vmin=-lim, vmax=lim, aspect='equal')
    ax.set_title(f'what the ViT sees: FFT period = {period}, so {period} rows x {ncycles} cycles,\nresized to 224 x 224',
                 fontsize=9, loc='left', color=INK2)
    ax.set_xlabel('cycle number (left = oldest)')
    ax.set_ylabel('position within the cycle')
    ax.set_xticks([])
    ax.set_yticks([])
    print(f'{title}: period found by FFT = {period}  ->  grid {period} x {ncycles}')

fig.suptitle("Aurora's 'endogenous image': the series folded by its strongest cycle, values shown as brightness",
             fontsize=11, x=0.01, ha='left', color=INK)
fig.tight_layout()
out = os.path.join(FIGS, 'fig5_endogenous_images.png')
fig.savefig(out, dpi=150)
print('saved', out)
