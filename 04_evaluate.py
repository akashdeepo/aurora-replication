"""Step 4: score every method on the same next-day targets and draw the figures.

Metrics (all on next-day returns, per asset and period):
  cov90 / cov98   share of realised returns inside the central 90% / 98% interval (ideal 90 / 98)
  VaR 1% / 5%     share of days the return fell below the 1% / 5% quantile (ideal 1 / 5), with
                  Kupiec's proportion-of-failures test (is the breach rate right?) and
                  Christoffersen's independence test (do breaches cluster?)
  pinball(tau)    quantile loss at tau = 0.01, 0.05, 0.95, 0.99 (lower is better)
  CRPS            approximated as 2 x mean pinball loss over tau = 0.01..0.99 (lower is better)
  width90         average width of the 90% interval, in percentage points
Outputs: results/metrics.csv, results/metrics_tables.md, figures/*.png
"""
import os
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from common import ASSETS, RESULTS, FIGS, PERIODS

METHODS = {   # file prefix -> (label, colour from the validated categorical palette, fixed order)
    'aurora_released': ('Aurora, released sampler', '#2a78d6'),
    'aurora_corrected': ('Aurora, corrected sampler', '#eb6834'),
    'garch': ('GARCH(1,1)-t', '#1baf7a'),
    'hs': ('Historical simulation', '#eda100'),
    'gauss': ('Rolling Gaussian', '#e87ba4'),
}
INK, INK2, GRID, SURF = '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb'


def load(prefix, asset):
    d = np.load(os.path.join(RESULTS, f'{prefix}_{asset}.npz'))
    return {k: d[k] for k in d.files}


def qcol(d, tau):
    return d['q'][:, int(np.argmin(np.abs(d['taus'] - tau)))]


def pinball(y, q, tau):
    u = y - q
    return np.mean(np.maximum(tau * u, (tau - 1) * u))


def kupiec(hits, p):
    n, x = len(hits), int(hits.sum())
    if n == 0:
        return np.nan
    phat = x / n
    ll0 = (n - x) * np.log(1 - p) + x * np.log(p)
    ll1 = (n - x) * np.log(1 - phat) + x * np.log(phat) if 0 < x < n else 0.0
    return float(stats.chi2.sf(-2 * (ll0 - ll1), 1))


def christoffersen(hits):
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


rows = []
for period, (a, b) in PERIODS.items():
    for asset in ASSETS:
        for prefix, (label, _) in METHODS.items():
            d = load(prefix, asset)
            m = (d['target'] >= np.datetime64(a)) & (d['target'] <= np.datetime64(b))
            y = d['y'][m].astype(float)
            sub = {k: (v[m] if isinstance(v, np.ndarray) and v.shape[:1] == d['y'].shape else v) for k, v in d.items()}
            q01, q05, q95, q99 = (qcol(sub, t) for t in (0.01, 0.05, 0.95, 0.99))
            hit1, hit5 = y < q01, y < q05
            taus_crps = np.round(np.arange(0.01, 1.0, 0.01), 2)
            crps = 2 * np.mean([pinball(y, qcol(sub, t), t) for t in taus_crps])
            rows.append(dict(
                period=period, asset=asset, method=label, n=int(m.sum()),
                cov90=100 * np.mean((y >= q05) & (y <= q95)),
                cov98=100 * np.mean((y >= q01) & (y <= q99)),
                var1_hits=100 * hit1.mean(), var1_kupiec_p=kupiec(hit1, 0.01),
                var5_hits=100 * hit5.mean(), var5_kupiec_p=kupiec(hit5, 0.05),
                var5_indep_p=christoffersen(hit5),
                pin01=pinball(y, q01, 0.01), pin05=pinball(y, q05, 0.05),
                pin95=pinball(y, q95, 0.95), pin99=pinball(y, q99, 0.99),
                crps=crps, width90=float(np.mean(q95 - q05)),
            ))

df = pd.DataFrame(rows)
df.to_csv(os.path.join(RESULTS, 'metrics.csv'), index=False)

# ---- markdown tables -------------------------------------------------------------
fmt = {'cov90': '{:.1f}', 'cov98': '{:.1f}', 'var1_hits': '{:.2f}', 'var1_kupiec_p': '{:.3f}',
       'var5_hits': '{:.2f}', 'var5_kupiec_p': '{:.3f}', 'var5_indep_p': '{:.3f}',
       'crps': '{:.4f}', 'pin01': '{:.4f}', 'width90': '{:.2f}'}
cols = ['asset', 'method', 'n'] + list(fmt)
heads = ['Asset', 'Method', 'n', 'Cov 90% (ideal 90)', 'Cov 98% (ideal 98)', 'VaR1 breach % (ideal 1)',
         'Kupiec p (1%)', 'VaR5 breach % (ideal 5)', 'Kupiec p (5%)', 'Indep. p (5%)', 'CRPS', 'Pinball 1%', 'Width 90%']
lines = []
for period in PERIODS:
    sub = df[df.period == period]
    lines.append(f'\n### {period}\n')
    lines.append('| ' + ' | '.join(heads) + ' |')
    lines.append('|' + '---|' * len(heads))
    for _, r in sub.iterrows():
        cells = [r['asset'], r['method'], str(r['n'])] + [fmt[c].format(r[c]) for c in fmt]
        lines.append('| ' + ' | '.join(cells) + ' |')
    avg = sub.groupby('method', sort=False)[['cov90', 'cov98', 'var1_hits', 'var5_hits', 'crps', 'width90']].mean()
    lines.append(f'\n**Average over the four assets ({period})**\n')
    lines.append('| Method | Cov 90% | Cov 98% | VaR1 breach % | VaR5 breach % | CRPS | Width 90% |')
    lines.append('|---|---|---|---|---|---|---|')
    for meth, r in avg.iterrows():
        lines.append(f"| {meth} | {r.cov90:.1f} | {r.cov98:.1f} | {r.var1_hits:.2f} | {r.var5_hits:.2f} | {r.crps:.4f} | {r.width90:.2f} |")
open(os.path.join(RESULTS, 'metrics_tables.md'), 'w', encoding='utf-8').write('\n'.join(lines) + '\n')
print('\n'.join(lines))

# ---- figures ------------------------------------------------------------------------
plt.rcParams.update({'font.size': 10, 'axes.edgecolor': GRID, 'axes.labelcolor': INK2,
                     'xtick.color': INK2, 'ytick.color': INK2, 'axes.titlecolor': INK,
                     'figure.facecolor': SURF, 'axes.facecolor': SURF, 'savefig.facecolor': SURF})
labels = [v[0] for v in METHODS.values()]
colors = [v[1] for v in METHODS.values()]
full = df[df.period == list(PERIODS)[0]]

# Fig 1: coverage of the 90% and 98% intervals, per asset, full period
fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=False)
for ax, col, nominal in ((axes[0], 'cov90', 90), (axes[1], 'cov98', 98)):
    x = np.arange(len(ASSETS))
    wbar = 0.15
    for k, (lab, colr) in enumerate(zip(labels, colors)):
        vals = [full[(full.asset == a) & (full.method == lab)][col].values[0] for a in ASSETS]
        ax.bar(x + (k - 2) * wbar, vals, wbar * 0.88, color=colr, label=lab)
    ax.axhline(nominal, color=INK, lw=1.2, ls='--')
    ax.text(len(ASSETS) - 0.45, nominal + 0.3, f'ideal {nominal}%', color=INK, fontsize=9, ha='right', va='bottom')
    ax.set_xticks(x, ASSETS)
    ax.set_ylim(0, 102)                     # start at zero so no method's bar is hidden below the axis
    ax.set_title(f'Share of next-day returns inside the {nominal}% interval', fontsize=10.5, loc='left')
    ax.grid(axis='y', color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
axes[0].set_ylabel('coverage (%)')
axes[1].legend(frameon=False, fontsize=8.5, loc='lower right')
fig.tight_layout()
fig.savefig(os.path.join(FIGS, 'fig1_coverage.png'), dpi=160)
plt.close(fig)

# Fig 2: PIT histograms for the S&P 500 (flat = calibrated, U-shape = intervals too narrow)
fig, axes = plt.subplots(1, 5, figsize=(13, 2.8), sharey=True)
for ax, (prefix, (lab, colr)) in zip(axes, METHODS.items()):
    d = load(prefix, 'SPX')
    ax.hist(d['pit'], bins=20, range=(0, 1), color=colr, edgecolor=SURF, linewidth=1.5)
    ax.axhline(len(d['pit']) / 20, color=INK, lw=1, ls='--')
    ax.set_title(lab, fontsize=9.5, loc='left')
    ax.set_xticks([0, 0.5, 1])
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
axes[0].set_ylabel('days')
fig.suptitle('S&P 500: where each realised return fell inside its forecast distribution (PIT); dashed = perfectly calibrated',
             fontsize=10, x=0.01, ha='left', color=INK)
fig.tight_layout()
fig.savefig(os.path.join(FIGS, 'fig2_pit_spx.png'), dpi=160)
plt.close(fig)

# Fig 3: the COVID crash, S&P 500 returns against three 1% VaR lines
fig, ax = plt.subplots(figsize=(11, 4))
ref = load('garch', 'SPX')
m = (ref['target'] >= np.datetime64('2020-01-15')) & (ref['target'] <= np.datetime64('2020-06-30'))
dates = ref['target'][m].astype('datetime64[D]')
ax.bar(dates, ref['y'][m], color='#b9b8b2', width=0.8, label='daily return')
for prefix in ('aurora_released', 'aurora_corrected', 'garch'):
    d = load(prefix, 'SPX')
    v = qcol(d, 0.01)[m]
    lab, colr = METHODS[prefix]
    nb = int(np.sum(d['y'][m] < v))
    ax.plot(dates, v, color=colr, lw=2, label=f'{lab}: 1% VaR ({nb} breaches)')
ax.axhline(0, color=GRID, lw=0.8)
ax.set_ylabel('return (%)')
ax.set_title('S&P 500, Jan to Jun 2020: next-day 1% value-at-risk from each model', fontsize=10.5, loc='left')
ax.legend(frameon=False, fontsize=8.5, loc='lower left')
for s in ('top', 'right'):
    ax.spines[s].set_visible(False)
fig.tight_layout()
fig.savefig(os.path.join(FIGS, 'fig3_covid_var.png'), dpi=160)
plt.close(fig)

# Fig 4: average interval width vs realised coverage, all assets, full period
fig, ax = plt.subplots(figsize=(6.5, 4.2))
for lab, colr, mk in zip(labels, colors, ['o', 's', '^', 'D', 'v']):   # shape as a second cue besides colour
    s = full[full.method == lab]
    ax.scatter(s.width90, s.cov90, s=60, color=colr, marker=mk, edgecolor=SURF, linewidth=2, label=lab, zorder=3)
    if lab.startswith('Aurora'):            # label only the Aurora points; the baseline cluster near 90% would collide
        for _, r in s.iterrows():
            ax.annotate(r.asset, (r.width90, r.cov90), textcoords='offset points', xytext=(6, -3), fontsize=8, color=INK2)
ax.axhline(90, color=INK, lw=1, ls='--')
ax.set_xlabel('average width of the 90% interval (percentage points)')
ax.set_ylabel('coverage of the 90% interval (%)')
ax.set_title('Narrow intervals buy their sharpness with missed coverage', fontsize=10.5, loc='left')
ax.grid(color=GRID, lw=0.8)
ax.set_axisbelow(True)
ax.legend(frameon=False, fontsize=8, loc='lower right')
for s in ('top', 'right'):
    ax.spines[s].set_visible(False)
fig.tight_layout()
fig.savefig(os.path.join(FIGS, 'fig4_width_vs_coverage.png'), dpi=160)
plt.close(fig)
print('figures written to', FIGS)
