"""Step 4: score every method on the same next-day targets and draw the figures.

Metrics, per asset and period, on next-day returns:
  cov90, cov98   share of returns inside the central 90% / 98% interval (ideal 90 / 98)
  VaR 1% / 5%    share of days below the 1% / 5% quantile (ideal 1 / 5), with Kupiec's test (is the breach
                 rate right?) and Christoffersen's test (do breaches cluster?)
  pinball        quantile loss at 1%, 5%, 95% and 99% (lower is better)
  CRPS           twice the mean pinball loss over levels 1% to 99% (lower is better)
  width90        average width of the 90% interval, in percentage points

Outputs: results/metrics.csv, results/metrics_tables.md, figures/fig1-fig4
"""
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import ASSETS, FIGS, PERIODS, RESULTS, load_forecasts
from metrics import christoffersen, crps_from_quantiles, kupiec, pinball, quantile_column

METHODS = {   # file prefix -> (label, colour), in display order
    'aurora_released': ('Aurora, released sampler', '#2a78d6'),
    'aurora_corrected': ('Aurora, corrected sampler', '#eb6834'),
    'garch': ('GARCH(1,1)-t', '#1baf7a'),
    'hs': ('Historical simulation', '#eda100'),
    'gauss': ('Rolling Gaussian', '#e87ba4'),
}
INK, INK2, GRID, SURF = '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb'


def restrict(forecast, mask):
    """The forecast's per-day arrays restricted to the days in mask."""
    n = len(forecast['y'])
    return {k: (v[mask] if isinstance(v, np.ndarray) and v.shape[:1] == (n,) else v) for k, v in forecast.items()}


def score(forecast):
    y = forecast['y'].astype(float)
    q01, q05, q95, q99 = (quantile_column(forecast, t) for t in (0.01, 0.05, 0.95, 0.99))
    hit1, hit5 = y < q01, y < q05
    return dict(
        n=len(y),
        cov90=100 * np.mean((y >= q05) & (y <= q95)),
        cov98=100 * np.mean((y >= q01) & (y <= q99)),
        var1_hits=100 * hit1.mean(), var1_kupiec_p=kupiec(hit1, 0.01),
        var5_hits=100 * hit5.mean(), var5_kupiec_p=kupiec(hit5, 0.05),
        var5_indep_p=christoffersen(hit5),
        pin01=pinball(y, q01, 0.01), pin05=pinball(y, q05, 0.05),
        pin95=pinball(y, q95, 0.95), pin99=pinball(y, q99, 0.99),
        crps=crps_from_quantiles(y, forecast), width90=float(np.mean(q95 - q05)),
    )


def metrics_table():
    rows = []
    for period, (a, b) in PERIODS.items():
        for asset in ASSETS:
            for prefix, (label, _) in METHODS.items():
                d = load_forecasts(prefix, asset)
                mask = (d['target'] >= np.datetime64(a)) & (d['target'] <= np.datetime64(b))
                rows.append(dict(period=period, asset=asset, method=label, **score(restrict(d, mask))))
    return pd.DataFrame(rows)


def write_markdown(df, path):
    fmt = {'cov90': '{:.1f}', 'cov98': '{:.1f}', 'var1_hits': '{:.2f}', 'var1_kupiec_p': '{:.3f}',
           'var5_hits': '{:.2f}', 'var5_kupiec_p': '{:.3f}', 'var5_indep_p': '{:.3f}',
           'crps': '{:.4f}', 'pin01': '{:.4f}', 'width90': '{:.2f}'}
    heads = ['Asset', 'Method', 'n', 'Cov 90% (ideal 90)', 'Cov 98% (ideal 98)', 'VaR1 breach % (ideal 1)',
             'Kupiec p (1%)', 'VaR5 breach % (ideal 5)', 'Kupiec p (5%)', 'Indep. p (5%)', 'CRPS', 'Pinball 1%', 'Width 90%']
    lines = []
    for period in PERIODS:
        sub = df[df.period == period]
        lines += [f'\n### {period}\n', '| ' + ' | '.join(heads) + ' |', '|' + '---|' * len(heads)]
        for _, r in sub.iterrows():
            cells = [r['asset'], r['method'], str(r['n'])] + [fmt[c].format(r[c]) for c in fmt]
            lines.append('| ' + ' | '.join(cells) + ' |')
        avg = sub.groupby('method', sort=False)[['cov90', 'cov98', 'var1_hits', 'var5_hits', 'crps', 'width90']].mean()
        lines += [f'\n**Average over the four assets ({period})**\n',
                  '| Method | Cov 90% | Cov 98% | VaR1 breach % | VaR5 breach % | CRPS | Width 90% |',
                  '|---|---|---|---|---|---|---|']
        for method, r in avg.iterrows():
            lines.append(f'| {method} | {r.cov90:.1f} | {r.cov98:.1f} | {r.var1_hits:.2f} | {r.var5_hits:.2f} '
                         f'| {r.crps:.4f} | {r.width90:.2f} |')
    text = '\n'.join(lines) + '\n'
    with open(path, 'w', encoding='utf-8') as f:
        f.write(text)
    return text


def tidy(ax):
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)


def save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, name), dpi=160)
    plt.close(fig)


def plot_coverage(full):
    """Fig 1: coverage of the 90% and 98% intervals, per asset."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    x, width = np.arange(len(ASSETS)), 0.15
    for ax, col, nominal in ((axes[0], 'cov90', 90), (axes[1], 'cov98', 98)):
        for k, (label, colour) in enumerate(METHODS.values()):
            vals = [full[(full.asset == a) & (full.method == label)][col].values[0] for a in ASSETS]
            ax.bar(x + (k - 2) * width, vals, width * 0.88, color=colour, label=label)
        ax.axhline(nominal, color=INK, lw=1.2, ls='--')
        ax.text(len(ASSETS) - 0.45, nominal + 0.3, f'ideal {nominal}%', color=INK, fontsize=9, ha='right', va='bottom')
        ax.set_xticks(x, ASSETS)
        ax.set_ylim(0, 102)                     # from zero, so no method's bar is hidden below the axis
        ax.set_title(f'Share of next-day returns inside the {nominal}% interval', fontsize=10.5, loc='left')
        ax.grid(axis='y', color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        tidy(ax)
    axes[0].set_ylabel('coverage (%)')
    axes[1].legend(frameon=False, fontsize=8.5, loc='lower right')
    save(fig, 'fig1_coverage.png')


def plot_pit():
    """Fig 2: PIT histograms for the S&P 500 (flat = calibrated, U-shape = intervals too narrow)."""
    fig, axes = plt.subplots(1, 5, figsize=(13, 2.8), sharey=True)
    for ax, (prefix, (label, colour)) in zip(axes, METHODS.items()):
        pit = load_forecasts(prefix, 'SPX')['pit']
        ax.hist(pit, bins=20, range=(0, 1), color=colour, edgecolor=SURF, linewidth=1.5)
        ax.axhline(len(pit) / 20, color=INK, lw=1, ls='--')
        ax.set_title(label, fontsize=9.5, loc='left')
        ax.set_xticks([0, 0.5, 1])
        tidy(ax)
    axes[0].set_ylabel('days')
    fig.suptitle('S&P 500: where each realised return fell inside its forecast distribution (PIT); dashed = perfectly calibrated',
                 fontsize=10, x=0.01, ha='left', color=INK)
    save(fig, 'fig2_pit_spx.png')


def plot_covid_var():
    """Fig 3: the COVID crash, S&P 500 returns against three 1% VaR lines."""
    fig, ax = plt.subplots(figsize=(11, 4))
    ref = load_forecasts('garch', 'SPX')
    mask = (ref['target'] >= np.datetime64('2020-01-15')) & (ref['target'] <= np.datetime64('2020-06-30'))
    dates = ref['target'][mask].astype('datetime64[D]')
    ax.bar(dates, ref['y'][mask], color='#b9b8b2', width=0.8, label='daily return')
    for prefix in ('aurora_released', 'aurora_corrected', 'garch'):
        d = load_forecasts(prefix, 'SPX')
        var1 = quantile_column(d, 0.01)[mask]
        label, colour = METHODS[prefix]
        ax.plot(dates, var1, color=colour, lw=2, label=f'{label}: 1% VaR ({int(np.sum(d["y"][mask] < var1))} breaches)')
    ax.axhline(0, color=GRID, lw=0.8)
    ax.set_ylabel('return (%)')
    ax.set_title('S&P 500, Jan to Jun 2020: next-day 1% value-at-risk from each model', fontsize=10.5, loc='left')
    ax.legend(frameon=False, fontsize=8.5, loc='lower left')
    tidy(ax)
    save(fig, 'fig3_covid_var.png')


def plot_width_coverage(full):
    """Fig 4: average interval width against realised coverage."""
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for (label, colour), marker in zip(METHODS.values(), ['o', 's', '^', 'D', 'v']):   # shape as a second cue
        s = full[full.method == label]
        ax.scatter(s.width90, s.cov90, s=60, color=colour, marker=marker, edgecolor=SURF, linewidth=2, label=label, zorder=3)
        if label.startswith('Aurora'):          # label only the Aurora points; the baselines cluster near 90%
            for _, r in s.iterrows():
                ax.annotate(r.asset, (r.width90, r.cov90), textcoords='offset points', xytext=(6, -3), fontsize=8, color=INK2)
    ax.axhline(90, color=INK, lw=1, ls='--')
    ax.set_xlabel('average width of the 90% interval (percentage points)')
    ax.set_ylabel('coverage of the 90% interval (%)')
    ax.set_title('Narrow intervals buy their sharpness with missed coverage', fontsize=10.5, loc='left')
    ax.grid(color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=8, loc='lower right')
    tidy(ax)
    save(fig, 'fig4_width_vs_coverage.png')


def main():
    df = metrics_table()
    df.to_csv(os.path.join(RESULTS, 'metrics.csv'), index=False)
    print(write_markdown(df, os.path.join(RESULTS, 'metrics_tables.md')))
    plt.rcParams.update({'font.size': 10, 'axes.edgecolor': GRID, 'axes.labelcolor': INK2,
                         'xtick.color': INK2, 'ytick.color': INK2, 'axes.titlecolor': INK,
                         'figure.facecolor': SURF, 'axes.facecolor': SURF, 'savefig.facecolor': SURF})
    full = df[df.period == next(iter(PERIODS))]
    plot_coverage(full)
    plot_pit()
    plot_covid_var()
    plot_width_coverage(full)
    print('figures written to figures/')


if __name__ == '__main__':
    main()
