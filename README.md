# Is Aurora calibrated on daily returns?

A zero-shot calibration test of **Aurora** (Wu et al., *Aurora: Towards Universal Generative Multimodal Time Series Forecasting*, ICLR 2026, [arXiv:2509.22295](https://arxiv.org/abs/2509.22295)) on daily returns of the S&P 500, Apple, JPMorgan and Bitcoin, 2020 to 2026, against standard econometric baselines.

Aurora produces probabilistic forecasts by flow matching. A forecast range is only useful if it is honest: a 90% interval should contain about 90% of outcomes, and a 1% value-at-risk should be breached about 1% of the time. This repository tests that, in the tails, on data the model was not built for.

## Headline results

Average over the four assets, next-day forecasts, January 2020 to October 2026 (1,699 to 2,472 forecasts per asset):

| Method | 90% interval covers (ideal 90%) | 98% interval covers (ideal 98%) | 1% VaR breached (ideal 1%) | 5% VaR breached (ideal 5%) | CRPS (lower is better) |
|---|---|---|---|---|---|
| Aurora, released sampler | **39.0%** | 64.6% | **24.6%** | 31.5% | 1.274 |
| Aurora, corrected sampler | 67.0% | 86.2% | 9.6% | 15.9% | 1.186 |
| GARCH(1,1) with Student-t errors | 89.2% | 98.0% | 1.4% | 6.3% | **1.032** |
| Historical simulation | 89.7% | 97.4% | 1.3% | 5.1% | 1.054 |
| Rolling Gaussian | 92.1% | 96.5% | 2.0% | 4.4% | 1.070 |

- **Window after the paper's release** (October 2025 to October 2026), which cannot be in pretraining: released 37.8% coverage, corrected 64.7%, GARCH-t 90.1%. The result is not an artefact of training-data overlap.
- **COVID crash** (15 February to 30 April 2020): 90% coverage of 16% (released), 30% (corrected) and 80% (GARCH-t).
- **Formal tests:** Kupiec's test rejects Aurora's breach rate at 1% and 5% for every asset and both samplers (all p < 0.0001).
- **Cost:** about 0.22 s per Aurora forecast (1,000 samples, RTX 3060 GPU) against about 30 ms for a GARCH-t refit on a CPU.

![Coverage per asset](figures/fig1_coverage.png)

## The sampler

The released code trains the network to predict the end point of the flow, and its sampler moves each step by `(prediction - starting noise) * dt`. Summed over the 50 steps, the final sample is exactly the **average of the 50 end-point predictions** along the path. Early predictions, made from almost pure noise, sit near the conditional mean, so averaging them in shrinks the spread.

The **corrected sampler** replaces this with the Euler step on the velocity implied by an end-point prediction, `(prediction - x) / (1 - t)`, which is the update the paper describes. Nothing else changes, and both samplers use identical random noise.

| Check | Released | Corrected | Target |
|---|---|---|---|
| Gaussian toy problem, sample standard deviation | 0.48 | 0.98 | 1.00 |
| Seasonal series, 90% interval coverage | 47.9% | 77.6% | 90% |
| Seasonal series, accuracy (MASE) | 0.761 | 0.785 | |
| Daily returns, 90% interval coverage | 39.0% | 67.0% | 90% |

So most of the under-dispersion comes from the sampler, it is not specific to finance, and fixing it costs almost no accuracy. The remaining gap is a modelling question: a Gaussian source distribution, a training loss that drops very large errors, little finance in pretraining, and per-window normalization that ignores volatility clustering. The same sampler update appears in the released code of Sundial.

## Design

| Choice | Value |
|---|---|
| Data | daily adjusted closes from Yahoo Finance, 2005 to October 2026; log returns in percent |
| Forecast origins | every trading day from 2 January 2020 to 6 October 2026, rolling one day at a time |
| Aurora input | the last 528 returns (its default context, 11 patches of 48); univariate, no text; no fine-tuning |
| Aurora output | 1,000 sampled 48-day paths per forecast; day 1 is scored |
| Batch size | 1, because the model picks its image period from the batch-averaged spectrum |
| Baselines | rolling Gaussian and historical simulation on the same 528 returns; GARCH(1,1)-t refit daily on the last 1,000 returns |
| Metrics | 90% and 98% coverage; 1% and 5% VaR breach rates with Kupiec and Christoffersen tests; pinball loss at 1%, 5%, 95%, 99%; CRPS (twice the mean pinball loss over the 1% to 99% grid); interval width |

Sanity check first: on 50 noisy seasonal series, the kind of data Aurora is built for, its median forecast tracked the truth (correlation 0.94) and beat a seasonal-naive forecast (MASE 0.76 against 0.83), so the pipeline calls the model correctly.

## Figures

- `figures/fig1_coverage.png`: coverage of the 90% and 98% intervals, per asset and method.
- `figures/fig2_pit_spx.png`: PIT histograms for the S&P 500 (flat means calibrated; a U means intervals too narrow).
- `figures/fig3_covid_var.png`: S&P 500, January to June 2020, with each method's 1% VaR line. The S&P 500 breached Aurora's line 43 times (released) and 26 times (corrected), and GARCH-t's 4 times.
- `figures/fig4_width_vs_coverage.png`: interval width against coverage.
- `figures/fig5_endogenous_images.png`: the image Aurora builds from a series. On returns the FFT finds no real period, so the image becomes a barcode.

## Limitations

- **Zero-shot only.** This is the released checkpoint with no fine-tuning, and the paper does not claim to forecast stock returns. What it shows is that universal zero-shot probabilistic forecasting does not carry over to financial returns without work.
- **The corrected sampler is my change.** It is the standard Euler step for an end-point-predicting flow, but the model was trained alongside the released sampler, so both are reported.
- **Quantile noise.** A 1% quantile from 1,000 samples rests on about 10 samples. That noise is small next to breach rates of 10% to 25%.
- **Scope.** One horizon, four assets, one seed per forecast, no text input. Christoffersen's test finds breach clustering in several series, so conditional coverage is worse than the raw rates suggest.

## Reproduce

```
python -m venv .venv                      # Python 3.11
pip install -r requirements.txt           # torch with CUDA 12.1 from download.pytorch.org
python setup_aurora.py                    # fetches the authors' model code (commit a247760) into vendor/aurora
python 01_get_data.py                     # downloads prices from Yahoo Finance into data/
python diag_one_forecast.py               # optional: one forecast, both samplers
python diag_sanity_seasonal.py            # optional: the seasonal sanity check
python 02_aurora_forecasts.py both        # about 55 minutes on an RTX 3060; weights download from Hugging Face
python 03_baselines.py                    # about 4 minutes on a CPU
python 04_evaluate.py                     # metrics, tests and figures
```

Price data and per-forecast outputs are not included; the scripts regenerate them. Full per-asset tables are in `results/metrics_tables.md` and raw numbers in `results/metrics.csv`.

| File | What it does |
|---|---|
| `common.py` | shared settings: assets, windows, quantile grid, periods |
| `01_get_data.py` | downloads prices, builds log returns |
| `02_aurora_forecasts.py` | Aurora forecasts with both samplers |
| `03_baselines.py` | rolling Gaussian, historical simulation, GARCH-t |
| `04_evaluate.py` | metrics, VaR backtests, figures |
| `diag_*.py` | one-forecast check, seasonal sanity check, endogenous-image figure |
| `setup_aurora.py` | fetches the released Aurora code from the authors' repository |

## Credits

Aurora's model code and weights are by the paper's authors: [github.com/decisionintelligence/Aurora](https://github.com/decisionintelligence/Aurora) and [huggingface.co/DecisionIntelligence/Aurora](https://huggingface.co/DecisionIntelligence/Aurora). This repository contains only the evaluation code, which is MIT-licensed (see `LICENSE`).

Akash Deep, October 2026.
