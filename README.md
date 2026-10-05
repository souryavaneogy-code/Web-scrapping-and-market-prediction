# Web-scrapping-and-FDX-prediction

A seven-phase, data-driven pipeline that models a currency pair's return
process as a **Generalized Langevin Equation (GLE) with memory and
colored noise** — estimated entirely from data, with no equilibrium or
Markov assumption imposed anywhere.

---

## 1. The physical and mathematical picture

A currency pair's log-return `r(t)` is treated as one observed degree of
freedom of a much larger, open system: order flow, other instruments,
central-bank behavior, and news are all degrees of freedom that are
either unobserved or only partially observed. The **Mori–Zwanzig
projection** formalism gives an exact way to reduce such a system onto
the variables you *do* observe: project the full dynamics onto `r(t)`
(and whatever other observed variables and drives you supply), and every
projected-out degree of freedom reappears in the reduced equation as two
terms — a **memory kernel** and a **noise process**. The result is the GLE:

```
dr/dt = Omega·r(t) − ∫₀ᵗ K(t−τ) r(τ) dτ + Σⱼ Ωⱼ xⱼ(t) + η(t)
```

- `K(t−τ)` — the memory kernel: how strongly the return's own past
  continues to influence it.
- `xⱼ(t)` — observed exogenous drives (other instruments, news).
- `η(t)` — the noise left over after everything explainable is removed.

Discretized on a daily grid, this becomes a linear autoregression with
exogenous inputs:

```
r(t+1) = Σₖ Kₖ·r(t−k+1) + Σⱼ Ωⱼ·xⱼ(t) + η(t)
```

**What is *not* assumed, anywhere in this pipeline:**

- **No equilibrium.** The projection itself is a mathematical identity
  that holds for *any* dynamics, in or out of equilibrium. No
  fluctuation–dissipation relation is imposed between `K` and `η` — the
  two are estimated independently from data, and how far the system
  actually sits from time-reversal symmetry (the signature of
  equilibrium) is *measured*, not presumed, in Phase 5.
- **No Markov property.** The memory term makes `r(t)` non-Markovian by
  construction. The memory depth is chosen from data (not fixed), and
  whether it's deep enough is statistically tested, not assumed.
- **Stationarity is used; equilibrium is not.** Fixed-coefficient
  estimation only requires that `r(t)`'s statistics don't drift over
  time — a property a *driven, dissipative, non-equilibrium steady
  state* satisfies just as well as an equilibrium state does. This is
  checked directly (Phase 2), and parameter drift is tested later
  (Phase 7) rather than assumed away.

---

## 2. What each phase does

| Phase | File | Role |
|---|---|---|
| 1 | `phase1_config.py` | Configuration + shared math (imported by all others, never run directly) |
| 2 | `phase2_state_space.py` | Builds the observed state `[r(t), σ(t)]` |
| 3 | `phase3_features.py` | Builds the candidate drives (neighbors + news) |
| 4 | `phase4_varx_fit.py` | Estimates the memory kernel `K` and couplings `Ω` |
| 5 | `phase5_noise_model.py` | Characterizes the noise `η(t)` + irreversibility diagnostic |
| 6 | `phase6_simulation.py` | Monte Carlo forward simulation → the actual forecast |
| 7 | `phase7_evaluation.py` | Backtest + adequacy diagnostics |

### Phase 1 — `phase1_config.py`
Not run on its own. Holds every tunable setting and three pieces of
shared math used identically everywhere else:
- `choose_lag_order()` — picks the memory depth `p` by AIC on `r(t)` alone.
- `build_design_matrix()` — assembles the regression table, with
  `r_lag1 = r(t)` (the most recent known value), `r_lag2 = r(t−1)`, etc.
- `make_scaled_elasticnet_pipeline()` — the sparse estimator: an
  elastic-net (L1+L2) penalty on *standardized* features, so that raw
  measurement-unit differences (e.g. a news count of 5 vs. a return of
  0.005) can't bias which coefficients survive.

### Phase 2 — `phase2_state_space.py`
**Algorithm:** fetches the price series, computes `r(t) = ln(P(t)/P(t−1))`
and `σ(t)` (rolling realized volatility), then runs an **Augmented
Dickey-Fuller test** to confirm `r(t)` is stationary.
**Physically:** `r(t)` is used instead of raw price because price is a
non-stationary, integrated process — the log-difference is what
produces a series whose statistics don't depend on calendar time, which
is what fixed-coefficient estimation actually requires.
**Output:** `state_space.csv`

### Phase 3 — `phase3_features.py`
**Algorithm:** fetches every candidate neighbor instrument (as its own
return series) and every news topic (daily count + sentiment), and runs
an **Engle-Granger cointegration test** between the currency's price
level and each neighbor's price level, as a separate diagnostic of the
data only.
**Physically:** these are the drives `xⱼ(t)` that couple to the
currency in the GLE. Nothing here decides which drives actually matter
— that's Phase 4's job.
**Output:** `features.csv`, `cointegration_report.csv`

### Phase 4 — `phase4_varx_fit.py`
**Algorithm:** a single elastic-net regression simultaneously estimates
every memory-kernel coefficient `Kₖ` and every drive coupling `Ωⱼ`:
```
minimize_β  (1/2n)‖y − Xβ‖² + α·[ l1_ratio·‖β‖₁ + (1−l1_ratio)/2·‖β‖² ]
```
with `α` and `l1_ratio` chosen by time-series cross-validation.
**Physically:** the surviving nonzero `Ωⱼ` coefficients *are* the
estimated network of what actually couples to the currency; everything
shrunk to exactly zero is treated as absorbed into the noise term. This
is the concrete, numerical realization of Mori–Zwanzig projection.
**Important honesty point stated in the file:** from a single observed
trajectory, linear memory and linear noise autocorrelation aren't
separately identifiable without assuming a fluctuation-dissipation
relation (which this pipeline deliberately doesn't assume) — so `{Kₖ}`
represents *all* linear temporal dependence, not a "pure" memory kernel
isolated from noise.
**Output:** `varx_model.pkl`, `residuals.csv`, `network_edges.csv`

### Phase 5 — `phase5_noise_model.py`
Two distinct jobs:

**Part A — noise characterization.** Fits `η(t)` as a
**GARCH(1,1) process with Student-t innovations**:
```
η(t) = √h(t)·z(t),   z(t) ~ Student-t(ν)
h(t) = ω + α·η(t−1)² + β·h(t−1)
```
`α+β` near 1 means long memory in volatility (clustering); `ν` measures
tail weight (small `ν` = fat tails, a well-documented feature of
currency returns that a plain Gaussian would miss).

**Part B — irreversibility diagnostic.** Since no equilibrium is
assumed, no fluctuation-dissipation test is run. Instead, this measures
how far the estimated dynamics is from **time-reversal symmetry** —
a path-level Kullback-Leibler divergence `⟨Σ⟩ = D_KL(P ‖ P∘J)` between
the forward and time-reversed path measures of a window of days,
computed from the joint covariance of the currency and its coupled
drives, and tested against a null ensemble of *reversible* Gaussian
surrogates (built by symmetrizing the cross-spectrum). A significant
result means genuine **directed, non-reciprocal coupling** was detected
— evidence the system is out of equilibrium, not just noisy.
**Output:** `sigma_eta.csv`, `garch_model.pkl`, `irreversibility_report.csv`

### Phase 6 — `phase6_simulation.py`
**This is where your actual forecast comes from.** Simulates `r(t)`
forward for `FORECAST_HORIZON` days using Monte Carlo, because the
system is non-Markovian and the noise variance is state-dependent —
no closed-form multi-day-ahead formula exists.

Per simulated path, per day:
1. A noise draw comes from the GARCH model's own **multi-step simulation
   forecast** (so volatility clustering compounds correctly across days).
2. Unknown future drive values are filled in by **block bootstrap** —
   one random contiguous historical window per path, preserving the
   drives' own day-to-day persistence (not independent daily draws,
   which would destroy it).
3. `σ(t+h)` is **recomputed from the path's own simulated history**
   at every step (never bootstrapped), since it's a deterministic
   function of the currency's own recent returns, not an independent
   drive.

The ensemble mean across all paths is the point forecast; the 5th/95th
percentiles form the confidence band, which should visibly *widen* with
horizon.
**Output:** `simulation_forecast.csv`

### Phase 7 — `phase7_evaluation.py`
**Algorithm:** a **walk-forward (expanding-window) backtest** — refit
every `REFIT_EVERY` days, predict one day ahead, advance — compared
against three baselines (naive persistence, OLS without the memory
kernel, Random Forest), then three residual diagnostics:
- **Ljung-Box** — is the memory depth `p` deep enough?
- **ARCH-LM** — is the GARCH(1,1) noise model sufficient?
- **CUSUM** (on *volatility-studentized* residuals, so known clustering
  isn't confused with real drift) — are the coefficients stable over
  time, or has the system's regime shifted?

A significant CUSUM result isn't necessarily a failure — for a driven,
non-equilibrium system whose forcing genuinely changes over time (e.g. a
managed-float currency under shifting central-bank intervention), that
is an expected and informative outcome, not a bug.
**Output:** `evaluation_report.txt`

---

## 3. How to run it

### Requirements
```bash
pip install pandas numpy scikit-learn statsmodels arch yfinance scipy requests
```
All seven files must sit in the same folder — they import each other by
filename (`import phase1_config as cfg`).

### Configuration (edit `phase1_config.py` only)
```python
CURRENCY_PAIR = "USDINR=X"          # any yfinance-compatible FX ticker

NEIGHBORS = [                        # candidate coupled instruments
    {"name": "dxy", "ticker": "DX-Y.NYB"},
    {"name": "brent_oil", "ticker": "BZ=F"},
    # ... keep this to a handful of genuinely distinct instruments;
    # avoid near-duplicates (e.g. both WTI and Brent), which only
    # muddy Phase 4's sparse selection
]

NEWS_CATEGORIES = [                  # free-text news search topics
    "Federal Reserve interest rate",
    "geopolitical risk",
]

START_DATE = "2020-01-01"
END_DATE   = "2025-01-01"
```
Leave `NEIGHBORS` / `NEWS_CATEGORIES` empty to run in **self-contained
demo mode**: a synthetic test process with known ground truth (a real
decaying memory kernel and directed drive coupling) is generated
automatically, so you can validate the whole pipeline before wiring up
real data.

For live news, set an API key as an environment variable (never hardcode
it in the file):
```bash
export NEWS_API_KEY="your_key_here"
```
**Caveat already documented in the code:** most free news-API tiers only
serve the last ~30 days — they cannot backfill years of historical news
for training. Without a key, or beyond that window, synthetic news
features are substituted automatically so the pipeline still runs end
to end; the news signal for older dates just won't be real.

### Run order
Each phase reads files the previous one wrote — run them in sequence:
```bash
python3 phase2_state_space.py
python3 phase3_features.py
python3 phase4_varx_fit.py
python3 phase5_noise_model.py
python3 phase6_simulation.py
python3 phase7_evaluation.py
```
(`phase1_config.py` is imported, never executed directly.)

If you change the configuration and re-run, delete the old `.csv`/`.pkl`
files first, or simply re-run all six phases — each one overwrites its
own output.

### Troubleshooting
- **yfinance returns empty data** — the ticker is likely wrong or
  delisted; verify it on finance.yahoo.com first.
- **Any fetch fails** — the pipeline automatically falls back to
  synthetic data for that piece so it keeps running; check the console
  for `"synthetic"` tags to see what didn't actually fetch.

---

## 4. What you get, and how to read it

| File | What it contains |
|---|---|
| `simulation_forecast.csv` | **The forecast itself** — one row per future day, with `pred_mean`, `pred_p05`, `pred_p95` |
| `evaluation_report.txt` | Backtest accuracy vs. baselines + the three diagnostic tests |
| `network_edges.csv` | Which drives the model found genuinely coupled |
| `irreversibility_report.csv` | Whether directed, non-equilibrium coupling was statistically detected |
| `cointegration_report.csv` | Long-run price-level relationships (diagnostic only) |

**Before trusting the forecast, check `evaluation_report.txt` for:**
1. The candidate model beating naive persistence on out-of-sample MAE/RMSE.
2. Ljung-Box and ARCH-LM coming back clean (no significant leftover structure).
3. What the CUSUM result says — a significant result is informative
   about regime change, not automatically disqualifying.

A forecast that doesn't clear bar 1 is telling you something true and
useful: at this frequency, with this drive set, the market is close to
unpredictable from these inputs — a legitimate finding, not a failure of
the pipeline.
