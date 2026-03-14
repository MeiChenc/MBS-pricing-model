# mbs_logit.py

`mbs_logit.py` restores the Colab workflow into a runnable Python script for:
- OLS `logit(SMM)` prepayment regression
- Vasicek stochastic rate simulation
- Path-dependent MBS cash-flow pricing
- Baseline vs Improved model comparison

## Framework

The script follows this structure:

1. **Baseline**
   - Model C: `logit_smm ~ spread + spread^2 + spread^3 + log_UPB + duration`

2. **Improved model selection**
   - Train Model A/B/C on train split
   - Pick best model by out-of-sample test `RMSE(SMM)` (tie-break by `MAE(SMM)`)

3. **Out-of-sample / valuation impact comparison**
   - Compare Baseline vs Improved on:
     - `P_base`
     - `P(+50bp)`
     - `P(-50bp)`
     - `effective_duration`

## Data source

Default input is:
- `PrepayData.txt` (same dataset used in this project)

## Models

- **Model A**: `logit_smm ~ spread`
- **Model B**: `logit_smm ~ spread + duration`
- **Model C**: `logit_smm ~ spread + I(spread**2) + I(spread**3) + log_UPB + duration`

## Time split

For target security type (default `30yr TBA Eligible`):
- Train = first 80% by time
- Test = last 20% by time

Split is done on monthly cohort panel sorted by `period`.

## Usage

Run with defaults:

```bash
python3 mbs_logit.py
```

Example with explicit options:

```bash
python3 mbs_logit.py \
  --data-path PrepayData.txt \
  --target-type "30yr TBA Eligible" \
  --train-ratio 0.8 \
  --n-paths 300 \
  --term-months 360 \
  --seed 1 \
  --kappa 0.25 \
  --sigma 0.012 \
  --min-rows 40 \
  --min-train-rows 30 \
  --min-test-rows 8 \
  --output-dir outputs
```

## CLI arguments

- `--data-path` (default: `PrepayData.txt`)
- `--target-type` (default: `30yr TBA Eligible`)
- `--train-ratio` (default: `0.8`)
- `--n-paths` (default: `300`)
- `--term-months` (default: `360`)
- `--seed` (default: `1`)
- `--kappa` (default: `0.25`)
- `--sigma` (default: `0.012`)
- `--min-rows` (default: `40`)
- `--min-train-rows` (default: `30`)
- `--min-test-rows` (default: `8`)
- `--output-dir` (default: `outputs`)

## Outputs

The script writes:

- `outputs/oos_metrics_30yr_tba.csv`
  - Model A/B/C test metrics (`rmse_smm`, `mae_smm`)
- `outputs/valuation_compare_30yr_tba.csv`
  - Baseline vs Improved pricing results
- `outputs/baseline_vs_improved_delta.csv`
  - Improved minus Baseline deltas

It also prints three tables to terminal:
- OOS metrics
- Baseline vs Improved valuation impact
- Delta table

## Notes

- `log_UPB` uses `ln(Cohort_Current_UPB.clip(lower=1))` for numerical stability.
- Logistic transform is implemented in numerically stable form.
- Path-loop prediction uses direct coefficient evaluation (faster than per-step patsy `predict`).

## Troubleshooting

### 1) `Insufficient rows for split/modeling`
Lower thresholds, for example:

```bash
python3 mbs_logit.py --min-rows 30 --min-train-rows 20 --min-test-rows 6
```

### 2) Environment / package issues
Install dependencies from `requirements.txt` in a clean virtual environment.

### 3) Slow runtime
Start with fewer paths:

```bash
python3 mbs_logit.py --n-paths 30
```

Then increase to production value (e.g. 300+).

## Project Extension (Next Phase)

To extend this project, the next objective is to build and compare three prepayment-model families under the same pricing engine:

1. **OLS Logit model** (current baseline in `mbs_logit.py`)
2. **Cox-based hazard model**
3. **Neural Network (NN) prepayment model**

### Extension goal

Evaluate which model performs best for **MBS pricing quality**, not only predictive fit.

### Planned comparison dimensions

- Out-of-sample prepayment metrics:
  - `RMSE(SMM)`
  - `MAE(SMM)`
- Valuation impact metrics (same cohorts, same scenarios/paths):
  - `P_base`
  - `P(+50bp)`
  - `P(-50bp)`
  - `effective_duration`
- Stability / robustness:
  - sensitivity to small spread/rate shocks
  - consistency across security types and time windows

### Success criterion

Select the model that provides the best balance of:
- predictive accuracy on unseen data, and
- stable, economically sensible pricing behavior under rate shocks.
