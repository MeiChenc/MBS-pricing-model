from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


EPS = 1e-6


MODEL_FORMULAS = {
    "A": "logit_smm ~ spread",
    "B": "logit_smm ~ spread + duration",
    "C": "logit_smm ~ spread + I(spread**2) + I(spread**3) + log_UPB + duration",
}


def to_decimal_rate(x: float) -> float:
    x = float(x)
    return x / 100.0 if x > 1.0 else x


def to_pct_points(x: float) -> float:
    x = float(x)
    return x * 100.0 if x <= 1.0 else x


def sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    # Numerically stable logistic transform to avoid overflow warnings.
    x_arr = np.asarray(x, dtype=float)
    x_arr = np.clip(x_arr, -60.0, 60.0)
    out = 1.0 / (1.0 + np.exp(-x_arr))
    return float(out) if np.isscalar(x) else out


def mortgage_payment(balance: float, wac_dec: float, remaining_term_months: int) -> float:
    r = wac_dec / 12.0
    if remaining_term_months <= 0:
        return balance
    if abs(r) < 1e-12:
        return balance / remaining_term_months
    return balance * r / (1.0 - (1.0 + r) ** (-remaining_term_months))


def load_and_prepare(data_path: Path) -> pd.DataFrame:
    df = pd.read_csv(data_path, sep="|", low_memory=False)
    df.columns = (
        df.columns.str.strip().str.replace(" ", "_", regex=False).str.replace("-", "_", regex=False)
    )

    num_cols = [
        "WA_Net_Interest_Rate",
        "Cohort_Current_UPB",
        "Cohort_WA_Current_Interest_Rate",
        "Cohort_WA_Current_Remaining_Months_to_Maturity",
        "Cohort_WA_Current_Loan_Age",
        "SMM",
        "Cumulative_SMM",
        "CPR",
        "Cumulative_CPR",
    ]
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "Date" not in df.columns:
        raise ValueError("Missing required column: Date")

    df["date_parsed"] = pd.to_datetime(df["Date"].astype(str), format="%Y%m%d", errors="coerce")
    df["period"] = df["date_parsed"].dt.to_period("M")

    # Feature engineering aligned with notebook conventions.
    df["duration"] = pd.to_numeric(df["Cohort_WA_Current_Loan_Age"], errors="coerce")
    df["market_rate"] = pd.to_numeric(df["WA_Net_Interest_Rate"], errors="coerce")
    df["spread"] = pd.to_numeric(df["Cohort_WA_Current_Interest_Rate"], errors="coerce") - df["market_rate"]
    df["spread2"] = df["spread"] ** 2
    df["spread3"] = df["spread"] ** 3

    upb = pd.to_numeric(df["Cohort_Current_UPB"], errors="coerce")
    df["log_UPB"] = np.log(upb.clip(lower=1))

    return df


def build_monthly_panel(df: pd.DataFrame) -> pd.DataFrame:
    required_cols = [
        "period",
        "Type_of_Security",
        "date_parsed",
        "SMM",
        "Cumulative_SMM",
        "spread",
        "duration",
        "log_UPB",
        "market_rate",
        "Cohort_WA_Current_Interest_Rate",
        "Cohort_Current_UPB",
    ]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    work = df.dropna(subset=["period", "Type_of_Security", "date_parsed"]).copy()
    # Keep one row per month x security (latest row in month).
    monthly = (
        work.sort_values("date_parsed")
        .groupby(["period", "Type_of_Security"], as_index=False)
        .last()
        .sort_values(["period", "Type_of_Security"])
        .reset_index(drop=True)
    )

    if "Cumulative_SMM" in monthly.columns:
        monthly["SMM_monthly"] = pd.to_numeric(monthly["Cumulative_SMM"], errors="coerce")
    else:
        monthly["SMM_monthly"] = pd.to_numeric(monthly["SMM"], errors="coerce")

    monthly["smm_adj"] = monthly["SMM_monthly"].clip(lower=EPS, upper=1 - EPS)
    monthly["logit_smm"] = np.log(monthly["smm_adj"] / (1 - monthly["smm_adj"]))
    monthly["spread2"] = monthly["spread"] ** 2
    monthly["spread3"] = monthly["spread"] ** 3
    return monthly


def time_split(df: pd.DataFrame, train_ratio: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    if df.empty:
        raise ValueError("Target dataset is empty after filtering.")
    n = len(df)
    split_idx = int(np.floor(train_ratio * n))
    split_idx = max(1, min(split_idx, n - 1))
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    if train_df["period"].max() >= test_df["period"].min():
        raise ValueError("Time split leakage detected: train end is not earlier than test start.")
    return train_df, test_df


def fit_models(train_df: pd.DataFrame) -> dict[str, object]:
    models: dict[str, object] = {}
    for name, formula in MODEL_FORMULAS.items():
        models[name] = smf.ols(formula=formula, data=train_df).fit()
    return models


def evaluate_models(models: dict[str, object], test_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    y_true = pd.to_numeric(test_df["SMM_monthly"], errors="coerce")
    mask = np.isfinite(y_true.values)
    if not np.any(mask):
        raise ValueError("No valid SMM_monthly values in test set.")

    test_valid = test_df.loc[mask].copy()
    y_true_valid = y_true.loc[mask].values.astype(float)

    for name, model in models.items():
        pred_logit = model.predict(test_valid)
        pred_smm = np.clip(sigmoid(np.asarray(pred_logit, dtype=float)), EPS, 1 - EPS)
        rmse = float(np.sqrt(np.mean((pred_smm - y_true_valid) ** 2)))
        mae = float(np.mean(np.abs(pred_smm - y_true_valid)))
        rows.append(
            {
                "model": name,
                "n_train": int(model.nobs),
                "n_test": int(len(y_true_valid)),
                "rmse_smm": rmse,
                "mae_smm": mae,
            }
        )

    return pd.DataFrame(rows).sort_values(["rmse_smm", "mae_smm"], ascending=True).reset_index(drop=True)


def pick_improved(metrics_df: pd.DataFrame) -> str:
    best = metrics_df.sort_values(["rmse_smm", "mae_smm"], ascending=[True, True]).iloc[0]
    return str(best["model"])


def simulate_vasicek_paths(
    r0: float,
    n_months: int,
    n_paths: int,
    kappa: float = 0.25,
    theta: float | None = None,
    sigma: float = 0.012,
    theta_shift_bps: float = 0.0,
    seed: int = 1,
) -> np.ndarray:
    dt = 1.0 / 12.0
    theta_eff = r0 if theta is None else float(theta)
    theta_eff = theta_eff + theta_shift_bps / 10000.0

    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n_paths, n_months))
    rates = np.zeros((n_paths, n_months), dtype=float)
    rates[:, 0] = r0

    for t in range(1, n_months):
        rt = rates[:, t - 1]
        drift = kappa * (theta_eff - rt) * dt
        diffusion = sigma * np.sqrt(dt) * z[:, t]
        rates[:, t] = np.maximum(rt + drift + diffusion, 0.0)

    return rates


def shift_paths_bps(rates: np.ndarray, bps: float) -> np.ndarray:
    return np.maximum(rates + bps / 10000.0, 0.0)


def _coef(params: pd.Series, *names: str) -> float:
    for n in names:
        if n in params.index:
            return float(params[n])
    return 0.0


def predict_logit_fast(model, spread: float, duration: float, log_upb: float) -> float:
    """
    Fast linear predictor for formula-trained OLS models A/B/C.
    Avoids patsy transformation overhead inside path loops.
    """
    p = model.params
    intercept = _coef(p, "Intercept", "const")
    b_spread = _coef(p, "spread")
    b_duration = _coef(p, "duration")
    b_log_upb = _coef(p, "log_UPB")
    b_spread2 = _coef(p, "I(spread ** 2)", "I(spread**2)")
    b_spread3 = _coef(p, "I(spread ** 3)", "I(spread**3)")

    return (
        intercept
        + b_spread * spread
        + b_duration * duration
        + b_log_upb * log_upb
        + b_spread2 * (spread ** 2)
        + b_spread3 * (spread ** 3)
    )


def price_mbs_simulation_path(
    model,
    initial_state: pd.Series,
    rate_path_dec: np.ndarray,
    oas_bps: float = 0.0,
    term_months: int = 360,
    face: float = 100.0,
) -> tuple[float, np.ndarray]:
    # Extract state variables
    wac_dec = to_decimal_rate(initial_state["Cohort_WA_Current_Interest_Rate"])
    wac_pct = to_pct_points(initial_state["Cohort_WA_Current_Interest_Rate"])
    age = float(initial_state["duration"]) if np.isfinite(initial_state["duration"]) else 0.0

    balance = float(face)
    cashflows: list[float] = []
    pv = 0.0

    oas_dec = oas_bps / 10000.0
    discount_df = 1.0

    for t in range(1, term_months + 1):
        r_t = float(rate_path_dec[t - 1]) + oas_dec
        discount_df *= 1.0 / (1.0 + r_t / 12.0)

        age += 1.0
        mr_pct_t = to_pct_points(r_t)
        spread = wac_pct - mr_pct_t

        pred_logit = predict_logit_fast(
            model=model,
            spread=spread,
            duration=age,
            log_upb=np.log(max(balance, 1.0)),
        )
        smm = float(np.clip(sigmoid(pred_logit), 0.0, 0.6))

        remaining = term_months - t + 1
        pmt = mortgage_payment(balance, wac_dec, remaining)
        interest = balance * wac_dec / 12.0
        sched_principal = max(pmt - interest, 0.0)
        bal_after_sched = max(balance - sched_principal, 0.0)
        prepay_principal = smm * bal_after_sched

        total_principal = min(balance, sched_principal + prepay_principal)
        cf = interest + total_principal

        cashflows.append(cf)
        pv += cf * discount_df

        balance -= total_principal
        if balance <= 1e-6:
            break

    return pv, np.asarray(cashflows, dtype=float)


def run_paths_and_average(
    model,
    latest_cohort: pd.Series,
    rates: np.ndarray,
    term_months: int,
) -> tuple[float, np.ndarray]:
    prices = []
    cf_list = []
    for i in range(rates.shape[0]):
        p, cf = price_mbs_simulation_path(
            model=model,
            initial_state=latest_cohort,
            rate_path_dec=rates[i],
            term_months=term_months,
            oas_bps=0.0,
        )
        prices.append(p)
        cf_list.append(cf)

    max_len = max(len(cf) for cf in cf_list)
    mat = np.zeros((len(cf_list), max_len), dtype=float)
    for i, cf in enumerate(cf_list):
        mat[i, : len(cf)] = cf

    return float(np.mean(prices)), np.mean(mat, axis=0)


def valuation_summary(
    model,
    latest_cohort: pd.Series,
    rates_base: np.ndarray,
    term_months: int,
) -> dict[str, float]:
    rates_up = shift_paths_bps(rates_base, +50)
    rates_down = shift_paths_bps(rates_base, -50)

    p_base, _ = run_paths_and_average(model, latest_cohort, rates_base, term_months)
    p_up, _ = run_paths_and_average(model, latest_cohort, rates_up, term_months)
    p_down, _ = run_paths_and_average(model, latest_cohort, rates_down, term_months)

    dy = 50 / 10000.0
    eff_duration = (p_down - p_up) / (2.0 * p_base * dy)

    return {
        "P_base": p_base,
        "P_plus_50bp": p_up,
        "P_minus_50bp": p_down,
        "effective_duration": float(eff_duration),
    }


def run_pipeline(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = load_and_prepare(Path(args.data_path))
    monthly = build_monthly_panel(df)

    target = monthly[monthly["Type_of_Security"] == args.target_type].copy()
    target = target.sort_values("period").reset_index(drop=True)
    needed = ["SMM_monthly", "logit_smm", "spread", "duration", "log_UPB", "market_rate", "Cohort_WA_Current_Interest_Rate"]
    target = target.dropna(subset=needed)

    if len(target) < args.min_rows:
        raise ValueError(
            f"Insufficient rows for split/modeling for {args.target_type}: {len(target)} "
            f"(required >= {args.min_rows})"
        )

    train_df, test_df = time_split(target, args.train_ratio)
    if len(train_df) < args.min_train_rows or len(test_df) < args.min_test_rows:
        raise ValueError(
            "Split produced too few rows: "
            f"train={len(train_df)} (required >= {args.min_train_rows}), "
            f"test={len(test_df)} (required >= {args.min_test_rows}). "
            "Adjust --train-ratio or min row thresholds."
        )
    models = fit_models(train_df)
    oos_metrics_df = evaluate_models(models, test_df)

    baseline_name = "C"
    improved_name = pick_improved(oos_metrics_df)

    latest_cohort = target.iloc[-1]
    r0_dec = to_decimal_rate(latest_cohort["market_rate"])
    rates_base = simulate_vasicek_paths(
        r0=r0_dec,
        n_months=args.term_months,
        n_paths=args.n_paths,
        kappa=args.kappa,
        sigma=args.sigma,
        theta_shift_bps=0.0,
        seed=args.seed,
    )

    rows = []
    for label, model_name in [("Baseline", baseline_name), ("Improved", improved_name)]:
        val = valuation_summary(models[model_name], latest_cohort, rates_base, args.term_months)
        rows.append(
            {
                "role": label,
                "model": model_name,
                **val,
            }
        )

    valuation_compare_df = pd.DataFrame(rows)

    baseline = valuation_compare_df[valuation_compare_df["role"] == "Baseline"].iloc[0]
    improved = valuation_compare_df[valuation_compare_df["role"] == "Improved"].iloc[0]

    delta_df = pd.DataFrame(
        [
            {
                "metric": "rmse_smm",
                "improved_minus_baseline": float(
                    oos_metrics_df.loc[oos_metrics_df["model"] == improved_name, "rmse_smm"].iloc[0]
                    - oos_metrics_df.loc[oos_metrics_df["model"] == baseline_name, "rmse_smm"].iloc[0]
                ),
            },
            {
                "metric": "mae_smm",
                "improved_minus_baseline": float(
                    oos_metrics_df.loc[oos_metrics_df["model"] == improved_name, "mae_smm"].iloc[0]
                    - oos_metrics_df.loc[oos_metrics_df["model"] == baseline_name, "mae_smm"].iloc[0]
                ),
            },
            {
                "metric": "P_base",
                "improved_minus_baseline": float(improved["P_base"] - baseline["P_base"]),
            },
            {
                "metric": "P_plus_50bp",
                "improved_minus_baseline": float(improved["P_plus_50bp"] - baseline["P_plus_50bp"]),
            },
            {
                "metric": "P_minus_50bp",
                "improved_minus_baseline": float(improved["P_minus_50bp"] - baseline["P_minus_50bp"]),
            },
            {
                "metric": "effective_duration",
                "improved_minus_baseline": float(improved["effective_duration"] - baseline["effective_duration"]),
            },
        ]
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    oos_metrics_df.to_csv(out_dir / "oos_metrics_30yr_tba.csv", index=False)
    valuation_compare_df.to_csv(out_dir / "valuation_compare_30yr_tba.csv", index=False)
    delta_df.to_csv(out_dir / "baseline_vs_improved_delta.csv", index=False)

    return oos_metrics_df, valuation_compare_df, delta_df


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Restore OLS-logit + Vasicek + path-dependent MBS pricing workflow with baseline/improved comparison."
    )
    parser.add_argument("--data-path", type=str, default="PrepayData.txt")
    parser.add_argument("--target-type", type=str, default="30yr TBA Eligible")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--n-paths", type=int, default=300)
    parser.add_argument("--term-months", type=int, default=360)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--kappa", type=float, default=0.25)
    parser.add_argument("--sigma", type=float, default=0.012)
    parser.add_argument("--min-rows", type=int, default=40)
    parser.add_argument("--min-train-rows", type=int, default=30)
    parser.add_argument("--min-test-rows", type=int, default=8)
    parser.add_argument("--output-dir", type=str, default="outputs")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    oos_metrics_df, valuation_compare_df, delta_df = run_pipeline(args)

    print("\n=== Out-of-Sample Metrics (30yr TBA Eligible) ===")
    print(oos_metrics_df.to_string(index=False, float_format=lambda x: f"{x:,.6f}"))

    print("\n=== Baseline vs Improved Valuation Impact ===")
    print(valuation_compare_df.to_string(index=False, float_format=lambda x: f"{x:,.6f}"))

    print("\n=== Improved - Baseline Delta ===")
    print(delta_df.to_string(index=False, float_format=lambda x: f"{x:,.6f}"))


if __name__ == "__main__":
    main()
