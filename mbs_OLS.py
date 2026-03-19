from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


EPSILON = 1e-6
TARGET_SECURITY_TYPES = [
    "15yr TBA Eligible",
    "20yr TBA Eligible",
    "30yr TBA Eligible",
]
MODEL_FORMULAS = {
    "M0": "logit_smm ~ bs(spread, df=5, degree=3, include_intercept=False) + loan_age + log_UPB_z",
    "M1": (
        "logit_smm ~ bs(spread, df=5, degree=3, include_intercept=False) + loan_age + log_UPB_z + "
        "bs(spread, df=5, degree=3, include_intercept=False):loan_age"
    ),
}
DEFAULT_MODEL_NAME = "M1"


def load_prepay_data(data_path: Path) -> pd.DataFrame:
    df = pd.read_csv(data_path, sep="|", low_memory=False)
    df.columns = (
        df.columns.str.strip().str.replace(" ", "_", regex=False).str.replace("-", "_", regex=False)
    )

    numeric_cols = [
        "Cohort_Current_UPB",
        "Cohort_WA_Current_Interest_Rate",
        "Cohort_WA_Current_Loan_Age",
        "Cohort_WA_Current_Remaining_Months_to_Maturity",
        "SMM",
        "Cumulative_SMM",
        "CPR",
        "Cumulative_CPR",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "Date" not in df.columns:
        raise ValueError("Missing required column: Date")

    df["date_parsed"] = pd.to_datetime(df["Date"].astype(str), format="%Y%m%d", errors="coerce")
    df["period"] = df["date_parsed"].dt.to_period("M")
    return df


def download_market_rates(df: pd.DataFrame) -> pd.DataFrame:
    try:
        from pandas_datareader import data as web
    except ImportError as exc:
        raise ImportError(
            "pandas_datareader is required to download FRED mortgage rates. "
            "Install dependencies from requirements.txt before running mbs_OLS.py."
        ) from exc

    start_year = int(df["date_parsed"].min().year) if df["date_parsed"].notna().any() else 2010
    end_year = int(df["date_parsed"].max().year) if df["date_parsed"].notna().any() else 2025

    mr30 = web.DataReader("MORTGAGE30US", "fred", f"{start_year}-01-01", f"{end_year}-12-31").rename(
        columns={"MORTGAGE30US": "mr30"}
    )
    mr15 = web.DataReader("MORTGAGE15US", "fred", f"{start_year}-01-01", f"{end_year}-12-31").rename(
        columns={"MORTGAGE15US": "mr15"}
    )

    market_monthly = mr30.join(mr15, how="inner").resample("ME").mean()
    market_monthly.index = market_monthly.index.to_period("M")
    market_monthly["mr20"] = market_monthly["mr15"] + (1.0 / 3.0) * (
        market_monthly["mr30"] - market_monthly["mr15"]
    )
    return market_monthly[["mr15", "mr20", "mr30"]]


def assign_market_rate(row: pd.Series) -> float:
    sec = str(row["Type_of_Security"]).lower()
    if "15" in sec:
        return row["mr15"]
    if "20" in sec:
        return row["mr20"]
    return row["mr30"]


def build_observation_panel(df: pd.DataFrame, market_monthly: pd.DataFrame) -> pd.DataFrame:
    panel = df.merge(market_monthly, left_on="period", right_index=True, how="left")
    panel["market_rate"] = panel.apply(assign_market_rate, axis=1)
    panel["loan_age"] = pd.to_numeric(panel["Cohort_WA_Current_Loan_Age"], errors="coerce")
    panel["remaining_months"] = pd.to_numeric(
        panel["Cohort_WA_Current_Remaining_Months_to_Maturity"], errors="coerce"
    )
    panel["coupon"] = pd.to_numeric(panel["Cohort_WA_Current_Interest_Rate"], errors="coerce")
    panel["spread"] = panel["coupon"] - pd.to_numeric(panel["market_rate"], errors="coerce")
    panel["log_UPB"] = np.log(pd.to_numeric(panel["Cohort_Current_UPB"], errors="coerce") + 1.0)
    panel["SMM_monthly"] = pd.to_numeric(panel["SMM"], errors="coerce")
    panel["smm_adj"] = panel["SMM_monthly"].clip(lower=EPSILON, upper=1 - EPSILON)
    panel["logit_smm"] = np.log(panel["smm_adj"] / (1.0 - panel["smm_adj"]))
    panel["actual_cpr_pct"] = 100.0 * (1.0 - (1.0 - panel["SMM_monthly"]) ** 12.0)

    panel = panel[panel["Type_of_Security"].isin(TARGET_SECURITY_TYPES)].copy()
    panel = panel.dropna(
        subset=[
            "period",
            "Type_of_Security",
            "market_rate",
            "coupon",
            "spread",
            "loan_age",
            "remaining_months",
            "Cohort_Current_UPB",
            "SMM_monthly",
            "logit_smm",
            "log_UPB",
        ]
    ).reset_index(drop=True)

    panel["log_UPB_z"] = (panel["log_UPB"] - panel["log_UPB"].mean()) / panel["log_UPB"].std()
    panel["obs_id"] = np.arange(len(panel), dtype=int)
    return panel


def assign_time_splits(
    panel: pd.DataFrame, train_ratio: float, val_ratio: float, test_ratio: float
) -> pd.DataFrame:
    ratio_sum = train_ratio + val_ratio + test_ratio
    if abs(ratio_sum - 1.0) > 1e-9:
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")

    periods = sorted(panel["period"].dropna().unique())
    n_periods = len(periods)
    if n_periods < 3:
        raise ValueError("Need at least 3 unique periods for train/validation/test split.")

    train_end = max(1, int(np.floor(n_periods * train_ratio)))
    val_end = max(train_end + 1, int(np.floor(n_periods * (train_ratio + val_ratio))))
    val_end = min(val_end, n_periods - 1)

    train_periods = set(periods[:train_end])
    val_periods = set(periods[train_end:val_end])
    test_periods = set(periods[val_end:])

    if not val_periods or not test_periods:
        raise ValueError("Time split produced an empty validation or test period block.")

    panel = panel.copy()
    panel["split"] = np.where(
        panel["period"].isin(train_periods),
        "train",
        np.where(panel["period"].isin(val_periods), "validation", "test"),
    )
    return panel


def fit_security_models(
    panel: pd.DataFrame,
) -> tuple[dict[str, dict[str, object]], pd.DataFrame, dict[str, str], dict[str, dict[str, float]]]:
    fitted_models: dict[str, dict[str, object]] = {}
    summary_rows: list[dict[str, float | int | str]] = []
    summary_text: dict[str, str] = {}
    train_support: dict[str, dict[str, float]] = {}

    train_panel = panel[panel["split"] == "train"].copy()

    for sec_type in TARGET_SECURITY_TYPES:
        subset = train_panel[train_panel["Type_of_Security"] == sec_type].copy()
        subset = subset.dropna(subset=["logit_smm", "spread", "loan_age", "log_UPB_z"])
        if len(subset) < 50:
            continue

        train_support[sec_type] = {
            "spread_min": float(subset["spread"].min()),
            "spread_max": float(subset["spread"].max()),
        }
        fitted_models[sec_type] = {}
        for model_name, formula in MODEL_FORMULAS.items():
            model = smf.ols(formula=formula, data=subset).fit()
            fitted_models[sec_type][model_name] = model
            summary_rows.append(
                {
                    "Type_of_Security": sec_type,
                    "model_name": model_name,
                    "formula": formula,
                    "nobs": int(model.nobs),
                    "rsquared": float(model.rsquared),
                    "adj_rsquared": float(model.rsquared_adj),
                    "aic": float(model.aic),
                    "bic": float(model.bic),
                }
            )
            summary_text[f"{model_name}__{sec_type}"] = model.summary().as_text()

    return fitted_models, pd.DataFrame(summary_rows), summary_text, train_support


def clip_spread_to_train_support(subset: pd.DataFrame, support: dict[str, float]) -> pd.DataFrame:
    clipped = subset.copy()
    clipped["spread"] = clipped["spread"].clip(lower=support["spread_min"], upper=support["spread_max"])
    return clipped


def choose_bin_count(n_obs: int, target_bin_size: int = 24, min_bins: int = 4, max_bins: int = 12) -> int:
    if n_obs <= 0:
        return min_bins
    return max(min_bins, min(max_bins, n_obs // target_bin_size))


def build_prediction_panel(
    panel: pd.DataFrame,
    fitted_models: dict[str, dict[str, object]],
    train_support: dict[str, dict[str, float]],
    model_name: str = DEFAULT_MODEL_NAME,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for sec_type in TARGET_SECURITY_TYPES:
        model = fitted_models.get(sec_type, {}).get(model_name)
        if model is None:
            continue
        support = train_support.get(sec_type)
        if support is None:
            continue

        subset = panel[panel["Type_of_Security"] == sec_type].copy()
        if subset.empty:
            continue

        pred_data = clip_spread_to_train_support(subset[["spread", "log_UPB_z", "loan_age"]], support)
        pred_logit = model.predict(pred_data)
        subset["pred_logit"] = pred_logit
        subset["pred_smm"] = 1.0 / (1.0 + np.exp(-pred_logit))
        subset["pred_cpr_pct"] = 100.0 * (1.0 - (1.0 - subset["pred_smm"]) ** 12.0)
        subset["actual_smm"] = subset["SMM_monthly"]
        subset["actual_cpr_pct"] = 100.0 * (1.0 - (1.0 - subset["actual_smm"]) ** 12.0)
        subset["model_name"] = model_name
        frames.append(subset)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def build_oos_predictions(prediction_panel: pd.DataFrame) -> pd.DataFrame:
    oos = prediction_panel[prediction_panel["split"].isin(["validation", "test"])].copy()
    return oos[
        [
            "Type_of_Security",
            "period",
            "spread",
            "loan_age",
            "log_UPB_z",
            "actual_smm",
            "pred_smm",
            "actual_cpr_pct",
            "pred_cpr_pct",
            "split",
            "Cohort_Current_UPB",
        ]
    ]


def annual_rate_to_decimal(rate: float) -> float:
    rate = float(rate)
    return rate / 100.0 if abs(rate) > 1.0 else rate


def mortgage_payment(balance: float, monthly_rate: float, remaining_months: int) -> float:
    if remaining_months <= 0:
        return balance
    if abs(monthly_rate) < 1e-12:
        return balance / remaining_months
    return balance * monthly_rate / (1.0 - (1.0 + monthly_rate) ** (-remaining_months))


def build_calibration_dataset(
    prediction_panel: pd.DataFrame,
    n_bins: int,
    model_name: str = DEFAULT_MODEL_NAME,
) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []

    for sec_type in TARGET_SECURITY_TYPES:
        subset = prediction_panel[
            (prediction_panel["Type_of_Security"] == sec_type)
            & (prediction_panel["split"].isin(["validation", "test"]))
        ].copy()
        if subset.empty:
            continue

        n_bins_eff = min(n_bins, max(2, subset["spread"].nunique()))
        subset["spread_bin"] = pd.qcut(subset["spread"], q=n_bins_eff, duplicates="drop")

        grouped = (
            subset.groupby("spread_bin", observed=False)
            .agg(
                mean_spread=("spread", "mean"),
                actual_cpr_mean=("actual_cpr_pct", "mean"),
                predicted_cpr_mean=("pred_cpr_pct", "mean"),
                actual_cpr_std=("actual_cpr_pct", "std"),
                n_obs=("spread", "size"),
            )
            .dropna(subset=["mean_spread", "actual_cpr_mean", "predicted_cpr_mean"])
            .reset_index()
            .sort_values("mean_spread")
        )

        grouped["actual_cpr_se"] = grouped["actual_cpr_std"] / np.sqrt(grouped["n_obs"].clip(lower=1))
        grouped["actual_cpr_ci95"] = 1.96 * grouped["actual_cpr_se"].fillna(0.0)

        for _, row in grouped.iterrows():
            rows.append(
                {
                    "Type_of_Security": sec_type,
                    "model_name": model_name,
                    "spread_bin": str(row["spread_bin"]),
                    "mean_spread": float(row["mean_spread"]),
                    "actual_cpr_mean": float(row["actual_cpr_mean"]),
                    "predicted_cpr_mean": float(row["predicted_cpr_mean"]),
                    "actual_cpr_std": float(row["actual_cpr_std"]) if pd.notna(row["actual_cpr_std"]) else np.nan,
                    "actual_cpr_se": float(row["actual_cpr_se"]) if pd.notna(row["actual_cpr_se"]) else np.nan,
                    "actual_cpr_ci95": float(row["actual_cpr_ci95"]),
                    "n_obs": int(row["n_obs"]),
                }
            )

    return pd.DataFrame(rows)


def predict_projected_smm(
    model,
    train_support: dict[str, float],
    spread: float,
    loan_age: float,
    log_upb_z: float,
) -> float:
    pred_df = pd.DataFrame(
        {
            "spread": [float(np.clip(spread, train_support["spread_min"], train_support["spread_max"]))],
            "loan_age": [float(loan_age)],
            "log_UPB_z": [float(log_upb_z)],
        }
    )
    pred_logit = float(model.predict(pred_df).iloc[0])
    pred_smm = 1.0 / (1.0 + np.exp(-pred_logit))
    return float(np.clip(pred_smm, EPSILON, 1.0 - EPSILON))


def project_ols_smm_path_for_observation(
    obs: pd.Series,
    model,
    train_support: dict[str, float],
    scenario_market_rate: float,
    log_upb_mean: float,
    log_upb_std: float,
    max_projection_months: int,
) -> pd.DataFrame:
    coupon_annual = annual_rate_to_decimal(obs["coupon"])
    scenario_annual = annual_rate_to_decimal(scenario_market_rate)
    balance = float(obs["Cohort_Current_UPB"])
    initial_balance = balance
    start_loan_age = float(obs["loan_age"])
    start_remaining = int(max(round(float(obs["remaining_months"])), 0))
    rows: list[dict[str, float | int | str]] = []

    for h in range(1, max_projection_months + 1):
        remaining_h = start_remaining - (h - 1)
        if remaining_h <= 0 or balance <= 1e-8:
            break

        projected_loan_age = start_loan_age + (h - 1)
        projected_spread = float(obs["coupon"]) - float(scenario_market_rate)
        projected_log_upb = np.log(balance + 1.0)
        projected_log_upb_z = (projected_log_upb - log_upb_mean) / log_upb_std
        projected_smm = predict_projected_smm(
            model=model,
            train_support=train_support,
            spread=projected_spread,
            loan_age=projected_loan_age,
            log_upb_z=projected_log_upb_z,
        )

        monthly_coupon_rate = coupon_annual / 12.0
        payment = mortgage_payment(balance, monthly_coupon_rate, remaining_h)
        sched_interest = balance * monthly_coupon_rate
        sched_principal = max(payment - sched_interest, 0.0)
        bal_after_sched = max(balance - sched_principal, 0.0)
        prepay_principal = projected_smm * bal_after_sched
        total_principal = min(balance, sched_principal + prepay_principal)
        total_cf = sched_interest + total_principal
        balance = balance - total_principal

        rows.append(
            {
                "month": h,
                "scenario_market_rate": scenario_market_rate,
                "projected_spread": projected_spread,
                "projected_loan_age": projected_loan_age,
                "projected_remaining_months": remaining_h,
                "projected_smm": projected_smm,
                "scheduled_interest": sched_interest,
                "scheduled_principal": sched_principal,
                "prepay_principal": prepay_principal,
                "total_principal": total_principal,
                "total_cashflow": total_cf,
                "ending_balance": max(balance, 0.0),
                "initial_balance": initial_balance,
                "discount_rate_annual": scenario_annual,
            }
        )

        if balance <= 1e-8:
            break

    return pd.DataFrame(rows)


def discount_cashflows(cashflow_df: pd.DataFrame) -> tuple[float, float]:
    if cashflow_df.empty:
        return 0.0, 0.0
    scenario_rate_m = float(cashflow_df["discount_rate_annual"].iloc[0]) / 12.0
    months = cashflow_df["month"].to_numpy(dtype=float)
    dfs = 1.0 / ((1.0 + scenario_rate_m) ** months)
    pv = float(np.sum(cashflow_df["total_cashflow"].to_numpy(dtype=float) * dfs))
    price_per_100 = 100.0 * pv / float(cashflow_df["initial_balance"].iloc[0])
    return pv, price_per_100


def compute_effective_duration(price_base: float, price_down: float, price_up: float, shock_bp: float) -> tuple[float, float]:
    shock_decimal = shock_bp / 10000.0
    if price_base <= 0 or shock_decimal <= 0:
        return np.nan, np.nan
    duration = (price_down - price_up) / (2.0 * price_base * shock_decimal)
    convexity = (price_down + price_up - 2.0 * price_base) / (price_base * shock_decimal**2)
    return float(duration), float(convexity)


def build_pricing_results(
    prediction_panel: pd.DataFrame,
    fitted_models: dict[str, dict[str, object]],
    train_support: dict[str, dict[str, float]],
    pricing_split: str,
    shock_bp: float,
    max_projection_months: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if pricing_split == "validation":
        pricing_splits = ["validation"]
    elif pricing_split == "both":
        pricing_splits = ["validation", "test"]
    else:
        pricing_splits = ["test"]

    pricing_panel = prediction_panel[prediction_panel["split"].isin(pricing_splits)].copy()
    result_rows: list[dict[str, float | int | str]] = []
    all_cf_frames: list[pd.DataFrame] = []

    for sec_type in TARGET_SECURITY_TYPES:
        model = fitted_models.get(sec_type, {}).get(DEFAULT_MODEL_NAME)
        support = train_support.get(sec_type)
        if model is None or support is None:
            continue

        sec_df = pricing_panel[pricing_panel["Type_of_Security"] == sec_type].sort_values(["period", "obs_id"]).copy()
        if sec_df.empty:
            continue

        for _, obs in sec_df.iterrows():
            base_rate = float(obs["market_rate"])
            down_rate = base_rate - shock_bp / 100.0
            up_rate = base_rate + shock_bp / 100.0

            scenario_paths: dict[str, pd.DataFrame] = {}
            scenario_prices: dict[str, tuple[float, float]] = {}
            for scenario_name, scenario_rate in [("base", base_rate), ("down", down_rate), ("up", up_rate)]:
                cf_df = project_ols_smm_path_for_observation(
                    obs=obs,
                    model=model,
                    train_support=support,
                    scenario_market_rate=scenario_rate,
                    log_upb_mean=float(prediction_panel["log_UPB"].mean()),
                    log_upb_std=float(prediction_panel["log_UPB"].std()),
                    max_projection_months=max_projection_months,
                )
                cf_df["Type_of_Security"] = sec_type
                cf_df["period"] = str(obs["period"])
                cf_df["split"] = obs["split"]
                cf_df["obs_id"] = int(obs["obs_id"])
                cf_df["scenario"] = scenario_name
                scenario_paths[scenario_name] = cf_df
                scenario_prices[scenario_name] = discount_cashflows(cf_df)
                all_cf_frames.append(cf_df)

            price_base = scenario_prices["base"][1]
            price_down = scenario_prices["down"][1]
            price_up = scenario_prices["up"][1]
            effective_duration, effective_convexity = compute_effective_duration(price_base, price_down, price_up, shock_bp)

            base_cf = scenario_paths["base"]
            total_principal = float(base_cf["total_principal"].sum()) if not base_cf.empty else 0.0
            wal_months = (
                float(np.sum(base_cf["month"] * base_cf["total_principal"]) / total_principal)
                if total_principal > 0
                else np.nan
            )
            result_rows.append(
                {
                    "Type_of_Security": sec_type,
                    "period": str(obs["period"]),
                    "split": obs["split"],
                    "obs_id": int(obs["obs_id"]),
                    "Cohort_Current_UPB": float(obs["Cohort_Current_UPB"]),
                    "coupon": float(obs["coupon"]),
                    "current_market_rate": float(obs["market_rate"]),
                    "spread": float(obs["spread"]),
                    "loan_age": float(obs["loan_age"]),
                    "remaining_months": float(obs["remaining_months"]),
                    "pv_dollar_base": float(scenario_prices["base"][0]),
                    "price_per_100_base": float(price_base),
                    "pv_dollar_down": float(scenario_prices["down"][0]),
                    "price_per_100_down": float(price_down),
                    "pv_dollar_up": float(scenario_prices["up"][0]),
                    "price_per_100_up": float(price_up),
                    "effective_duration": effective_duration,
                    "effective_convexity": effective_convexity,
                    "total_projected_interest": float(base_cf["scheduled_interest"].sum()) if not base_cf.empty else 0.0,
                    "total_projected_principal": total_principal,
                    "wal_months": wal_months,
                    "wal_years": wal_months / 12.0 if np.isfinite(wal_months) else np.nan,
                    "final_projected_balance": float(base_cf["ending_balance"].iloc[-1]) if not base_cf.empty else float(obs["Cohort_Current_UPB"]),
                }
            )

    pricing_results = pd.DataFrame(result_rows)
    all_cashflows = pd.concat(all_cf_frames, ignore_index=True) if all_cf_frames else pd.DataFrame()
    avg_cf = (
        all_cashflows[all_cashflows["scenario"] == "base"]
        .groupby(["Type_of_Security", "month"], as_index=False)["total_cashflow"]
        .mean()
        .rename(columns={"total_cashflow": "avg_total_cashflow"})
        if not all_cashflows.empty
        else pd.DataFrame()
    )
    return pricing_results, all_cashflows, avg_cf, all_cashflows


def get_spread_grid(subset: pd.DataFrame, num_points: int = 80) -> np.ndarray:
    lo = float(subset["spread"].quantile(0.01))
    hi = float(subset["spread"].quantile(0.99))
    if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi:
        lo = float(subset["spread"].quantile(0.05))
        hi = float(subset["spread"].quantile(0.95))
    if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi:
        lo = float(subset["spread"].min())
        hi = float(subset["spread"].max())
    return np.linspace(lo, hi, num_points)


def get_spread_grid_from_quantiles(subset: pd.DataFrame, q_low: float, q_high: float, num_points: int = 80) -> np.ndarray:
    lo = float(subset["spread"].quantile(q_low))
    hi = float(subset["spread"].quantile(q_high))
    if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi:
        return get_spread_grid(subset, num_points=num_points)
    return np.linspace(lo, hi, num_points)


def summarize_counterfactual_predictions(
    subset: pd.DataFrame,
    model,
    spread_grid: np.ndarray,
    curve_family: str,
    slice_name: str,
    model_name: str,
) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []

    for spread_value in spread_grid:
        pred_data = subset[["log_UPB_z", "loan_age"]].copy()
        pred_data["spread"] = float(spread_value)
        pred_logit = model.predict(pred_data)
        pred_smm = 1.0 / (1.0 + np.exp(-pred_logit))
        pred_cpr_pct = 100.0 * (1.0 - (1.0 - pred_smm) ** 12.0)

        rows.append(
            {
                "spread": float(spread_value),
                "model_name": model_name,
                "curve_family": curve_family,
                "slice_name": slice_name,
                "predicted_cpr_mean": float(np.mean(pred_cpr_pct)),
                "predicted_cpr_p25": float(np.percentile(pred_cpr_pct, 25)),
                "predicted_cpr_p75": float(np.percentile(pred_cpr_pct, 75)),
                "n_obs": int(len(subset)),
            }
        )

    return pd.DataFrame(rows)


def assign_loan_age_slices(subset: pd.DataFrame) -> pd.Series:
    q1 = subset["loan_age"].quantile(1.0 / 3.0)
    q2 = subset["loan_age"].quantile(2.0 / 3.0)
    labels = np.where(
        subset["loan_age"] <= q1,
        "low_loan_age",
        np.where(subset["loan_age"] <= q2, "median_loan_age", "high_loan_age"),
    )
    return pd.Series(labels, index=subset.index, dtype="object")


def build_theoretical_curves(
    panel: pd.DataFrame,
    fitted_models: dict[str, dict[str, object]],
    train_support: dict[str, dict[str, float]],
    spread_q_low: float,
    spread_q_high: float,
    model_name: str = DEFAULT_MODEL_NAME,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    overall_frames: list[pd.DataFrame] = []
    slice_frames: list[pd.DataFrame] = []

    for sec_type in TARGET_SECURITY_TYPES:
        model = fitted_models.get(sec_type, {}).get(model_name)
        if model is None:
            continue
        support = train_support.get(sec_type)
        if support is None:
            continue

        subset = panel[
            (panel["Type_of_Security"] == sec_type) & (panel["split"].isin(["validation", "test"]))
        ].copy()
        if subset.empty:
            continue

        spread_grid = get_spread_grid_from_quantiles(subset, spread_q_low, spread_q_high)
        spread_grid = np.clip(spread_grid, support["spread_min"], support["spread_max"])
        overall_df = summarize_counterfactual_predictions(subset, model, spread_grid, "overall", "all", model_name)
        overall_df.insert(0, "Type_of_Security", sec_type)
        overall_frames.append(overall_df)

        subset["loan_age_slice"] = assign_loan_age_slices(subset)
        for slice_name, slice_df in subset.groupby("loan_age_slice"):
            if slice_df.empty:
                continue
            slice_curve = summarize_counterfactual_predictions(
                slice_df,
                model,
                spread_grid,
                "loan_age_slice",
                str(slice_name),
                model_name,
            )
            slice_curve.insert(0, "Type_of_Security", sec_type)
            slice_frames.append(slice_curve)

    overall = pd.concat(overall_frames, ignore_index=True) if overall_frames else pd.DataFrame()
    sliced = pd.concat(slice_frames, ignore_index=True) if slice_frames else pd.DataFrame()
    return overall, sliced


def build_slope_diagnostics(theoretical_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for sec_type in theoretical_df["Type_of_Security"].drop_duplicates():
        sec_df = (
            theoretical_df[theoretical_df["Type_of_Security"] == sec_type]
            .groupby("spread", as_index=False)["predicted_cpr_mean"]
            .mean()
            .sort_values("spread")
            .copy()
        )
        if len(sec_df) < 2:
            continue
        slopes = np.gradient(sec_df["predicted_cpr_mean"].to_numpy(), sec_df["spread"].to_numpy())
        sec_df["dCPR_dspread"] = slopes
        sec_df["Type_of_Security"] = sec_type
        rows.extend(sec_df[["Type_of_Security", "spread", "predicted_cpr_mean", "dCPR_dspread"]].to_dict("records"))
    return pd.DataFrame(rows)


def build_oos_metrics(
    oos_predictions: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []

    def metric_row(df: pd.DataFrame, level: str, security_type: str) -> dict[str, float | int | str]:
        err = df["pred_smm"] - df["actual_smm"]
        w = pd.to_numeric(df["Cohort_Current_UPB"], errors="coerce").fillna(0.0)
        w = np.where(w > 0, w, 0.0)
        w_sum = float(np.sum(w))
        if w_sum <= 0:
            w = np.ones(len(df), dtype=float)
            w_sum = float(np.sum(w))
        return {
            "level": level,
            "Type_of_Security": security_type,
            "nobs": int(len(df)),
            "rmse": float(np.sqrt(np.mean(err**2))),
            "mae": float(np.mean(np.abs(err))),
            "upb_weighted_rmse": float(np.sqrt(np.sum(w * (err**2)) / w_sum)),
            "upb_weighted_mae": float(np.sum(w * np.abs(err)) / w_sum),
        }

    if not oos_predictions.empty:
        rows.append(metric_row(oos_predictions, "overall", "ALL"))
    for sec_type in TARGET_SECURITY_TYPES:
        subset = oos_predictions[oos_predictions["Type_of_Security"] == sec_type]
        if subset.empty:
            continue
        rows.append(metric_row(subset, "security_type", sec_type))

    return pd.DataFrame(rows)


def summarize_pricing_results(pricing_results: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | str | int]] = []

    def summary_row(df: pd.DataFrame, level: str, security_type: str) -> dict[str, float | str | int]:
        return {
            "level": level,
            "Type_of_Security": security_type,
            "nobs": int(len(df)),
            "mean_price_per_100_base": float(df["price_per_100_base"].mean()),
            "median_price_per_100_base": float(df["price_per_100_base"].median()),
            "std_price_per_100_base": float(df["price_per_100_base"].std()),
            "mean_effective_duration": float(df["effective_duration"].mean()),
            "median_effective_duration": float(df["effective_duration"].median()),
            "std_effective_duration": float(df["effective_duration"].std()),
            "mean_wal_years": float(df["wal_years"].mean()),
            "mean_projected_total_principal": float(df["total_projected_principal"].mean()),
            "mean_projected_total_interest": float(df["total_projected_interest"].mean()),
        }

    if not pricing_results.empty:
        rows.append(summary_row(pricing_results, "overall", "ALL"))
    for sec_type in TARGET_SECURITY_TYPES:
        subset = pricing_results[pricing_results["Type_of_Security"] == sec_type]
        if subset.empty:
            continue
        rows.append(summary_row(subset, "security_type", sec_type))
    return pd.DataFrame(rows)


def plot_calibration_curves(calibration_df: pd.DataFrame, plots_dir: Path, show_plots: bool) -> None:
    sec_types = calibration_df["Type_of_Security"].drop_duplicates().tolist()
    if not sec_types:
        return
    fig, axes = plt.subplots(nrows=len(sec_types), ncols=1, figsize=(12, 5 * max(len(sec_types), 1)))
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])

    for ax, sec_type in zip(axes, sec_types):
        sec_df = calibration_df[calibration_df["Type_of_Security"] == sec_type].sort_values("mean_spread")
        ax.errorbar(
            sec_df["mean_spread"],
            sec_df["actual_cpr_mean"],
            yerr=sec_df["actual_cpr_ci95"],
            fmt="o",
            color="#1f77b4",
            ecolor="#1f77b4",
            elinewidth=1.2,
            capsize=3,
            label="Actual bin mean",
        )
        ax.plot(
            sec_df["mean_spread"],
            sec_df["predicted_cpr_mean"],
            color="#d62728",
            linewidth=2.2,
            marker="o",
            label="Predicted bin mean",
        )
        for _, row in sec_df.iterrows():
            ax.annotate(
                f"n={int(row['n_obs'])}",
                (row["mean_spread"], row["actual_cpr_mean"]),
                textcoords="offset points",
                xytext=(0, 7),
                ha="center",
                fontsize=8,
                color="#444444",
            )
        ax.set_title(sec_type)
        ax.set_xlabel("Spread (WAC - Market Rate) in %")
        ax.set_ylabel("CPR (%)")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper left")

    fig.suptitle(f"Calibration Curves by Security Type ({DEFAULT_MODEL_NAME})", fontsize=16)
    fig.tight_layout()
    fig.savefig(plots_dir / "calibration_curves.png", dpi=160, bbox_inches="tight")
    if show_plots:
        plt.show()
    else:
        plt.close(fig)


def plot_theoretical_curves(theoretical_df: pd.DataFrame, plots_dir: Path, show_plots: bool) -> None:
    if theoretical_df.empty:
        return
    fig, ax = plt.subplots(figsize=(12, 7))
    for sec_type in theoretical_df["Type_of_Security"].drop_duplicates():
        sec_df = theoretical_df[theoretical_df["Type_of_Security"] == sec_type].sort_values("spread")
        ax.plot(sec_df["spread"], sec_df["predicted_cpr_mean"], linewidth=2.5, label=sec_type)
        ax.fill_between(sec_df["spread"], sec_df["predicted_cpr_p25"], sec_df["predicted_cpr_p75"], alpha=0.12)

    ax.set_title(f"Theoretical Counterfactual S-Curves by Security Type ({DEFAULT_MODEL_NAME})")
    ax.set_xlabel("Spread (WAC - Market Rate) in %")
    ax.set_ylabel("Average Predicted CPR (%)")
    ax.axvline(0.0, color="black", linestyle="--", alpha=0.35)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(plots_dir / "theoretical_s_curves.png", dpi=160, bbox_inches="tight")
    if show_plots:
        plt.show()
    else:
        plt.close(fig)


def plot_loan_age_slice_curves(loan_age_df: pd.DataFrame, plots_dir: Path, show_plots: bool) -> None:
    sec_types = loan_age_df["Type_of_Security"].drop_duplicates().tolist()
    if not sec_types:
        return
    fig, axes = plt.subplots(nrows=len(sec_types), ncols=1, figsize=(12, 5 * max(len(sec_types), 1)))
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])

    for ax, sec_type in zip(axes, sec_types):
        sec_df = loan_age_df[loan_age_df["Type_of_Security"] == sec_type]
        for slice_name in ["low_loan_age", "median_loan_age", "high_loan_age"]:
            slice_df = sec_df[sec_df["slice_name"] == slice_name].sort_values("spread")
            if slice_df.empty:
                continue
            ax.plot(slice_df["spread"], slice_df["predicted_cpr_mean"], linewidth=2.2, label=slice_name)

        ax.set_title(f"{sec_type} - Loan Age Slice Theoretical Curves")
        ax.set_xlabel("Spread (WAC - Market Rate) in %")
        ax.set_ylabel("Average Predicted CPR (%)")
        ax.axvline(0.0, color="black", linestyle="--", alpha=0.35)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper left")

    fig.suptitle(f"Loan Age Slice Theoretical S-Curves ({DEFAULT_MODEL_NAME})", fontsize=16)
    fig.tight_layout()
    fig.savefig(plots_dir / "loan_age_slice_theoretical_s_curves.png", dpi=160, bbox_inches="tight")
    if show_plots:
        plt.show()
    else:
        plt.close(fig)


def plot_slope_diagnostics(slope_df: pd.DataFrame, plots_dir: Path, show_plots: bool) -> None:
    if slope_df.empty:
        return
    fig, ax = plt.subplots(figsize=(12, 7))
    for sec_type in slope_df["Type_of_Security"].drop_duplicates():
        sec_df = slope_df[slope_df["Type_of_Security"] == sec_type].sort_values("spread")
        ax.plot(sec_df["spread"], sec_df["dCPR_dspread"], linewidth=2.0, label=sec_type)

    ax.set_title(f"Theoretical S-Curve Slope Diagnostics ({DEFAULT_MODEL_NAME})")
    ax.set_xlabel("Spread (WAC - Market Rate) in %")
    ax.set_ylabel("dCPR / dspread")
    ax.axhline(0.0, color="black", linestyle="--", alpha=0.35)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(plots_dir / "theoretical_slope_diagnostics.png", dpi=160, bbox_inches="tight")
    if show_plots:
        plt.show()
    else:
        plt.close(fig)


def plot_pricing_validation_outputs(
    sample_cashflows: pd.DataFrame,
    avg_cf: pd.DataFrame,
    pricing_results: pd.DataFrame,
    plots_dir: Path,
    show_plots: bool,
) -> None:
    if not sample_cashflows.empty:
        fig, axes = plt.subplots(nrows=len(TARGET_SECURITY_TYPES), ncols=1, figsize=(12, 5 * len(TARGET_SECURITY_TYPES)))
        if not isinstance(axes, np.ndarray):
            axes = np.array([axes])
        for ax, sec_type in zip(axes, TARGET_SECURITY_TYPES):
            sec_df = sample_cashflows[
                (sample_cashflows["Type_of_Security"] == sec_type) & (sample_cashflows["scenario"] == "base")
            ]
            for obs_id in sec_df["obs_id"].drop_duplicates():
                obs_df = sec_df[sec_df["obs_id"] == obs_id].sort_values("month")
                ax.plot(obs_df["month"], obs_df["total_cashflow"], linewidth=1.8, label=f"obs {obs_id}")
            ax.set_title(sec_type)
            ax.set_xlabel("Month")
            ax.set_ylabel("Projected Cash Flow")
            ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(plots_dir / "cashflow_path_samples.png", dpi=160, bbox_inches="tight")
        if show_plots:
            plt.show()
        else:
            plt.close(fig)

    if not avg_cf.empty:
        fig, ax = plt.subplots(figsize=(12, 7))
        for sec_type in avg_cf["Type_of_Security"].drop_duplicates():
            sec_df = avg_cf[avg_cf["Type_of_Security"] == sec_type].sort_values("month")
            ax.plot(sec_df["month"], sec_df["avg_total_cashflow"], linewidth=2.2, label=sec_type)
        ax.set_title("Average Base-Scenario Cash Flow by Security Type")
        ax.set_xlabel("Month")
        ax.set_ylabel("Average Cash Flow")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right")
        fig.tight_layout()
        fig.savefig(plots_dir / "avg_cashflow_by_security.png", dpi=160, bbox_inches="tight")
        if show_plots:
            plt.show()
        else:
            plt.close(fig)

    if not pricing_results.empty:
        fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(12, 10))
        for sec_type in pricing_results["Type_of_Security"].drop_duplicates():
            sec_df = pricing_results[pricing_results["Type_of_Security"] == sec_type]
            axes[0].hist(sec_df["price_per_100_base"], bins=20, alpha=0.45, label=sec_type)
            axes[1].hist(sec_df["effective_duration"].dropna(), bins=20, alpha=0.45, label=sec_type)
        axes[0].set_title("Price per 100 Distribution by Security Type")
        axes[0].set_xlabel("Price per 100")
        axes[0].set_ylabel("Count")
        axes[0].legend(loc="upper right")
        axes[1].set_title("Effective Duration Distribution by Security Type")
        axes[1].set_xlabel("Effective Duration")
        axes[1].set_ylabel("Count")
        axes[1].legend(loc="upper right")
        for ax in axes:
            ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(plots_dir / "price_distribution_by_security.png", dpi=160, bbox_inches="tight")
        if show_plots:
            plt.show()
        else:
            plt.close(fig)

        fig, ax = plt.subplots(figsize=(12, 7))
        for sec_type in pricing_results["Type_of_Security"].drop_duplicates():
            sec_df = pricing_results[pricing_results["Type_of_Security"] == sec_type]
            ax.hist(sec_df["effective_duration"].dropna(), bins=20, alpha=0.45, label=sec_type)
        ax.set_title("Effective Duration Distribution by Security Type")
        ax.set_xlabel("Effective Duration")
        ax.set_ylabel("Count")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right")
        fig.tight_layout()
        fig.savefig(plots_dir / "duration_distribution_by_security.png", dpi=160, bbox_inches="tight")
        if show_plots:
            plt.show()
        else:
            plt.close(fig)


def save_model_summaries(data_dir: Path, summary_df: pd.DataFrame, summary_text: dict[str, str]) -> None:
    summary_df.to_csv(data_dir / "fitted_model_summaries.csv", index=False)
    for key, text in summary_text.items():
        model_name, sec_type = key.split("__", 1)
        slug = f"{model_name.lower()}_{sec_type.lower().replace(' ', '_')}"
        (data_dir / f"model_summary_{slug}.txt").write_text(text)


def save_outputs(
    output_root: Path,
    summary_df: pd.DataFrame,
    summary_text: dict[str, str],
    oos_predictions: pd.DataFrame,
    oos_metrics: pd.DataFrame,
    calibration_df: pd.DataFrame,
    theoretical_df: pd.DataFrame,
    loan_age_df: pd.DataFrame,
    slope_df: pd.DataFrame,
    pricing_results: pd.DataFrame,
    pricing_summary: pd.DataFrame,
    sample_cashflows: pd.DataFrame,
    avg_cf: pd.DataFrame,
) -> None:
    data_dir = output_root / "data"
    plots_dir = output_root / "plots"
    data_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    save_model_summaries(data_dir, summary_df, summary_text)
    oos_predictions.to_csv(data_dir / "oos_predictions.csv", index=False)
    oos_metrics.to_csv(data_dir / "oos_metrics.csv", index=False)
    calibration_df.to_csv(data_dir / "calibration_dataset.csv", index=False)
    theoretical_df.to_csv(data_dir / "theoretical_s_curve_dataset.csv", index=False)
    loan_age_df.to_csv(data_dir / "loan_age_slice_theoretical_s_curve_dataset.csv", index=False)
    slope_df.to_csv(data_dir / "theoretical_slope_diagnostics.csv", index=False)
    pricing_results.to_csv(data_dir / "pricing_results_oos.csv", index=False)
    pricing_summary.to_csv(data_dir / "pricing_summary.csv", index=False)
    sample_cashflows.to_csv(data_dir / "cashflow_paths_sample.csv", index=False)
    avg_cf.to_csv(data_dir / "cashflow_paths_avg_by_security.csv", index=False)


def print_model_results(summary_df: pd.DataFrame) -> None:
    print(f"\n=== Fitted Model Summaries (default production model: {DEFAULT_MODEL_NAME}) ===")
    if summary_df.empty:
        print("No models were fitted.")
    else:
        print(summary_df.to_string(index=False, float_format=lambda x: f"{x:,.6f}"))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="MBS logit pipeline with calibration curves and theoretical counterfactual S-curves."
    )
    parser.add_argument("--data-path", type=str, default="PrepayData.txt")
    parser.add_argument("--output-dir", type=str, default="outputs/ols")
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--n-bins", type=int, default=8)
    parser.add_argument("--spread-q-low", type=float, default=0.01)
    parser.add_argument("--spread-q-high", type=float, default=0.99)
    parser.add_argument("--pricing-split", type=str, default="test", choices=["test", "validation", "both"])
    parser.add_argument("--shock-bp", type=float, default=50.0)
    parser.add_argument("--max-cf-plots", type=int, default=3)
    parser.add_argument("--max-projection-months", type=int, default=360)
    parser.add_argument("--show-plots", action="store_true", default=True)
    parser.add_argument("--no-show-plots", action="store_false", dest="show_plots")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    output_root = Path(args.output_dir)

    df = load_prepay_data(Path(args.data_path))
    market_monthly = download_market_rates(df)
    panel = build_observation_panel(df, market_monthly)
    panel = assign_time_splits(panel, args.train_ratio, args.val_ratio, args.test_ratio)
    fitted_models, summary_df, summary_text, train_support = fit_security_models(panel)
    prediction_panel = build_prediction_panel(
        panel, fitted_models, train_support, model_name=DEFAULT_MODEL_NAME
    )
    oos_predictions = build_oos_predictions(prediction_panel)
    oos_metrics = build_oos_metrics(oos_predictions)

    calibration_df = build_calibration_dataset(prediction_panel, n_bins=args.n_bins, model_name=DEFAULT_MODEL_NAME)
    theoretical_df, loan_age_df = build_theoretical_curves(
        prediction_panel,
        fitted_models,
        train_support,
        spread_q_low=args.spread_q_low,
        spread_q_high=args.spread_q_high,
        model_name=DEFAULT_MODEL_NAME,
    )
    slope_df = build_slope_diagnostics(theoretical_df)
    pricing_results, sample_cashflows, avg_cf, all_cashflows = build_pricing_results(
        prediction_panel=prediction_panel,
        fitted_models=fitted_models,
        train_support=train_support,
        pricing_split=args.pricing_split,
        shock_bp=args.shock_bp,
        max_projection_months=args.max_projection_months,
    )
    if not sample_cashflows.empty:
        sample_cashflows = (
            sample_cashflows.sort_values(["Type_of_Security", "obs_id", "scenario", "month"])
            .groupby(["Type_of_Security", "obs_id", "scenario"], as_index=False, group_keys=False)
            .head(args.max_cf_plots * args.max_projection_months)
        )
        sample_obs = (
            sample_cashflows[["Type_of_Security", "obs_id"]]
            .drop_duplicates()
            .groupby("Type_of_Security", as_index=False, group_keys=False)
            .head(args.max_cf_plots)
        )
        sample_cashflows = sample_cashflows.merge(sample_obs, on=["Type_of_Security", "obs_id"])
    pricing_summary = summarize_pricing_results(pricing_results)

    save_outputs(
        output_root=output_root,
        summary_df=summary_df,
        summary_text=summary_text,
        oos_predictions=oos_predictions,
        oos_metrics=oos_metrics,
        calibration_df=calibration_df,
        theoretical_df=theoretical_df,
        loan_age_df=loan_age_df,
        slope_df=slope_df,
        pricing_results=pricing_results,
        pricing_summary=pricing_summary,
        sample_cashflows=sample_cashflows,
        avg_cf=avg_cf,
    )

    print_model_results(summary_df)
    plots_dir = output_root / "plots"
    plot_calibration_curves(calibration_df, plots_dir, args.show_plots)
    plot_theoretical_curves(theoretical_df, plots_dir, args.show_plots)
    if not loan_age_df.empty:
        plot_loan_age_slice_curves(loan_age_df, plots_dir, args.show_plots)
    if not slope_df.empty:
        plot_slope_diagnostics(slope_df, plots_dir, args.show_plots)
    plot_pricing_validation_outputs(sample_cashflows, avg_cf, pricing_results, plots_dir, args.show_plots)


if __name__ == "__main__":
    main()
