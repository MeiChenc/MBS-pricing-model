from __future__ import annotations

import argparse
from dataclasses import dataclass
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
HAZARD_FORMULA = (
    "cloglog_smm ~ bs(spread, df=5, degree=3, include_intercept=False) + "
    "loan_age + log_UPB_z + "
    "bs(spread, df=5, degree=3, include_intercept=False):loan_age"
)


@dataclass
class SecurityModelResult:
    security_type: str
    model: object
    panel: pd.DataFrame
    train_mean_log_upb: float
    train_std_log_upb: float
    spread_min: float
    spread_max: float
    summary_text: str


def load_prepay_data(data_path: Path) -> pd.DataFrame:
    df = pd.read_csv(data_path, sep="|", low_memory=False)
    df.columns = (
        df.columns.str.strip().str.replace(" ", "_", regex=False).str.replace("-", "_", regex=False)
    )

    numeric_cols = [
        "Cohort_Current_UPB",
        "Cohort_WA_Current_Interest_Rate",
        "Cohort_WA_Current_Loan_Age",
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
            "Install dependencies from requirements.txt before running mbs_Cox.py."
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
    panel["coupon"] = pd.to_numeric(panel["Cohort_WA_Current_Interest_Rate"], errors="coerce")
    panel["spread"] = panel["coupon"] - pd.to_numeric(panel["market_rate"], errors="coerce")
    panel["log_UPB"] = np.log(pd.to_numeric(panel["Cohort_Current_UPB"], errors="coerce") + 1.0)
    panel["SMM_monthly"] = pd.to_numeric(panel["SMM"], errors="coerce")
    panel["smm_adj"] = panel["SMM_monthly"].clip(lower=EPSILON, upper=1.0 - EPSILON)
    panel["cloglog_smm"] = np.log(-np.log(1.0 - panel["smm_adj"]))
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
            "Cohort_Current_UPB",
            "log_UPB",
            "SMM_monthly",
            "smm_adj",
            "cloglog_smm",
        ]
    ).reset_index(drop=True)
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


def standardize_log_upb(panel: pd.DataFrame) -> tuple[pd.DataFrame, float, float]:
    train_mask = panel["split"] == "train"
    train_mean = float(panel.loc[train_mask, "log_UPB"].mean())
    train_std = float(panel.loc[train_mask, "log_UPB"].std())
    if not np.isfinite(train_std) or train_std <= 0:
        train_std = 1.0

    panel = panel.copy()
    panel["log_UPB_z"] = (panel["log_UPB"] - train_mean) / train_std
    return panel, train_mean, train_std


def fit_security_models(
    panel: pd.DataFrame,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> dict[str, SecurityModelResult]:
    results: dict[str, SecurityModelResult] = {}

    for sec_type in TARGET_SECURITY_TYPES:
        sec_panel = panel[panel["Type_of_Security"] == sec_type].copy()
        if sec_panel.empty:
            continue

        sec_panel = assign_time_splits(sec_panel, train_ratio, val_ratio, test_ratio)
        sec_panel, train_mean, train_std = standardize_log_upb(sec_panel)

        train_df = sec_panel[sec_panel["split"] == "train"].dropna(
            subset=["cloglog_smm", "spread", "loan_age", "log_UPB_z"]
        )
        if len(train_df) < 50:
            continue

        spread_min = float(train_df["spread"].min())
        spread_max = float(train_df["spread"].max())

        model = smf.ols(formula=HAZARD_FORMULA, data=train_df).fit()
        results[sec_type] = SecurityModelResult(
            security_type=sec_type,
            model=model,
            panel=sec_panel,
            train_mean_log_upb=train_mean,
            train_std_log_upb=train_std,
            spread_min=spread_min,
            spread_max=spread_max,
            summary_text=model.summary().as_text(),
        )

    return results


def inverse_cloglog(eta_hat: pd.Series | np.ndarray) -> np.ndarray:
    return 1.0 - np.exp(-np.exp(np.asarray(eta_hat, dtype=float)))


def smm_to_cpr_pct(smm: pd.Series | np.ndarray) -> np.ndarray:
    smm_arr = np.asarray(smm, dtype=float)
    return 100.0 * (1.0 - (1.0 - smm_arr) ** 12.0)


def clip_spread_to_train_support(pred_data: pd.DataFrame, result: SecurityModelResult) -> pd.DataFrame:
    clipped = pred_data.copy()
    clipped["spread"] = clipped["spread"].clip(lower=result.spread_min, upper=result.spread_max)
    return clipped


def build_prediction_panel(results: dict[str, SecurityModelResult]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for sec_type in TARGET_SECURITY_TYPES:
        result = results.get(sec_type)
        if result is None:
            continue

        subset = result.panel.copy()
        pred_data = clip_spread_to_train_support(subset[["spread", "loan_age", "log_UPB_z"]], result)
        eta_hat = result.model.predict(pred_data)
        subset["eta_hat"] = eta_hat
        subset["pred_smm"] = inverse_cloglog(eta_hat)
        subset["actual_smm"] = subset["SMM_monthly"]
        subset["pred_cpr_pct"] = smm_to_cpr_pct(subset["pred_smm"])
        subset["actual_cpr_pct"] = smm_to_cpr_pct(subset["actual_smm"])
        frames.append(subset)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def build_oos_predictions(prediction_panel: pd.DataFrame) -> pd.DataFrame:
    oos = prediction_panel[prediction_panel["split"].isin(["validation", "test"])].copy()
    if oos.empty:
        return pd.DataFrame(
            columns=[
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
        )
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


def build_oos_metrics(oos_predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []

    if oos_predictions.empty or "Type_of_Security" not in oos_predictions.columns:
        return pd.DataFrame(rows)

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
            "n_obs": int(len(df)),
            "RMSE": float(np.sqrt(np.mean(err**2))),
            "MAE": float(np.mean(np.abs(err))),
            "UPB-weighted RMSE": float(np.sqrt(np.sum(w * (err**2)) / w_sum)),
            "UPB-weighted MAE": float(np.sum(w * np.abs(err)) / w_sum),
        }

    if not oos_predictions.empty:
        rows.append(metric_row(oos_predictions, "overall", "ALL"))
    for sec_type in TARGET_SECURITY_TYPES:
        subset = oos_predictions[oos_predictions["Type_of_Security"] == sec_type]
        if subset.empty:
            continue
        rows.append(metric_row(subset, "security_type", sec_type))

    return pd.DataFrame(rows)


def build_calibration_dataset(prediction_panel: pd.DataFrame, n_bins: int) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []

    if prediction_panel.empty or "Type_of_Security" not in prediction_panel.columns:
        return pd.DataFrame(rows)

    for sec_type in TARGET_SECURITY_TYPES:
        subset = prediction_panel[
            (prediction_panel["Type_of_Security"] == sec_type) & (prediction_panel["split"] == "test")
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
                total_upb=("Cohort_Current_UPB", "sum"),
            )
            .dropna(subset=["mean_spread", "actual_cpr_mean", "predicted_cpr_mean"])
            .reset_index()
            .sort_values("mean_spread")
        )
        if grouped.empty:
            continue

        grouped["actual_cpr_se"] = grouped["actual_cpr_std"] / np.sqrt(grouped["n_obs"].clip(lower=1))
        grouped["actual_cpr_ci95"] = 1.96 * grouped["actual_cpr_se"].fillna(0.0)
        grouped["bin_sq_error"] = (grouped["actual_cpr_mean"] - grouped["predicted_cpr_mean"]) ** 2
        total_upb = grouped["total_upb"].sum()
        if pd.notna(total_upb) and float(total_upb) > 0:
            grouped["upb_weight"] = grouped["total_upb"] / total_upb
            grouped["weighted_bin_sq_error"] = grouped["bin_sq_error"] * grouped["upb_weight"]
        else:
            grouped["upb_weight"] = np.nan
            grouped["weighted_bin_sq_error"] = np.nan

        for _, row in grouped.iterrows():
            rows.append(
                {
                    "Type_of_Security": sec_type,
                    "spread_bin": str(row["spread_bin"]),
                    "mean_spread": float(row["mean_spread"]),
                    "actual_cpr_mean": float(row["actual_cpr_mean"]),
                    "predicted_cpr_mean": float(row["predicted_cpr_mean"]),
                    "actual_cpr_std": float(row["actual_cpr_std"]) if pd.notna(row["actual_cpr_std"]) else np.nan,
                    "actual_cpr_se": float(row["actual_cpr_se"]) if pd.notna(row["actual_cpr_se"]) else np.nan,
                    "actual_cpr_ci95": float(row["actual_cpr_ci95"]),
                    "n_obs": int(row["n_obs"]),
                    "total_upb": float(row["total_upb"]) if pd.notna(row["total_upb"]) else np.nan,
                    "bin_sq_error": float(row["bin_sq_error"]),
                    "weighted_bin_sq_error": (
                        float(row["weighted_bin_sq_error"]) if pd.notna(row["weighted_bin_sq_error"]) else np.nan
                    ),
                }
            )

    return pd.DataFrame(rows)


def get_spread_grid(subset: pd.DataFrame, q_low: float, q_high: float, num_points: int = 80) -> np.ndarray:
    lo = float(subset["spread"].quantile(q_low))
    hi = float(subset["spread"].quantile(q_high))
    if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi:
        lo = float(subset["spread"].quantile(0.05))
        hi = float(subset["spread"].quantile(0.95))
    if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi:
        lo = float(subset["spread"].min())
        hi = float(subset["spread"].max())
    return np.linspace(lo, hi, num_points)


def build_theoretical_curves(
    prediction_panel: pd.DataFrame,
    results: dict[str, SecurityModelResult],
    spread_q_low: float,
    spread_q_high: float,
) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []

    if prediction_panel.empty or "Type_of_Security" not in prediction_panel.columns:
        return pd.DataFrame(rows)

    for sec_type in TARGET_SECURITY_TYPES:
        result = results.get(sec_type)
        if result is None:
            continue

        subset = prediction_panel[
            (prediction_panel["Type_of_Security"] == sec_type) & (prediction_panel["split"] == "test")
        ].copy()
        if subset.empty:
            continue

        spread_grid = get_spread_grid(subset, spread_q_low, spread_q_high)
        spread_grid = np.clip(spread_grid, result.spread_min, result.spread_max)
        for spread_value in spread_grid:
            pred_data = subset[["loan_age", "log_UPB_z"]].copy()
            pred_data["spread"] = float(spread_value)
            pred_data = clip_spread_to_train_support(pred_data, result)
            eta_hat = result.model.predict(pred_data)
            pred_smm = inverse_cloglog(eta_hat)
            pred_cpr_pct = smm_to_cpr_pct(pred_smm)

            rows.append(
                {
                    "Type_of_Security": sec_type,
                    "spread": float(spread_value),
                    "predicted_cpr_mean": float(np.mean(pred_cpr_pct)),
                    "predicted_cpr_p25": float(np.percentile(pred_cpr_pct, 25)),
                    "predicted_cpr_p75": float(np.percentile(pred_cpr_pct, 75)),
                    "n_obs": int(len(subset)),
                }
            )

    return pd.DataFrame(rows)


def build_slope_diagnostics(theoretical_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    if theoretical_df.empty:
        return pd.DataFrame(rows)

    for sec_type in theoretical_df["Type_of_Security"].drop_duplicates():
        sec_df = theoretical_df[theoretical_df["Type_of_Security"] == sec_type].sort_values("spread").copy()
        if len(sec_df) < 2:
            continue
        sec_df["dCPR_dspread"] = np.gradient(
            sec_df["predicted_cpr_mean"].to_numpy(),
            sec_df["spread"].to_numpy(),
        )
        rows.extend(
            sec_df[
                [
                    "Type_of_Security",
                    "spread",
                    "predicted_cpr_mean",
                    "predicted_cpr_p25",
                    "predicted_cpr_p75",
                    "dCPR_dspread",
                    "n_obs",
                ]
            ].to_dict("records")
        )

    return pd.DataFrame(rows)


def plot_calibration_curves(calibration_df: pd.DataFrame, output_dir: Path, show_plots: bool) -> None:
    if calibration_df.empty or "Type_of_Security" not in calibration_df.columns:
        return

    sec_types = [sec for sec in TARGET_SECURITY_TYPES if sec in calibration_df["Type_of_Security"].unique()]
    if not sec_types:
        return

    fig, axes = plt.subplots(nrows=len(sec_types), ncols=1, figsize=(12, 5 * len(sec_types)))
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
        ax.set_title(sec_type)
        ax.set_xlabel("Spread (WAC - Market Rate) in %")
        ax.set_ylabel("CPR (%)")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper left")

    fig.tight_layout()
    fig.savefig(output_dir / "calibration_curves.png", dpi=160, bbox_inches="tight")
    if show_plots:
        plt.show()
    else:
        plt.close(fig)


def plot_theoretical_curves(theoretical_df: pd.DataFrame, output_dir: Path, show_plots: bool) -> None:
    if theoretical_df.empty or "Type_of_Security" not in theoretical_df.columns:
        return

    fig, ax = plt.subplots(figsize=(12, 7))
    for sec_type in TARGET_SECURITY_TYPES:
        sec_df = theoretical_df[theoretical_df["Type_of_Security"] == sec_type].sort_values("spread")
        if sec_df.empty:
            continue
        ax.plot(sec_df["spread"], sec_df["predicted_cpr_mean"], linewidth=2.5, label=sec_type)
        ax.fill_between(sec_df["spread"], sec_df["predicted_cpr_p25"], sec_df["predicted_cpr_p75"], alpha=0.12)

    ax.set_title("Theoretical Counterfactual S-Curves")
    ax.set_xlabel("Spread (WAC - Market Rate) in %")
    ax.set_ylabel("Average Predicted CPR (%)")
    ax.axvline(0.0, color="black", linestyle="--", alpha=0.35)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(output_dir / "theoretical_scurve.png", dpi=160, bbox_inches="tight")
    if show_plots:
        plt.show()
    else:
        plt.close(fig)


def plot_slope_diagnostics(slope_df: pd.DataFrame, output_dir: Path, show_plots: bool) -> None:
    if slope_df.empty or "Type_of_Security" not in slope_df.columns:
        return

    fig, ax = plt.subplots(figsize=(12, 7))
    for sec_type in TARGET_SECURITY_TYPES:
        sec_df = slope_df[slope_df["Type_of_Security"] == sec_type].sort_values("spread")
        if sec_df.empty:
            continue
        ax.plot(sec_df["spread"], sec_df["dCPR_dspread"], linewidth=2.2, label=sec_type)

    ax.set_title("Theoretical S-Curve Slope Diagnostics")
    ax.set_xlabel("Spread (WAC - Market Rate) in %")
    ax.set_ylabel("dCPR / dspread")
    ax.axhline(0.0, color="black", linestyle="--", alpha=0.35)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(output_dir / "theoretical_slope_diagnostics.png", dpi=160, bbox_inches="tight")
    if show_plots:
        plt.show()
    else:
        plt.close(fig)


def save_model_summaries(output_dir: Path, results: dict[str, SecurityModelResult]) -> None:
    for sec_type, result in results.items():
        slug = sec_type.lower().replace(" ", "_")
        (output_dir / f"model_summary_{slug}.txt").write_text(result.summary_text)


def save_outputs(
    output_dir: Path,
    oos_predictions: pd.DataFrame,
    oos_metrics: pd.DataFrame,
    calibration_df: pd.DataFrame,
    theoretical_df: pd.DataFrame,
    slope_df: pd.DataFrame,
    results: dict[str, SecurityModelResult],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    save_model_summaries(output_dir, results)
    oos_predictions.to_csv(output_dir / "oos_predictions.csv", index=False)
    oos_metrics.to_csv(output_dir / "oos_metrics.csv", index=False)
    calibration_df.to_csv(output_dir / "calibration_curve_bins.csv", index=False)
    theoretical_df.to_csv(output_dir / "theoretical_scurve.csv", index=False)
    slope_df.to_csv(output_dir / "theoretical_slope_diagnostics.csv", index=False)


def print_model_results(results: dict[str, SecurityModelResult]) -> None:
    print("\n=== Hazard Benchmark Model Summaries ===")
    if not results:
        print("No models were fitted.")
        return

    summary_rows = []
    for sec_type, result in results.items():
        summary_rows.append(
            {
                "Type_of_Security": sec_type,
                "nobs": int(result.model.nobs),
                "rsquared": float(result.model.rsquared),
                "adj_rsquared": float(result.model.rsquared_adj),
                "aic": float(result.model.aic),
                "bic": float(result.model.bic),
            }
        )
    summary_df = pd.DataFrame(summary_rows)
    print(summary_df.to_string(index=False, float_format=lambda x: f"{x:,.6f}"))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="MBS hazard-style benchmark using a complementary log-log transformed monthly SMM target."
    )
    parser.add_argument("--data-path", type=str, default="PrepayData.txt")
    parser.add_argument("--output-dir", type=str, default="outputs/cox")
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--n-bins", type=int, default=8)
    parser.add_argument("--spread-q-low", type=float, default=0.01)
    parser.add_argument("--spread-q-high", type=float, default=0.99)
    parser.add_argument("--show-plots", action="store_true", default=True)
    parser.add_argument("--no-show-plots", action="store_false", dest="show_plots")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    df = load_prepay_data(Path(args.data_path))
    market_monthly = download_market_rates(df)
    panel = build_observation_panel(df, market_monthly)
    results = fit_security_models(
        panel=panel,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
    )
    prediction_panel = build_prediction_panel(results)
    oos_predictions = build_oos_predictions(prediction_panel)
    oos_metrics = build_oos_metrics(oos_predictions)
    calibration_df = build_calibration_dataset(prediction_panel, n_bins=args.n_bins)
    theoretical_df = build_theoretical_curves(
        prediction_panel=prediction_panel,
        results=results,
        spread_q_low=args.spread_q_low,
        spread_q_high=args.spread_q_high,
    )
    slope_df = build_slope_diagnostics(theoretical_df)

    output_dir = Path(args.output_dir)
    save_outputs(
        output_dir=output_dir,
        oos_predictions=oos_predictions,
        oos_metrics=oos_metrics,
        calibration_df=calibration_df,
        theoretical_df=theoretical_df,
        slope_df=slope_df,
        results=results,
    )

    print_model_results(results)
    plot_calibration_curves(calibration_df, output_dir, args.show_plots)
    plot_theoretical_curves(theoretical_df, output_dir, args.show_plots)
    plot_slope_diagnostics(slope_df, output_dir, args.show_plots)


if __name__ == "__main__":
    main()
