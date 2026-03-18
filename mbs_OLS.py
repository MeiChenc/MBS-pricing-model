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
            "Cohort_Current_UPB",
            "SMM_monthly",
            "logit_smm",
            "log_UPB",
        ]
    ).reset_index(drop=True)

    panel["log_UPB_z"] = (panel["log_UPB"] - panel["log_UPB"].mean()) / panel["log_UPB"].std()
    return panel


def fit_security_models(panel: pd.DataFrame) -> tuple[dict[str, dict[str, object]], pd.DataFrame, dict[str, str]]:
    fitted_models: dict[str, dict[str, object]] = {}
    summary_rows: list[dict[str, float | int | str]] = []
    summary_text: dict[str, str] = {}

    for sec_type in TARGET_SECURITY_TYPES:
        subset = panel[panel["Type_of_Security"] == sec_type].copy()
        subset = subset.dropna(subset=["logit_smm", "spread", "loan_age", "log_UPB_z"])
        if len(subset) < 50:
            continue

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

    return fitted_models, pd.DataFrame(summary_rows), summary_text


def choose_bin_count(n_obs: int, target_bin_size: int = 24, min_bins: int = 4, max_bins: int = 12) -> int:
    if n_obs <= 0:
        return min_bins
    return max(min_bins, min(max_bins, n_obs // target_bin_size))


def build_calibration_dataset(
    panel: pd.DataFrame, fitted_models: dict[str, dict[str, object]], model_name: str = DEFAULT_MODEL_NAME
) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []

    for sec_type in TARGET_SECURITY_TYPES:
        model = fitted_models.get(sec_type, {}).get(model_name)
        if model is None:
            continue

        subset = panel[panel["Type_of_Security"] == sec_type].copy()
        if subset.empty:
            continue

        pred_logit = model.predict(subset[["spread", "log_UPB_z", "loan_age"]])
        subset["predicted_smm"] = 1.0 / (1.0 + np.exp(-pred_logit))
        subset["predicted_cpr_pct"] = 100.0 * (1.0 - (1.0 - subset["predicted_smm"]) ** 12.0)
        subset["actual_cpr_pct"] = 100.0 * (1.0 - (1.0 - subset["SMM_monthly"]) ** 12.0)

        n_bins = choose_bin_count(len(subset))
        subset["spread_bin"] = pd.qcut(subset["spread"], q=n_bins, duplicates="drop")

        grouped = (
            subset.groupby("spread_bin", observed=False)
            .agg(
                mean_spread=("spread", "mean"),
                actual_cpr_mean=("actual_cpr_pct", "mean"),
                predicted_cpr_mean=("predicted_cpr_pct", "mean"),
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
    model_name: str = DEFAULT_MODEL_NAME,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    overall_frames: list[pd.DataFrame] = []
    slice_frames: list[pd.DataFrame] = []

    for sec_type in TARGET_SECURITY_TYPES:
        model = fitted_models.get(sec_type, {}).get(model_name)
        if model is None:
            continue

        subset = panel[panel["Type_of_Security"] == sec_type].copy()
        if subset.empty:
            continue

        spread_grid = get_spread_grid(subset)
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
        sec_df = theoretical_df[theoretical_df["Type_of_Security"] == sec_type].sort_values("spread").copy()
        slopes = np.gradient(sec_df["predicted_cpr_mean"].to_numpy(), sec_df["spread"].to_numpy())
        sec_df["dCPR_dspread"] = slopes
        rows.extend(sec_df[["Type_of_Security", "spread", "predicted_cpr_mean", "dCPR_dspread"]].to_dict("records"))
    return pd.DataFrame(rows)


def build_model_comparison_metrics(
    panel: pd.DataFrame, fitted_models: dict[str, dict[str, object]]
) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for sec_type in TARGET_SECURITY_TYPES:
        subset = panel[panel["Type_of_Security"] == sec_type].dropna(subset=["spread", "loan_age", "log_UPB_z", "SMM_monthly"])
        if subset.empty:
            continue
        actual_cpr_pct = 100.0 * (1.0 - (1.0 - subset["SMM_monthly"]) ** 12.0)

        for model_name, model in fitted_models.get(sec_type, {}).items():
            pred_logit = model.predict(subset[["spread", "log_UPB_z", "loan_age"]])
            pred_smm = 1.0 / (1.0 + np.exp(-pred_logit))
            pred_cpr_pct = 100.0 * (1.0 - (1.0 - pred_smm) ** 12.0)
            rows.append(
                {
                    "Type_of_Security": sec_type,
                    "model_name": model_name,
                    "nobs": int(len(subset)),
                    "rmse_cpr_pct": float(np.sqrt(np.mean((pred_cpr_pct - actual_cpr_pct) ** 2))),
                    "mae_cpr_pct": float(np.mean(np.abs(pred_cpr_pct - actual_cpr_pct))),
                }
            )

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
    fig.savefig(plots_dir / "theoretical_s_curve_slope_diagnostics.png", dpi=160, bbox_inches="tight")
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
    comparison_df: pd.DataFrame,
    calibration_df: pd.DataFrame,
    theoretical_df: pd.DataFrame,
    loan_age_df: pd.DataFrame,
    slope_df: pd.DataFrame,
) -> None:
    data_dir = output_root / "data"
    plots_dir = output_root / "plots"
    data_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    save_model_summaries(data_dir, summary_df, summary_text)
    comparison_df.to_csv(data_dir / "model_fit_comparison.csv", index=False)
    calibration_df.to_csv(data_dir / "calibration_dataset.csv", index=False)
    theoretical_df.to_csv(data_dir / "theoretical_s_curve_dataset.csv", index=False)
    loan_age_df.to_csv(data_dir / "loan_age_slice_theoretical_s_curve_dataset.csv", index=False)
    slope_df.to_csv(data_dir / "theoretical_s_curve_slope_diagnostics.csv", index=False)


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
    parser.add_argument("--show-plots", action="store_true", default=True)
    parser.add_argument("--no-show-plots", action="store_false", dest="show_plots")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    output_root = Path(args.output_dir)

    df = load_prepay_data(Path(args.data_path))
    market_monthly = download_market_rates(df)
    panel = build_observation_panel(df, market_monthly)
    fitted_models, summary_df, summary_text = fit_security_models(panel)
    comparison_df = build_model_comparison_metrics(panel, fitted_models)

    calibration_df = build_calibration_dataset(panel, fitted_models, model_name=DEFAULT_MODEL_NAME)
    theoretical_df, loan_age_df = build_theoretical_curves(panel, fitted_models, model_name=DEFAULT_MODEL_NAME)
    slope_df = build_slope_diagnostics(theoretical_df)

    save_outputs(
        output_root=output_root,
        summary_df=summary_df,
        summary_text=summary_text,
        comparison_df=comparison_df,
        calibration_df=calibration_df,
        theoretical_df=theoretical_df,
        loan_age_df=loan_age_df,
        slope_df=slope_df,
    )

    print_model_results(summary_df)
    plots_dir = output_root / "plots"
    plot_calibration_curves(calibration_df, plots_dir, args.show_plots)
    plot_theoretical_curves(theoretical_df, plots_dir, args.show_plots)
    if not loan_age_df.empty:
        plot_loan_age_slice_curves(loan_age_df, plots_dir, args.show_plots)
    if not slope_df.empty:
        plot_slope_diagnostics(slope_df, plots_dir, args.show_plots)


if __name__ == "__main__":
    main()
