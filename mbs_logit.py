from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


EPSILON = 1e-6

MODEL_FORMULAS = {
    "A": "logit_smm ~ spread",
    "B": "logit_smm ~ spread + duration",
    # TODO: If right-tail inversion persists, replace the cubic spread terms with a spline specification.
    "C": "logit_smm ~ spread + I(spread**2) + I(spread**3) + log_UPB_z + duration",
}


def load_prepay_data(data_path: Path) -> pd.DataFrame:
    df = pd.read_csv(data_path, sep="|", low_memory=False)
    df.columns = (
        df.columns.str.strip().str.replace(" ", "_", regex=False).str.replace("-", "_", regex=False)
    )

    numeric_cols = [
        "Cohort_Current_UPB",
        "Cohort_WA_Current_Interest_Rate",
        "Cohort_WA_Current_Remaining_Months_to_Maturity",
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
            "Install dependencies from requirements.txt before running mbs_logit.py."
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


def prepare_features(df: pd.DataFrame, market_monthly: pd.DataFrame) -> pd.DataFrame:
    merged = df.merge(market_monthly, left_on="period", right_index=True, how="left")
    merged["market_rate"] = merged.apply(assign_market_rate, axis=1)

    merged["duration"] = pd.to_numeric(merged["Cohort_WA_Current_Loan_Age"], errors="coerce")
    merged["SMM"] = pd.to_numeric(merged["SMM"], errors="coerce")
    merged["smm_adj"] = merged["SMM"].clip(lower=EPSILON, upper=1 - EPSILON)
    merged["logit_smm"] = np.log(merged["smm_adj"] / (1 - merged["smm_adj"]))

    merged["spread"] = (
        pd.to_numeric(merged["Cohort_WA_Current_Interest_Rate"], errors="coerce")
        - pd.to_numeric(merged["market_rate"], errors="coerce")
    )
    merged["spread2"] = merged["spread"] ** 2
    merged["spread3"] = merged["spread"] ** 3
    merged["log_UPB"] = np.log(pd.to_numeric(merged["Cohort_Current_UPB"], errors="coerce") + 1.0)
    merged["coupon"] = pd.to_numeric(merged["Cohort_WA_Current_Interest_Rate"], errors="coerce")
    merged["log_UPB_z"] = (merged["log_UPB"] - merged["log_UPB"].mean()) / merged["log_UPB"].std()

    merged["actual_cpr_pct"] = 100.0 * (1.0 - (1.0 - merged["SMM"]) ** 12.0)

    return merged


def build_monthly_panel(df: pd.DataFrame) -> pd.DataFrame:
    weight_col = "Cohort_Current_UPB"
    work = df.dropna(subset=["period", "Type_of_Security", weight_col]).copy()
    work[weight_col] = pd.to_numeric(work[weight_col], errors="coerce")
    work = work[work[weight_col] > 0].copy()
    if work.empty:
        raise ValueError("No rows available to build the monthly weighted panel.")

    weighted_avg_cols = [
        "SMM",
        "Cumulative_SMM",
        "CPR",
        "Cumulative_CPR",
        "market_rate",
        "Cohort_WA_Current_Interest_Rate",
        "duration",
    ]

    def weighted_mean(values: pd.Series, weights: pd.Series) -> float:
        mask = values.notna() & weights.notna()
        if not mask.any():
            return np.nan
        vals = values[mask].astype(float)
        wts = weights[mask].astype(float)
        total_weight = wts.sum()
        if total_weight <= 0:
            return np.nan
        return float(np.average(vals, weights=wts))

    def aggregate_group(group: pd.DataFrame) -> pd.Series:
        period, security_type = group.name
        weights = pd.to_numeric(group[weight_col], errors="coerce")
        row: dict[str, float | str | pd.Period | int] = {
            "period": period,
            "Type_of_Security": security_type,
            "Date": int(group["date_parsed"].max().strftime("%Y%m%d")),
            "Cohort_Current_UPB": float(weights.sum()),
            "n_raw_rows": int(len(group)),
        }
        for col in weighted_avg_cols:
            row[col] = weighted_mean(pd.to_numeric(group[col], errors="coerce"), weights)
        return pd.Series(row)

    monthly = (
        work.groupby(["period", "Type_of_Security"])
        .apply(aggregate_group)
        .reset_index(drop=True)
    )

    monthly["period"] = pd.PeriodIndex(monthly["period"], freq="M")
    monthly["date_parsed"] = monthly["period"].dt.to_timestamp("M")
    monthly["coupon"] = pd.to_numeric(monthly["Cohort_WA_Current_Interest_Rate"], errors="coerce")
    monthly["spread"] = monthly["coupon"] - pd.to_numeric(monthly["market_rate"], errors="coerce")
    monthly["spread2"] = monthly["spread"] ** 2
    monthly["spread3"] = monthly["spread"] ** 3
    monthly["log_UPB"] = np.log(pd.to_numeric(monthly["Cohort_Current_UPB"], errors="coerce") + 1.0)
    monthly["log_UPB_z"] = (monthly["log_UPB"] - monthly["log_UPB"].mean()) / monthly["log_UPB"].std()
    monthly["SMM_monthly"] = pd.to_numeric(monthly["SMM"], errors="coerce")
    monthly["smm_adj"] = monthly["SMM_monthly"].clip(lower=EPSILON, upper=1 - EPSILON)
    monthly["logit_smm"] = np.log(monthly["smm_adj"] / (1 - monthly["smm_adj"]))
    monthly["actual_cpr_pct"] = 100.0 * (1.0 - (1.0 - monthly["SMM_monthly"]) ** 12.0)
    monthly["reported_cpr_pct"] = pd.to_numeric(monthly["CPR"], errors="coerce")

    return monthly


def select_top_security_types(monthly: pd.DataFrame, top_n: int = 3) -> list[str]:
    return monthly["Type_of_Security"].value_counts().nlargest(top_n).index.tolist()


def fit_security_models(
    monthly: pd.DataFrame, security_types: list[str]
) -> tuple[dict[str, dict[str, object]], pd.DataFrame]:
    fitted_models: dict[str, dict[str, object]] = {}
    summary_rows: list[dict[str, float | int | str]] = []

    needed_cols = ["logit_smm", "spread", "duration", "log_UPB_z"]

    for sec_type in security_types:
        subset = monthly[monthly["Type_of_Security"] == sec_type].copy()
        subset = subset.dropna(subset=needed_cols)
        if len(subset) < 50:
            continue

        fitted_models[sec_type] = {}
        for model_name, formula in MODEL_FORMULAS.items():
            model = smf.ols(formula=formula, data=subset).fit()
            fitted_models[sec_type][model_name] = model
            summary_rows.append(
                {
                    "Type_of_Security": sec_type,
                    "model": model_name,
                    "formula": formula,
                    "nobs": int(model.nobs),
                    "rsquared": float(model.rsquared),
                    "adj_rsquared": float(model.rsquared_adj),
                    "aic": float(model.aic),
                    "bic": float(model.bic),
                }
            )

    return fitted_models, pd.DataFrame(summary_rows)


def build_scurve_predictions(
    monthly: pd.DataFrame,
    fitted_models: dict[str, dict[str, object]],
    security_types: list[str],
    spread_grid: np.ndarray,
) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []

    for sec_type in security_types:
        if sec_type not in fitted_models or "C" not in fitted_models[sec_type]:
            continue

        subset = monthly[monthly["Type_of_Security"] == sec_type].dropna(subset=["duration", "log_UPB_z", "spread"])
        if subset.empty:
            continue

        model = fitted_models[sec_type]["C"]
        for spread_value in spread_grid:
            pred_data = subset[["duration", "log_UPB_z"]].copy()
            pred_data["spread"] = float(spread_value)
            pred_logit = model.predict(pred_data)
            pred_smm = 1.0 / (1.0 + np.exp(-pred_logit))
            pred_cpr_pct = 100.0 * (1.0 - (1.0 - pred_smm) ** 12.0)
            rows.append(
                {
                    "Type_of_Security": sec_type,
                    "spread": float(spread_value),
                    "curve_family": "overall",
                    "slice_name": "all",
                    "n_obs": int(len(subset)),
                    "pred_logit": float(np.mean(pred_logit)),
                    "pred_smm": float(np.mean(pred_smm)),
                    "pred_cpr_pct": float(np.mean(pred_cpr_pct)),
                }
            )

    return pd.DataFrame(rows)


def assign_duration_slices(subset: pd.DataFrame) -> pd.Series:
    q1 = subset["duration"].quantile(1.0 / 3.0)
    q2 = subset["duration"].quantile(2.0 / 3.0)
    labels = np.where(
        subset["duration"] <= q1,
        "low_duration",
        np.where(subset["duration"] <= q2, "median_duration", "high_duration"),
    )
    return pd.Series(labels, index=subset.index, dtype="object")


def build_duration_slice_scurves(
    monthly: pd.DataFrame,
    fitted_models: dict[str, dict[str, object]],
    security_types: list[str],
    spread_grid: np.ndarray,
) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []

    for sec_type in security_types:
        model = fitted_models.get(sec_type, {}).get("C")
        if model is None:
            continue

        subset = monthly[monthly["Type_of_Security"] == sec_type].dropna(subset=["duration", "log_UPB_z", "spread"]).copy()
        if subset.empty:
            continue

        subset["duration_slice"] = assign_duration_slices(subset)

        for slice_name, slice_df in subset.groupby("duration_slice"):
            if slice_df.empty:
                continue
            for spread_value in spread_grid:
                pred_data = slice_df[["duration", "log_UPB_z"]].copy()
                pred_data["spread"] = float(spread_value)
                pred_logit = model.predict(pred_data)
                pred_smm = 1.0 / (1.0 + np.exp(-pred_logit))
                pred_cpr_pct = 100.0 * (1.0 - (1.0 - pred_smm) ** 12.0)
                rows.append(
                    {
                        "Type_of_Security": sec_type,
                        "spread": float(spread_value),
                        "curve_family": "duration_slice",
                        "slice_name": str(slice_name),
                        "n_obs": int(len(slice_df)),
                        "pred_logit": float(np.mean(pred_logit)),
                        "pred_smm": float(np.mean(pred_smm)),
                        "pred_cpr_pct": float(np.mean(pred_cpr_pct)),
                    }
                )

    return pd.DataFrame(rows)


def build_historical_cpr(monthly: pd.DataFrame, security_types: list[str]) -> pd.DataFrame:
    history = monthly[monthly["Type_of_Security"].isin(security_types)].copy()
    history = history.sort_values(["Type_of_Security", "period"])
    return history[
        [
            "period",
            "Date",
            "Type_of_Security",
            "spread",
            "SMM_monthly",
            "Cumulative_SMM",
            "actual_cpr_pct",
            "reported_cpr_pct",
            "log_UPB",
            "log_UPB_z",
            "coupon",
            "duration",
            "Cohort_Current_UPB",
            "n_raw_rows",
        ]
    ]


def build_curve_comparison(
    monthly: pd.DataFrame,
    fitted_models: dict[str, dict[str, object]],
    security_types: list[str],
    n_bins: int = 12,
) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []

    for sec_type in security_types:
        model = fitted_models.get(sec_type, {}).get("C")
        if model is None:
            continue

        subset = monthly[monthly["Type_of_Security"] == sec_type].dropna(
            subset=["spread", "duration", "log_UPB_z", "SMM_monthly"]
        ).copy()
        if subset.empty:
            continue

        pred_logit = model.predict(subset[["spread", "duration", "log_UPB_z"]])
        subset["pred_smm"] = 1.0 / (1.0 + np.exp(-pred_logit))
        subset["pred_cpr_pct"] = 100.0 * (1.0 - (1.0 - subset["pred_smm"]) ** 12.0)
        subset["actual_cpr_pct"] = 100.0 * (1.0 - (1.0 - subset["SMM_monthly"]) ** 12.0)

        try:
            subset["spread_bin"] = pd.qcut(subset["spread"], q=n_bins, duplicates="drop")
        except ValueError:
            continue

        binned = (
            subset.groupby("spread_bin", observed=False)
            .agg(
                spread=("spread", "mean"),
                actual_cpr_pct=("actual_cpr_pct", "mean"),
                pred_cpr_pct=("pred_cpr_pct", "mean"),
                actual_cpr_std=("actual_cpr_pct", "std"),
                pred_cpr_std=("pred_cpr_pct", "std"),
                n_obs=("spread", "size"),
            )
            .dropna(subset=["spread", "actual_cpr_pct", "pred_cpr_pct"])
            .reset_index(drop=True)
            .sort_values("spread")
        )
        if binned.empty:
            continue

        binned["actual_cpr_se"] = binned["actual_cpr_std"] / np.sqrt(binned["n_obs"].clip(lower=1))
        binned["pred_cpr_se"] = binned["pred_cpr_std"] / np.sqrt(binned["n_obs"].clip(lower=1))
        binned["actual_cpr_ci95"] = 1.96 * binned["actual_cpr_se"].fillna(0.0)
        binned["pred_cpr_ci95"] = 1.96 * binned["pred_cpr_se"].fillna(0.0)

        for _, row in binned.iterrows():
            rows.append(
                {
                    "Type_of_Security": sec_type,
                    "spread": float(row["spread"]),
                    "actual_cpr_pct": float(row["actual_cpr_pct"]),
                    "pred_cpr_pct": float(row["pred_cpr_pct"]),
                    "actual_cpr_std": float(row["actual_cpr_std"]) if pd.notna(row["actual_cpr_std"]) else np.nan,
                    "pred_cpr_std": float(row["pred_cpr_std"]) if pd.notna(row["pred_cpr_std"]) else np.nan,
                    "actual_cpr_se": float(row["actual_cpr_se"]) if pd.notna(row["actual_cpr_se"]) else np.nan,
                    "pred_cpr_se": float(row["pred_cpr_se"]) if pd.notna(row["pred_cpr_se"]) else np.nan,
                    "actual_cpr_ci95": float(row["actual_cpr_ci95"]),
                    "pred_cpr_ci95": float(row["pred_cpr_ci95"]),
                    "n_obs": int(row["n_obs"]),
                }
            )

    return pd.DataFrame(rows)


def plot_predicted_cpr_curves(scurve_df: pd.DataFrame, output_dir: Path, show_plots: bool) -> None:
    fig, ax = plt.subplots(figsize=(12, 7))
    overall = scurve_df[scurve_df["curve_family"] == "overall"]
    for sec_type in overall["Type_of_Security"].drop_duplicates():
        sec_df = overall[overall["Type_of_Security"] == sec_type]
        ax.plot(sec_df["spread"], sec_df["pred_cpr_pct"], linewidth=2.5, label=sec_type)

    ax.set_title("Empirical Theoretical CPR S-Curves by Security Type")
    ax.set_xlabel("Incentive (WAC - Market Rate) in %")
    ax.set_ylabel("Average Predicted CPR (%)")
    ax.axvline(0.0, color="black", linestyle="--", alpha=0.35, label="At-The-Money")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(output_dir / "empirical_theoretical_s_curves.png", dpi=160, bbox_inches="tight")

    if show_plots:
        plt.show()
    else:
        plt.close(fig)


def plot_duration_slice_scurves(scurve_df: pd.DataFrame, output_dir: Path, show_plots: bool) -> None:
    sliced = scurve_df[scurve_df["curve_family"] == "duration_slice"]
    sec_types = sliced["Type_of_Security"].drop_duplicates().tolist()
    fig, axes = plt.subplots(nrows=len(sec_types), ncols=1, figsize=(12, 5 * max(len(sec_types), 1)))
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])

    for ax, sec_type in zip(axes, sec_types):
        sec_df = sliced[sliced["Type_of_Security"] == sec_type]
        for slice_name in ["low_duration", "median_duration", "high_duration"]:
            slice_df = sec_df[sec_df["slice_name"] == slice_name]
            if slice_df.empty:
                continue
            ax.plot(slice_df["spread"], slice_df["pred_cpr_pct"], linewidth=2.2, label=slice_name)

        ax.set_title(f"{sec_type} - Duration Slice Curves")
        ax.set_xlabel("Incentive (WAC - Market Rate) in %")
        ax.set_ylabel("Average Predicted CPR (%)")
        ax.axvline(0.0, color="black", linestyle="--", alpha=0.35)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper left")

    fig.suptitle("Heterogeneity-Aware Empirical Theoretical S-Curves", fontsize=16)
    fig.tight_layout()
    fig.savefig(output_dir / "duration_slice_empirical_theoretical_s_curves.png", dpi=160, bbox_inches="tight")

    if show_plots:
        plt.show()
    else:
        plt.close(fig)


def plot_calibration_curves(curve_df: pd.DataFrame, output_dir: Path, show_plots: bool) -> None:
    sec_types = curve_df["Type_of_Security"].drop_duplicates().tolist()
    fig, axes = plt.subplots(nrows=len(sec_types), ncols=1, figsize=(12, 5 * max(len(sec_types), 1)))
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])

    for ax, sec_type in zip(axes, sec_types):
        sec_df = curve_df[curve_df["Type_of_Security"] == sec_type].sort_values("spread")

        ax.errorbar(
            sec_df["spread"],
            sec_df["actual_cpr_pct"],
            yerr=sec_df["actual_cpr_ci95"],
            fmt="o",
            color="#1f77b4",
            ecolor="#1f77b4",
            elinewidth=1.2,
            capsize=3,
            label="Actual bin mean",
        )
        ax.plot(
            sec_df["spread"],
            sec_df["pred_cpr_pct"],
            color="#d62728",
            linewidth=2.2,
            marker="o",
            label="Predicted bin mean",
        )

        for _, row in sec_df.iterrows():
            ax.annotate(
                f"n={int(row['n_obs'])}",
                (row["spread"], row["actual_cpr_pct"]),
                textcoords="offset points",
                xytext=(0, 7),
                ha="center",
                fontsize=8,
                color="#444444",
            )

        ax.set_title(sec_type)
        ax.set_xlabel("Incentive (WAC - Market Rate) in %")
        ax.set_ylabel("CPR (%)")
        ax.axvline(0.0, color="black", linestyle="--", alpha=0.35)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper left")

    fig.suptitle("Bin-Level CPR Calibration by Security Type", fontsize=16)
    fig.tight_layout()
    fig.savefig(output_dir / "bin_level_cpr_calibration.png", dpi=160, bbox_inches="tight")

    if show_plots:
        plt.show()
    else:
        plt.close(fig)


def print_model_results(
    fitted_models: dict[str, dict[str, object]],
    summary_df: pd.DataFrame,
    security_types: list[str],
    history_df: pd.DataFrame,
) -> None:
    print("\nSelected Top Security Types for Modeling:")
    for sec_type in security_types:
        print(f"- {sec_type}")

    print("\n=== Why the x-axis is 'Incentive (WAC - Market Rate)' ===")
    print(
        "WAC is the borrower coupon on the existing mortgage pool. "
        "Market Rate is the current refinance rate proxy for the same maturity. "
        "When WAC - Market Rate is positive, borrowers have more incentive to refinance, "
        "so expected prepayment CPR should usually rise."
    )

    print("\n=== Model Comparison Summary ===")
    if summary_df.empty:
        print("No models were fitted.")
    else:
        print(summary_df.to_string(index=False, float_format=lambda x: f"{x:,.6f}"))

    cumulative_cols = ["Type_of_Security", "period", "Cumulative_SMM", "SMM_monthly"]
    cumulative_view = history_df[cumulative_cols].dropna(subset=["Cumulative_SMM", "SMM_monthly"]).copy()
    print("\n=== Cumulative SMM Sample ===")
    if cumulative_view.empty:
        print("No cumulative SMM rows available.")
    else:
        print(cumulative_view.head(12).to_string(index=False))

    print("\n=== Cumulative SMM Summary ===")
    if cumulative_view.empty:
        print("No cumulative SMM summary available.")
    else:
        cumulative_summary = (
            cumulative_view.groupby("Type_of_Security")[["Cumulative_SMM", "SMM_monthly"]]
            .describe()
            .round(6)
        )
        print(cumulative_summary.to_string())

    standard_cols = ["spread", "coupon", "log_UPB", "log_UPB_z"]
    standard_view = history_df[["Type_of_Security", *standard_cols]].dropna().copy()
    print("\n=== Standardized Column Summary ===")
    if standard_view.empty:
        print("No standardized-column rows available.")
    else:
        standard_summary = standard_view.groupby("Type_of_Security")[standard_cols].agg(["mean", "std", "min", "max"])
        print(standard_summary.round(6).to_string())

    for sec_type in security_types:
        if sec_type not in fitted_models:
            continue
        for model_name in ["A", "B", "C"]:
            if model_name not in fitted_models[sec_type]:
                continue
            print(f"\n=== OLS Regression Results: {sec_type} | Model {model_name} ===")
            print(fitted_models[sec_type][model_name].summary())


def save_outputs(
    output_dir: Path,
    monthly: pd.DataFrame,
    history_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    scurve_df: pd.DataFrame,
    curve_compare_df: pd.DataFrame,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    monthly.to_csv(output_dir / "monthly_modeling_panel.csv", index=False)
    history_df.to_csv(output_dir / "historical_cpr_top3.csv", index=False)
    summary_df.to_csv(output_dir / "ols_model_comparison.csv", index=False)
    scurve_df.to_csv(output_dir / "empirical_theoretical_scurve_grid.csv", index=False)
    curve_compare_df.to_csv(output_dir / "actual_vs_predicted_cpr_curve_bins.csv", index=False)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Notebook-aligned OLS/logit MBS prepayment modeling with top-3 security types."
    )
    parser.add_argument("--data-path", type=str, default="PrepayData.txt")
    parser.add_argument("--output-dir", type=str, default="outputs/logit_model")
    parser.add_argument("--show-plots", action="store_true", default=True)
    parser.add_argument("--no-show-plots", action="store_false", dest="show_plots")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    data_path = Path(args.data_path)
    output_dir = Path(args.output_dir)

    df = load_prepay_data(data_path)
    market_monthly = download_market_rates(df)
    featured = prepare_features(df, market_monthly)
    monthly = build_monthly_panel(featured)
    security_types = select_top_security_types(monthly, top_n=3)
    fitted_models, summary_df = fit_security_models(monthly, security_types)
    scurve_df = build_scurve_predictions(
        monthly=monthly,
        fitted_models=fitted_models,
        security_types=security_types,
        spread_grid=np.linspace(-2.0, 3.0, 100),
    )
    duration_slice_scurve_df = build_duration_slice_scurves(
        monthly=monthly,
        fitted_models=fitted_models,
        security_types=security_types,
        spread_grid=np.linspace(-2.0, 3.0, 100),
    )
    scurve_df = pd.concat([scurve_df, duration_slice_scurve_df], ignore_index=True)
    history_df = build_historical_cpr(monthly, security_types)
    curve_compare_df = build_curve_comparison(monthly, fitted_models, security_types)

    save_outputs(
        output_dir=output_dir,
        monthly=monthly,
        history_df=history_df,
        summary_df=summary_df,
        scurve_df=scurve_df,
        curve_compare_df=curve_compare_df,
    )
    print_model_results(fitted_models, summary_df, security_types, history_df)
    plot_predicted_cpr_curves(scurve_df, output_dir, args.show_plots)
    plot_duration_slice_scurves(scurve_df, output_dir, args.show_plots)
    plot_calibration_curves(curve_compare_df, output_dir, args.show_plots)


if __name__ == "__main__":
    main()
