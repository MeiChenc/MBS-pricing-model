from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from lifelines import CoxPHFitter


def load_and_prepare(data_path: Path) -> pd.DataFrame:
    df = pd.read_csv(data_path, sep="|", low_memory=False)

    df.columns = df.columns.str.strip().str.replace(" ", "_", regex=False).str.replace("-", "_", regex=False)

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
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.replace([np.inf, -np.inf], np.nan)

    # Requested setup
    # Define survival duration (loan age in months)
    df["Loan age"] = pd.to_numeric(df["Cohort_WA_Current_Loan_Age"], errors="coerce")

    # Logit SMM
    df["SMM"] = pd.to_numeric(df["SMM"], errors="coerce")
    epsilon = 1e-6
    df["smm_adj"] = df["SMM"].clip(lower=epsilon, upper=1 - epsilon)
    df["logit_smm"] = np.log(df["smm_adj"] / (1 - df["smm_adj"]))

    # market_rate column fallback (cohort-level proxy if not provided externally)
    if "market_rate" not in df.columns:
        df["market_rate"] = pd.to_numeric(df["WA_Net_Interest_Rate"], errors="coerce")
    else:
        df["market_rate"] = pd.to_numeric(df["market_rate"], errors="coerce")

    # Define key covariates
    df["spread"] = pd.to_numeric(df["Cohort_WA_Current_Interest_Rate"], errors="coerce") - df["market_rate"]
    df["spread2"] = df["spread"] ** 2
    df["spread3"] = df["spread"] ** 3
    df["log_UPB"] = np.log(pd.to_numeric(df["Cohort_Current_UPB"], errors="coerce") + 1)
    df["coupon"] = pd.to_numeric(df["Cohort_WA_Current_Interest_Rate"], errors="coerce")

    df["event"] = (df["SMM"].fillna(0) > 0).astype(int)

    model_cols = ["Loan age", "event", "spread", "spread2", "spread3", "log_UPB", "coupon", "logit_smm"]
    df = df.dropna(subset=model_cols).copy()

    return df


def fit_cox(df: pd.DataFrame):
    feature_cols = ["spread", "spread2", "spread3", "log_UPB", "coupon"]

    mu = df[feature_cols].mean()
    sigma = df[feature_cols].std().replace(0, 1.0)

    z = (df[feature_cols] - mu) / sigma

    model_df = pd.concat([df[["Loan age", "event"]], z], axis=1)

    # Mild penalization improves numerical stability for polynomial covariates.
    cph = CoxPHFitter(penalizer=0.01)
    cph.fit(model_df, duration_col="Loan age", event_col="event")

    beta = cph.params_.loc[feature_cols].values
    h0 = cph.baseline_cumulative_hazard_.values.flatten()

    return cph, mu, sigma, beta, h0


def risk_multiplier(x_raw: pd.Series, mu: pd.Series, sigma: pd.Series, beta: np.ndarray) -> float:
    z = (x_raw[["spread", "spread2", "spread3", "log_UPB", "coupon"]] - mu) / sigma
    return float(np.exp(np.dot(z.values, beta)))


def smm_from_survival(survival: np.ndarray) -> np.ndarray:
    s_prev = np.concatenate(([1.0], survival[:-1]))
    ratio = np.divide(survival, s_prev, out=np.ones_like(survival), where=s_prev > 0)
    smm = 1.0 - ratio
    return np.clip(smm, 0.0, 1.0)


def amortizing_cashflows(
    smm: np.ndarray,
    pass_through_coupon_annual: float,
    term_months: int,
    balance0: float = 1.0,
) -> pd.DataFrame:
    r_m = pass_through_coupon_annual / 12.0
    if r_m == 0:
        sched_pmt = balance0 / term_months
    else:
        sched_pmt = balance0 * r_m / (1.0 - (1.0 + r_m) ** (-term_months))

    n = min(len(smm), term_months)

    records = []
    bal = balance0
    for t in range(1, n + 1):
        if bal <= 1e-12:
            break

        interest = bal * r_m
        sched_prin = max(min(sched_pmt - interest, bal), 0.0)
        bal_after_sched = bal - sched_prin

        # Prepay applied to post-scheduled balance (pass-through convention simplification)
        prepay_prin = min(smm[t - 1] * bal_after_sched, bal_after_sched)

        total_prin = sched_prin + prepay_prin
        total_cf = interest + total_prin
        bal_end = bal - total_prin

        records.append(
            {
                "month": t,
                "beg_balance": bal,
                "interest": interest,
                "scheduled_principal": sched_prin,
                "prepay_principal": prepay_prin,
                "total_principal": total_prin,
                "total_cashflow": total_cf,
                "end_balance": max(bal_end, 0.0),
            }
        )

        bal = max(bal_end, 0.0)

    return pd.DataFrame(records)


def pv(cashflows: np.ndarray, rate_annual: float) -> float:
    r_m = rate_annual / 12.0
    months = np.arange(1, len(cashflows) + 1)
    return float(np.sum(cashflows / (1.0 + r_m) ** months))


def build_cashflow_from_hazard(
    h0: np.ndarray,
    x_raw: pd.Series,
    mu: pd.Series,
    sigma: pd.Series,
    beta: np.ndarray,
    pass_through_coupon_annual: float,
    term_months: int,
) -> pd.DataFrame:
    risk = np.clip(risk_multiplier(x_raw, mu, sigma, beta), 0.05, 20.0)
    h = h0 * risk
    survival = np.exp(-h)
    smm = smm_from_survival(survival)
    return amortizing_cashflows(smm=smm, pass_through_coupon_annual=pass_through_coupon_annual, term_months=term_months)


def run_attribution(
    h0: np.ndarray,
    x0: pd.Series,
    mu: pd.Series,
    sigma: pd.Series,
    beta: np.ndarray,
    y0_annual: float,
    rate_shock_bp: float,
    spread_beta_to_rate: float,
    coupon_annual: float,
    term_months: int,
):
    dr = rate_shock_bp / 10000.0

    cf0 = build_cashflow_from_hazard(
        h0=h0,
        x_raw=x0,
        mu=mu,
        sigma=sigma,
        beta=beta,
        pass_through_coupon_annual=coupon_annual,
        term_months=term_months,
    )
    p0 = pv(cf0["total_cashflow"].values, y0_annual)

    # A) Discount-only: fixed cashflow, shocked discount rate
    p_discount = pv(cf0["total_cashflow"].values, y0_annual + dr)
    dP_discount = p_discount - p0

    # B) Prepay-only: shocked prepay driver, fixed discount rate
    x_prepay = x0.copy()
    # spread is percentage-point; dr is decimal.
    x_prepay["spread"] = x_prepay["spread"] + spread_beta_to_rate * dr
    x_prepay["spread2"] = x_prepay["spread"] ** 2
    x_prepay["spread3"] = x_prepay["spread"] ** 3

    cf_prepay = build_cashflow_from_hazard(
        h0=h0,
        x_raw=x_prepay,
        mu=mu,
        sigma=sigma,
        beta=beta,
        pass_through_coupon_annual=coupon_annual,
        term_months=term_months,
    )
    p_prepay = pv(cf_prepay["total_cashflow"].values, y0_annual)
    dP_prepay = p_prepay - p0

    # C) Joint: discount + prepay shocked together
    p_joint = pv(cf_prepay["total_cashflow"].values, y0_annual + dr)
    dP_joint = p_joint - p0

    interaction = dP_joint - dP_discount - dP_prepay

    return {
        "shock_bp": rate_shock_bp,
        "P0": p0,
        "P_discount_only": p_discount,
        "P_prepay_only": p_prepay,
        "P_joint": p_joint,
        "dP_discount": dP_discount,
        "dP_prepay": dP_prepay,
        "dP_joint": dP_joint,
        "interaction": interaction,
    }


def central_diff_metrics(out: pd.DataFrame, price_col: str, shift_bp: int = 50):
    down = out.loc[out["shock_bp"] == -shift_bp, price_col]
    up = out.loc[out["shock_bp"] == shift_bp, price_col]
    base = out.loc[out["shock_bp"] == 0, price_col]
    y_shift = shift_bp / 10000.0

    if down.empty or up.empty or base.empty:
        return None

    p_rate_down = float(down.iloc[0])  # y - shift
    p_rate_up = float(up.iloc[0])      # y + shift
    p0 = float(base.iloc[0])           # y + 0

    duration = (p_rate_down - p_rate_up) / (2.0 * p0 * y_shift)
    convexity = (p_rate_down + p_rate_up - 2.0 * p0) / (p0 * y_shift * y_shift)
    return p0, p_rate_down, p_rate_up, duration, convexity


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def summarize_smm_logit_diagnostics(df: pd.DataFrame):
    s = df["smm_adj"].dropna()
    l = df["logit_smm"].dropna()
    if s.empty or l.empty:
        print("\nSMM/logit diagnostics: insufficient data")
        return
    near_zero = int((s < 1e-4).sum())
    near_one = int((s > 1 - 1e-4).sum())
    print("\nSMM/logit diagnostics")
    print(f"smm_adj min/median/max: {s.min():.8f} / {s.median():.8f} / {s.max():.8f}")
    print(f"logit_smm min/median/max: {l.min():.6f} / {l.median():.6f} / {l.max():.6f}")
    print(f"smm_adj near-0 count (<1e-4): {near_zero}")
    print(f"smm_adj near-1 count (>1-1e-4): {near_one}")


def summarize_rate_unit_diagnostics(df: pd.DataFrame):
    print("\nRate unit diagnostics")
    show_cols = [c for c in ["market_rate", "coupon", "spread", "Loan age", "log_UPB"] if c in df.columns]
    if show_cols:
        print(df[show_cols].head(5).to_string(index=False, float_format=lambda x: f"{x:,.6f}"))
    for c in ["market_rate", "coupon", "spread"]:
        if c not in df.columns:
            continue
        s = pd.to_numeric(df[c], errors="coerce").dropna()
        if s.empty:
            continue
        print(f"{c} min/median/max: {s.min():.6f} / {s.median():.6f} / {s.max():.6f}")
    print("Convention: spread is handled in percentage-point units.")


def _month_key_from_df(df: pd.DataFrame) -> pd.Series:
    if "Factor_Date" in df.columns:
        return pd.to_numeric(df["Factor_Date"], errors="coerce")
    date_num = pd.to_numeric(df.get("Date"), errors="coerce")
    dt = pd.to_datetime(date_num.astype("Int64").astype(str), format="%Y%m%d", errors="coerce")
    return dt.dt.year * 100 + dt.dt.month


def fit_logit_smm_by_security_type(df: pd.DataFrame, min_obs: int = 100):
    work = df.copy()
    if "Type_of_Security" not in work.columns:
        work["Type_of_Security"] = "ALL"
    work["Cohort_Month"] = _month_key_from_df(work)

    # cohort-month aggregation to avoid daily duplicate overweight.
    agg_spec = {
        "spread": "mean",
        "Loan age": "mean",
        "log_UPB": "mean",
        "coupon": "mean",
        "smm_adj": "mean",
        "logit_smm": "mean",
        "Cohort_Current_UPB": "sum",
    }
    macro_cols = [c for c in ["CPI", "Unemployment_Rate", "New_Home_Sales"] if c in work.columns]
    for c in macro_cols:
        agg_spec[c] = "mean"

    agg = (
        work.groupby(["Type_of_Security", "Cohort_Month"], dropna=False, as_index=False)
        .agg(agg_spec)
        .dropna(subset=["spread", "Loan age", "log_UPB", "coupon", "smm_adj", "logit_smm", "Cohort_Current_UPB"])
    )
    # Rebuild polynomial terms after aggregation for formula models.
    agg["spread2"] = agg["spread"] ** 2
    agg["spread3"] = agg["spread"] ** 3
    agg["smm_positive"] = (agg["smm_adj"] > 1e-4).astype(float)
    agg["logit_smm"] = np.log(np.clip(agg["smm_adj"], 1e-6, 1 - 1e-6) / (1 - np.clip(agg["smm_adj"], 1e-6, 1 - 1e-6)))
    agg["logit_smm_hat"] = np.nan
    agg["smm_hat"] = np.nan
    agg["cpr_hat"] = np.nan
    agg["cpr_hat_pct"] = np.nan

    if macro_cols:
        macro_terms = " + ".join(macro_cols)
        common_terms = f"spread + spread2 + spread3 + Q('Loan age') + log_UPB + coupon + {macro_terms}"
        time_effect_mode = "macro_continuous"
    else:
        common_terms = "spread + spread2 + spread3 + Q('Loan age') + log_UPB + coupon + C(Cohort_Month)"
        time_effect_mode = "factor_date_dummy"

    formula_pos = f"smm_positive ~ {common_terms}"
    formula_cond = f"logit_smm ~ {common_terms}"

    pooled_df = agg.dropna(subset=["smm_positive", "logit_smm", "Cohort_Current_UPB"]).copy()
    if pooled_df.empty:
        return pd.DataFrame(), work.assign(logit_smm_hat=np.nan, smm_hat=np.nan, cpr_hat=np.nan, cpr_hat_pct=np.nan)
    global_pos_model = smf.wls(formula_pos, data=pooled_df, weights=pooled_df["Cohort_Current_UPB"]).fit()
    pooled_pos_df = pooled_df[pooled_df["smm_positive"] > 0.5].copy()
    if len(pooled_pos_df) >= 8:
        global_cond_model = smf.wls(formula_cond, data=pooled_pos_df, weights=pooled_pos_df["Cohort_Current_UPB"]).fit()
    else:
        global_cond_model = smf.wls(formula_cond, data=pooled_df, weights=pooled_df["Cohort_Current_UPB"]).fit()

    coef_rows = []
    for sec_type, g in agg.groupby("Type_of_Security", dropna=False):
        g = g.dropna(subset=["smm_positive", "logit_smm", "Cohort_Current_UPB"]).copy()
        if g.empty:
            continue
        if len(g) >= min_obs:
            try:
                pos_model = smf.wls(formula_pos, data=g, weights=g["Cohort_Current_UPB"]).fit()
                g_pos = g[g["smm_positive"] > 0.5].copy()
                if len(g_pos) >= max(8, min_obs // 2):
                    cond_model = smf.wls(formula_cond, data=g_pos, weights=g_pos["Cohort_Current_UPB"]).fit()
                else:
                    cond_model = smf.wls(formula_cond, data=g, weights=g["Cohort_Current_UPB"]).fit()
                coef_source = "group_specific"
            except Exception:
                pos_model = global_pos_model
                cond_model = global_cond_model
                coef_source = "pooled_fallback"
        else:
            pos_model = global_pos_model
            cond_model = global_cond_model
            coef_source = "pooled_fallback"

        p_pos = np.clip(pos_model.predict(g), 1e-6, 1 - 1e-6)
        smm_cond = np.clip(sigmoid(cond_model.predict(g)), 1e-6, 1 - 1e-6)
        smm_floor = float(np.clip(g["smm_adj"].quantile(0.05), 1e-6, 0.2))
        smm_hat = np.clip((1.0 - p_pos) * smm_floor + p_pos * smm_cond, 1e-6, 1 - 1e-6)
        cpr_hat = 1.0 - (1.0 - smm_hat) ** 12

        agg.loc[g.index, "logit_smm_hat"] = np.log(smm_hat / (1.0 - smm_hat))
        agg.loc[g.index, "smm_hat"] = smm_hat
        agg.loc[g.index, "cpr_hat"] = cpr_hat
        agg.loc[g.index, "cpr_hat_pct"] = cpr_hat * 100.0

        coef_map_1 = pos_model.params.to_dict()
        coef_map_2 = cond_model.params.to_dict()
        w = pd.to_numeric(g["Cohort_Current_UPB"], errors="coerce").fillna(0.0).values.astype(float)
        w_norm = np.clip(w, 0.0, None)
        w_norm = w_norm / np.sum(w_norm) if np.sum(w_norm) > 0 else np.repeat(1.0 / len(g), len(g))

        coef_rows.append(
            {
                "Type_of_Security": sec_type,
                "coef_source": coef_source,
                "n_obs_monthly": len(g),
                "min_obs_threshold": min_obs,
                "time_effect_mode": time_effect_mode,
                "n_month_effects": int(
                    sum(str(name).startswith("C(Cohort_Month)") for name in pos_model.params.index)
                ),
                "total_upb_weight": float(np.sum(g["Cohort_Current_UPB"].values)),
                "smm_floor": smm_floor,
                "pct_positive_smm": float(np.mean(g["smm_positive"].values)),
                "mean_loan_age": float(np.sum(g["Loan age"].values * w_norm)),
                "mean_log_UPB": float(np.sum(g["log_UPB"].values * w_norm)),
                "mean_coupon": float(np.sum(g["coupon"].values * w_norm)),
                "mean_smm_hat": float(np.sum(smm_hat * w_norm)),
                "mean_cpr_hat_pct": float(np.sum((cpr_hat * 100.0) * w_norm)),
                "r2_pos": float(getattr(pos_model, "rsquared", np.nan)),
                "r2_cond": float(getattr(cond_model, "rsquared", np.nan)),
                "beta1_intercept": float(coef_map_1.get("Intercept", 0.0)),
                "beta1_spread_lin": float(coef_map_1.get("spread", 0.0)),
                "beta1_spread2": float(coef_map_1.get("spread2", 0.0)),
                "beta1_spread3": float(coef_map_1.get("spread3", 0.0)),
                "beta1_loan_age": float(coef_map_1.get("Q('Loan age')", 0.0)),
                "beta1_log_UPB": float(coef_map_1.get("log_UPB", 0.0)),
                "beta1_coupon": float(coef_map_1.get("coupon", 0.0)),
                "beta2_intercept": float(coef_map_2.get("Intercept", 0.0)),
                "beta2_spread_lin": float(coef_map_2.get("spread", 0.0)),
                "beta2_spread2": float(coef_map_2.get("spread2", 0.0)),
                "beta2_spread3": float(coef_map_2.get("spread3", 0.0)),
                "beta2_loan_age": float(coef_map_2.get("Q('Loan age')", 0.0)),
                "beta2_log_UPB": float(coef_map_2.get("log_UPB", 0.0)),
                "beta2_coupon": float(coef_map_2.get("coupon", 0.0)),
            }
        )

    coef_df = pd.DataFrame(coef_rows)
    if not coef_df.empty:
        coef_df = coef_df.sort_values("n_obs_monthly", ascending=False)

    pred_df = work.merge(
        agg[
            [
                "Type_of_Security",
                "Cohort_Month",
                "logit_smm_hat",
                "smm_hat",
                "cpr_hat",
                "cpr_hat_pct",
            ]
        ],
        on=["Type_of_Security", "Cohort_Month"],
        how="left",
    )
    return coef_df, pred_df


def safe_slug(text: str) -> str:
    allowed = []
    for ch in str(text):
        if ch.isalnum() or ch in ("-", "_"):
            allowed.append(ch)
        elif ch in (" ", "/", "\\"):
            allowed.append("_")
    slug = "".join(allowed).strip("_")
    return slug or "unknown"


def build_scurve_grid_predictions(coef_df: pd.DataFrame) -> pd.DataFrame:
    if coef_df.empty:
        return pd.DataFrame()

    # Fixed grid in percentage points.
    spread_grid = np.linspace(-2, 3, 200)
    spread2 = spread_grid ** 2
    spread3 = spread_grid ** 3

    rows = []
    for _, r in coef_df.iterrows():
        # Hold controls at group weighted means; time FE set to baseline (0).
        lp1 = (
            r["beta1_intercept"]
            + r["beta1_spread_lin"] * spread_grid
            + r.get("beta1_spread2", 0.0) * spread2
            + r.get("beta1_spread3", 0.0) * spread3
            + r["beta1_loan_age"] * r["mean_loan_age"]
            + r["beta1_log_UPB"] * r["mean_log_UPB"]
            + r["beta1_coupon"] * r["mean_coupon"]
        )
        lp2 = (
            r["beta2_intercept"]
            + r["beta2_spread_lin"] * spread_grid
            + r.get("beta2_spread2", 0.0) * spread2
            + r.get("beta2_spread3", 0.0) * spread3
            + r["beta2_loan_age"] * r["mean_loan_age"]
            + r["beta2_log_UPB"] * r["mean_log_UPB"]
            + r["beta2_coupon"] * r["mean_coupon"]
        )
        p_pos = np.clip(sigmoid(lp1), 1e-6, 1 - 1e-6)
        smm_cond = np.clip(sigmoid(lp2), 1e-6, 1 - 1e-6)
        smm_hat = np.clip((1.0 - p_pos) * r["smm_floor"] + p_pos * smm_cond, 1e-6, 1 - 1e-6)
        smm_hat = np.clip(smm_hat, 1e-6, 1 - 1e-6)
        cpr_hat = 1.0 - (1.0 - smm_hat) ** 12

        grid_df = pd.DataFrame(
            {
                "Type_of_Security": r["Type_of_Security"],
                "spread_grid_pct": spread_grid,
                "spread2": spread2,
                "spread3": spread3,
                "p_positive_hat": p_pos,
                "smm_cond_hat": smm_cond,
                "smm_hat": smm_hat,
                "cpr_hat": cpr_hat,
                "cpr_hat_pct": cpr_hat * 100.0,
            }
        )
        rows.append(grid_df)

    return pd.concat(rows, axis=0, ignore_index=True)


def save_scurve_plots(curve_df: pd.DataFrame, out_dir: Path):
    if curve_df.empty:
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    group_cols = ["Type_of_Security"]
    for keys, g in curve_df.groupby(group_cols, dropna=False):
        sec_type = keys
        g = g.sort_values("spread_grid_pct")
        plt.figure(figsize=(6, 4))
        plt.plot(g["spread_grid_pct"], g["cpr_hat_pct"], lw=2)
        plt.title(f"S-curve CPR vs Spread ({sec_type})")
        plt.xlabel("Spread (%)")
        plt.ylabel("Predicted CPR (%)")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        file_stem = f"{safe_slug(sec_type)}"
        plt.savefig(out_dir / f"scurve_{file_stem}.png", dpi=150)
        plt.close()


def _to_datetime_safe(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    return pd.to_datetime(s.astype("Int64").astype(str), format="%Y%m%d", errors="coerce")


def plot_quality_of_fit(logit_pred_df: pd.DataFrame, out_dir: Path):
    fit_df = logit_pred_df.dropna(subset=["SMM", "smm_hat"]).copy()
    if fit_df.empty:
        return

    if "Date" in fit_df.columns:
        fit_df["Date_dt"] = _to_datetime_safe(fit_df["Date"])
    else:
        fit_df["Date_dt"] = pd.NaT

    # 1) Time-series comparison (actual vs predicted SMM)
    ts = fit_df.dropna(subset=["Date_dt"]).groupby("Date_dt", as_index=False)[["SMM", "smm_hat"]].mean()
    if not ts.empty:
        plt.figure(figsize=(8, 4))
        plt.plot(ts["Date_dt"], ts["SMM"], label="Actual SMM", lw=1.8)
        plt.plot(ts["Date_dt"], ts["smm_hat"], label="Predicted SMM", lw=1.8)
        plt.title("Quality of Fit: Actual vs Predicted SMM (Time Series)")
        plt.xlabel("Date")
        plt.ylabel("SMM")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_dir / "fit_quality_timeseries_smm.png", dpi=150)
        plt.close()

    # 2) Scatter + residual diagnostics
    fit_df["residual_smm"] = fit_df["SMM"] - fit_df["smm_hat"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].scatter(fit_df["SMM"], fit_df["smm_hat"], s=8, alpha=0.25)
    axes[0].set_title("Actual vs Predicted SMM")
    axes[0].set_xlabel("Actual SMM")
    axes[0].set_ylabel("Predicted SMM")
    axes[0].grid(True, alpha=0.3)

    axes[1].scatter(fit_df["Loan age"], fit_df["residual_smm"], s=8, alpha=0.25)
    axes[1].axhline(0.0, color="k", lw=1)
    axes[1].set_title("Residual vs Loan age")
    axes[1].set_xlabel("Loan age")
    axes[1].set_ylabel("Residual (SMM - SMM_hat)")
    axes[1].grid(True, alpha=0.3)

    axes[2].scatter(fit_df["log_UPB"], fit_df["residual_smm"], s=8, alpha=0.25)
    axes[2].axhline(0.0, color="k", lw=1)
    axes[2].set_title("Residual vs log_UPB")
    axes[2].set_xlabel("log_UPB")
    axes[2].set_ylabel("Residual (SMM - SMM_hat)")
    axes[2].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "fit_quality_scatter_residuals.png", dpi=150)
    plt.close()


def plot_hazard_survival(h0_cum: np.ndarray, x0: pd.Series, mu: pd.Series, sigma: pd.Series, beta: np.ndarray, out_dir: Path):
    risk = np.clip(risk_multiplier(x0, mu, sigma, beta), 0.05, 20.0)
    h_cum = np.clip(h0_cum * risk, 0.0, None)
    lambda_monthly = np.diff(np.concatenate(([0.0], h_cum)))
    lambda_monthly = np.clip(lambda_monthly, 0.0, None)
    survival = np.exp(-h_cum)

    m = np.arange(1, len(h_cum) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(m, lambda_monthly, lw=1.8)
    axes[0].set_title("Hazard Curve (lambda_t)")
    axes[0].set_xlabel("Month index")
    axes[0].set_ylabel("Monthly intensity")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(m, survival, lw=1.8)
    axes[1].set_title("Survival Curve S(t)")
    axes[1].set_xlabel("Month index")
    axes[1].set_ylabel("Survival probability")
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "hazard_survival_curves.png", dpi=150)
    plt.close()


def plot_price_yield_negative_convexity(
    h0: np.ndarray,
    x0: pd.Series,
    mu: pd.Series,
    sigma: pd.Series,
    beta: np.ndarray,
    base_yield: float,
    spread_beta_to_rate: float,
    coupon_annual: float,
    term_months: int,
    out_dir: Path,
):
    y_grid = np.linspace(base_yield - 0.03, base_yield + 0.03, 61)

    # Standard amortizing bond benchmark (no voluntary prepayment option)
    cf_bond = amortizing_cashflows(
        smm=np.zeros(term_months),
        pass_through_coupon_annual=coupon_annual,
        term_months=term_months,
        balance0=1.0,
    )

    p_mbs = []
    p_bond = []
    for y in y_grid:
        dr = y - base_yield
        x = x0.copy()
        # spread in model is percentage-point; dr is decimal.
        x["spread"] = x0["spread"] + spread_beta_to_rate * dr
        x["spread2"] = x["spread"] ** 2
        x["spread3"] = x["spread"] ** 3
        cf_mbs = build_cashflow_from_hazard(
            h0=h0,
            x_raw=x,
            mu=mu,
            sigma=sigma,
            beta=beta,
            pass_through_coupon_annual=coupon_annual,
            term_months=term_months,
        )
        p_mbs.append(pv(cf_mbs["total_cashflow"].values, y))
        p_bond.append(pv(cf_bond["total_cashflow"].values, y))

    curve_df = pd.DataFrame({"yield_pct": y_grid * 100.0, "price_mbs": p_mbs, "price_bond_no_prepay": p_bond})
    curve_df.to_csv(out_dir / "price_yield_curves.csv", index=False)

    plt.figure(figsize=(7, 4))
    plt.plot(curve_df["yield_pct"], curve_df["price_mbs"], label="MBS (rate-prepay feedback)", lw=2)
    plt.plot(curve_df["yield_pct"], curve_df["price_bond_no_prepay"], label="Amortizing Bond (no prepay)", lw=2)
    plt.title("Price-Yield: MBS vs No-Prepay Benchmark")
    plt.xlabel("Yield (%)")
    plt.ylabel("Price (par=1)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "price_yield_negative_convexity.png", dpi=150)
    plt.close()


def plot_cashflow_breakdown(
    h0: np.ndarray,
    x0: pd.Series,
    mu: pd.Series,
    sigma: pd.Series,
    beta: np.ndarray,
    coupon_annual: float,
    term_months: int,
    out_dir: Path,
):
    cf = build_cashflow_from_hazard(
        h0=h0,
        x_raw=x0,
        mu=mu,
        sigma=sigma,
        beta=beta,
        pass_through_coupon_annual=coupon_annual,
        term_months=term_months,
    )
    if cf.empty:
        return
    cf.to_csv(out_dir / "baseline_cashflow_breakdown.csv", index=False)

    x = cf["month"].values
    y_interest = cf["interest"].values
    y_sched = cf["scheduled_principal"].values
    y_prepay = cf["prepay_principal"].values

    plt.figure(figsize=(8, 4.5))
    plt.stackplot(
        x,
        y_interest,
        y_sched,
        y_prepay,
        labels=["Interest", "Scheduled Principal", "Prepay Principal"],
        alpha=0.9,
    )
    plt.title("Cash Flow Breakdown (Pass-through)")
    plt.xlabel("Month")
    plt.ylabel("Cash flow amount (par=1)")
    plt.legend(loc="upper right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "cashflow_breakdown_stack.png", dpi=150)
    plt.close()


def plot_scenario_sensitivity_matrix(out: pd.DataFrame, out_dir: Path):
    if out.empty:
        return
    m = out.copy()
    p0 = m["P0"].replace(0, np.nan)
    m["discount_only_pct"] = 100.0 * m["dP_discount"] / p0
    m["prepay_only_pct"] = 100.0 * m["dP_prepay"] / p0
    m["joint_pct"] = 100.0 * m["dP_joint"] / p0

    matrix = (
        m.set_index("shock_bp")[["discount_only_pct", "prepay_only_pct", "joint_pct"]]
        .sort_index()
        .T
    )
    matrix.to_csv(out_dir / "scenario_sensitivity_matrix_pct.csv")

    fig, ax = plt.subplots(figsize=(8, 3.5))
    im = ax.imshow(matrix.values, aspect="auto")
    ax.set_xticks(np.arange(matrix.shape[1]))
    ax.set_xticklabels(matrix.columns.astype(int))
    ax.set_yticks(np.arange(matrix.shape[0]))
    ax.set_yticklabels(["Discount-only", "Prepay-only", "Full (Joint)"])
    ax.set_xlabel("Rate shock (bp)")
    ax.set_title("Scenario Sensitivity Matrix (% Price Change)")
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("%")
    plt.tight_layout()
    plt.savefig(out_dir / "scenario_sensitivity_heatmap.png", dpi=150)
    plt.close()


def main():
    data_path = Path(__file__).resolve().parent / "PrepayData.txt"
    df = load_and_prepare(data_path)

    cph, mu, sigma, beta, h0 = fit_cox(df)

    x0 = df[["spread", "spread2", "spread3", "log_UPB", "coupon"]].mean()

    # Baseline pricing setup
    y0_annual = 0.045
    coupon_annual = 0.05
    term_months = int(np.clip(df["Cohort_WA_Current_Remaining_Months_to_Maturity"].median(), 60, 360))

    # Simple mapping from rate shock to prepay incentive proxy (spread)
    spread_beta_to_rate = -1.0

    scenarios = []
    for shock_bp in [-100, -50, -25, 0, 25, 50, 100]:
        scenarios.append(
            run_attribution(
                h0=h0,
                x0=x0,
                mu=mu,
                sigma=sigma,
                beta=beta,
                y0_annual=y0_annual,
                rate_shock_bp=shock_bp,
                spread_beta_to_rate=spread_beta_to_rate,
                coupon_annual=coupon_annual,
                term_months=term_months,
            )
        )

    out = pd.DataFrame(scenarios)
    logit_coef_df, logit_pred_df = fit_logit_smm_by_security_type(df, min_obs=100)
    scurve_df = build_scurve_grid_predictions(logit_coef_df)
    out_dir = Path(__file__).resolve().parent
    fig_dir = out_dir / "outputs"
    fig_dir.mkdir(parents=True, exist_ok=True)

    print("\nModel fit summary")
    print(f"Observations: {len(df)}")
    print(f"Concordance Index: {cph.concordance_index_:.3f}")
    print(f"Median remaining term used for CF engine: {term_months} months")
    print(f"Baseline discount rate: {y0_annual:.2%}")
    print(f"Baseline pass-through coupon: {coupon_annual:.2%}")
    print(f"Spread-to-rate shock beta: {spread_beta_to_rate:+.2f}")

    print("\nRate vs Prepay Attribution")
    print(out.to_string(index=False, float_format=lambda x: f"{x:,.6f}"))
    summarize_smm_logit_diagnostics(df)
    summarize_rate_unit_diagnostics(df)
    if logit_coef_df.empty:
        print("\nWarning: no group reached min_obs for Logit-SMM fit; coefficients table is empty.")

    if not logit_coef_df.empty:
        print("\nPer-Type two-part SMM model (Type grouping + Factor_Date time effects)")
        print(
            logit_coef_df[
                [
                    "Type_of_Security",
                    "coef_source",
                    "n_obs_monthly",
                    "min_obs_threshold",
                    "n_month_effects",
                    "time_effect_mode",
                    "total_upb_weight",
                    "pct_positive_smm",
                    "smm_floor",
                    "beta1_spread_lin",
                    "beta2_spread_lin",
                    "mean_smm_hat",
                    "mean_cpr_hat_pct",
                ]
            ]
            .head(10)
            .to_string(index=False, float_format=lambda x: f"{x:,.6f}")
        )

        # Save full model outputs for research workflow.
        logit_coef_df.to_csv(out_dir / "security_type_logit_smm_coefficients.csv", index=False)
        pred_cols = [
            "Type_of_Security",
            "Cohort_Month",
            "Date",
            "spread",
            "logit_smm",
            "logit_smm_hat",
            "smm_hat",
            "cpr_hat",
            "cpr_hat_pct",
        ]
        pred_cols = [c for c in pred_cols if c in logit_pred_df.columns]
        logit_pred_df[pred_cols].to_csv(out_dir / "security_type_logit_smm_predictions.csv", index=False)

    if not scurve_df.empty:
        scurve_df.to_csv(out_dir / "security_type_scurve_grid_predictions.csv", index=False)
        save_scurve_plots(scurve_df, out_dir=fig_dir)
        print("\nS-curve outputs written:")
        print(f"- {out_dir / 'security_type_scurve_grid_predictions.csv'}")
        print(f"- {fig_dir} (per-security CPR-vs-spread PNGs)")

    # Visual outputs requested in assignment
    plot_quality_of_fit(logit_pred_df, out_dir=fig_dir)
    plot_hazard_survival(h0_cum=h0, x0=x0, mu=mu, sigma=sigma, beta=beta, out_dir=fig_dir)
    plot_price_yield_negative_convexity(
        h0=h0,
        x0=x0,
        mu=mu,
        sigma=sigma,
        beta=beta,
        base_yield=y0_annual,
        spread_beta_to_rate=spread_beta_to_rate,
        coupon_annual=coupon_annual,
        term_months=term_months,
        out_dir=fig_dir,
    )
    plot_cashflow_breakdown(
        h0=h0,
        x0=x0,
        mu=mu,
        sigma=sigma,
        beta=beta,
        coupon_annual=coupon_annual,
        term_months=term_months,
        out_dir=fig_dir,
    )
    plot_scenario_sensitivity_matrix(out, out_dir=fig_dir)
    print("\nAdditional visual outputs written to:")
    print(f"- {fig_dir}")

    discount_metrics = central_diff_metrics(out, price_col="P_discount_only", shift_bp=50)
    joint_metrics = central_diff_metrics(out, price_col="P_joint", shift_bp=50)

    if discount_metrics is not None:
        p0_d, p_rate_down_d, p_rate_up_d, duration_d, convexity_d = discount_metrics
        print("\nDiscount-only sensitivity (central difference, +/-50bp around 0bp)")
        print(f"p0          (y +   0bp): {p0_d:.6f}")
        print(f"p_rate_down (y -  50bp): {p_rate_down_d:.6f}")
        print(f"p_rate_up   (y +  50bp): {p_rate_up_d:.6f}")
        print(f"duration_discount_only: {duration_d:.6f}")
        print(f"convexity_discount_only: {convexity_d:.6f}")

    if joint_metrics is not None:
        p0_j, p_rate_down_j, p_rate_up_j, duration_j, convexity_j = joint_metrics
        print("\nFull MBS sensitivity (joint rate+prepay, central difference, +/-50bp around 0bp)")
        print(f"p0          (y +   0bp): {p0_j:.6f}")
        print(f"p_rate_down (y -  50bp): {p_rate_down_j:.6f}")
        print(f"p_rate_up   (y +  50bp): {p_rate_up_j:.6f}")
        print(f"effective_duration_full: {duration_j:.6f}")
        print(f"effective_convexity_full: {convexity_j:.6f}")


if __name__ == "__main__":
    main()
