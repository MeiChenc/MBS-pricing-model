from pathlib import Path

import numpy as np
import pandas as pd
try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    plt = None


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

    # Standard logit target: ln(SMM/(1-SMM))
    df["SMM"] = pd.to_numeric(df["SMM"], errors="coerce")
    epsilon = 1e-6
    df["smm_adj"] = df["SMM"].clip(lower=epsilon, upper=1 - epsilon)
    df["logit_smm_target"] = np.log(df["smm_adj"] / (1 - df["smm_adj"]))

    # market_rate column fallback (cohort-level proxy if not provided externally)
    if "market_rate" not in df.columns:
        df["market_rate"] = pd.to_numeric(df["WA_Net_Interest_Rate"], errors="coerce")
    else:
        df["market_rate"] = pd.to_numeric(df["market_rate"], errors="coerce")

    # Define key covariates
    df["spread"] = pd.to_numeric(df["Cohort_WA_Current_Interest_Rate"], errors="coerce") - df["market_rate"]
    df["spread2"] = df["spread"] ** 2
    df["spread3"] = df["spread"] ** 3
    # Freddie Mac Attribute 4: Cohort Current UPB -> log_UPB = ln(UPB), clipped for numerical stability.
    upb = pd.to_numeric(df["Cohort_Current_UPB"], errors="coerce")
    df["log_UPB"] = np.log(upb.clip(lower=1))
    df["coupon"] = pd.to_numeric(df["Cohort_WA_Current_Interest_Rate"], errors="coerce")

    model_cols = ["Loan age", "spread", "spread2", "spread3", "log_UPB", "coupon", "logit_smm_target", "smm_adj"]
    df = df.dropna(subset=model_cols).copy()

    return df


def fit_cox(df: pd.DataFrame):
    feature_cols = ["spread", "spread2", "spread3", "log_UPB", "coupon"]

    mu = df[feature_cols].mean()
    sigma = df[feature_cols].std().replace(0, 1.0)
    z = (df[feature_cols] - mu) / sigma

    # Cohort-level "Cox-like" intensity fit:
    # log(SMM_t) = alpha(age_t) + z_t * beta + eps_t, weighted by UPB.
    age = pd.to_numeric(df["Loan age"], errors="coerce").round().astype(int)
    age = age.clip(lower=1)
    y = np.log(np.clip(pd.to_numeric(df["smm_adj"], errors="coerce").values.astype(float), 1e-6, 1 - 1e-6))
    X = z.values.astype(float)
    w = pd.to_numeric(df["Cohort_Current_UPB"], errors="coerce").values.astype(float)
    w = np.where(np.isfinite(w) & (w > 0), w, np.nan)
    if np.all(np.isnan(w)):
        w = np.ones(len(df), dtype=float)
    else:
        median_w = float(np.nanmedian(w))
        if not np.isfinite(median_w) or median_w <= 0:
            median_w = 1.0
        w = np.where(np.isnan(w), 1.0, w / median_w)
        w = np.clip(w, 1e-6, 1e6)

    beta = np.zeros(X.shape[1], dtype=float)
    for _ in range(8):
        resid = y - X @ beta
        age_df = pd.DataFrame({"age": age.values, "resid": resid, "w": w})
        alpha_by_age = age_df.groupby("age").apply(lambda g: np.average(g["resid"], weights=g["w"]))
        alpha_vec = age.map(alpha_by_age).values.astype(float)
        y_tilde = y - alpha_vec
        sqrt_w = np.sqrt(w)
        beta, _, _, _ = np.linalg.lstsq(X * sqrt_w[:, None], y_tilde * sqrt_w, rcond=None)

    resid = y - X @ beta
    age_df = pd.DataFrame({"age": age.values, "resid": resid, "w": w})
    alpha_by_age = age_df.groupby("age").apply(lambda g: np.average(g["resid"], weights=g["w"]))
    max_age = int(age.max())
    alpha_full = alpha_by_age.reindex(range(1, max_age + 1)).ffill().bfill().fillna(0.0).values
    lambda0 = np.clip(np.exp(alpha_full), 1e-8, 0.95)
    h0 = np.cumsum(lambda0)

    y_hat = age.map(alpha_by_age).values.astype(float) + X @ beta
    mse_w = float(np.average((y - y_hat) ** 2, weights=w))
    model_diag = {"weighted_rmse_log_smm": float(np.sqrt(mse_w))}
    return model_diag, mu, sigma, beta, h0


def risk_multiplier(x_raw: pd.Series, mu: pd.Series, sigma: pd.Series, beta: np.ndarray) -> float:
    z = (x_raw[["spread", "spread2", "spread3", "log_UPB", "coupon"]] - mu) / sigma
    return float(np.exp(np.dot(z.values, beta)))


def smm_from_survival(survival: np.ndarray) -> np.ndarray:
    s_prev = np.concatenate(([1.0], survival[:-1]))
    ratio = np.divide(survival, s_prev, out=np.ones_like(survival), where=s_prev > 0) #survival ratio
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


def build_top3_security_monthly_smm(df: pd.DataFrame) -> pd.DataFrame:
    monthly = df.copy()
    if "Date" not in monthly.columns:
        return pd.DataFrame(columns=["Type_of_Security", "period", "Date", "Cumulative_SMM", "SMM_monthly"])

    if "Type_of_Security" not in monthly.columns:
        monthly["Type_of_Security"] = "ALL"

    monthly["Date_dt"] = _to_datetime_safe(monthly["Date"])
    monthly = monthly.dropna(subset=["Date_dt", "Type_of_Security"]).copy()
    if monthly.empty:
        return pd.DataFrame(columns=["Type_of_Security", "period", "Date", "SMM_monthly"])

    monthly["period"] = monthly["Date_dt"].dt.to_period("M")
    monthly["Year"] = monthly["Date_dt"].dt.year
    monthly = (
        monthly.sort_values("Date_dt")
        .groupby(["period", "Type_of_Security", "Year"], as_index=False)
        .last()
    )

    # Keep only the top 3 security types by available monthly observations.
    top_types = monthly["Type_of_Security"].value_counts().nlargest(3).index.tolist()
    monthly = monthly[monthly["Type_of_Security"].isin(top_types)].copy()

    if "Cumulative_SMM" in monthly.columns:
        monthly["SMM_monthly"] = monthly["Cumulative_SMM"]
    elif "SMM" in monthly.columns:
        monthly["SMM_monthly"] = monthly["SMM"]
    else:
        monthly["SMM_monthly"] = np.nan
    return monthly[["Type_of_Security", "period", "Date", "SMM_monthly"]].sort_values(
        ["Type_of_Security", "period"]
    )


def fit_logit_smm_by_security_type(df: pd.DataFrame, min_obs: int = 1000):
    if "Type_of_Security" not in df.columns:
        work = df.copy()
        work["Type_of_Security"] = "ALL"
    else:
        work = df.copy()

    pred_df = work.copy()
    pred_df["coef_source"] = np.nan
    pred_df["logit_smm_hat"] = np.nan
    pred_df["smm_hat"] = np.nan
    pred_df["cpr_hat"] = np.nan
    pred_df["cpr_hat_pct"] = np.nan

    fit_cols = ["spread", "spread2", "spread3", "Loan age", "log_UPB", "logit_smm_target", "Cohort_Current_UPB"]

    def _stable_upb_weights(series: pd.Series) -> np.ndarray:
        # Rescale UPB by median so sqrt(weights) stays numerically stable while preserving relative weighting.
        raw = pd.to_numeric(series, errors="coerce").values.astype(float)
        raw = np.where(np.isfinite(raw) & (raw > 0), raw, np.nan)
        if np.all(np.isnan(raw)):
            return np.ones(len(series), dtype=float)
        median_raw = float(np.nanmedian(raw))
        if not np.isfinite(median_raw) or median_raw <= 0:
            median_raw = 1.0
        scaled = np.where(np.isnan(raw), 1.0, raw / median_raw)
        return np.clip(scaled, 1e-6, 1e6)

    def _fit_wls_beta(g_fit: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        w_local = _stable_upb_weights(g_fit["Cohort_Current_UPB"])
        X_local = np.column_stack(
            [
                np.ones(len(g_fit)),
                g_fit["spread"].values,
                g_fit["spread2"].values,
                g_fit["spread3"].values,
                g_fit["Loan age"].values,
                g_fit["log_UPB"].values,
            ]
        )
        y_local = g_fit["logit_smm_target"].values
        sqrt_w = np.sqrt(w_local)
        beta_local, _, _, _ = np.linalg.lstsq(X_local * sqrt_w[:, None], y_local * sqrt_w, rcond=None)
        return beta_local, w_local

    pooled = work.dropna(subset=fit_cols).copy()
    if pooled.empty:
        return pd.DataFrame(), pred_df
    pooled_beta, _ = _fit_wls_beta(pooled)

    coef_rows = []
    for sec_type, g in work.groupby("Type_of_Security"):
        g = g.dropna(subset=fit_cols).copy()
        if g.empty:
            continue

        if len(g) >= min_obs:
            try:
                beta = _fit_wls_beta(g)[0]
                coef_source = "group_specific"
            except Exception:
                beta = pooled_beta
                coef_source = "pooled_fallback"
        else:
            beta = pooled_beta
            coef_source = "pooled_fallback"

        X = np.column_stack(
            [
                np.ones(len(g)),
                g["spread"].values,
                g["spread2"].values,
                g["spread3"].values,
                g["Loan age"].values,
                g["log_UPB"].values,
            ]
        )
        y_hat = X @ beta
        smm_hat = np.clip(sigmoid(y_hat), 1e-6, 1 - 1e-6)
        cpr_hat = 1.0 - (1.0 - smm_hat) ** 12

        pred_df.loc[g.index, "coef_source"] = coef_source
        pred_df.loc[g.index, "logit_smm_hat"] = y_hat
        pred_df.loc[g.index, "smm_hat"] = smm_hat
        pred_df.loc[g.index, "cpr_hat"] = cpr_hat
        pred_df.loc[g.index, "cpr_hat_pct"] = cpr_hat * 100.0

        w = _stable_upb_weights(g["Cohort_Current_UPB"])
        coef_rows.append(
            {
                "Type_of_Security": sec_type,
                "coef_source": coef_source,
                "n_obs": len(g),
                "min_obs_threshold": min_obs,
                "beta_0": beta[0],
                "beta_spread": beta[1],
                "beta_spread2": beta[2],
                "beta_spread3": beta[3],
                "beta_loan_age": beta[4],
                "beta_log_upb": beta[5],
                "w_scale_median_raw_upb": float(np.nanmedian(pd.to_numeric(g["Cohort_Current_UPB"], errors="coerce"))),
                "w_scaled_min": float(np.nanmin(w)),
                "w_scaled_max": float(np.nanmax(w)),
                "mean_loan_age": float(np.average(g["Loan age"].values, weights=w)),
                "mean_log_UPB": float(np.average(g["log_UPB"].values, weights=w)),
                "mean_smm_hat": float(np.mean(smm_hat)),
                "mean_cpr_hat": float(np.mean(cpr_hat)),
                "mean_cpr_hat_pct": float(np.mean(cpr_hat * 100.0)),
            }
        )

    coef_df = pd.DataFrame(coef_rows).sort_values("n_obs", ascending=False)
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

    # Fixed instruction grid in percentage points.
    spread_grid = np.linspace(-2, 3, 200)
    spread2 = spread_grid ** 2
    spread3 = spread_grid ** 3
    rows = []
    for _, r in coef_df.iterrows():
        logit_smm_hat = (
            r["beta_0"]
            + r["beta_spread"] * spread_grid
            + r["beta_spread2"] * spread2
            + r["beta_spread3"] * spread3
            + r["beta_loan_age"] * r["mean_loan_age"]
            + r["beta_log_upb"] * r["mean_log_UPB"]
        )
        smm_hat = np.clip(sigmoid(logit_smm_hat), 1e-6, 1 - 1e-6)
        cpr_hat = 1.0 - (1.0 - smm_hat) ** 12

        grid_df = pd.DataFrame(
            {
                "Type_of_Security": r["Type_of_Security"],
                "spread_grid_pct": spread_grid,
                "spread2": spread2,
                "spread3": spread3,
                "logit_smm_hat": logit_smm_hat,
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
    for sec_type, g in curve_df.groupby("Type_of_Security"):
        g = g.sort_values("spread_grid_pct")
        plt.figure(figsize=(6, 4))
        plt.plot(g["spread_grid_pct"], g["cpr_hat_pct"], lw=2)
        plt.title(f"S-curve CPR vs Spread ({sec_type})")
        plt.xlabel("Spread (%)")
        plt.ylabel("Predicted CPR (%)")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_dir / f"scurve_{safe_slug(sec_type)}.png", dpi=150)
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
    logit_coef_df, logit_pred_df = fit_logit_smm_by_security_type(df, min_obs=1000)
    out_dir = Path(__file__).resolve().parent
    print("\nWLS model build summary")
    print(f"Observations: {len(df)}")

    if not logit_coef_df.empty:
        print("\nPer-Security WLS Logit (target=ln(SMM/(1-SMM)), features=spread/spread2/spread3/Loan age/log_UPB)")
        print(
            logit_coef_df[
                [
                    "Type_of_Security",
                    "coef_source",
                    "n_obs",
                    "min_obs_threshold",
                    "beta_0",
                    "beta_spread",
                    "beta_spread2",
                    "beta_spread3",
                    "beta_loan_age",
                    "beta_log_upb",
                    "w_scaled_min",
                    "w_scaled_max",
                    "mean_smm_hat",
                    "mean_cpr_hat",
                    "mean_cpr_hat_pct",
                ]
            ]
            .head(10)
            .to_string(index=False, float_format=lambda x: f"{x:,.6f}")
        )

        # Save full model outputs for research workflow.
        logit_coef_df.to_csv(out_dir / "security_type_logit_smm_coefficients.csv", index=False)
        logit_pred_df[
            [
                "Type_of_Security",
                "coef_source",
                "Date",
                "spread",
                "Loan age",
                "log_UPB",
                "logit_smm_target",
                "logit_smm_hat",
                "smm_hat",
                "cpr_hat",
                "cpr_hat_pct",
            ]
        ].to_csv(out_dir / "security_type_logit_smm_predictions.csv", index=False)
        print("\nWLS outputs written:")
        print(f"- {out_dir / 'security_type_logit_smm_coefficients.csv'}")
        print(f"- {out_dir / 'security_type_logit_smm_predictions.csv'}")
    else:
        print("\nNo valid rows for WLS fitting after preprocessing.")


if __name__ == "__main__":
    main()
