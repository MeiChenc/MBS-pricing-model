from pathlib import Path

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter


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
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.replace([np.inf, -np.inf], np.nan)

    # Model features
    df["duration"] = pd.to_numeric(df["Cohort_WA_Current_Loan_Age"], errors="coerce")
    df["event"] = (pd.to_numeric(df["SMM"], errors="coerce").fillna(0) > 0).astype(int)
    df["spread"] = (
        pd.to_numeric(df["Cohort_WA_Current_Interest_Rate"], errors="coerce")
        - pd.to_numeric(df["WA_Net_Interest_Rate"], errors="coerce")
    )
    df["log_UPB"] = np.log(pd.to_numeric(df["Cohort_Current_UPB"], errors="coerce") + 1)
    df["coupon"] = pd.to_numeric(df["WA_Net_Interest_Rate"], errors="coerce")

    model_cols = ["duration", "event", "spread", "log_UPB", "coupon"]
    df = df.dropna(subset=model_cols).copy()

    return df


def fit_cox(df: pd.DataFrame):
    feature_cols = ["spread", "log_UPB", "coupon"]

    mu = df[feature_cols].mean()
    sigma = df[feature_cols].std().replace(0, 1.0)

    z = (df[feature_cols] - mu) / sigma

    model_df = pd.concat([df[["duration", "event"]], z], axis=1)

    cph = CoxPHFitter()
    cph.fit(model_df, duration_col="duration", event_col="event")

    beta = cph.params_.loc[feature_cols].values
    h0 = cph.baseline_cumulative_hazard_.values.flatten()

    return cph, mu, sigma, beta, h0


def risk_multiplier(x_raw: pd.Series, mu: pd.Series, sigma: pd.Series, beta: np.ndarray) -> float:
    z = (x_raw[["spread", "log_UPB", "coupon"]] - mu) / sigma
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
    x_prepay["spread"] = x_prepay["spread"] + spread_beta_to_rate * (rate_shock_bp / 100.0)

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


def main():
    data_path = Path(__file__).resolve().parent / "PrepayData.txt"
    df = load_and_prepare(data_path)

    cph, mu, sigma, beta, h0 = fit_cox(df)

    x0 = df[["spread", "log_UPB", "coupon"]].mean()

    # Baseline pricing setup
    y0_annual = 0.05
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

    print("\nModel fit summary")
    print(f"Observations: {len(df)}")
    print(f"Concordance Index: {cph.concordance_index_:.3f}")
    print(f"Median remaining term used for CF engine: {term_months} months")
    print(f"Baseline discount rate: {y0_annual:.2%}")
    print(f"Baseline pass-through coupon: {coupon_annual:.2%}")
    print(f"Spread-to-rate shock beta: {spread_beta_to_rate:+.2f}")

    print("\nRate vs Prepay Attribution")
    print(out.to_string(index=False, float_format=lambda x: f"{x:,.6f}"))

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
