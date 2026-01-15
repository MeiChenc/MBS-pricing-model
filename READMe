# Agency MBS Prepayment Modeling & Valuation Engine
### Stochastic Survival Analysis (Cox PH) & Monte Carlo Pricing

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Model](https://img.shields.io/badge/Model-Cox%20Proportional%20Hazard-orange)
![Status](https://img.shields.io/badge/Status-Research%20Prototype-yellow)

## ⚡ Executive Summary

This project implements a quantitative framework to price Agency Mortgage-Backed Securities (MBS) by modeling borrower prepayment behavior. Using a **Cox Proportional Hazards Model**, I quantify the sensitivity of prepayment speeds (SMM/CPR) to refinancing incentives and loan characteristics.

The model is integrated into a cash-flow engine to compute **Option-Adjusted Duration (OAD)** and **Convexity**. The research highlights the "S-Curve" behavior of prepayments and critically evaluates the limitations of applying survival analysis to aggregated cohort data.

## 📊 Empirical Results & Factor Analysis

The model was calibrated on historical Agency MBS cohort data. The concordance index of **0.78** indicates strong discriminatory power in ranking prepayment risks.

| Covariate | Coef $\beta$ | Exp($\beta$) | z-score | Desk Interpretation |
| :--- | :--- | :--- | :--- | :--- |
| **Refi Incentive (Spread)** | `0.70` | `2.01` | `233.7` | **High Sensitivity**. A 100bps increase in incentive doubles the conditional hazard rate. Primary driver of negative convexity. |
| **Loan Size (log_UPB)** | `0.38` | `1.47` | `162.3` | **Size Effect**. Larger loan balances correlate with higher prepayment efficiency (fixed costs of refinancing are less burdensome). |
| **Coupon** | `-0.38` | `0.68` | `-148.2` | **Base Rate Effect**. Controlling for spread, lower coupon pools exhibit naturally lower turnover (stickier money). |

## 📉 Visualization of Risk Profiles

### 1. The Prepayment S-Curve (CPR vs. Incentive)
This chart demonstrates the non-linear response of borrowers. The steep slope around the "At-the-Money" point (Spread = 0) indicates where the portfolio's duration is most unstable.

![Prepayment S-Curve](images/s_curve_plot.png)
*(Suggested: Insert the Spread vs CPR plot here)*

### 2. Price-Yield Curve & Negative Convexity
The valuation engine reveals the "Price Compression" effect. Unlike standard bonds (grey line), the MBS price (blue line) is capped as rates rally due to accelerated prepayments.

![Price Yield](images/price_yield_plot.png)
*(Suggested: Insert the Price vs Rate plot here)*

## 🧐 Model Diagnostics & Critical Observations

A key part of this research involved critically analyzing the statistical anomalies arising from applying biological survival models to financial cohort data.

### 1. The "High Event Frequency" Anomaly
* **Observation**: In the dataset, `Event = 1` (defined as SMM > 0) occurs in nearly **99%** of the observations.
* **Quant Insight**: Unlike clinical trials where a patient dies once, an MBS pool "bleeds" principal continuously. Since we are modeling *cohorts* rather than *individual loans*, the "Event" is effectively continuous.
* **Implication**: The Cox model here functions less as a "Time-to-Death" predictor and more as a **Conditional Intensity Model**. The high event rate is a feature of the aggregation level, not a data error.

### 2. Magnitude of Coefficients
* **Observation**: The raw coefficients appear small (e.g., Spread $\beta = 0.70$, Coupon $\beta = -0.38$).
* **Quant Insight**: In a proportional hazards framework, the effect is exponential ($\exp(\beta)$). A coefficient of 0.70 implies an $\exp(0.70) \approx 2.01$ multiplier.
* **Implication**: Small nominal shifts in coefficients result in massive changes in projected CPR. This confirms the **high model risk**—a slight miss-estimation of the spread coefficient can lead to significant errors in duration hedging.

### 3. Baseline Cumulative Hazard Shape
* **Observation**: The baseline cumulative hazard grows linearly and aggressively, lacking the typical "flattening" seen in biological survival.
* **Quant Insight**: This reflects the absence of "true" survival behavior in the early life of a pool. An MBS pool creates cash flows every month.
* **Correction**: In the pricing engine, we convert this cumulative hazard into a monthly **Single Monthly Mortality (SMM)** probability to prevent the "probability > 1" fallacy in long-dated projections.

## 💰 Valuation Scenarios (Sensitivity Analysis)

We stressed the model under instantaneous rate shifts to derive hedge ratios:

* **Base Case**: Price `100.00` (Par)
* **Rally (-50bps)**: Price `102.43` (Limited Upside)
* **Sell-off (+50bps)**: Price `97.65` (Extended Duration)
* **Effective Duration**: `4.78` years

## 🛠 Repository Structure

```text
.
├── notebooks/
│   └── Cox_MBS_Analysis.ipynb   # Model fitting & Diagnostic plots
├── src/
│   └── valuation_engine.py      # Monte Carlo cash flow logic
├── images/                      # S-Curve & Convexity visualizations
└── README.md
