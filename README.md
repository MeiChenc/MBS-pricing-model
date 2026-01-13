# Modeling Prepayment Risk and Negative Convexity in Mortgage-Backed Securities

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Lifelines](https://img.shields.io/badge/Library-Lifelines-orange)
![Finance](https://img.shields.io/badge/Domain-Quant%20Finance-green)
![Status](https://img.shields.io/badge/Status-Research%20Prototype-success)

---

## Executive Summary

Mortgage-Backed Securities (MBS) differ fundamentally from standard fixed-income instruments due to **borrower prepayment behavior**, which embeds a call option into the cashflow structure and gives rise to **negative convexity**.

This project develops a **survival-analysis-based framework** to model prepayment risk using a **Cox Proportional Hazards model**, linking borrower refinancing incentives and loan characteristics to monthly prepayment dynamics (SMM/CPR).  
The estimated prepayment behavior is then embedded into a cashflow discounting engine to evaluate MBS prices and key risk metrics, including duration and convexity.

Empirically, **refinancing spread** is identified as the dominant driver of prepayment risk. The model achieves strong out-of-sample ranking performance with a **Concordance Index (C-index) of 0.78**, indicating substantial explanatory power in distinguishing early versus late prepayment behavior.

This framework connects borrower-level prepayment behavior directly to **cashflow timing, negative convexity, and price sensitivity under interest rate shocks**, which are central concerns in MBS valuation and risk management.

---

## Methodology

### 1. Data Processing and Feature Engineering

The dataset consists of historical monthly performance of agency MBS. Key modeling choices include:

- **Event Definition**  
  A prepayment event is defined when the observed Single Monthly Mortality (SMM) exceeds zero.

- **Time Dimension**  
  Loan age (in months) is used as the duration variable for survival analysis.

- **Key Covariates**
  - **Spread (Refinancing Incentive)**  
    Defined as:  
    `Current Mortgage Rate − Net Interest Rate`
  - **log(UPB)**  
    Log-transformed unpaid principal balance, serving as a proxy for loan size.
  - **Coupon**  
    Security coupon rate.

These variables are chosen to capture refinancing incentives, borrower scale effects, and contractual interest rate exposure.

---
### Baseline Hazard Estimation

The Cox model estimates a non-parametric baseline hazard using the Breslow method.
The cumulative baseline hazard captures the unconditional evolution of prepayment
intensity over loan age.

<img width="376" height="316" alt="image" src="https://github.com/user-attachments/assets/b46129f5-e11f-44a5-a33a-a924ec877949" />

*Figure 1. Estimated cumulative baseline hazard as a function of loan age.*

---

### 2. Survival Analysis Model

Prepayment behavior is modeled using a semi-parametric **Cox Proportional Hazards model**:

\[
h(t \mid x) = h_0(t)\exp\left(
\beta_1 \cdot \text{Spread} +
\beta_2 \cdot \log(\text{UPB}) +
\beta_3 \cdot \text{Coupon}
\right)
\]

where:
- \( h(t \mid x) \) denotes the conditional prepayment hazard,
- \( h_0(t) \) is the baseline hazard,
- covariates enter multiplicatively through relative hazard shifts.

The proportional hazards assumption is employed as a **first-order approximation** to capture relative refinancing incentives across borrowers, rather than to model absolute prepayment timing with full structural realism.

---

## Empirical Results

The model demonstrates strong discriminatory power with a **Concordance Index (C-index) of 0.78**, indicating effective ranking of loans by prepayment risk.

| Feature | Coefficient | Exp(Coef) | z-score | Economic Interpretation |
|------|------------|-----------|--------|-------------------------|
| **Spread** | **0.70** | **2.01** | 233.66 | A unit increase in refinancing spread approximately doubles the monthly prepayment hazard, making it the primary driver of prepayment behavior. |
| **log(UPB)** | **0.38** | **1.47** | 162.26 | Larger loan balances exhibit higher sensitivity to refinancing incentives. |
| **Coupon** | **-0.38** | **0.68** | -148.16 | Conditional on spread, higher coupon loans display lower incremental prepayment risk. |

All coefficients are statistically significant with p-values below 0.005.

---

## Valuation and Risk Metrics

The estimated survival curve is transformed into an expected CPR term structure and embedded into a discounted cashflow engine under a zero-OAS assumption.

**Baseline Valuation**
- **Theoretical Price**: 100.00 (At Par)
- **Effective Duration**: 4.78 years
- **Convexity**: 32.07

**Interest Rate Sensitivity**
- Rates −50 bps → Price = 102.43  
- Rates +50 bps → Price = 97.65  

The asymmetric price response to interest rate shocks highlights the embedded prepayment option and the resulting **negative convexity** characteristic of MBS.

---

## Repository Structure

```text
├── notebooks/
│   └── Cox_MBS_Valuation.ipynb   # Core research notebook
├── src/
│   └── cox_mbs_pricing.py        # Modular pricing and modeling logic
├── results/
│   ├── survival_curves.png       # Estimated survival functions
│   └── cpr_projection.png        # Modeled CPR paths
├── README.md
└── requirements.txt
