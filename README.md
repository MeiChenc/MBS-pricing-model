# MBS Pricing and Negative Convexity under Stochastic Interest Rates

This project implements a Monte Carlo pricing framework for Mortgage-Backed Securities (MBS),
focusing on prepayment risk and negative convexity under stochastic interest rate dynamics.

The goal is to demonstrate how borrower behavior, interest rate uncertainty,
and embedded prepayment options jointly affect MBS valuation and risk.

---

## Motivation

Unlike standard fixed-income instruments, MBS cashflows are path-dependent.
Borrowers may prepay their mortgages when interest rates fall, effectively embedding
a call option that leads to **negative convexity**.

Understanding this interaction is central to:
- MBS valuation and OAS analysis
- Interest rate risk management
- Structured fixed-income research

This project provides a transparent, model-driven implementation
to illustrate these mechanisms.

---

## Model Overview

**Interest Rate Model**
- Single-factor short-rate model (Cox-type dynamics)
- Monte Carlo simulation of interest rate paths

**Mortgage Cashflow Engine**
- Level-payment fixed-rate mortgage
- Monthly amortization schedule
- Remaining balance updated dynamically

**Prepayment Modeling**
- Rule-based PSA-style prepayment assumption
- Prepayment rate linked to interest rate incentives
- Endogenous cashflow truncation due to prepayment

**Valuation**
- Monte Carlo discounted cashflows
- Pathwise aggregation of principal and interest payments
- MBS price computed as expectation across simulated paths

**Risk Analytics**
- Price sensitivity to interest rate changes
- Illustration of negative convexity
- Comparison across different prepayment assumptions

---

## Methodology

1. Simulate short-rate paths under the Cox model
2. Generate mortgage cashflows along each rate path
3. Apply prepayment logic at each payment date
4. Discount realized cashflows back to time 0
5. Aggregate results across Monte Carlo scenarios
6. Analyze price behavior and convexity effects

All results are obtained under consistent model assumptions
to isolate the impact of prepayment and rate dynamics.

---

## Outputs

The project produces:
- MBS price estimates under stochastic interest rates
- Cashflow profiles with and without prepayment
- Price–rate relationships highlighting negative convexity
- Sensitivity of valuation to prepayment assumptions (PSA levels)

Plots and summary tables are exported to the `outputs/` directory.

---

## Reproducibility

To reproduce the results:

```bash
python notebooks/Cox-MBS_pricing.ipynb
