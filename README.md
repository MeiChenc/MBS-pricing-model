# MBS Pricing — Logit-SMM + Amortization Cashflows + OAS

Notebook: `MBS_Pricing.ipynb`

This notebook implements a **pass-through MBS pricing engine** with:
- **Monthly amortization cashflows** (scheduled interest + scheduled principal)
- **Prepayment modeling** via **Logit → SMM** (0–1) to generate **CPR/SMM-style** prepayment speed
- **Pathwise discounting** and **OAS** (option-adjusted spread) calibration to match a target market price

> Scope: educational / research prototype. Not production-ready (see Limitations).

---

## 1) Model Logic (What’s being priced)

### Cashflow engine (monthly)
For month \(t\), with balance \(B_{t-1}\), mortgage rate \(c\), maturity \(N\), and monthly rate \(i=c/12\):

1. **Scheduled payment**
\[
PMT = B_0 \cdot \frac{i(1+i)^N}{(1+i)^N - 1}
\]

2. **Scheduled interest**
\[
I_t = i \cdot B_{t-1}
\]

3. **Scheduled principal**
\[
SP_t = PMT - I_t
\]

4. **Balance after scheduled principal**
\[
\tilde{B}_t = B_{t-1} - SP_t
\]

5. **Prepayment principal (SMM)**
\[
PP_t = SMM_t \cdot \tilde{B}_t
\]

6. **Ending balance**
\[
B_t = \tilde{B}_t - PP_t
\]

7. **Total cashflow**
\[
CF_t = I_t + SP_t + PP_t
\]

### Discounting + OAS
Discount factors are computed along rate paths using a monthly short rate \(r_t\) (and optional spread/OAS):
\[
DF_t = \prod_{k=1}^{t}\frac{1}{1 + (r_k + OAS)/12}
\]
\[
PV = \sum_{t=1}^{T} CF_t \cdot DF_t
\]

---

## 2) Prepayment Model (Logit → SMM)

The notebook estimates a regression on **logit(SMM)**:
\[
\text{logit}(SMM_t) = \log\left(\frac{SMM_t}{1-SMM_t}\right)
\approx \beta^\top x_t
\]

Then predicts:
\[
\widehat{SMM}_t = \sigma(\beta^\top x_t) = \frac{1}{1+\exp(-\beta^\top x_t)}
\]

Typical predictors in the notebook include polynomial terms of **refinancing incentive / spread** (and optionally other covariates if added).

---

## 3) Repository Structure

```text
.
├── MBS_Pricing.ipynb
├── data/                      # optional: raw/processed inputs (not required)
├── outputs/                   # optional: figures, tables
└── README.md
