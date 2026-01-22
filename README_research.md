# MBS Prepayment Pricing & Risk Engine

## 1. Executive Summary

This project develops a comprehensive **Logit-based Prepayment Model** and a path-dependent
**Pricing & Risk Engine** for Agency Mortgage-Backed Securities (MBS).

The primary objective is to quantify borrower **S-Curve prepayment behavior** and translate it into
**tradable price, duration, and convexity risk** at the pool level.

The modeling framework adopts a reduced-form **Logit S-curve specification on monthly SMM**,
augmented with cubic incentive terms to capture non-linear refinancing dynamics and borrower burnout.
A maturity-aware valuation engine then simulates cash flows and measures **Negative Convexity**
through effective duration analysis.

Using a 30-year TBA-eligible cohort, the model estimates an **Effective Duration of 4.06**, confirming
significant **price compression in rate rallies** and validating the framework’s ability to connect
borrower behavior with investor risk exposure.

---

## 2. Methodology

### Step 1: Data Processing (ETL)

- **Aggregation:** Raw loan-level or pool-level records are aggregated to a monthly frequency.
- **Target Variable:** Monthly **SMM (Single Monthly Mortality)** is constructed from cumulative
  prepayment measures using end-of-period values to ensure internal consistency.
- **Key Features:**
  - **Incentive (Spread):**  
    \[
    \text{Spread} = \text{WAC} - \text{Market Rate}_{\text{matched maturity}}
    \]
  - **Seasoning:** Loan age (months).
  - **Burnout / Size Proxy:** Logarithm of current UPB.

---

### Step 2: Interest Rate Construction (Borrower-Facing)

Refinancing incentives are constructed using **Freddie Mac Primary Mortgage Market Survey (PMMS)**
mortgage rates sourced from FRED:

- **30-year pools:** `MORTGAGE30US`
- **15-year pools:** `MORTGAGE15US`
- **20-year pools:** Proxy via maturity interpolation:
  \[
  r_{20} = r_{15} + \frac{1}{3}(r_{30} - r_{15})
  \]

These rates reflect **borrower-facing refinancing conditions** and are used consistently both to
define prepayment incentives and as the base discount rate in valuation.

---

### Step 3: Prepayment Modeling (Logit S-Curve)

Monthly SMM is modeled using a Logistic specification:

\[
\ln\left(\frac{SMM}{1-SMM}\right)
=
\alpha
+ \beta_1 S
+ \beta_2 S^2
+ \beta_3 S^3
+ \beta_4 \text{Age}
+ \beta_5 \ln(\text{UPB})
+ \varepsilon
\]

- **Cubic incentive terms** capture the empirical S-curve:
  - Flat response when out-of-the-money
  - High sensitivity near at-the-money
  - Burnout at deep in-the-money levels
- **Segmentation:** The framework supports separate models by security type (e.g., 15yr, 20yr, 30yr).

---

### Step 4: Valuation & Risk Engine

The valuation framework is explicitly **path-dependent**:

- Monthly cash flows are simulated over a 360-month horizon.
- **Scheduled amortization and prepayments** jointly reduce outstanding balance.
- Prepayment behavior feeds back into future cash flows through updated loan age and UPB.
- Interest-rate shocks (±50 bps) are applied to the maturity-matched market rate.

---

## 3. Key Findings

### 3.1 Prepayment S-Curve Behavior

The estimated Logit models capture economically consistent prepayment dynamics across maturities:

- **15-year pools** exhibit the steepest S-curve slope, indicating high refinancing sensitivity.
- **20-year pools** show intermediate behavior.
- **30-year pools** display a flatter response at high incentive levels, consistent with borrower burnout.

<img width="1000" height="630" alt="image" src="https://github.com/user-attachments/assets/e5f65e8d-abc4-4172-8951-e606ee6d6666" />

*(Figure 1: Estimated Prepayment S-Curves by Security Type.)*

---

### 3.2 Pricing & Negative Convexity

**Scenario Analysis (30yr TBA Eligible, Current Spread ≈ +40bps)**

| Scenario | Rate Shock | Price ($) |
| :--- | :--- | :--- |
| Rates −50 bps | −50 bps | **103.64** |
| Base Case | 0 bps | **101.79** |
| Rates +50 bps | +50 bps | **99.50** |

- **Effective Duration:** **4.06**

The asymmetric price response confirms pronounced **Negative Convexity**:
price appreciation in a rally is capped by accelerated prepayments, while sell-offs extend duration.

<img width="854" height="630" alt="image" src="https://github.com/user-attachments/assets/95833f8c-bfd8-4987-a69b-a200da2d18c3" />

*(Figure 2: MBS Price–Yield Profile vs. a Standard Bond.)*

---

## 4. Repository Structure

- **`MBS_pricing.ipynb`**
  - Data loading and monthly aggregation
  - Logit model estimation and S-curve visualization
  - Path-dependent pricing, cash-flow projection, and duration analysis

---

## 5. Usage

1. Place raw data locally (not included due to size/proprietary constraints).
2. Set data path via environment variable if required.
3. Run all cells in `MBS_pricing.ipynb` to reproduce figures and tables.

---

## 6. Model Limitations & Risk Considerations

- **Static Rate Environment:**  
  Interest rates are shocked deterministically; stochastic refinancing waves are not modeled.
- **Reduced-Form Discounting:**  
  Valuation uses a flat, borrower-facing market rate rather than a full Treasury/swap term structure or calibrated OAS.
- **Identification Risk:**  
  Multiple S-curve parameterizations can fit historical data yet imply materially different convexity under stress.

These limitations emphasize that **MBS risk management depends on scenario robustness, not point estimates alone**.

---

## 7. Takeaway

This project demonstrates how borrower prepayment behavior can be systematically translated into
**price sensitivity, duration shortening, and negative convexity** using a transparent,
research-grade framework consistent with buy-side MBS analysis.

