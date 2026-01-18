# MBS Prepayment Pricing & Risk Engine


## 1. Executive Summary

This project develops a comprehensive **Prepayment Model** (Logit-Based) and a path-dependent **Pricing Engine** for Agency Mortgage-Backed Securities (MBS). The primary goal is to quantify the "S-Curve" behavior of borrowers and value the embedded prepayment option.

The modeling framework adopts a Logit-based S-curve specification on monthly SMM, which is standard in empirical MBS prepayment modeling and well-suited for aggregate pool-level data.

The Valuation Engine reveals significant **Negative Convexity** in the analyzed 30-year cohort. The estimated **Effective Duration of 6.66** confirms "Price Compression" in rallying rate environments, validating the framework’s ability to translate borrower behavior into **tradable convexity and duration risk**.

---

## 2. Methodology

We model conditional prepayment behavior using a Logit specification on monthly SMM, augmented with cubic incentive terms to capture the empirical S-curve observed in MBS markets.

### Step 1: Data Processing (ETL)
* **Aggregation:** Raw daily reports are aggregated to a monthly frequency.
* **Target Variable:** Monthly **SMM** (Single Monthly Mortality) is derived from the *last* record of `Cumulative_SMM` for each period to ensure accuracy.
* **Features:**
    * **Incentive (Spread):** $WAC - MarketRate$ (The primary driver).
    * **Seasoning (Duration):** Loan Age (months).
    * **Burnout/Size:** Logarithm of Current UPB.

### Step 2: Prepayment Modeling (Logit-Based S-Curve Specification)
We model the Conditional Prepayment Rate (CPR) using a Logistic transformation to bound predictions within $[0, 1]$:

$$\ln\left(\frac{SMM}{1-SMM}\right) = \alpha + \beta_1 S + \beta_2 S^2 + \beta_3 S^3 + \beta_4 Age + \beta_5 \ln(UPB) + \epsilon$$

* **S-Curve Logic:** Cubic terms ($S^2, S^3$) are included to capture the non-linear convexity of prepayments.
* **Segmentation:** The code supports splitting models by security type (e.g., **30yr TBA Eligible** vs. **RPL**) to account for behavioral heterogeneity.

### Step 3: Valuation & Risk Analysis
The valuation framework is explicitly path-dependent, with prepayment behavior feeding back into future cash flows via loan age and outstanding balance dynamics.

A path-dependent **Pricing Engine** simulates cash flows over a 360-month horizon:
* **Dynamic Simulation:** Updates `Spread` and `Loan Age` iteratively.
* **Scenario Analysis:** Shocks interest rates ($\pm 50bps$) to calculate Price Asymmetry and Effective Duration.

---

## 3. Key Findings

### 3.1 Model Performance (Cohort: 30yr TBA Eligible)
The reported R-squared reflects in-sample fit on the logit-transformed SMM and is used as a diagnostic rather than a structural goodness-of-fit metric.


* **Primary Driver:** The **Spread** coefficient is positive and highly significant (**t-stat > 23**).
* **S-Curve Validation:**

As shown in the figure below, the model captures the expected **non-linear S-shaped relationship**
between refinancing incentive and prepayment rates across different MBS tenors.

Key economically consistent patterns emerge:

- **Strong term-structure heterogeneity:**  
  The **15yr cohort (orange line)** exhibits the steepest slope, indicating the highest sensitivity
  to refinancing incentives.  
  The **20yr cohort (green line)** shows a similar but moderately less aggressive response.

- **Burnout effects in longer tenors:**  
  The **30yr cohort (blue line)** displays a noticeably flatter S-curve at higher incentive levels,
  consistent with borrower burnout and refinancing frictions commonly observed in long-duration mortgages.

- **At-the-money differentiation:**  
  Near zero incentive, baseline CPR levels differ materially across cohorts, suggesting that loan
  maturity and borrower composition influence prepayment behavior beyond rate incentive alone.

Overall, the estimated S-curves align with well-documented MBS prepayment dynamics and support the
model’s ability to translate refinancing incentives into economically meaningful prepayment responses.

**Pricing implication:**  
Differences in S-curve slope and saturation translate directly into **distinct duration and convexity
profiles** across MBS tenors, particularly in rate rally environments.


<img width="1000" height="630" alt="image" src="https://github.com/user-attachments/assets/d1e35341-daaf-435e-b107-f4a648ee9ed7" />
*(Figure 1: Estimated Prepayment S-Curves. The non-linear shape confirms the effectiveness of the cubic spread terms in the Logit model.)*

### 3.2 Pricing & Negative Convexity
The pricing engine highlights the unique risks of MBS compared to standard bonds.

**Scenario Analysis (Current Spread: +40bps)**

| Scenario | Rate Shock | Price ($) | Change |
| :--- | :--- | :--- | :--- |
| **Rally (Rates Down)** | -50 bps | **$101.71** | +$2.85 |
| **Base Case** | 0 bps | **$98.86** | - |
| **Sell-off (Rates Up)** | +50 bps | **$95.12** | -$3.74 |

* **Price Compression:** The chart below illustrates **Negative Convexity**. In the rally scenario (left side), the MBS price (Blue) underperforms the theoretical standard bond (Grey) because accelerated prepayments cap the upside potential.

<img width="854" height="630" alt="image" src="https://github.com/user-attachments/assets/95833f8c-bfd8-4987-a69b-a200da2d18c3" />

*(Figure 2: MBS Price-Yield Curve vs. Standard Bond. The divergence in the rally scenario quantifies the cost of the embedded prepayment option.)*
* **Effective Duration:** **6.66**. This is significantly lower than a standard 30-year bond, quantifying the "Duration Shortening" effect caused by the prepayment option.

---

## 4. Repository Structure

This project is contained within a single computational notebook:

* **`MBS_pricing.ipynb`**:
    * **Part 1:** Data Loading & Monthly Aggregation.
    * **Part 2:** Logit Model Training & S-Curve Visualization.
    * **Part 3:** Pricing Engine, Cash Flow Projection, and Duration Analysis.

---

## 5. Usage

To replicate the results:
1.  Ensure the raw data file is available in the working directory.
2.  Run all cells in `MBS_pricing.ipynb`.
3.  The notebook will output the OLS summary statistics, the S-Curve plot, and the Price-Yield (Negative Convexity) chart.

---

## 6. Model Limitations & Risk Considerations

- **Static Spread Assumption:**  
  Interest rates are shocked deterministically rather than modeled as stochastic processes, which understates path-dependent refinancing waves.

- **Reduced-Form Prepayment Specification:**  
  The Logit S-curve captures average behavioral responses but does not explicitly model borrower credit constraints, servicer capacity, or policy interventions.

- **Identification Risk:**  
  Multiple S-curve parameterizations can fit historical CPR equally well but imply materially different duration and convexity under stress scenarios.

These limitations highlight that MBS pricing accuracy depends as much on **scenario robustness** as on in-sample model fit.

