# Economic Policy Uncertainty and US Stock Market Returns

**Course:** Econometrics for Accounting and Finance
**Institution:** University of Bath  
**Year:** 2025  
**Program:** MSc Finance

## Project Overview

This empirical study investigates how Economic Policy Uncertainty (EPU) influences excess returns in the US stock market, examining the role of financial and economic variables over a 30-year period (1985-2015). Using classical regression techniques with robust diagnostic testing, the analysis explores the relationship between policy uncertainty and market performance while controlling for term spreads, credit spreads, interest rates, and structural breaks.

## Research Question

**How does Economic Policy Uncertainty (EPU) affect the excess returns of the U.S stock market, and what role do other financial and economic variables play in explaining this relationship?**

## Files in This Project

- `epu_stock_returns_analysis.ipynb` - Jupyter notebook with complete analysis
- `epu_analysis_report.docx` - Formal write-up with results and interpretation
- `requirements.txt` - Python dependencies
- `README.md` - This documentation

## Note on Files

The Jupyter notebook contains the data processing, regression analysis, and diagnostic tests. Some outputs have been truncated in the GitHub version for file size compatibility. The Word document (`epu_analysis_report.docx`) contains the complete formal write-up with all results, interpretations, figures, and tables.

## Theoretical Motivation

### Economic Policy Uncertainty

The EPU index (Baker et al., 2016) measures uncertainty about economic policy through newspaper coverage, tax code expiration dates, and forecaster disagreement. Economic theory suggests that increased policy uncertainty:
- Reduces investor confidence
- Leads to delayed investment decisions  
- Increases market volatility
- Dampens asset returns

**Hypothesis:** EPU negatively correlates with stock returns, reflecting risk aversion and weaker market performance.

### Financial Indicators

**Term Spread (TMS):**
- Difference between long-term and short-term interest rates
- Predicts economic conditions (Froot, 1989)
- Higher spreads signal growth expectations

**Default Return Spread (DFR):**
- Gauges credit risk in corporate bonds
- Wider spreads indicate risk aversion (Investopedia, 2024)
- Reflects default risk premium

**Long-term Returns (ltr):**
- Government bond returns
- Reflect inflation expectations and economic stability
- Influence stock valuations through discount rates

**S&P 500 Index:**
- Justified by Efficient Market Hypothesis (EMH)
- Proxy for market-wide influences and sentiment

### Structural Breaks

Major market events significantly affect dynamics:
- **Black Monday (October 1987):** 22% single-day market drop
- **Asian Financial Crisis (August 1998):** Contagion effects
- **Great Recession (October 2008):** Financial system collapse

Dummy variables account for these discrete events that continuous variables cannot capture.

## Data Description

### Sample Period
**370 monthly observations:** March 1985 - December 2015

**Structure:** Pooled cross-sectional data combining monthly observations over 30 years, capturing both stable and crisis periods.

### Variables

**Dependent Variable:**
- **Excess_Return:** S&P 500 returns (CRSP_SPvw) minus risk-free rate (Rfree)

**Key Independent Variables:**
- **EPU_change:** Monthly change in Economic Policy Uncertainty Index (scaled by 100)
- **DFR_w1:** Default Return Spread (winsorized at 1%)
- **TMS:** Term Spread (lty - tbl)
- **ltr_w1:** Long-term government bond returns (winsorized at 1%)
- **Index:** S&P 500 index scaled by 1000

**Control Variables (constructed from Goyal & Welsch, 2008):**
- **DP:** Dividend Price Ratio = log(D12/Index)
- **DY:** Dividend Yield = log(D12/Index_{t-1})
- **EP:** Earnings Price Ratio = log(E12/Index)
- **DE:** Dividend Payout Ratio = log(D12/E12)
- **DFY:** Default Yield Spread = BAA - AAA

**Dummy Variables:**
- **dummy_Black_Monday_1987:** October 1987 market crash
- **dummy_Asia_1998:** August 1998 Asian Financial Crisis
- **dummy_Recession_2008:** October 2008 Great Recession
- **recession_dummy:** NBER-dated recessions (1990-91, 2001, 2007-09)

### Data Processing

**Winsorization:**
- Applied at 1% level to all variables except Excess_Return and EPU_change
- Extreme values identified using 3.5 standard deviation rule
- Persistent outliers in EP, DE, and DFY deemed acceptable

**Multicollinearity Prevention:**
- Variables used to construct new variables were excluded
- Correlation matrix and VIF analysis conducted
- All VIF values below 7.5 (threshold: 10)

### Data Limitations
- Financial data subject to revisions
- May not fully capture unforeseen events (geopolitical crises, natural disasters)
- Investor sentiment and behavioral biases difficult to quantify

## Methodology

### Model Specification Process

#### Stage 1: Kitchen Sink Model
Initial regression with all variables after VIF testing:
- **R²:** 0.133
- **Adjusted R²:** 0.114
- **F-statistic:** 6.930 (significant at 5%)

**Diagnostic Issues Identified:**
- ✅ **No Autocorrelation:** Durbin-Watson = 2.170
- ❌ **Heteroscedasticity:** Breusch-Pagan, White, Goldfeld-Quandt tests failed
- ❌ **Non-normal Residuals:** Jarque-Bera test failed
- ⚠️ **Functional Form:** RESET test passed for degrees 2-3, failed at degree 4

**Solution:** Employed robust standard errors (HC3) to correct for heteroscedasticity.

#### Stage 2: Structural Break Analysis
After adding dummy variables for structural breaks:
- **R²:** 0.258
- **Adjusted R²:** 0.233
- **F-statistic:** 4.278 (significant at 5%)

**Improvement:**
- Explanatory power nearly doubled
- Breusch-Pagan test now passed
- Some heteroscedasticity concerns alleviated

#### Stage 3: Stepwise Refinement
Removed insignificant variables (DE, nits, infl) based on z-tests while retaining theoretically important variables.

### Final Model

```
Excess_Return = 0.0753 + 0.6055×DFR_w1 - 0.2843×TMS - 0.4254×ltr_w1 
                - 0.0200×Index - 0.0234×EPU_change 
                - 0.1979×dummy_Black_Monday_1987 
                - 0.1414×dummy_Asia_1998 
                - 0.1475×dummy_Recession_2008 
                - 0.0148×recession_dummy
```

**Model Performance:**
- **R²:** 0.255
- **Adjusted R²:** 0.237
- **F-statistic:** 4.605 (significant at 5%)
- **Durbin-Watson:** 2.289 (no autocorrelation)
- **Jarque-Bera:** 7.641 (p = 0.0219, improved normality)

### Diagnostic Tests Conducted

1. **Autocorrelation:**
   - Durbin-Watson test
   - Residuals vs. lagged residuals plot

2. **Heteroscedasticity:**
   - Breusch-Pagan test
   - White test
   - Goldfeld-Quandt test
   - Robust standard errors (HC3) employed

3. **Normality:**
   - Jarque-Bera test
   - Residual histogram
   - Residuals vs. fitted values plot

4. **Functional Form:**
   - Ramsey RESET test (degrees 2, 3, 4)
   - Residual plots for patterns

5. **Structural Breaks:**
   - Chow test
   - Visual inspection of excess returns over time

## Key Results & Interpretation

### Economic Policy Uncertainty Effect

**EPU_change coefficient: -0.0234**

A 1-point increase in the EPU index leads to a decrease in excess returns by **0.000234 percentage points**.

**Interpretation:**
- Negative relationship confirms theoretical prediction
- Higher policy uncertainty → lower market performance
- Consistent with uncertainty dampening investor confidence
- Effect economically small but statistically significant

### Financial Variables

**Default Return Spread (DFR_w1): +0.6055**
- Positive and highly significant
- Wider credit spreads associate with higher equity risk premiums
- Reflects compensation for bearing default risk

**Term Spread (TMS): -0.2843**
- Negative relationship (unexpected)
- May reflect flight-to-quality during uncertainty
- When long rates fall relative to short rates, stocks underperform

**Long-term Returns (ltr_w1): -0.4254**
- Negative relationship
- Rising bond yields compete with equity returns
- Reflects opportunity cost of equity investment

**Index: -0.0200**
- Small negative coefficient
- May capture mean reversion in scaled returns

### Structural Break Impacts

All structural break dummies are negative and significant:

- **Black Monday 1987: -0.1979** (largest negative impact)
- **Great Recession 2008: -0.1475**
- **Asian Crisis 1998: -0.1414**
- **Recession periods: -0.0148**

**Interpretation:**
- Major crises cause discrete negative shifts in returns
- Effects beyond what continuous variables capture
- Black Monday had the most severe immediate impact
- Recession dummy captures persistent downturn effects

## Model Assumptions & Limitations

### Maintained Assumptions
1. **Linearity:** Relationship linear in parameters (RESET test passed for main terms)
2. **No perfect multicollinearity:** VIF values all below 7.5
3. **Zero conditional mean:** E(ε|X) = 0
4. **Robust to heteroscedasticity:** HC3 standard errors employed

### Violated/Weak Assumptions
1. **Homoscedasticity:** Some tests still indicate heteroscedasticity
2. **Normality:** Residuals not perfectly normal (Jarque-Bera p = 0.0219)
3. **Functional form:** Potential misspecification at higher polynomial degrees

### Limitations

**Omitted Variables:**
- Monetary policy stance not directly included
- Global economic conditions limited representation
- Market sentiment and behavioral factors
- Liquidity conditions
- Fiscal policy variables

**Causality:**
- Regression shows correlation, not causation
- EPU may be endogenous (market crashes can increase uncertainty)
- Reverse causality possible

**Sample Period:**
- Ends in 2015 (misses recent events: COVID-19, 2022 inflation shock)
- May not generalize to post-2015 period
- Different policy regimes (QE era, zero rates) not fully captured

**Structural Stability:**
- Parameter constancy assumed outside dummy variable periods
- Relationship may have evolved over 30 years

## Technical Skills Demonstrated

- **Classical Regression Analysis:** OLS estimation, interpretation
- **Diagnostic Testing:** Comprehensive model validation
- **Robust Inference:** HC3 standard errors for heteroscedasticity
- **Structural Break Analysis:** Chow tests, dummy variable specification
- **Multicollinearity Diagnostics:** VIF analysis, correlation matrices
- **Data Processing:** Winsorization, variable construction
- **Python Programming:** Statsmodels, pandas, data manipulation
- **Economic Interpretation:** Translating coefficients to meaningful insights

## Extensions & Further Analysis

### Potential Enhancements:

1. **Time-Varying Parameters:**
   - Rolling window regressions
   - Markov-switching models
   - Regime-dependent coefficients

2. **Additional Variables:**
   - VIX index (market volatility)
   - Federal Funds Rate
   - Oil prices
   - Exchange rates
   - Global EPU indices

3. **Advanced Models:**
   - GARCH models for volatility clustering
   - Quantile regression for distributional effects
   - VAR/VECM for dynamic relationships
   - Non-linear models (threshold effects)

4. **Causality Analysis:**
   - Granger causality tests
   - Instrumental variable estimation
   - Natural experiments

5. **Extended Sample:**
   - Update to include 2016-2024 data
   - Analyze COVID-19 pandemic effects
   - Recent inflation/rate hiking cycle

6. **Subsample Analysis:**
   - Pre/post financial crisis comparison
   - High vs. low EPU regime analysis
   - Bull vs. bear market periods

## Dependencies

```
pandas>=1.3.0
numpy>=1.20.0
matplotlib>=3.5.0
statsmodels>=0.13.0
seaborn>=0.11.0
scipy>=1.7.0
jupyter>=1.0.0
```

## Academic Context

This project demonstrates proficiency in:

- **Econometric Theory:** Classical linear regression model (CLRM)
- **Applied Econometrics:** Model specification, testing, refinement
- **Financial Economics:** Asset pricing, risk premiums, market efficiency
- **Empirical Research:** Hypothesis testing, diagnostic validation
- **Policy Analysis:** Understanding macro-financial linkages

## Real-World Applications

**Investment Management:**
- Incorporating EPU into asset allocation models
- Risk management during high-uncertainty periods
- Tactical positioning based on policy environment

**Risk Assessment:**
- Quantifying uncertainty's impact on portfolios
- Stress testing for policy shocks
- Scenario analysis

**Policy Analysis:**
- Feedback from markets to policymakers
- Understanding market response to policy changes
- Timing of policy announcements

**Portfolio Construction:**
- Hedging strategies during high EPU
- Factor investing (uncertainty factor)
- Dynamic asset allocation

## Key References

- **Baker, S.R., Bloom, N. & Davies, S.J. (2016).** "Measuring Economic Policy Uncertainty." *Quarterly Journal of Economics*, 131, 1593-1636.

- **Goyal, A. & Welsch, I. (2008).** "A Comprehensive Look at the Empirical Performance of Equity Premium Prediction." *Review of Financial Studies*, 21, 1455-1508.

- **Froot, K.A. (1989).** "New Hope for the Expectations Hypothesis of the Term Structure of Interest Rates." *Journal of Finance*, 44(2).

- **Knight, F.H. (1957).** *Risk, Uncertainty and Profit.* Boston: Houghton Mifflin Company.

## Conclusion

This analysis provides empirical evidence that Economic Policy Uncertainty negatively affects US stock market excess returns, even after controlling for traditional financial variables and structural breaks. The finding that a 1-point increase in EPU reduces excess returns by 0.000234 percentage points confirms theoretical predictions about uncertainty's dampening effect on investor confidence and market performance.

The model explains approximately 25% of variation in excess returns, with robust standard errors addressing heteroscedasticity concerns. Structural break dummies prove essential, nearly doubling explanatory power and capturing discrete crisis effects that continuous variables cannot.

While limitations exist—including potential omitted variables, non-normal residuals, and causality concerns—the analysis demonstrates rigorous application of econometric techniques to an important question in financial economics. The results have practical implications for portfolio management, risk assessment, and understanding the macro-financial nexus.

---

**Note:** This analysis employs classical regression techniques with robust diagnostic testing appropriate for graduate-level econometrics coursework. Results should be interpreted within the context of model assumptions and identified limitations.
