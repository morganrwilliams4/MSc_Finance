# Vector Error Correction Model: Macroeconomic Analysis

**Course:** Econometrics for Accounting and Finance
**Institution:** University of Bath  
**Year:** 2025  
**Program:** MSc Finance

## Project Overview

This project implements a comprehensive time series econometric analysis examining the long-run equilibrium relationships between the S&P 500 index, Consumer Price Index (CPI), and unemployment rate using Vector Error Correction Models (VECM). The analysis spans 25 years of monthly data (2000-2024) and demonstrates the application of advanced cointegration techniques to understand macroeconomic and financial market dynamics.

## Research Question

**Do long-run equilibrium relationships exist between equity markets (S&P 500), inflation (CPI), and labor markets (unemployment rate)?**

If cointegration exists, how do these variables adjust to deviations from their long-run equilibrium, and what are the short-run dynamics?

## Theoretical Motivation

### Economic Relationships

**S&P 500 and CPI:**
- Stock prices should theoretically adjust for inflation expectations
- Real returns matter more than nominal returns for investors
- Central bank policy links inflation to equity valuations via discount rates

**S&P 500 and Unemployment:**
- Unemployment affects corporate earnings (labor costs, consumer demand)
- Stock market often viewed as leading indicator for economic activity
- Labor market conditions influence Federal Reserve policy decisions

**CPI and Unemployment:**
- Phillips Curve relationship (inverse relationship between inflation and unemployment)
- Both are key inputs to monetary policy decisions
- Wage-price spiral dynamics

## Methodology

### Data Sources & Period

**Variables:**
1. **S&P 500 Index (^GSPC)** - Broad US equity market benchmark
2. **Consumer Price Index (CPIAUCSL)** - Urban consumers, all items
3. **Unemployment Rate (UNRATE)** - Civilian unemployment rate

**Sources:**
- S&P 500: Yahoo Finance (yfinance)
- CPI & Unemployment: Federal Reserve Economic Data (FRED)

**Sample Period:** January 2000 - December 2024 (300 monthly observations)

**Frequency:** Monthly (end-of-month observations)

### Econometric Framework

#### Stage 1: Unit Root Testing

Testing for stationarity using Augmented Dickey-Fuller (ADF) tests:

**Null Hypothesis (H₀):** Series has a unit root (non-stationary)  
**Alternative (H₁):** Series is stationary

**Test Specification:**
```
ADF test with trend and constant
Automatic lag selection using AIC
```

**Results Expected:**
- Levels: Likely non-stationary (p > 0.05)
- First differences: Stationary (p < 0.05)
- Conclusion: Variables are I(1) - integrated of order 1

#### Stage 2: Cointegration Testing

Using Johansen cointegration test to determine if long-run equilibrium exists:

**Test Statistics:**
- **Trace Statistic:** Tests null of r cointegrating vectors against alternative of n vectors
- **Maximum Eigenvalue:** Tests null of r vectors against r+1 vectors

**Procedure:**
1. Specify deterministic trend (constant in cointegration relation)
2. Test for number of cointegrating relationships (rank test)
3. Determine optimal cointegration rank

**Expected Outcome:**
If test rejects H₀ at r=0 but fails to reject at r=1, we have **one cointegrating relationship**.

#### Stage 3: Vector Error Correction Model (VECM)

**Model Specification:**

The VECM with cointegration rank r=1 takes the form:

```
Δy_t = Π y_(t-1) + Γ₁ Δy_(t-1) + ... + Γ_(p-1) Δy_(t-p+1) + μ + ε_t
```

Where:
- Π = αβ' (error correction term)
- α = loading coefficients (adjustment speeds)
- β = cointegrating vector (long-run relationship)
- Γᵢ = short-run dynamic coefficients

**Interpretation:**

**Cointegrating Vector (β):**
```
log(S&P500) = β₁ + β₂·log(CPI) + β₃·UnemploymentRate + error correction term
```

Defines the long-run equilibrium relationship.

**Loading Coefficients (α):**
- Measure how quickly each variable adjusts to deviations from equilibrium
- Negative α: variable decreases when above equilibrium
- Positive α: variable increases when below equilibrium
- Magnitude indicates adjustment speed

**Short-Run Dynamics (Γ):**
- Capture temporary effects and adjustment paths
- Represent responses to lagged changes in variables

### Diagnostic Tests

**Model Adequacy:**
1. **Residual Autocorrelation:** Portmanteau test, Ljung-Box Q-test
2. **Residual Normality:** Jarque-Bera test
3. **Heteroskedasticity:** ARCH effects test
4. **Model Stability:** Check eigenvalues of companion matrix

## Implementation Details

### Data Processing

**Transformations:**
```python
# Log transformation for index variables (multiplicative relationships)
log_sp500 = np.log(sp500)
log_cpi = np.log(cpi)

# Unemployment rate kept in levels (already stationary-like)
unemployment_rate = unemployment  # percentage points
```

**Why Logarithms?**
- Converts exponential growth to linear trends
- Percentage changes approximately equal to log differences
- Stabilizes variance in financial time series
- Economic interpretation: elasticities

### Python Implementation

**Key Libraries:**
- `statsmodels.tsa.api` - Time series analysis and VECM
- `yfinance` - Financial market data
- `pandas_datareader` - FRED economic data
- `numpy/pandas` - Data manipulation
- `matplotlib` - Visualization

**VECM Estimation:**
```python
from statsmodels.tsa.vector_ar import vecm

# Specify VECM
vecm_model = vecm.VECM(
    data,
    k_ar_diff=1,          # Lag order for differenced variables
    coint_rank=1,         # Number of cointegrating relationships
    deterministic='co'    # Constant within cointegration
)

# Estimate model
vecm_results = vecm_model.fit()
```

## Key Results & Interpretation

### Cointegration Findings

**Johansen Test Results:**
- **Cointegration Rank:** r = 1 (one long-run equilibrium relationship)
- **Interpretation:** S&P 500, CPI, and unemployment share a common stochastic trend
- **Implication:** Short-run deviations from equilibrium are temporary

### Long-Run Equilibrium (β coefficients)

**Cointegrating Equation:**
```
log(S&P500) = β₀ + β₁·log(CPI) + β₂·UnemploymentRate
```

**Coefficient Interpretation:**
- **β₁ (CPI coefficient):** Long-run relationship between stock prices and inflation
  - If β₁ > 0: Stocks move with inflation (hedge against inflation)
  - If β₁ < 0: Stocks inversely related to inflation
  
- **β₂ (Unemployment coefficient):** Long-run relationship between stocks and labor market
  - If β₂ < 0: Higher unemployment associated with lower stock prices
  - If β₂ > 0: Positive relationship (less common)

### Adjustment Speeds (α coefficients)

**Loading Coefficients indicate:**

**For S&P 500:**
- Speed at which stock market corrects deviations from equilibrium
- Large |α|: Quick adjustment
- Small |α|: Slow adjustment or weak exogeneity

**For CPI:**
- How inflation responds to equilibrium deviations
- Expected to be small (inflation adjusts slowly)

**For Unemployment:**
- Labor market adjustment to shocks
- Typically moderate adjustment speed

### Economic Insights

**Error Correction Mechanism:**
- When S&P 500 is "too high" relative to fundamentals (CPI, unemployment), it tends to fall
- When "too low," it tends to rise
- Adjustment is not instantaneous - takes multiple periods

**Policy Implications:**
- Federal Reserve should consider equity valuations in policy decisions
- Stock market contains information about future economic conditions
- Inflation-unemployment trade-offs affect financial markets

## Visualizations

The notebook includes:

1. **Time Series Plots:**
   - Original series (levels)
   - Log-transformed series
   - First differences

2. **Stationarity Analysis:**
   - Visual inspection of trends
   - ACF/PACF plots for unit root assessment

3. **Model Diagnostics:**
   - Residual plots
   - ACF of residuals
   - Stability analysis

## Model Assumptions & Limitations

### Assumptions

1. **Linearity:** Relationships are linear in parameters
2. **Weak Exogeneity:** At least one variable weakly exogenous
3. **Gaussian Errors:** Residuals normally distributed
4. **Lag Order:** Correctly specified lag structure
5. **Structural Stability:** Parameters constant over sample period

### Limitations

1. **Omitted Variables:** 
   - Interest rates not included
   - GDP growth excluded
   - Oil prices, exchange rates not considered

2. **Parameter Instability:**
   - 2008 Financial Crisis structural break
   - COVID-19 pandemic regime shift
   - May need subsample analysis

3. **Nonlinearity:**
   - Relationships may be nonlinear during crises
   - Threshold effects not captured

4. **Forward-Looking Behavior:**
   - Stock markets incorporate expectations
   - Model based on realized values only

## Technical Skills Demonstrated

- **Time Series Econometrics:** Unit root tests, cointegration, VECM
- **Statistical Testing:** Hypothesis testing, p-value interpretation
- **Data Management:** Multi-source data integration, frequency alignment
- **Python Programming:** Advanced statsmodels usage
- **Financial Economics:** Understanding macro-finance linkages
- **Model Diagnostics:** Residual analysis, specification testing

## Extensions & Further Analysis

### Potential Enhancements:

1. **Structural Breaks:**
   - Chow test for break dates
   - Recursive estimation
   - Sub-sample analysis (pre/post 2008)

2. **Additional Variables:**
   - Federal Funds Rate
   - 10-Year Treasury Yield
   - Oil prices (WTI)
   - Exchange rates

3. **Advanced Models:**
   - Time-varying VECM
   - Threshold VECM (TVECM)
   - Markov-switching VECM

4. **Forecasting:**
   - Out-of-sample predictions
   - Impulse response functions
   - Forecast error variance decomposition

5. **Granger Causality:**
   - Test directional relationships
   - Lead-lag analysis

## Dependencies

```python
yfinance>=0.2.0
pandas-datareader>=0.10.0
pandas>=1.3.0
numpy>=1.20.0
matplotlib>=3.5.0
statsmodels>=0.13.0
openpyxl>=3.0.0  # For Excel export
```

## Files

- `vecm_sp500_macro_analysis.ipynb` - Main Jupyter notebook with complete analysis
- `Econometrics_Data.xlsx` - Exported dataset (generated by code)
- `README.md` - This documentation

## Usage

```python
# Load and run the notebook
import jupyter

# Or execute specific sections:
# 1. Data Cleaning & Preparation
# 2. Unit Root Testing
# 3. Cointegration Analysis
# 4. VECM Estimation
# 5. Model Diagnostics
```

## Academic Context

This project demonstrates understanding of:

- **Econometric Theory:** Cointegration, error correction models
- **Applied Econometrics:** Model specification, estimation, interpretation
- **Financial Economics:** Macro-finance relationships
- **Empirical Research:** Data collection, hypothesis testing, inference

## Real-World Applications

**Investment Management:**
- Asset allocation based on macro regime
- Risk management using equilibrium deviations
- Tactical trading signals from error correction

**Policy Analysis:**
- Central bank decision-making
- Fiscal policy impacts on markets
- Unemployment-inflation-equity nexus

**Risk Assessment:**
- Systemic risk monitoring
- Early warning indicators
- Scenario analysis

## Theoretical Background

### Key Papers:

- **Engle & Granger (1987):** "Co-integration and Error Correction"
- **Johansen (1988, 1991):** "Statistical Analysis of Cointegration Vectors"
- **Stock & Watson (1988):** "Testing for Common Trends"
- **Fama & French (1989):** "Business Conditions and Expected Returns"

### Economic Theory:

- **Efficient Markets Hypothesis:** Stock prices reflect all available information
- **Phillips Curve:** Unemployment-inflation trade-off
- **Fisher Effect:** Nominal returns and inflation
- **Present Value Models:** Stock valuation via discounted cash flows

## Conclusion

This VECM analysis provides empirical evidence of long-run equilibrium relationships among key financial and macroeconomic variables. The error correction framework reveals how markets and the economy adjust to temporary deviations from equilibrium, offering insights for both investment decisions and policy analysis.

The finding of cointegration supports the view that financial markets are fundamentally linked to macroeconomic conditions, with implications for portfolio management, risk assessment, and economic forecasting.

---

**Note:** This analysis uses real-world data and advanced econometric techniques appropriate for graduate-level coursework in financial econometrics. Results should be interpreted within the context of model assumptions and limitations.
