# Market Value Growth Momentum Strategy

**Course:** Programming for Finance  
**Grade:** 80%  
**Institution:** University of Bath  
**Year:** 2025

## Project Overview

This project implements and evaluates a momentum trading strategy based on market value growth, using CRSP (Center for Research in Security Prices) data from 2019-2023. The analysis examines whether stocks with high past market capitalization growth continue to outperform, or if a reversal effect occurs.

## Methodology

### Data Source & Preparation
- **Database:** CRSP monthly stock file (MSF) via WRDS
- **Period:** January 2019 - December 2023
- **Universe:** Common stocks (share codes 10, 11) listed on major exchanges (NYSE, AMEX, NASDAQ)
- **Sample Size:** Filtered dataset containing major exchange-traded common stocks

### Strategy Design

1. **Signal Construction**
   - Calculate 11-month market value growth: (MV_t-1 - MV_t-11) / MV_t-11
   - Market value = |Price| × Shares Outstanding
   - Winsorize extreme growth rates at 0.1% and 99.9% percentiles

2. **Portfolio Formation**
   - Sort stocks into 10 decile portfolios based on market value growth
   - Portfolio 1: Lowest growth (losers)
   - Portfolio 10: Highest growth (winners)
   - Rebalance monthly

3. **Return Calculation**
   - **Equal-weighted portfolios:** Average return across all stocks in each portfolio
   - **Value-weighted portfolios:** Returns weighted by market capitalization
   - Hold period: 1 month forward

4. **Long-Short Strategy**
   - **Equal-weighted:** Long Portfolio 1 (losers), Short Portfolio 10 (winners)
   - **Value-weighted:** Long Portfolio 10 (winners), Short Portfolio 1 (losers)

## Key Findings

### Portfolio Performance

#### Equal-Weighted Strategy
- Shows evidence of **reversal effect**
- Lowest growth stocks (Portfolio 1) outperform highest growth stocks (Portfolio 10)
- Long-short strategy: Long losers, Short winners

#### Value-Weighted Strategy
- Shows evidence of **momentum effect**
- Highest growth stocks (Portfolio 10) outperform lowest growth stocks (Portfolio 1)
- Long-short strategy: Long winners, Short losers
- The momentum effect is stronger when weighted by market cap, suggesting large-cap growth stocks drive performance

### Risk-Adjusted Performance

The value-weighted long-short portfolio demonstrates:
- **Positive alpha** after controlling for market risk (CAPM)
- **Volatility analysis:** Strategy volatility compared to market benchmark
- **Sharpe Ratio:** Risk-adjusted return measurement
- **Treynor Ratio:** Return per unit of systematic risk
- **Beta exposure:** Sensitivity to market movements

### CAPM Analysis
- Regression of excess portfolio returns on market risk premium
- Alpha estimates across all 10 portfolios
- Security Market Line (SML) comparison: Beta vs. Mean Return relationship

## Technical Implementation

### Tools & Libraries
- **Python 3.x**
- **WRDS:** Database connection and SQL queries
- **pandas:** Data manipulation and analysis
- **numpy:** Numerical computations
- **matplotlib:** Data visualization
- **statsmodels:** Statistical modeling and regression analysis

### Key Features
- Automated data extraction from CRSP via SQL
- Robust handling of missing data and outliers
- Monthly portfolio rebalancing logic
- Cumulative return tracking for performance visualization
- Risk metrics calculation (volatility, Sharpe ratio, Treynor ratio)
- CAPM regression framework

## Visualizations

The project includes several key visualizations:
1. **Average returns** by portfolio (bar charts)
2. **Cumulative returns** for long and short positions (time series)
3. **Long-short vs. market** performance comparison
4. **Security Market Line:** Beta-return relationship across portfolios

## Results Interpretation

The contrasting results between equal-weighted and value-weighted portfolios suggest:
- **Small-cap reversal:** Smaller stocks exhibit mean-reversion in market value growth
- **Large-cap momentum:** Larger stocks show persistence in growth trends
- **Market microstructure:** Different return dynamics across market cap segments
- **Practical implications:** Strategy performance highly dependent on weighting scheme and implementation

## Files in this Repository

- `momentum_strategy.py` - Main analysis code
- `README.md` - This file

## Usage

```python
# Requires WRDS account and credentials
import wrds
db = wrds.Connection()

# Run the analysis
# (Execute the main script)
```

## Academic Context

This project demonstrates proficiency in:
- Empirical asset pricing research methods
- Trading strategy development and backtesting
- Factor model implementation (CAPM)
- Financial data handling at scale
- Quantitative portfolio construction
- Performance attribution and risk analysis

## References

- CRSP (Center for Research in Security Prices), University of Chicago Booth School of Business
- Fama-French Research Data Library
- Jegadeesh, N., & Titman, S. (1993). Returns to Buying Winners and Selling Losers: Implications for Stock Market Efficiency. *Journal of Finance*.

---

**Note:** This project was completed as coursework and achieved a grade of 80%. The analysis is for educational purposes and should not be considered investment advice.
