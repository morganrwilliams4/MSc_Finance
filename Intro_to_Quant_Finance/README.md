# Introduction to Quantitative Finance

**Course:** Introduction to Quantitative Finance  
**Institution:** University of Bath  
**Year:** 2025  
**Program:** MSc Finance

## Course Overview

This folder contains coursework projects from Introduction to Quantitative Finance, a foundational course covering essential quantitative methods in finance. The projects demonstrate proficiency in financial modeling, derivatives pricing, fixed income analysis, and empirical market research using Python.

## Projects

### 1. Bond Pricing and Duration Calculator
**Folder:** [`Bond_Pricing_Duration/`](./Bond_Pricing_Duration/)

Implementation of fixed income valuation models with tax considerations.

**Key Topics:**
- Present value analysis and discounted cash flows
- Macaulay duration calculation
- Tax-adjusted bond pricing
- Flexible payment frequency handling

**Skills:** Python, NumPy, financial mathematics, fixed income analysis

---

### 2. Binomial Options Pricing Model
**Folder:** [`Binomial_Options_Pricing/`](./Binomial_Options_Pricing/)

Binomial tree implementation for pricing European and American options with comparative analysis.

**Key Topics:**
- Binomial tree methodology
- Risk-neutral valuation
- Early exercise premium for American options
- Options payoff and profit visualization

**Skills:** Python, numerical methods, derivatives pricing, matplotlib visualization

---

### 3. Market Analysis and Options Trading Strategies
**Folder:** [`Market_Analysis_Options_Strategies/`](./Market_Analysis_Options_Strategies/)

**Project Type:** Final Project

Comprehensive analysis combining historical market data analysis with options trading strategy construction and evaluation.

**Case Study 1 - Historical Market Analysis:**
- 20-year S&P 500 and NASDAQ performance analysis
- Technical analysis using moving averages
- Return calculations and performance metrics

**Case Study 2 - Options Trading Strategies:**
- Black-Scholes pricing implementation
- Bull call spread analysis
- Bear put spread analysis
- Long straddle construction and evaluation

**Skills:** Python, yfinance, Black-Scholes model, strategy analysis, data visualization

---

## Core Competencies Demonstrated

### Quantitative Methods
- Numerical methods (binomial trees, iterative algorithms)
- Stochastic modeling and probability distributions
- Present value and time value of money calculations
- Statistical analysis and hypothesis testing

### Financial Theory
- Options pricing theory (Black-Scholes, binomial model)
- Fixed income mathematics (duration, convexity concepts)
- Risk-neutral valuation principles
- No-arbitrage pricing framework

### Derivatives & Trading
- European vs. American option valuation
- Multi-leg options strategies (spreads, straddles)
- Payoff diagram construction
- Risk-reward analysis and breakeven calculation

### Programming & Tools
- Python for financial modeling
- NumPy for numerical computation
- Matplotlib for data visualization
- SciPy for statistical functions
- yfinance for market data acquisition
- Pandas for time series analysis

### Risk Management
- Duration as interest rate risk measure
- Options Greeks (implicit understanding)
- Position risk profiles
- Hedging strategies

## Technical Stack
```python
# Core Libraries
numpy>=1.20.0
matplotlib>=3.5.0
scipy>=1.7.0
pandas>=1.3.0

# Data & Finance
yfinance>=0.2.0
```

## Project Structure
```
intro_to_quant_finance/
├── README.md                                    # This file
│
├── Bond_Pricing_Duration/
│   ├── README.md
│   └── bond_calculator.py
│
├── Binomial_Options_Pricing/
│   ├── README.md
│   └── binomial_options_pricing.py
│
└── Market_Analysis_Options_Strategies/
    ├── README.md
    └── market_analysis_options_strategies.py
```

## Key Learning Outcomes

By completing these projects, the following competencies were developed:

1. **Pricing Models:** Implementation of industry-standard valuation models
2. **Computational Finance:** Translating financial theory into working code
3. **Data Analysis:** Processing and analyzing real market data
4. **Risk Assessment:** Evaluating risk-reward profiles of financial instruments
5. **Visualization:** Communicating quantitative results effectively
6. **Financial Engineering:** Constructing synthetic positions and trading strategies

## Course Themes

### Fixed Income Markets
- Bond valuation with varying payment frequencies
- Duration and interest rate sensitivity
- Tax implications on bond returns

### Derivatives Markets
- Options pricing methodologies (closed-form and numerical)
- Understanding early exercise premiums
- Strategy construction and analysis
- Volatility trading concepts

### Empirical Finance
- Historical data analysis
- Technical indicators and trading signals
- Performance measurement and attribution
- Return calculation methodologies

### Quantitative Modeling
- Numerical methods implementation
- Model validation and testing
- Assumption analysis
- Computational efficiency considerations

## Applications

These projects demonstrate skills applicable to:

- **Quantitative Analysis:** Asset pricing, model validation
- **Trading:** Options strategy implementation, technical analysis
- **Risk Management:** Duration matching, options hedging
- **Portfolio Management:** Performance analysis, tactical allocation
- **Financial Engineering:** Structured products, derivatives design

## Theoretical Foundations

### Mathematical Finance
- Stochastic calculus (implicit in pricing models)
- Probability theory (risk-neutral measures)
- Optimization (option exercise decisions)

### Economic Principles
- No-arbitrage conditions
- Risk-neutral valuation
- Market efficiency considerations
- Time preferences and discounting

## Validation & Testing

All implementations include:
- Input validation and error handling
- Comparison with theoretical benchmarks
- Visual verification of results
- Edge case testing

## Further Reading

### Key References:
- Hull, J. C. (2018). *Options, Futures, and Other Derivatives*
- Luenberger, D. G. (1997). *Investment Science*
- Shreve, S. E. (2004). *Stochastic Calculus for Finance*
- Black, F., & Scholes, M. (1973). "The Pricing of Options and Corporate Liabilities"
- Cox, J. C., Ross, S. A., & Rubinstein, M. (1979). "Option Pricing: A Simplified Approach"

## Future Extensions

Potential enhancements across projects:
- **Greeks Calculation:** Delta, gamma, theta, vega, rho for all strategies
- **Monte Carlo Methods:** Alternative pricing approach for path-dependent options
- **Volatility Modeling:** GARCH models, implied volatility surfaces
- **Portfolio Optimization:** Mean-variance framework, risk budgeting
- **Machine Learning:** Predictive models for returns, volatility forecasting

---

## Contact & Usage

These projects were completed as part of the MSc Finance program at the University of Bath. The code is provided for educational and portfolio demonstration purposes.

For questions or collaboration opportunities, please refer to the main repository README.

---

*All projects demonstrate practical applications of quantitative finance theory using Python, showcasing both technical programming skills and deep understanding of financial markets.*
