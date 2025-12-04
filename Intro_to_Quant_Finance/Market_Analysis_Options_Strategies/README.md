# Market Analysis and Options Trading Strategies

**Course:** Introduction to Quantitative Finance  
**Institution:** University of Bath  
**Year:** 2025  
**Project Type:** Final Project

## Project Overview

This comprehensive project consists of two major case studies that demonstrate proficiency in empirical market analysis and derivatives trading strategies. The project combines real-world market data analysis with theoretical options pricing and strategy construction, showcasing both descriptive analytics and quantitative modeling skills.

## Case Study 1: Historical Market Analysis (2004-2024)

### Objective
Analyze 20 years of US equity market data to identify trends, calculate performance metrics, and visualize market behavior using moving averages.

### Data Sources
- **S&P 500 Index (^SPX):** Broad market benchmark
- **NASDAQ Composite (^IXIC):** Technology-heavy index
- **VIX Index (^VIX):** Market volatility indicator
- **13-Week Treasury Bill (^IRX):** Risk-free rate proxy

**Period:** December 2004 - November 2024 (20 years)  
**Frequency:** Monthly (month-end observations)  
**Source:** Yahoo Finance via yfinance API

### Methodology

#### 1.1 Return Calculations

**Monthly Returns:**
```
R_t = (P_t / P_t-1 - 1) × 100
```

**20-Year Compounded Return:**
```
Compounded Return = (P_final / P_initial - 1) × 100
```

**Annualized Return (Log Method):**
```
Annualized Return = [ln(P_final / P_initial) / T] × 100
```
where T = number of years

This log-return method provides continuous compounding and is more appropriate for long-term performance measurement.

#### 1.2 Technical Analysis - Moving Averages

**3-Month Moving Average:** Short-term trend indicator
```
MA_3 = (P_t + P_t-1 + P_t-2) / 3
```

**36-Month Moving Average:** Long-term trend indicator
```
MA_36 = Σ(P_t-i) / 36, for i = 0 to 35
```

**Trading Signals:**
- **Golden Cross:** 3-month MA crosses above 36-month MA (bullish signal)
- **Death Cross:** 3-month MA crosses below 36-month MA (bearish signal)

### Key Findings

The analysis reveals:
- Long-term equity market performance over two decades
- Impact of major market events (2008 Financial Crisis, COVID-19, etc.)
- Cyclical patterns in NASDAQ Composite
- Effectiveness of moving average crossovers as momentum indicators

### Visualizations

- Time series plot of NASDAQ Composite with overlaid moving averages
- Clear identification of trend changes and market regimes
- Visual confirmation of technical trading signals

## Case Study 2: Options Trading Strategies

### Objective
Implement and analyze three popular options trading strategies using Black-Scholes pricing, evaluating their risk-reward profiles across different market scenarios.

### Black-Scholes Model Implementation

#### Call Option Pricing:
```
C = S₀N(d₁) - Ke^(-rT)N(d₂)

where:
d₁ = [ln(S/K) + (r + σ²/2)T] / (σ√T)
d₂ = d₁ - σ√T
```

#### Put Option Pricing:
```
P = Ke^(-rT)N(-d₂) - S₀N(-d₁)
```

### Parameters
- **Current Stock Price (S₀):** $100
- **Risk-Free Rate (r):** 5%
- **Volatility (σ):** 25%
- **Time to Maturity (T):** 1 year
- **Strike Prices:** $90, $100, $110

### Strategy 1: Bull Call Spread

**Construction:**
- **Long 1 Call** at K₁ = $90 (lower strike)
- **Short 1 Call** at K₂ = $100 (higher strike)

**Market View:** Moderately bullish

**Payoff at Expiration:**
```
Payoff = max(S - K₁, 0) - max(S - K₂, 0)
```

**Characteristics:**
- **Maximum Profit:** K₂ - K₁ - Net Premium = Limited
- **Maximum Loss:** Net Premium Paid = Limited
- **Breakeven:** K₁ + Net Premium
- **Best Case:** Stock rises to or above K₂

**When to Use:**
- Expect moderate price increase
- Want to reduce upfront cost vs. naked long call
- Willing to cap maximum profit for lower risk

### Strategy 2: Bear Put Spread

**Construction:**
- **Long 1 Put** at K₃ = $110 (higher strike)
- **Short 1 Put** at K₂ = $100 (lower strike)

**Market View:** Moderately bearish

**Payoff at Expiration:**
```
Payoff = max(K₃ - S, 0) - max(K₂ - S, 0)
```

**Characteristics:**
- **Maximum Profit:** K₃ - K₂ - Net Premium = Limited
- **Maximum Loss:** Net Premium Paid = Limited
- **Breakeven:** K₃ - Net Premium
- **Best Case:** Stock falls to or below K₂

**When to Use:**
- Expect moderate price decrease
- Want defined risk vs. short stock
- Reduce cost compared to naked long put

### Strategy 3: Long Straddle

**Construction:**
- **Long 1 Call** at K = $100
- **Long 1 Put** at K = $100

**Market View:** Expect high volatility (direction-neutral)

**Payoff at Expiration:**
```
Payoff = max(S - K, 0) + max(K - S, 0) = |S - K|
```

**Characteristics:**
- **Maximum Profit:** Unlimited (theoretically)
- **Maximum Loss:** Total Premium Paid = Limited
- **Breakeven Points:** 
  - Lower: K - Total Premium
  - Upper: K + Total Premium
- **Best Case:** Large price movement in either direction

**When to Use:**
- Expect significant volatility (earnings, FDA approval, etc.)
- Uncertain about direction but confident about magnitude
- Implied volatility is relatively low

## Comparative Analysis

| Strategy | Market View | Max Profit | Max Loss | Breakeven Points | Complexity |
|----------|-------------|------------|----------|------------------|------------|
| Bull Call Spread | Moderately Bullish | Limited | Limited | 1 | Medium |
| Bear Put Spread | Moderately Bearish | Limited | Limited | 1 | Medium |
| Long Straddle | High Volatility | Unlimited | Limited | 2 | Medium |

### Risk-Reward Trade-offs

**Spreads (Bull Call & Bear Put):**
- ✅ Lower cost than naked options
- ✅ Defined maximum risk
- ❌ Capped profit potential
- ✅ Better risk-reward ratio for moderate moves

**Straddle:**
- ✅ Profits from large moves in either direction
- ✅ No directional risk
- ❌ High upfront cost (buying two options)
- ❌ Time decay works against position
- ✅ Unlimited profit potential (on upside)

## Technical Implementation

### Key Features

1. **Data Acquisition:** Automated download from Yahoo Finance
2. **Data Processing:** Monthly resampling and return calculations
3. **Options Pricing:** Black-Scholes implementation with scipy
4. **Strategy Construction:** Modular payoff and profit calculations
5. **Visualization:** Comprehensive plotting of strategies

### Functions Implemented

#### `black_scholes_call(S, K, T, r, sigma)`
Calculates European call option price using Black-Scholes formula.

**Parameters:**
- `S`: Current stock price
- `K`: Strike price
- `T`: Time to maturity (years)
- `r`: Risk-free rate
- `sigma`: Volatility

**Returns:** Call option price

#### `black_scholes_put(S, K, T, r, sigma)`
Calculates European put option price using Black-Scholes formula.

**Parameters:** Same as call function

**Returns:** Put option price

### Visualizations Generated

For each strategy, the code produces:
1. **Payoff Diagram:** Shows intrinsic value at expiration
2. **Profit/Loss Diagram:** Incorporates premium costs
3. **Breakeven Analysis:** Visual markers for breakeven points

## Key Results Summary

### Case Study 1 Output
- Monthly return series for S&P 500
- 20-year compounded return
- Annualized return (log method)
- NASDAQ moving average crossover signals

### Case Study 2 Output

**Bull Call Spread:**
- Breakeven point
- Maximum profit and loss
- Optimal exercise scenario

**Bear Put Spread:**
- Breakeven point
- Maximum profit and loss
- Optimal exercise scenario

**Long Straddle:**
- Two breakeven points
- Maximum loss (at K = $100)
- Profit regions (S < lower breakeven or S > upper breakeven)

## Financial Concepts Demonstrated

### Case Study 1
- Return calculation methodologies
- Compound vs. simple returns
- Log returns for multi-period analysis
- Technical analysis and momentum indicators
- Moving average trading systems

### Case Study 2
- Options pricing theory (Black-Scholes)
- Synthetic position construction
- Risk management through spreads
- Volatility trading (straddle)
- Payoff diagrams and profit analysis
- Option Greeks implications (implicitly)

## Technical Skills

- **Python Libraries:** yfinance, numpy, scipy, matplotlib
- **Statistical Methods:** Probability distributions (normal CDF)
- **Data Manipulation:** Time series resampling, pandas operations
- **Quantitative Modeling:** Black-Scholes formula implementation
- **Visualization:** Multi-panel plots, overlay charts
- **Financial Engineering:** Derivatives strategy construction

## Dependencies
```python
yfinance>=0.2.0
numpy>=1.20.0
matplotlib>=3.5.0
scipy>=1.7.0
pandas>=1.3.0
```

## Usage Example
```python
# Case Study 1: Download and analyze market data
import yfinance as yf

tickers = ["^SPX", "^IXIC", "^VIX", "^IRX"]
data = yf.download(tickers, start="2004-12-01", end="2024-11-30")

# Case Study 2: Price options and analyze strategies
call_price = black_scholes_call(S=100, K=100, T=1, r=0.05, sigma=0.25)
put_price = black_scholes_put(S=100, K=100, T=1, r=0.05, sigma=0.25)

# Construct long straddle
straddle_cost = call_price + put_price
```

## Extensions & Applications

### Possible Enhancements:

**Case Study 1:**
- Correlation analysis between indices and VIX
- Regime detection algorithms
- Backtesting of moving average crossover strategy
- Risk-adjusted performance metrics (Sharpe ratio, maximum drawdown)

**Case Study 2:**
- Greeks calculation (delta, gamma, theta, vega)
- Implied volatility surface analysis
- Monte Carlo simulation for strategy outcomes
- Transaction costs and bid-ask spread considerations
- Dynamic hedging strategies

### Real-World Applications:

- **Portfolio Management:** Using moving averages for tactical asset allocation
- **Options Trading:** Implementing spreads and volatility strategies
- **Risk Management:** Hedging equity positions with options
- **Market Analysis:** Identifying trends and regime changes
- **Volatility Trading:** Exploiting mispriced volatility through straddles

## Model Assumptions

### Black-Scholes Assumptions:
- Constant volatility and risk-free rate
- No dividends during option life
- European-style exercise
- No transaction costs
- Continuous trading
- Log-normal stock price distribution

### Market Data Assumptions:
- Month-end prices representative of monthly performance
- No adjustments for dividends or splits (handled by yfinance)
- Data quality from Yahoo Finance is reliable

## Validation & Robustness

- Black-Scholes prices validated against market standards
- Strategy payoffs verified at key price points
- Moving averages calculated using pandas built-in functions (tested framework)
- Visual inspection of results for reasonableness

## Conclusion

This project demonstrates comprehensive skills in:
1. **Empirical Finance:** Working with real market data
2. **Derivatives Pricing:** Implementing Black-Scholes model
3. **Trading Strategies:** Constructing and analyzing multi-leg options positions
4. **Technical Analysis:** Using moving averages for trend identification
5. **Python Programming:** Data acquisition, numerical methods, visualization

The combination of historical analysis and forward-looking options strategies showcases both descriptive and prescriptive analytics capabilities essential for quantitative finance roles.

---

**References:**
- Black, F., & Scholes, M. (1973). "The Pricing of Options and Corporate Liabilities." *Journal of Political Economy*, 81(3), 637-654.
- Hull, J. C. (2018). *Options, Futures, and Other Derivatives* (10th ed.). Pearson.
- Yahoo Finance Data: Retrieved via yfinance Python library.
