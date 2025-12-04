# Binomial Tree Options Pricing Model

**Course:** Introduction to Quantitative Finance  
**Institution:** University of Bath  
**Year:** 2025

## Project Overview

This project implements the binomial tree model for pricing both European and American options (calls and puts). The model demonstrates the fundamental differences between European-style options (exercisable only at maturity) and American-style options (exercisable at any time before maturity), with visual comparison of option prices against their intrinsic payoff values.

## Methodology

### Binomial Tree Model

The binomial model discretizes the continuous stock price process into a series of up and down movements over N time steps, creating a recombining tree of possible stock prices.

**Key Parameters:**
- **Up factor (u):** `u = exp(σ√Δt)`
- **Down factor (d):** `d = 1/u`
- **Risk-neutral probability (p):** `p = (e^(rΔt) - d) / (u - d)`

### European Options

European options can only be exercised at maturity. Pricing uses backward induction through the tree without considering early exercise:
```
V(i,j) = e^(-rΔt) × [p × V(i+1,j) + (1-p) × V(i+1,j+1)]
```

### American Options

American options allow early exercise at any node. At each node, the option value is the maximum of:
1. **Intrinsic value** (immediate exercise)
2. **Continuation value** (holding the option)
```
V(i,j) = max(Intrinsic Value, Continuation Value)
```

This reflects the early exercise premium inherent in American options.

## Implementation Features

### Two Pricing Functions

#### `binomial_tree_eu(S, K, T, r, sigma, N, option_type)`
Prices European call and put options using standard backward induction.

**Parameters:**
- `S` (float): Current stock price
- `K` (float): Strike price
- `T` (float): Time to maturity (years)
- `r` (float): Risk-free interest rate
- `sigma` (float): Volatility (annualized)
- `N` (int): Number of time steps (default: 100)
- `option_type` (str): 'call' or 'put'

**Returns:** Option price (float)

#### `binomial_tree_am(S0, K, T, r, sigma, N, option_type)`
Prices American call and put options with early exercise consideration.

**Parameters:** Same as European version

**Returns:** Option price (float)

### Visualization

The code generates comparative plots showing:
1. **Call Options:** Payoff diagram vs European and American call prices
2. **Put Options:** Payoff diagram vs European and American put prices

## Example Usage
```python
import numpy as np
import matplotlib.pyplot as plt

# Parameters (Xmazon stock example)
S = 100          # Current stock price
K = 100          # Strike price (at-the-money)
T = 1            # 1 year to maturity
r = 0.05         # 5% risk-free rate
sigma = 0.3      # 30% volatility
N = 100          # 100 time steps

# Price European call
eu_call_price = binomial_tree_eu(S, K, T, r, sigma, N, "call")
print(f"European Call Price: ${eu_call_price:.2f}")

# Price American put
am_put_price = binomial_tree_am(S, K, T, r, sigma, N, "put")
print(f"American Put Price: ${am_put_price:.2f}")
```

## Key Results & Insights

### Stock Price Range Analysis
The code evaluates options across stock prices from $50 to $150 (strike = $100):

**Call Options:**
- Deep out-of-the-money (S << K): Both European and American calls worth near zero
- At-the-money (S ≈ K): Time value maximized
- Deep in-the-money (S >> K): Converges toward intrinsic value (S - K)
- **European = American for calls on non-dividend stocks** (no early exercise advantage)

**Put Options:**
- Deep out-of-the-money (S >> K): Minimal value
- At-the-money (S ≈ K): Maximum time value
- Deep in-the-money (S << K): American puts show **early exercise premium**
- **American > European for puts** (early exercise is optimal when sufficiently ITM)

### Early Exercise Premium

The most important finding: **American puts are worth more than European puts** because:
- Exercising a deep ITM put gives immediate access to cash (K - S)
- This cash can be invested at the risk-free rate
- The benefit outweighs the loss of time value when S is sufficiently low

For calls on non-dividend stocks, early exercise is never optimal (time value dominates).

## Visualizations

The generated plots show:

1. **Call Option Analysis**
   - Dashed line: Intrinsic payoff max(S - K, 0)
   - Circle markers: European call prices
   - X markers: American call prices
   - Observation: Lines overlap (no early exercise benefit)

2. **Put Option Analysis**
   - Dashed line: Intrinsic payoff max(K - S, 0)
   - Circle markers: European put prices
   - X markers: American put prices
   - Observation: American puts trade at premium when deep ITM

## Financial Concepts Demonstrated

- **Risk-neutral valuation:** Using risk-neutral probabilities for pricing
- **No-arbitrage pricing:** Model prevents arbitrage opportunities
- **Early exercise premium:** American option value over European
- **Time value of options:** Difference between option price and intrinsic value
- **Moneyness:** Impact of stock price relative to strike on option value
- **Put-call relationships:** Different early exercise characteristics

## Technical Skills

- Python programming for financial modeling
- Numerical methods (binomial trees)
- Dynamic programming (backward induction)
- Object-oriented function design
- Data visualization with matplotlib
- NumPy for efficient array operations

## Model Assumptions

- Constant volatility (σ) over the option's life
- Constant risk-free rate (r)
- No dividends paid during option life
- No transaction costs or taxes
- Continuous trading possible
- Log-normal stock price distribution

## Dependencies
```python
numpy>=1.20.0
matplotlib>=3.5.0
```

## Files

- `binomial_options_pricing.py` - Main implementation and visualization
- `README.md` - This documentation

## Extensions & Applications

### Possible Enhancements:
- **Dividends:** Incorporate discrete or continuous dividend yields
- **Greeks calculation:** Delta, gamma, theta, vega, rho
- **Convergence analysis:** Show how price converges as N → ∞
- **Exotic options:** Barrier, Asian, lookback options
- **Implied volatility:** Solve for σ given market price
- **Comparison with Black-Scholes:** Validate against closed-form solutions

### Real-World Applications:
- Equity options trading strategies
- Employee stock option valuation
- Real options in capital budgeting
- Risk management and hedging
- Convertible bond pricing

## Theoretical Foundation

The binomial model, developed by Cox, Ross, and Rubinstein (1979), provides:
- Intuitive discrete-time framework
- Foundation for understanding continuous-time models (Black-Scholes)
- Flexibility for path-dependent and American features
- Convergence to Black-Scholes as N → ∞

## Validation

For European options with sufficient time steps (N ≥ 100), the binomial model converges to Black-Scholes prices within acceptable tolerance, validating the implementation.

---

**Reference:** Cox, J. C., Ross, S. A., & Rubinstein, M. (1979). "Option Pricing: A Simplified Approach." *Journal of Financial Economics*, 7(3), 229-263.
