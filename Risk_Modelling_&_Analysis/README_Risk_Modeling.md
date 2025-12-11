# Risk Modelling and Analysis

**Course:** Risk Modelling and Analysis  
**Institution:** University of Bath  
**Year:** 2025  
**Program:** MSc Finance

## Project Overview

This comprehensive risk modeling project demonstrates proficiency in both **market risk** and **credit risk** analysis using modern quantitative techniques. The project combines Monte Carlo simulation for portfolio risk assessment with machine learning for credit default prediction, showcasing the application of advanced statistical and computational methods to real-world financial risk management.

## Project Structure

This coursework consists of two major components:

### Part 1: Portfolio Risk Analysis (VaR & CVaR)
Monte Carlo simulation-based portfolio risk assessment using real market data from S&P 500, Gold (GC=F), and Real Estate (VNQ).

### Part 2: Credit Default Prediction
Machine learning classification models (Logistic Regression, Random Forest) to predict loan defaults using borrower characteristics and credit history.

## Part 1: Portfolio Risk Analysis

### Objective
Quantify market risk for a multi-asset portfolio using Value-at-Risk (VaR) and Conditional Value-at-Risk (CVaR) through Monte Carlo simulation.

### Data Sources
- **S&P 500 Index (^GSPC):** US large-cap equities
- **Gold Futures (GC=F):** Commodity exposure
- **Real Estate (VNQ):** Vanguard Real Estate ETF

**Period:** January 2019 - December 2024 (6 years of daily prices)  
**Source:** Yahoo Finance via yfinance API

### Portfolio Specification
```python
Initial Portfolio Value: $1,000,000
Asset Weights:
  - S&P 500: 30%
  - Gold: 20%
  - Real Estate: 50%
```

### Methodology

#### 1. Data Preparation
**Log Returns Calculation:**
```
r_t = ln(P_t / P_{t-1})
```

**Why Log Returns?**
- Time-additive (multi-period returns = sum of log returns)
- Approximately normally distributed
- Symmetry between gains and losses
- Mathematically convenient for modeling

**Summary Statistics:**
- Mean daily returns (μ)
- Volatility (σ) - standard deviation of returns
- Skewness - asymmetry of distribution
- Kurtosis - tail heaviness (fat tails)

#### 2. Covariance Matrix & Correlation
```python
V = log_returns.cov()  # Covariance matrix
```

**Purpose:**
- Captures linear dependencies between assets
- Essential for portfolio variance calculation
- Determines diversification benefits

**Correlation Insights:**
- High correlation → Less diversification
- Low/negative correlation → Better risk reduction
- Gold often negatively correlated with equities

#### 3. Monte Carlo Simulation

**Cholesky Decomposition:**
```python
C = np.linalg.cholesky(V)
```

**Purpose:** Decompose covariance matrix to generate correlated random returns

**Simulation Process:**
1. Generate independent standard normal random variables
2. Apply Cholesky matrix to induce correlation
3. Scale by volatility and add drift
4. Compound returns over time horizon
5. Calculate portfolio value evolution

**Parameters:**
- **Time Horizon (T):** 1 year
- **Time Steps:** 250 (daily intervals)
- **Number of Simulations:** 10,000 paths
- **Interval Size:** T / 250 = 1/250 year

**Formula for Correlated Returns:**
```
Y = X @ C.T * √(Δt) + (μ - σ²/2) * Δt
```

Where:
- X ~ N(0,1) independent normals
- C = Cholesky factor of covariance matrix
- μ = mean return vector
- σ² = variance vector
- Δt = time interval

**Portfolio Value Evolution:**
```
P_{t+1} = P_t * exp(w · Y_t)
```

Where:
- w = weight vector [0.3, 0.2, 0.5]
- Y_t = correlated asset returns at time t

#### 4. Risk Metrics

**Value-at-Risk (VaR):**
```
VaR_α = -Quantile(P_T, α)
```

**Interpretation:** Maximum expected loss at α confidence level over horizon T

**Common Confidence Levels:**
- 95% VaR: Loss exceeded 5% of the time
- 99% VaR: Loss exceeded 1% of the time

**Conditional Value-at-Risk (CVaR / Expected Shortfall):**
```
CVaR_α = E[L | L > VaR_α]
```

**Interpretation:** Average loss given that loss exceeds VaR threshold

**Why CVaR?**
- Coherent risk measure (satisfies subadditivity)
- Captures tail risk beyond VaR
- More informative for extreme events
- Preferred by regulators (Basel III)

### Key Findings (Expected)

**Portfolio Characteristics:**
- Simulated final portfolio values show distribution
- Mean final value indicates expected return
- Standard deviation quantifies portfolio risk
- Skewness reveals asymmetry (crashes vs gains)

**Risk Metrics:**
- **95% VaR:** Maximum loss at 95% confidence
- **99% VaR:** Extreme loss threshold
- **CVaR:** Average loss in worst scenarios
- **VaR/CVaR spread:** Tail risk quantification

**Diversification Benefits:**
- Portfolio volatility < weighted average of individual volatilities
- Correlation effects visible in simulation spread
- Gold potentially reduces downside risk

## Part 2: Credit Default Prediction

### Objective
Build machine learning classification models to predict loan defaults based on borrower characteristics and credit history.

### Data
**Source:** `view.csv` dataset (likely containing loan/credit data)

**Expected Features (typical credit risk dataset):**
- **Borrower Demographics:** Age, income, employment status
- **Credit History:** Credit score, previous defaults, inquiries
- **Loan Characteristics:** Amount, term, interest rate, purpose
- **Financial Ratios:** Debt-to-income, loan-to-value
- **Payment History:** Delinquencies, late payments

**Target Variable:** Default (binary: 0 = no default, 1 = default)

### Data Challenges

**Class Imbalance:**
- Defaults are rare events (typically <10% of loans)
- Accuracy is misleading as a metric
- Model may predict "no default" for all cases and achieve high accuracy

**Implication:** Must use appropriate evaluation metrics (precision, recall, F1, AUC-ROC)

### Methodology

#### 1. Exploratory Data Analysis

**Correlation Analysis:**
```python
corr = data.corr()
sns.heatmap(corr, cmap='coolwarm', center=0)
```

**Purpose:**
- Identify multicollinearity
- Understand feature relationships
- Guide feature selection

**Distribution Analysis:**
- Check for skewness in financial variables
- Identify outliers
- Assess normality assumptions

#### 2. Data Preprocessing

**Train-Test Split:**
```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
```

**Standardization (if needed):**
- Scale features to have mean 0, std 1
- Important for logistic regression
- Less critical for tree-based models

#### 3. Machine Learning Models

**Logistic Regression:**
```
P(Default = 1 | X) = 1 / (1 + exp(-β₀ - β₁X₁ - ... - βₙXₙ))
```

**Advantages:**
- Interpretable coefficients
- Fast training
- Probabilistic output
- Baseline model

**Random Forest:**
Ensemble of decision trees voting on classification

**Advantages:**
- Handles non-linear relationships
- Feature importance built-in
- Robust to outliers
- No feature scaling needed
- Handles interactions automatically

**Model Training:**
```python
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
```

#### 4. Model Evaluation

**Confusion Matrix:**
```
                Predicted
                No    Yes
Actual  No     TN    FP
        Yes    FN    TP
```

**Metrics:**

**Accuracy:**
```
Accuracy = (TP + TN) / Total
```
⚠️ Misleading for imbalanced data!

**Precision:**
```
Precision = TP / (TP + FP)
```
"Of predicted defaults, what % actually defaulted?"

**Recall (Sensitivity):**
```
Recall = TP / (TP + FN)
```
"Of actual defaults, what % did we catch?"

**F1 Score:**
```
F1 = 2 * (Precision * Recall) / (Precision + Recall)
```
Harmonic mean balancing precision and recall

**ROC-AUC:**
- Plots True Positive Rate vs False Positive Rate
- AUC = 0.5: Random classifier
- AUC = 1.0: Perfect classifier
- AUC > 0.7: Generally acceptable
- AUC > 0.8: Good performance

### Business Context

**Cost of Errors:**

**Type I Error (False Positive):**
- Predict default when borrower won't default
- Cost: Lost revenue from rejected good loan
- Typically lower cost

**Type II Error (False Negative):**
- Predict no default when borrower will default
- Cost: Full loan loss + collection costs
- Typically much higher cost

**Optimal Threshold:**
- May not be 0.5
- Should minimize expected cost:
  ```
  Expected Cost = P(FP) * Cost(FP) + P(FN) * Cost(FN)
  ```

### Key Findings (Expected)

**Model Performance:**
- Random Forest likely outperforms Logistic Regression
- AUC-ROC scores for both models
- Precision-recall tradeoff analysis

**Important Features:**
- Credit score (likely #1 predictor)
- Debt-to-income ratio
- Employment length
- Loan amount
- Previous delinquencies

**Business Recommendations:**
- Threshold selection based on cost-benefit
- Feature importance for underwriting rules
- Model deployment considerations

## Technical Implementation

### Libraries Used

**Data & Analysis:**
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew, kurtosis
```

**Financial Data:**
```python
import yfinance as yf
```

**Machine Learning:**
```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, confusion_matrix, 
    precision_score, recall_score, 
    roc_auc_score, roc_curve
)
```

### Computational Techniques

**Monte Carlo Simulation:**
- 10,000 scenarios for statistical significance
- Vectorized operations for efficiency
- Cholesky decomposition for correlation

**Matrix Operations:**
- Covariance matrix computation
- Linear algebra for portfolio calculations
- NumPy for fast array operations

**Machine Learning:**
- Supervised learning (classification)
- Ensemble methods (Random Forest)
- Cross-validation (implied)
- Hyperparameter tuning (potential)

## Integration of Both Parts

### Holistic Risk Management

**Portfolio Risk (Part 1):**
- **Market Risk:** External factors affecting asset values
- **Systematic Risk:** Cannot be diversified away
- **Measurement:** VaR, CVaR, volatility

**Credit Risk (Part 2):**
- **Idiosyncratic Risk:** Borrower-specific factors
- **Can be diversified:** Through loan portfolio
- **Measurement:** Probability of default, loss given default

**Combined View:**
A comprehensive risk management framework needs both:
1. **Top-down:** Market conditions affecting portfolio (Part 1)
2. **Bottom-up:** Individual credit quality assessment (Part 2)

### Real-World Application

**Bank Risk Management:**
1. **Trading Book:** Part 1 methods for market risk
2. **Loan Book:** Part 2 methods for credit risk
3. **Capital Requirements:** Both feed into Basel III calculations

**Investment Firm:**
1. **Asset Allocation:** Portfolio risk optimization
2. **Security Selection:** Credit analysis for fixed income
3. **Risk Budgeting:** Allocate risk across strategies

## Skills Demonstrated

### Quantitative Finance
✅ Monte Carlo simulation for risk assessment  
✅ VaR and CVaR calculation and interpretation  
✅ Portfolio theory and diversification  
✅ Correlation and covariance analysis  
✅ Time series analysis of financial returns  

### Statistical Methods
✅ Probability distributions (normal, empirical)  
✅ Hypothesis testing (implicit in model evaluation)  
✅ Summary statistics and moment analysis  
✅ Cholesky decomposition  
✅ Random number generation with correlation  

### Machine Learning
✅ Classification algorithms (Logistic, Random Forest)  
✅ Train-test split methodology  
✅ Model evaluation metrics  
✅ Handling imbalanced datasets  
✅ Feature importance analysis  
✅ ROC-AUC interpretation  

### Programming
✅ Python for financial analysis  
✅ NumPy for numerical computation  
✅ Pandas for data manipulation  
✅ Scikit-learn for machine learning  
✅ Matplotlib/Seaborn for visualization  
✅ API integration (yfinance)  

### Financial Acumen
✅ Understanding of market risk vs credit risk  
✅ Portfolio construction principles  
✅ Credit underwriting concepts  
✅ Risk-return tradeoff analysis  
✅ Regulatory awareness (Basel, VaR requirements)  

## Model Limitations & Assumptions

### Part 1: Portfolio Risk

**Assumptions:**
- Log returns are normally distributed (may not hold during crises)
- Correlations are constant (they change over time)
- No transaction costs or taxes
- Continuous trading (not realistic)
- Historical data predicts future (regime shifts possible)

**Limitations:**
- Doesn't capture fat tails and extreme events well
- Linear correlations miss non-linear dependencies
- Single time horizon (1 year) - may want multiple
- No liquidity risk consideration

### Part 2: Credit Default

**Assumptions:**
- Features are predictive of future defaults
- Training data representative of future loans
- Feature relationships are stable
- Missing data is missing at random

**Limitations:**
- Model drift over time (needs retraining)
- Doesn't capture macroeconomic shocks
- May not generalize to new loan products
- Interpretability-performance tradeoff (RF vs LogReg)

## Potential Enhancements

### Portfolio Risk Enhancements
- **GARCH Models:** Time-varying volatility
- **Copulas:** Non-linear dependencies
- **Extreme Value Theory:** Better tail risk modeling
- **Historical Simulation:** Non-parametric approach
- **Stress Testing:** Scenario analysis
- **Multiple Horizons:** 1-day, 10-day, 1-year VaR

### Credit Risk Enhancements
- **Feature Engineering:** Create interaction terms, ratios
- **Advanced Models:** XGBoost, Neural Networks
- **SMOTE:** Synthetic oversampling for imbalance
- **Calibration:** Probability calibration curves
- **Explainability:** SHAP values for interpretability
- **Time-to-Default:** Survival analysis instead of binary

### Integration
- **Credit-Adjusted VaR:** Combine both risk types
- **Economic Capital:** Unified risk measure
- **Expected Loss:** PD × LGD × EAD framework
- **Portfolio Credit Risk:** Correlation of defaults

## Files

- `risk_modeling_analysis.ipynb` - Jupyter notebook with complete analysis
- `README.md` - This documentation

**Generated Outputs (when run):**
- `stock_data.csv` - Downloaded market data
- Monte Carlo simulation plots
- VaR/CVaR visualizations
- Confusion matrices
- ROC curves
- Feature importance plots

## Running the Analysis

### Prerequisites
```bash
pip install numpy pandas matplotlib seaborn scipy scikit-learn yfinance
```

### Execution
```bash
jupyter notebook risk_modeling_analysis.ipynb
```

### Expected Runtime
- Data download: ~30 seconds
- Monte Carlo simulation: 1-2 minutes (10,000 paths)
- ML model training: 1-2 minutes
- Total: ~5 minutes

## Academic Context

This coursework demonstrates proficiency appropriate for:
- **Graduate-level risk management** courses
- **Quantitative finance** programs
- **Financial engineering** specializations
- **Data science in finance** roles

The combination of simulation-based risk assessment and machine learning classification showcases both traditional quantitative finance methods and modern data science techniques.

## Real-World Relevance

### Regulatory Compliance
- **Basel III:** VaR for market risk capital
- **IFRS 9 / CECL:** Expected credit loss models
- **Stress Testing:** Fed/ECB requirements

### Industry Applications
- **Risk Management:** Portfolio risk monitoring
- **Trading Desks:** Position sizing, limit setting
- **Credit Underwriting:** Automated loan decisions
- **Asset Management:** Risk-adjusted performance

---

**Project Type:** Coursework / Portfolio  
**Complexity:** Advanced  
**Time Investment:** 15-20 hours for complete implementation and analysis  
**Key Skills:** Monte Carlo Simulation, VaR/CVaR, Machine Learning, Credit Risk Modeling
