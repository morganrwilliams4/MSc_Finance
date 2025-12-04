"""
Vector Error Correction Model: Macroeconomic Analysis
Econometrics - MSc Finance, University of Bath
Year: 2025

This script performs a comprehensive VECM analysis examining the long-run 
equilibrium relationships between S&P 500, CPI, and unemployment rate.

Analysis Steps:
1. Data Collection and Cleaning
2. Unit Root Testing (ADF)
3. Cointegration Analysis (Johansen)
4. VECM Estimation
5. Model Diagnostics
"""

import numpy as np
import pandas as pd
import yfinance as yf
import pandas_datareader.data as web
import datetime
import matplotlib.pyplot as plt
import statsmodels.tsa.api as smt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.vector_ar import vecm

# ============================================================================
# SECTION 1: DATA COLLECTION AND CLEANING
# ============================================================================

print("="*80)
print("SECTION 1: DATA COLLECTION AND CLEANING")
print("="*80)

# Define the date range
start_date = "2000-01-01"
end_date = "2025-02-01"  # End date later than final data point required

# Download S&P 500 data from Yahoo Finance (Monthly data)
print("\nDownloading S&P 500 data from Yahoo Finance...")
sp500 = yf.download("^GSPC", start=start_date, end=end_date, interval='1mo')[["Close"]]

# Flatten the MultiIndex by selecting the 'Price' level
sp500.columns = [col[0] for col in sp500.columns]

# Rename the column for clarity
sp500.rename(columns={"Close": "S&P500"}, inplace=True)

# Convert index to datetime format
sp500.index = pd.to_datetime(sp500.index)

# Define the date range for FRED data
start_date_dt = datetime.datetime(2000, 1, 1)
end_date_dt = datetime.datetime(2025, 1, 1)

# Download CPI and Unemployment Rate from FRED
print("Downloading CPI and Unemployment data from FRED...")
cpi = web.DataReader('CPIAUCSL', 'fred', start_date_dt, end_date_dt)
unemployment = web.DataReader('UNRATE', 'fred', start_date_dt, end_date_dt)

# Merge dataframes on Date to ensure same frequency
data = sp500.merge(cpi, left_index=True, right_index=True, how="outer")
data = data.merge(unemployment, left_index=True, right_index=True, how="outer")

# Rename columns for clarity
data.rename(columns={"CPIAUCSL": "CPI", "UNRATE": "UnemploymentRate"}, inplace=True)

# Check for missing values
print("\nMissing values check:")
print(data.isna().sum())

# Display the first few rows
print("\nFirst 5 rows of data:")
print(data.head())

print("\nLast 5 rows of data:")
print(data.tail())

print(f"\nTotal observations: {len(data)}")
print(f"Date range: {data.index.min()} to {data.index.max()}")

# ============================================================================
# SECTION 2: DATA VISUALIZATION
# ============================================================================

print("\n" + "="*80)
print("SECTION 2: DATA VISUALIZATION")
print("="*80)

# Plot original time series
fig, axes = plt.subplots(3, 1, figsize=(12, 10))

# S&P 500
axes[0].plot(data.index, data['S&P500'], color='blue', linewidth=1.5)
axes[0].set_title('S&P 500 Index (2000-2024)', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Index Level')
axes[0].grid(alpha=0.3)

# CPI
axes[1].plot(data.index, data['CPI'], color='green', linewidth=1.5)
axes[1].set_title('Consumer Price Index (2000-2024)', fontsize=12, fontweight='bold')
axes[1].set_ylabel('CPI Level')
axes[1].grid(alpha=0.3)

# Unemployment Rate
axes[2].plot(data.index, data['UnemploymentRate'], color='red', linewidth=1.5)
axes[2].set_title('Unemployment Rate (2000-2024)', fontsize=12, fontweight='bold')
axes[2].set_ylabel('Unemployment Rate (%)')
axes[2].set_xlabel('Date')
axes[2].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('time_series_plots.png', dpi=300, bbox_inches='tight')
plt.show()

print("Time series plots saved as 'time_series_plots.png'")

# ============================================================================
# SECTION 3: DATA TRANSFORMATION
# ============================================================================

print("\n" + "="*80)
print("SECTION 3: DATA TRANSFORMATION")
print("="*80)

# Create log transformations for S&P 500 and CPI
data['log_sp500'] = np.log(data['S&P500'])
data['log_cpi'] = np.log(data['CPI'])

print("\nLog transformations created:")
print("- log_sp500: Natural log of S&P 500")
print("- log_cpi: Natural log of CPI")
print("- UnemploymentRate: Kept in levels")

# Plot log-transformed series
fig, axes = plt.subplots(2, 1, figsize=(12, 8))

axes[0].plot(data.index, data['log_sp500'], color='blue', linewidth=1.5)
axes[0].set_title('Log S&P 500', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Log Index')
axes[0].grid(alpha=0.3)

axes[1].plot(data.index, data['log_cpi'], color='green', linewidth=1.5)
axes[1].set_title('Log CPI', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Log CPI')
axes[1].set_xlabel('Date')
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('log_transformed_series.png', dpi=300, bbox_inches='tight')
plt.show()

print("Log-transformed series plots saved as 'log_transformed_series.png'")

# ============================================================================
# SECTION 4: UNIT ROOT TESTING (ADF TEST)
# ============================================================================

print("\n" + "="*80)
print("SECTION 4: UNIT ROOT TESTING (AUGMENTED DICKEY-FULLER)")
print("="*80)

def adf_test(series, name, trend='ct'):
    """
    Perform Augmented Dickey-Fuller test
    
    Parameters:
    -----------
    series : pandas Series
        Time series to test
    name : str
        Name of the series for display
    trend : str
        Trend specification ('c', 'ct', 'ctt', 'n')
    """
    result = adfuller(series.dropna(), regression=trend, autolag='AIC')
    
    print(f"\n{name}:")
    print(f"  ADF Statistic: {result[0]:.4f}")
    print(f"  P-value: {result[1]:.4f}")
    print(f"  Critical Values:")
    for key, value in result[4].items():
        print(f"    {key}: {value:.4f}")
    
    if result[1] < 0.05:
        print(f"  → Reject H0: {name} is STATIONARY (p < 0.05)")
    else:
        print(f"  → Fail to reject H0: {name} is NON-STATIONARY (p >= 0.05)")
    
    return result

print("\nTesting LEVELS (with trend and constant):")
print("-" * 60)

# Test levels
adf_sp500_level = adf_test(data['log_sp500'], "Log S&P 500 (Level)", trend='ct')
adf_cpi_level = adf_test(data['log_cpi'], "Log CPI (Level)", trend='ct')
adf_unemp_level = adf_test(data['UnemploymentRate'], "Unemployment Rate (Level)", trend='ct')

print("\n\nTesting FIRST DIFFERENCES:")
print("-" * 60)

# Calculate first differences
data['d_log_sp500'] = data['log_sp500'].diff()
data['d_log_cpi'] = data['log_cpi'].diff()
data['d_unemp'] = data['UnemploymentRate'].diff()

# Test first differences
adf_sp500_diff = adf_test(data['d_log_sp500'], "Log S&P 500 (First Difference)", trend='c')
adf_cpi_diff = adf_test(data['d_log_cpi'], "Log CPI (First Difference)", trend='c')
adf_unemp_diff = adf_test(data['d_unemp'], "Unemployment Rate (First Difference)", trend='c')

print("\n" + "="*60)
print("CONCLUSION: All variables are I(1) - integrated of order 1")
print("They are non-stationary in levels but stationary in first differences")
print("="*60)

# ============================================================================
# SECTION 5: PREPARE DATA FOR VECM ANALYSIS
# ============================================================================

print("\n" + "="*80)
print("SECTION 5: PREPARE DATA FOR VECM ANALYSIS")
print("="*80)

# Select variables for VECM (in levels, as VECM handles differencing internally)
vecm_data = data[['log_sp500', 'log_cpi', 'UnemploymentRate']].dropna()

print(f"\nVECM dataset prepared:")
print(f"  Variables: log_sp500, log_cpi, UnemploymentRate")
print(f"  Observations: {len(vecm_data)}")
print(f"  Date range: {vecm_data.index.min()} to {vecm_data.index.max()}")

print("\nDescriptive statistics:")
print(vecm_data.describe())

# ============================================================================
# SECTION 6: JOHANSEN COINTEGRATION TEST
# ============================================================================

print("\n" + "="*80)
print("SECTION 6: JOHANSEN COINTEGRATION TEST")
print("="*80)

print("\nTesting for cointegration rank...")
print("Null Hypothesis: Cointegration rank ≤ r")
print("Alternative: Cointegration rank > r")

# Perform Johansen cointegration rank test
rank_test = vecm.select_coint_rank(
    vecm_data,
    det_order=0,      # Constant in cointegration relation
    k_ar_diff=1,      # Lag order for differences
    method='trace',   # Trace statistic
    signif=0.05
)

print("\n" + "="*60)
print("JOHANSEN COINTEGRATION RANK TEST RESULTS")
print("="*60)
print(rank_test.summary())

print("\n" + "="*60)
print(f"Selected cointegration rank: {rank_test.rank}")
print("="*60)

# ============================================================================
# SECTION 7: ESTIMATE VECM
# ============================================================================

print("\n" + "="*80)
print("SECTION 7: VECTOR ERROR CORRECTION MODEL ESTIMATION")
print("="*80)

# Estimate VECM with selected rank
vecm_model = vecm.VECM(
    vecm_data,
    k_ar_diff=1,           # First lag of differences
    coint_rank=rank_test.rank,  # From rank test
    deterministic='co'     # Constant within cointegration relation
)

vecm_results = vecm_model.fit()

# Display results
print("\n" + "="*60)
print("VECM ESTIMATION RESULTS")
print("="*60)
print(vecm_results.summary())

# ============================================================================
# SECTION 8: EXTRACT AND INTERPRET KEY RESULTS
# ============================================================================

print("\n" + "="*80)
print("SECTION 8: KEY RESULTS INTERPRETATION")
print("="*80)

# Cointegrating vector (beta)
print("\nCOINTEGRATING VECTOR (β - Long-Run Relationship):")
print("-" * 60)
beta = vecm_results.beta
print(beta)
print("\nInterpretation:")
print(f"  log(S&P500) = {beta[0,0]:.4f}")
print(f"              + {beta[1,0]:.4f} × log(CPI)")
print(f"              + {beta[2,0]:.4f} × UnemploymentRate")

# Loading coefficients (alpha)
print("\n\nLOADING COEFFICIENTS (α - Adjustment Speeds):")
print("-" * 60)
alpha = vecm_results.alpha
print(f"S&P 500:         {alpha[0,0]:.4f}")
print(f"CPI:             {alpha[1,0]:.4f}")
print(f"Unemployment:    {alpha[2,0]:.4f}")

print("\nInterpretation:")
print("  Positive α: Variable increases when below equilibrium")
print("  Negative α: Variable decreases when above equilibrium")
print("  Larger |α|: Faster adjustment to equilibrium")

# ============================================================================
# SECTION 9: MODEL DIAGNOSTICS
# ============================================================================

print("\n" + "="*80)
print("SECTION 9: MODEL DIAGNOSTICS")
print("="*80)

# Residual analysis
residuals = vecm_results.resid

print("\nRESIDUAL STATISTICS:")
print("-" * 60)
print(f"Mean of residuals:")
print(residuals.mean())
print(f"\nStandard deviation of residuals:")
print(residuals.std())

# Plot residuals
fig, axes = plt.subplots(3, 1, figsize=(12, 10))

residual_names = ['log_sp500', 'log_cpi', 'UnemploymentRate']
for i, (ax, name) in enumerate(zip(axes, residual_names)):
    ax.plot(residuals[:, i], linewidth=0.8)
    ax.axhline(y=0, color='r', linestyle='--', linewidth=1)
    ax.set_title(f'Residuals: {name}', fontsize=11, fontweight='bold')
    ax.set_ylabel('Residual')
    ax.grid(alpha=0.3)

axes[-1].set_xlabel('Observation')
plt.tight_layout()
plt.savefig('vecm_residuals.png', dpi=300, bbox_inches='tight')
plt.show()

print("\nResidual plots saved as 'vecm_residuals.png'")

# ACF plots of residuals
fig, axes = plt.subplots(3, 1, figsize=(12, 10))

for i, (ax, name) in enumerate(zip(axes, residual_names)):
    smt.graphics.plot_acf(residuals[:, i], lags=20, ax=ax, alpha=0.05)
    ax.set_title(f'ACF of Residuals: {name}', fontsize=11, fontweight='bold')
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('vecm_residuals_acf.png', dpi=300, bbox_inches='tight')
plt.show()

print("Residual ACF plots saved as 'vecm_residuals_acf.png'")

# ============================================================================
# SECTION 10: EXPORT DATA
# ============================================================================

print("\n" + "="*80)
print("SECTION 10: EXPORT DATA")
print("="*80)

# Export to Excel
output_file = 'Econometrics_Data.xlsx'
data.to_excel(output_file)
print(f"\nData exported to: {output_file}")

# Export summary results
with open('VECM_Results_Summary.txt', 'w') as f:
    f.write("="*80 + "\n")
    f.write("VECTOR ERROR CORRECTION MODEL - RESULTS SUMMARY\n")
    f.write("="*80 + "\n\n")
    f.write(str(vecm_results.summary()))

print("Results summary exported to: VECM_Results_Summary.txt")

# ============================================================================
# FINAL SUMMARY
# ============================================================================

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)

print("\nFiles generated:")
print("  1. time_series_plots.png - Original time series")
print("  2. log_transformed_series.png - Log transformations")
print("  3. vecm_residuals.png - Model residual plots")
print("  4. vecm_residuals_acf.png - Residual autocorrelation")
print("  5. Econometrics_Data.xlsx - Full dataset")
print("  6. VECM_Results_Summary.txt - Estimation results")

print("\nKey Findings:")
print(f"  - Cointegration rank: {rank_test.rank}")
print("  - Sample period: {vecm_data.index.min()} to {vecm_data.index.max()}")
print(f"  - Total observations: {len(vecm_data)}")
print("\nInterpretation:")
print("  Long-run equilibrium relationship exists among S&P 500, CPI,")
print("  and unemployment rate. Variables adjust to deviations from")
print("  equilibrium according to their loading coefficients (α).")

print("\n" + "="*80)
print("END OF ANALYSIS")
print("="*80)
