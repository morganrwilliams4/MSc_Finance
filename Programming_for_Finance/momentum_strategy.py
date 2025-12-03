"""
Market Value Growth Momentum Strategy Analysis
Programming for Finance - MSc Finance, University of Bath
Grade: 80%

This script implements a momentum trading strategy based on market value growth
using CRSP data from 2019-2023. It analyzes both equal-weighted and value-weighted
portfolios to test for momentum and reversal effects.
"""

# ============================================================================
# SETUP AND DATA EXTRACTION
# ============================================================================

import wrds
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import numpy as np

# Connect to WRDS database
db = wrds.Connection()

# Extract CRSP monthly stock data
sql_query = """
SELECT a.permno, a.date, a.ret, a.prc, a.shrout, b.shrcd, b.exchcd
FROM crsp.msf a
LEFT JOIN crsp.msenames b 
    ON a.permno = b.permno 
    AND a.date >= b.namedt 
    AND a.date <= b.nameendt
WHERE a.date >= '2019-01-01' 
    AND a.date <= '2023-12-31'
"""

raw_data = db.raw_sql(sql_query, date_cols=['date'])
data = raw_data.copy()
print(f"Initial dataset contains {data.shape[0]} rows.")

# ============================================================================
# DATA FILTERING AND PREPROCESSING
# ============================================================================

# Filter for common stocks on major exchanges
crsp = data.loc[
    (data['shrcd'].isin([10, 11])) &  # Common stocks only
    (data['exchcd'].isin([1, 2, 3]))   # NYSE, AMEX, NASDAQ
].copy()
print(f"Filtered dataset contains {crsp.shape[0]} rows.")

# Calculate market value (market cap in $ millions)
crsp['mv'] = (crsp['prc'].abs() * crsp['shrout']) / 1000

# Sort data for lagged calculations
crsp = crsp.sort_values(['permno', 'date'], ignore_index=True)

# ============================================================================
# SIGNAL CONSTRUCTION: MARKET VALUE GROWTH
# ============================================================================

# Calculate lagged market values (t-1 to t-11)
for i in range(1, 12):
    crsp[f'mv_lag{i}'] = crsp.groupby('permno')['mv'].shift(i)

# Calculate 11-month market value growth rate
crsp['mv_growth'] = (crsp['mv_lag1'] - crsp['mv_lag11']) / crsp['mv_lag11']

# Winsorize extreme values at 0.1% and 99.9% percentiles
pctl_low = crsp['mv_growth'].quantile(0.001)
pctl_high = crsp['mv_growth'].quantile(0.999)
crsp.loc[crsp['mv_growth'] < pctl_low, 'mv_growth'] = pctl_low
crsp.loc[crsp['mv_growth'] > pctl_high, 'mv_growth'] = pctl_high

# ============================================================================
# PORTFOLIO FORMATION
# ============================================================================

# Prepare data for portfolio formation
port_df = crsp.dropna(subset=['mv_growth', 'mv']).copy()
port_df = port_df[['permno', 'date', 'ret', 'mv_growth', 'mv', 'exchcd']]
port_df = port_df.sort_values(['permno', 'date'], ignore_index=True)

# Calculate decile breakpoints for each month
pctls = port_df.groupby('date')['mv_growth'].quantile(
    [i/10 for i in range(1, 10)]
).unstack().reset_index()

# Merge breakpoints back to main data
port_df = port_df.merge(pctls, how='inner', on='date')

# Assign stocks to portfolios (1-10 based on mv_growth deciles)
port_df.loc[port_df['mv_growth'] <= port_df[0.1], 'port'] = 1
port_df.loc[(port_df['mv_growth'] > port_df[0.1]) & (port_df['mv_growth'] <= port_df[0.2]), 'port'] = 2
port_df.loc[(port_df['mv_growth'] > port_df[0.2]) & (port_df['mv_growth'] <= port_df[0.3]), 'port'] = 3
port_df.loc[(port_df['mv_growth'] > port_df[0.3]) & (port_df['mv_growth'] <= port_df[0.4]), 'port'] = 4
port_df.loc[(port_df['mv_growth'] > port_df[0.4]) & (port_df['mv_growth'] <= port_df[0.5]), 'port'] = 5
port_df.loc[(port_df['mv_growth'] > port_df[0.5]) & (port_df['mv_growth'] <= port_df[0.6]), 'port'] = 6
port_df.loc[(port_df['mv_growth'] > port_df[0.6]) & (port_df['mv_growth'] <= port_df[0.7]), 'port'] = 7
port_df.loc[(port_df['mv_growth'] > port_df[0.7]) & (port_df['mv_growth'] <= port_df[0.8]), 'port'] = 8
port_df.loc[(port_df['mv_growth'] > port_df[0.8]) & (port_df['mv_growth'] <= port_df[0.9]), 'port'] = 9
port_df.loc[port_df['mv_growth'] > port_df[0.9], 'port'] = 10

# ============================================================================
# PORTFOLIO STATISTICS
# ============================================================================

# Average number of stocks per portfolio
port_avg_n = port_df.groupby(['date', 'port'])['mv_growth'].count().groupby('port').mean()
print("\nAverage number of stocks per portfolio:")
print(port_avg_n)

# Average market value growth by portfolio
port_avg_mvg = port_df.groupby(['date', 'port'])['mv_growth'].mean().groupby('port').mean()
print("\nAverage market value growth by portfolio:")
print(port_avg_mvg)

# Average size (market cap) by portfolio
port_avg_size = port_df.groupby(['date', 'port'])['mv'].mean().groupby('port').mean()
print("\nAverage market cap by portfolio:")
print(port_avg_size)

# ============================================================================
# EQUAL-WEIGHTED PORTFOLIO RETURNS
# ============================================================================

# Calculate forward returns
port_df = port_df.sort_values(['permno', 'date'], ignore_index=True)
port_df['ret_lead1m'] = port_df.groupby('permno')['ret'].shift(-1)

# Calculate equal-weighted returns
ew_month = port_df.dropna().groupby(['date', 'port'])['ret_lead1m'].mean().unstack().reset_index()
ew_month.columns.name = ''
ew_month['date'] = ew_month['date'] + pd.offsets.MonthEnd(0)
ew_month['date'] = ew_month['date'] + pd.offsets.MonthEnd(1)

# Plot average returns
ew_ret = ew_month.iloc[:, 1:].mean()
plt.figure(figsize=(8, 5))
ew_ret.plot(kind='bar', rot=0)
plt.title('Equal-Weighted Portfolio Average Returns')
plt.xlabel('Portfolio')
plt.ylabel('Average Returns (%)')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('ew_portfolio_returns.png', dpi=300, bbox_inches='tight')
plt.show()

print("\nEqual-weighted portfolio returns:")
print(ew_ret)

# Calculate cumulative returns for long and short positions
ew_cum = ew_month.copy()
ew_cum['long_plus'] = ew_cum[1] + 1
ew_cum['short_plus'] = ew_cum[10] + 1
ew_cum['cum_long'] = ew_cum['long_plus'].cumprod()
ew_cum['cum_short'] = ew_cum['short_plus'].cumprod()

plt.figure(figsize=(8, 5))
ew_cum.set_index('date')[['cum_long', 'cum_short']].plot()
plt.title('Equal-Weighted Strategy: Cumulative Returns')
plt.ylabel('Cumulative Return')
plt.xlabel('Date')
plt.legend(['Long Portfolio 1 (Losers)', 'Short Portfolio 10 (Winners)'])
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('ew_cumulative_returns.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================================================
# VALUE-WEIGHTED PORTFOLIO RETURNS
# ============================================================================

# Calculate total market value by portfolio-date
mv_total = port_df.groupby(['date', 'port'])['mv'].sum().to_frame('value_tot').reset_index()

# Merge and calculate weights
port_vw = port_df.merge(mv_total, how='inner', on=['date', 'port'])
port_vw['weight'] = port_vw['mv'] / port_vw['value_tot']
port_vw['ret_weight'] = port_vw['ret_lead1m'] * port_vw['weight']

# Calculate value-weighted returns
vw_month = port_vw.dropna().groupby(['date', 'port'])['ret_weight'].sum().unstack().reset_index()
vw_month.columns.name = ''
vw_month['date'] = vw_month['date'] + pd.offsets.MonthEnd(0)
vw_month['date'] = vw_month['date'] + pd.offsets.MonthEnd(1)

# Plot average returns
vw_ret = vw_month.iloc[:, 1:].mean()
plt.figure(figsize=(8, 5))
vw_ret.plot(kind='bar', rot=0)
plt.title('Value-Weighted Portfolio Average Returns')
plt.xlabel('Portfolio')
plt.ylabel('Average Return (%)')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('vw_portfolio_returns.png', dpi=300, bbox_inches='tight')
plt.show()

print("\nValue-weighted portfolio returns:")
print(vw_ret)

# Calculate cumulative returns
vw_cum = vw_month.copy()
vw_cum['long_plus'] = vw_cum[10] + 1
vw_cum['short_plus'] = vw_cum[1] + 1
vw_cum['cum_long'] = vw_cum['long_plus'].cumprod()
vw_cum['cum_short'] = vw_cum['short_plus'].cumprod()

plt.figure(figsize=(8, 5))
vw_cum.set_index('date')[['cum_long', 'cum_short']].plot()
plt.title('Value-Weighted Strategy: Cumulative Returns')
plt.ylabel('Cumulative Return')
plt.xlabel('Date')
plt.legend(['Long Portfolio 10 (Winners)', 'Short Portfolio 1 (Losers)'])
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('vw_cumulative_returns.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================================================
# LONG-SHORT PORTFOLIO ANALYSIS - EQUAL-WEIGHTED
# ============================================================================

# Get Fama-French factors
query = """
SELECT dateff as date, mktrf, rf
FROM ff.factors_monthly
"""
mkt_ret = db.raw_sql(query)
mkt_ret['date'] = pd.to_datetime(mkt_ret['date'], format='%Y%m%d')
mkt_ret['date'] = mkt_ret['date'] + pd.offsets.MonthEnd(0)
mkt_ret['mktret'] = mkt_ret['mktrf'] + mkt_ret['rf']

# Equal-weighted long-short portfolio
ls_ret_ew = ew_month.copy()
ls_ret_ew['lsret'] = ls_ret_ew[1] - ls_ret_ew[10]  # Long losers, short winners
ls_ret_ew = ls_ret_ew.merge(mkt_ret, how='inner', on='date')

# Calculate cumulative returns
ls_ret_ew['ls_plus'] = ls_ret_ew['lsret'] + 1
ls_ret_ew['mkt_plus'] = ls_ret_ew['mktret'] + 1
ls_ret_ew['cum_ls'] = ls_ret_ew['ls_plus'].cumprod()
ls_ret_ew['cum_mkt'] = ls_ret_ew['mkt_plus'].cumprod()

plt.figure(figsize=(8, 5))
ls_ret_ew[['date', 'cum_ls', 'cum_mkt']].set_index('date').plot()
plt.title('Equal-Weighted Long-Short vs Market')
plt.ylabel('Cumulative Return')
plt.xlabel('Date')
plt.legend(['Long-Short Portfolio', 'Market'])
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('ew_longshort_vs_market.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================================================
# LONG-SHORT PORTFOLIO ANALYSIS - VALUE-WEIGHTED
# ============================================================================

# Value-weighted long-short portfolio
ls_ret_vw = vw_month.copy()
ls_ret_vw['lsret'] = ls_ret_vw[10] - ls_ret_vw[1]  # Long winners, short losers
ls_ret_vw = ls_ret_vw.merge(mkt_ret, how='inner', on='date')

# Calculate cumulative returns
ls_ret_vw['ls_plus'] = ls_ret_vw['lsret'] + 1
ls_ret_vw['mkt_plus'] = ls_ret_vw['mktret'] + 1
ls_ret_vw['cum_ls'] = ls_ret_vw['ls_plus'].cumprod()
ls_ret_vw['cum_mkt'] = ls_ret_vw['mkt_plus'].cumprod()

plt.figure(figsize=(8, 5))
ls_ret_vw[['date', 'cum_ls', 'cum_mkt']].set_index('date').plot()
plt.title('Value-Weighted Long-Short vs Market')
plt.ylabel('Cumulative Return')
plt.xlabel('Date')
plt.legend(['Long-Short Portfolio', 'Market'])
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('vw_longshort_vs_market.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================================================
# CAPM REGRESSION - VALUE-WEIGHTED LONG-SHORT
# ============================================================================

# Run CAPM regression
capm_model = sm.formula.ols('lsret ~ mktrf', ls_ret_vw, missing='drop').fit()
print("\n" + "="*80)
print("CAPM Regression Results: Value-Weighted Long-Short Portfolio")
print("="*80)
print(capm_model.summary())

# Calculate performance metrics
vw_ls_avg = ls_ret_vw['lsret'].mean()
vw_ls_se = ls_ret_vw['lsret'].std() / np.sqrt(ls_ret_vw['lsret'].count())
vw_ls_t = vw_ls_avg / vw_ls_se

print(f"\nMean Long-Short Return: {vw_ls_avg:.4f}")
print(f"T-statistic: {vw_ls_t:.4f}")

# Calculate volatility metrics
volatility_ls_ret_vw = ls_ret_vw['lsret'].std()
market_volatility = mkt_ret['mktret'].std()

print(f"\nVolatility of value-weighted long-short portfolio: {volatility_ls_ret_vw:.4f}")
print(f"Volatility of the market: {market_volatility:.4f}")

# Calculate risk-adjusted performance metrics
risk_free_rate = ls_ret_vw['rf'].mean()
average_return_ls = ls_ret_vw['lsret'].mean()
average_return_mkt = ls_ret_vw['mktret'].mean()

sharpe_ratio_ls = (average_return_ls - risk_free_rate) / volatility_ls_ret_vw
sharpe_ratio_mkt = (average_return_mkt - risk_free_rate) / market_volatility

print(f"\nSharpe Ratio of value-weighted long-short portfolio: {sharpe_ratio_ls:.4f}")
print(f"Sharpe Ratio of the market: {sharpe_ratio_mkt:.4f}")

# Get beta from regression for Treynor ratio
ls_beta = capm_model.params['mktrf']
mkt_beta = 1.0

treynor_ratio_ls = (average_return_ls - risk_free_rate) / ls_beta
treynor_ratio_mkt = (average_return_mkt - risk_free_rate) / mkt_beta

print(f"\nTreynor Ratio of value-weighted long-short portfolio: {treynor_ratio_ls:.4f}")
print(f"Treynor Ratio of the market: {treynor_ratio_mkt:.4f}")

# ============================================================================
# CAPM ANALYSIS FOR ALL PORTFOLIOS
# ============================================================================

ls_vw_est = vw_month.copy()
ls_vw_est['ls'] = ls_vw_est[10] - ls_vw_est[1]
ls_vw_est = ls_vw_est.merge(mkt_ret, how='left', on='date')

# Initialize list to store results
vw_est = []

# Perform CAPM regression for each of the 10 portfolios
for i in range(1, 11):
    avg_ret = ls_vw_est[i].mean()
    ls_vw_est['y'] = ls_vw_est[i] - ls_vw_est['rf']
    est = sm.formula.ols('y ~ mktrf', ls_vw_est, missing='drop').fit().params
    alpha = est.iloc[0]
    beta = est.iloc[1]
    vw_est.append((i, avg_ret, alpha, beta))

# Perform CAPM regression for the long-short portfolio
ls_avg_ret = ls_vw_est['ls'].mean()
ls_vw_est['y'] = ls_vw_est['ls'] - ls_vw_est['rf']
ls_est = sm.formula.ols('y ~ mktrf', ls_vw_est, missing='drop').fit().params
ls_alpha = ls_est.iloc[0]
ls_beta = ls_est.iloc[1]
vw_est.append(('LS', ls_avg_ret, ls_alpha, ls_beta))

# Compile results into DataFrame
vw_est_df = pd.DataFrame(vw_est, columns=['port', 'mean', 'alpha', 'beta'])

print("\n" + "="*80)
print("CAPM Results for All Portfolios")
print("="*80)
print(vw_est_df.to_string(index=False))

# Plot Security Market Line
plt.figure(figsize=(8, 5))
vw_est_df[vw_est_df['port'] != 'LS'].plot(kind='scatter', x='beta', y='mean', figsize=(8, 5))
plt.title('Security Market Line: Beta vs Mean Return')
plt.xlabel('Beta')
plt.ylabel('Mean Return')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('security_market_line.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n" + "="*80)
print("Analysis Complete!")
print("="*80)
