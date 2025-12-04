# Bond Pricing and Duration Calculator

**Course:** [Course Name]  
**Institution:** University of Bath  
**Year:** 2025

## Project Overview

This project implements a bond valuation tool that calculates both the fair price and Macaulay duration of a bond, accounting for tax effects on coupon payments. The implementation uses numpy for efficient numerical computation and includes comprehensive input validation.

## Features

### 1. Bond Price Calculation
Calculates the present value of a bond by discounting all future cash flows (coupon payments and principal) at the required rate of return.

**Formula:**
```
Bond Price = Σ[C(1-T) / (1+r)^t] + [FV / (1+r)^n]
```

Where:
- C = Coupon payment per period
- T = Tax rate on coupon income
- r = Required return per period
- FV = Face value
- n = Total number of periods

### 2. Bond Duration Calculation
Computes the Macaulay duration, which measures the weighted average time to receive the bond's cash flows.

**Formula:**
```
Duration = Σ[t × PV(CF_t)] / Bond Price
```

Where:
- t = Time period
- PV(CF_t) = Present value of cash flow at time t

## Key Functionality

### Flexible Payment Frequencies
- Monthly
- Quarterly
- Semi-annually
- Annually

### Tax Adjustment
Incorporates tax effects on coupon income, providing after-tax bond valuations relevant for taxable investors.

### Robust Input Validation
- Validates all numerical inputs (positive values, valid percentages)
- Checks payment frequency against allowed options
- Clear error messages for invalid inputs

## Technical Implementation

### Functions

#### `bond_price(face_value, coupon_rate, tax_rate, maturity, payment_frequency, required_return)`

**Parameters:**
- `face_value` (float): Par value of the bond
- `coupon_rate` (float): Annual coupon rate (e.g., 0.0625 for 6.25%)
- `tax_rate` (float): Tax rate on coupon income (e.g., 0.20 for 20%)
- `maturity` (int/float): Years to maturity
- `payment_frequency` (str): Payment frequency ('monthly', 'quarterly', 'semi-annually', 'annually')
- `required_return` (float): Required annual rate of return

**Returns:** Bond price (float)

#### `bond_duration(face_value, coupon_rate, tax_rate, maturity, payment_frequency, required_return, price=None)`

**Parameters:** Same as `bond_price()`, with optional `price` parameter

**Returns:** Macaulay duration in years (float)

## Example Usage
```python
import numpy as np

# Define bond characteristics
face_value = 1000
coupon_rate = 0.0625          # 6.25%
tax_rate = 0.20               # 20%
maturity = 15                 # 15 years
payment_frequency = 'quarterly'
required_return = 0.01        # 1%

# Calculate bond price
price = bond_price(face_value, coupon_rate, tax_rate, maturity, 
                   payment_frequency, required_return)
print(f'Bond Price: £{price:.3f}')
# Output: Bond Price: £1556.524

# Calculate duration
duration = bond_duration(face_value, coupon_rate, tax_rate, maturity, 
                        payment_frequency, required_return)
print(f'Duration: {duration:.3f} years')
# Output: Duration: 11.620 years
```

## Results Interpretation

For the given example:
- **Bond Price: £1,556.52** - The bond trades at a significant premium (55.6% above par) because the coupon rate (6.25%) far exceeds the required return (1%)
- **Duration: 11.62 years** - The weighted average time to receive cash flows is approximately 11.6 years, shorter than the 15-year maturity due to intermediate coupon payments

### Key Insights:
1. **Premium Bond:** Low required return relative to coupon rate results in premium pricing
2. **Duration < Maturity:** Coupon payments reduce duration below the bond's maturity
3. **Tax Impact:** The 20% tax rate on coupons reduces effective cash flows, lowering the bond price from what it would be in a tax-free scenario

## Financial Concepts Demonstrated

- **Present Value Analysis:** Discounting future cash flows to determine fair value
- **Time Value of Money:** Core principle underlying bond valuation
- **Duration:** Interest rate risk measurement
- **Tax Effects:** Real-world consideration for taxable bond investments
- **Payment Frequency:** Impact of compounding periods on valuation

## Technical Skills

- Python programming
- Financial mathematics implementation
- NumPy for numerical computation
- Input validation and error handling
- Function design and documentation

## Dependencies
```python
numpy>=1.20.0
```

## Files

- `bond_calculator.py` - Main implementation file
- `README.md` - This documentation

## Potential Extensions

- Modified duration calculation (interest rate sensitivity)
- Convexity measurement
- Yield-to-maturity solver
- Bond portfolio analysis
- Visualization of price-yield relationship
- Callable bond pricing

## Applications

This tool is useful for:
- Fixed income portfolio management
- Bond investment analysis
- Interest rate risk assessment
- Comparative bond valuation
- Educational purposes in fixed income markets

---

**Note:** This implementation assumes:
- Fixed coupon rate throughout bond life
- Tax rate applies to all coupon payments
- No default risk (risk-free discounting)
- Perfect divisibility of payment periods
