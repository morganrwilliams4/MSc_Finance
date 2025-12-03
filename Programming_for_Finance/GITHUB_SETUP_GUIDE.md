# Project Structure for GitHub

## Suggested Folder Organization

```
MSc_Finance/
│
├── README.md                          # Main repository overview
├── .gitignore                        # Git ignore file
│
└── Programming_for_Finance/           # This coursework folder
    ├── README.md                     # Project-specific documentation
    ├── momentum_strategy.py          # Main analysis code
    ├── requirements.txt              # Python dependencies (create this)
    └── outputs/                      # Generated figures (optional)
        ├── ew_portfolio_returns.png
        ├── vw_portfolio_returns.png
        └── ...
```

## Steps to Upload to GitHub

1. **Create the main repository:**
   ```bash
   git init
   git add .
   git commit -m "Initial commit: MSc Finance coursework portfolio"
   ```

2. **Create repository on GitHub:**
   - Go to github.com
   - Click "New repository"
   - Name it "MSc_Finance"
   - Add the description you created earlier
   - Don't initialize with README (you already have one)

3. **Push to GitHub:**
   ```bash
   git remote add origin https://github.com/YOUR_USERNAME/MSc_Finance.git
   git branch -M main
   git push -u origin main
   ```

## Requirements.txt

Create a `requirements.txt` file with:

```
pandas>=1.5.0
numpy>=1.23.0
matplotlib>=3.5.0
statsmodels>=0.13.0
wrds>=3.1.0
```

## Notes

- **Do NOT upload WRDS credentials** - they're already in .gitignore
- Consider whether to include output plots (they can be regenerated)
- You may want to add a sample output or screenshot to the README
- Keep data files local unless they're publicly available and small

## Main Repository README

Create a main README.md in the root that lists all your courseworks:

```markdown
# MSc Finance Coursework Portfolio

University of Bath (2025)

## Projects

### 1. Programming for Finance - Market Value Growth Momentum Strategy
Grade: 80%

[View Project →](./Programming_for_Finance/)

A quantitative trading strategy analysis examining momentum and reversal effects...

---

*Additional courseworks will be added as completed*
```
