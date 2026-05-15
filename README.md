# 📊 Black-Litterman × DCF Portfolio Optimiser

A personal portfolio research tool that combines **Discounted Cash Flow (DCF) analysis** with the **Black-Litterman model** to build scenario-driven, optimally-weighted stock portfolios. 

> ⚠️ This app is for personal research and educational purposes only. Nothing here constitutes financial advice.

---

## What This App Does

The core idea is simple: intrinsic valuation dictates what a stock is worth, but portfolio theory dictates how much of it you should hold. 

This tool takes 1-year forward price targets (derived from fundamental DCF models) and translates them into an **optimal portfolio allocation**. It uses the Black-Litterman model to blend these subjective analyst views with the market's implied equilibrium, refereed by user-defined confidence levels (the Idzorek method). 

### Key Features
- **Idzorek Confidence Blending** — weights your DCF views against the market prior based on your conviction level.
- **Margin of Safety Integration** — systemically haircuts DCF price targets to ensure allocations are only driven by highly asymmetric return profiles.
- **Long-Only Max-Sharpe Optimisation** — fully invested, no-shorting constraints with adjustable position size floors and ceilings.
- **Correlated GBM Monte Carlo** — 1-year forward simulations preserving cross-stock correlations via Cholesky decomposition.
- **Historical Stress Testing** — evaluates the BL allocation against the COVID crash, 2022 rate hikes, tech selloffs, and tariff shocks.
- **Strategy Comparison** — rolling out-of-sample backtests benchmarking BL against Equal-Weighted, Cap-Weighted, Global Minimum Variance (GMV), and Risk Parity strategies.

---

## Stock Universe

The portfolio consists of a high-conviction, 17-stock universe concentrated in technology, semiconductors, payments, and financial infrastructure. These were screened for durable moats, high historical ROIC, and manageable debt profiles:

`AAPL` `ADBE` `AMAT` `AMZN` `ASML` `CPRT` `FICO` `GOOGL` `LRCX` `MA` `META` `MSCI` `MSFT` `NFLX` `NVDA` `TSM` `V`

---

## Files in This Repo

| File | Purpose |
|------|---------|
| `app.py` | Main Streamlit application and UI logic |
| `requirements.txt` | Python dependencies for Streamlit Cloud deployment |
| `edhec_risk_kit_final.py` | Custom portfolio analytics helper module (derived from EDHEC) |
| `README.md` | This file |

---

## Roadmap

- [ ] **DCF Automation:** Integrate `xlwings` to pull price targets dynamically from local Wall Street Prep Excel models.
- [ ] **Fundamental Dashboard:** Display 5-year average ROIC and Debt-to-Equity ratios directly in the UI to contextualize capital structures.
- [ ] **Scenario Toggles:** Add Bear / Base / Bull toggle switches to instantly stress-test DCF assumptions and view the resulting allocation shifts.
- [ ] **Export Functionality:** Allow users to export the final optimized weights to CSV for execution.

---

## Built With

- **[Streamlit](https://streamlit.io)** — Frontend UI
- **[yfinance](https://github.com/ranaroussi/yfinance)** — Market data and analyst consensus
- **[SciPy](https://scipy.org)** — Non-linear optimization (SLSQP)
- **[Plotly](https://plotly.com/python/)** — Interactive financial charting
- **EDHEC Risk Institute** — Core matrix operations and shrinkage estimators

---
*Work in progress — continuously refined alongside ongoing financial modeling coursework.*
