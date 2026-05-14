[README.md](https://github.com/user-attachments/files/27754818/README.md)
# Black-Litterman × DCF Portfolio Optimiser

A personal portfolio research tool that combines **Discounted Cash Flow (DCF) analysis** with the **Black-Litterman model** to build scenario-driven, optimally-weighted stock portfolios.

Disclaimer: This app is for personal research and educational purposes only. Nothing here constitutes financial advice.

---

## What This App Does

The core idea is simple: your DCF models produce a **price target** for each stock. That price target implies a **return view**. Black-Litterman takes that view, blends it with what the market already implies, and outputs an **optimal portfolio allocation**.

The app lets you stress-test three scenarios — Bear, Base, and Bull — and immediately see how each shifts the weights.

### Features
- **Bear / Base / Bull scenario switching** — toggle between DCF cases and see real-time impact on allocation
- **Black-Litterman posterior returns** — blends your analyst views with market-implied equilibrium returns using the Idzorek confidence method
- **Long-only Max-Sharpe optimisation** — no short positions, fully invested
- **Correlated GBM Monte Carlo** — 1-year forward simulation preserving cross-stock correlations via Cholesky decomposition
- **Historical stress tests** — COVID crash, 2022 rate hikes, tech selloff, Trump tariffs
- **Strategy comparison** — BL vs. Equal-Weighted, Cap-Weighted, GMV, and Risk Parity

---

## Stock Universe

30 conviction holdings across technology, semiconductors, financials, and healthcare:

`AAPL` `ADBE` `AMAT` `AMD` `AMZN` `AVGO` `CLS` `CPRT` `DUOL` `FICO`
`GE` `GOOGL` `IBM` `IONQ` `LLY` `LRCX` `MA` `MSCI` `MSFT` `MU`
`NFLX` `NOW` `NVDA` `PANW` `PYPL` `SOFI` `TSM` `UBER` `V` `WDC`

---

## Roadmap

- [ ] Integrate xlwings to pull price targets directly from Excel DCF models
- [ ] Add three-statement model outputs (revenue, EBITDA, FCF) per stock
- [ ] Update DCF models to latest 10-K filings post Wall Street Prep course
- [ ] Per-stock DCF assumption editor (WACC, terminal growth rate, margin assumptions)
- [ ] Export optimised weights to CSV

---

## Built With

- [Streamlit](https://streamlit.io)
- [yfinance](https://github.com/ranaroussi/yfinance)
- [pandas-datareader](https://pandas-datareader.readthedocs.io) (FRED risk-free rate)
- [Plotly](https://plotly.com/python/)
- [SciPy](https://scipy.org) (optimisation)
- EDHEC Risk Institute portfolio analytics toolkit

---

*Work in progress — updating as the Wall Street Prep course progresses.*
