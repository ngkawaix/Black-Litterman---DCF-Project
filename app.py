"""
DCF + Black-Litterman Portfolio Optimiser
==========================================
A Streamlit app for setting Bear / Base / Bull price targets per stock
(derived from DCF work) and seeing how each scenario shifts the Black-Litterman
optimal portfolio weights, Monte Carlo distribution, and stress-test results.

How to run locally:
    streamlit run app.py

Deployment note:
    edhec_risk_kit_final.py  <-- must sit in the same folder as this file
    requirements.txt         <-- must also be in the repo root
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf
import requests
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, date
from scipy.optimize import minimize

# Custom EDHEC Risk Kit Module from EDHEC Course
import edhec_risk_kit_final as erk

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG  (must be the very first Streamlit call)
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DCF × Black-Litterman",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
TICKERS = sorted([
    "AAPL", "ADBE", "AMAT", "AMZN", "ASML", "CPRT",
    "FICO", "GOOGL","LRCX", "MA",   "META", "MSCI",
    "MSFT", "NFLX", "NVDA", "TSM",  "V",
])

# My base case price targets
BASE_TARGETS = {
    "AAPL": 305.00, "ADBE": 328.00, "AMAT": 444.00,
    "AMZN": 312.00, "ASML": 1661.00, "CPRT":  43.00,
    "FICO": 1562.00, "GOOGL": 428.00, "LRCX": 310.00,
    "MA":   650.00, "META": 827.00, "MSCI": 685.00,
    "MSFT": 562.00, "NFLX": 115.00, "NVDA": 270.00,
    "TSM":  463.00, "V":    399.00,
}

# FOR FUTURE IMPLEMENTATION: Bear = 20 % below base;  Bull = 25 % above base  (reserved for future DCF scenario toggle)

BASE_CONFIDENCE = {
    # Updated May 2026 — reflects latest analyst narratives and earnings.
    # AAPL  ↓ tariff uncertainty + supply chain mid-transition to India
    # ADBE  ↓ AI commoditisation fears; stock -42% from 52-week high
    # AMAT  ↓ revenue -3.5% YoY last quarter; China headwinds persist
    # ASML  ↑ raised 2026 sales forecast Apr; EUV monopoly + AI tailwind clear
    # LRCX  ↑ revenue +28% YoY; high-margin recurring revenue insulates cycle risk
    # META  ↑ fastest-growing AI hyperscaler; ad integration playing out in numbers
    # NVDA  ↑ Q4 FY2026 $68B rev (+73% YoY); Q1 FY2027 guided ~$78B
    "AAPL": 0.40, "ADBE": 0.30, "AMAT": 0.35,
    "AMZN": 0.65, "ASML": 0.65, "CPRT": 0.50,
    "FICO": 0.35, "GOOGL":0.60, "LRCX": 0.65,
    "MA":   0.50, "META": 0.70, "MSCI": 0.55,
    "MSFT": 0.60, "NFLX": 0.60, "NVDA": 0.75,
    "TSM":  0.65, "V":    0.65,
}

STRESS_PERIODS = {
    "COVID Crash (Feb--Mar 2020)":        ("2020-02-19", "2020-03-23"),
    "Post-COVID Rate Hikes (2022)":      ("2022-01-01", "2022-12-31"),
    "Tech Selloff (Nov 2021--May 2022)":  ("2021-11-19", "2022-05-20"),
    "GFC Echo (Aug--Oct 2015)":           ("2015-08-18", "2015-10-01"),
    "Trump Tariffs (Apr 2025)":          ("2025-04-02", "2025-04-09"),
}

# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING  (cached so Streamlit doesn't re-download on every interaction)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Fetching market data from Yahoo Finance…")
def load_market_data(tickers, start="2012-01-01"):
    data = yf.download(tickers, start=start, interval="1d", auto_adjust=True, progress=False)
    price_data = data["Close"]
    price_data.index = price_data.index.tz_localize(None)
    price_data = price_data.loc[~price_data.index.duplicated(keep="first")]

    # Drop tickers whose entire column is NaN (failed/rate-limited downloads).
    failed = price_data.columns[price_data.isna().all()].tolist()
    if failed:
        st.warning(
            f"⚠️ Could not download price data for: **{', '.join(failed)}**. "
            "These tickers have been excluded from this run. "
            "Yahoo Finance likely rate-limited pull request — refresh in a minute to retry.",
        )
        price_data = price_data.drop(columns=failed)

    # Forward-fill intra-series gaps, then use dropna(how="all") so a single
    # missing ticker on one day does not wipe out the entire returns history.
    price_data = price_data.ffill()
    tick_rets  = price_data.pct_change().dropna(how="all")
    return price_data, tick_rets

# Retrieval of Market Cap, Consensus Estimates and Earnings Data
@st.cache_data(show_spinner="Fetching ticker metadata (mcap, consensus, earnings)…", ttl=86400)
def load_ticker_metadata(tickers):
    import time
    import os

    today  = pd.Timestamp.today().normalize()
    cutoff = today - pd.Timedelta(days=30)

    mcap            = {}
    consensus       = {}
    recent_earnings = {}
    
    CACHE_FILE = "cached_market_caps.csv"

    for ticker in tickers:
        t = yf.Ticker(ticker)

        # ── LAYER 1: Fast Info Property ──
        try:
            mcap[ticker] = t.fast_info.market_cap
        except Exception:
            mcap[ticker] = None

        # ── LAYER 2: Standard Info Dictionary ──
        if mcap[ticker] is None:
            try:
                mcap[ticker] = t.info.get("marketCap", None)
            except Exception:
                pass
        
        # ── LAYER 3: Programmatic Derivation (Shares * Price) ──
        if mcap[ticker] is None:
            try:
                # Fallback to reconstructing from components if main field is rate-limited
                shares = None
                try: shares = t.fast_info.shares_outstanding
                except Exception: shares = t.info.get("sharesOutstanding")
                
                price = None
                try: price = t.fast_info.last_price
                except Exception: price = t.info.get("previousClose")
                
                if shares and price:
                    mcap[ticker] = shares * price
            except Exception:
                pass

        # ── Consensus & Earnings Target Fetching ──
        info = {}
        try:
            info = t.info
        except Exception:
            pass

        mean_t = None
        n_analysts = info.get("numberOfAnalystOpinions", None)
        try:
            apt = t.analyst_price_targets
            mean_t = apt.get("mean", None) if isinstance(apt, dict) else None
        except Exception:
            pass
        if mean_t is None:
            mean_t = info.get("targetMeanPrice", None)

        consensus[ticker] = {"mean": mean_t, "n_analysts": n_analysts}

        flagged = False
        try:
            cal = t.calendar
            dates = []
            if isinstance(cal, dict):
                raw = cal.get("Earnings Date", [])
                dates = raw if isinstance(raw, list) else [raw]
            elif isinstance(cal, pd.DataFrame) and not cal.empty:
                col_name = "Earnings Date" if "Earnings Date" in cal.columns else None
                dates = cal[col_name].dropna().tolist() if col_name else cal.iloc[0].dropna().tolist()
            flagged = any(
                cutoff <= pd.Timestamp(d).normalize() <= today
                for d in dates if d is not None
            )
        except Exception:
            pass
        recent_earnings[ticker] = flagged

        time.sleep(0.2)   

    mcap_series = pd.Series(mcap)
    missing     = mcap_series[mcap_series.isna()].index.tolist()
    
    # ── LAYER 4: Local Persistent Cache Fallback ──
    if missing and os.path.exists(CACHE_FILE):
        try:
            cached_df = pd.read_csv(CACHE_FILE, index_col=0)
            for tkr in missing:
                if tkr in cached_df.index:
                    mcap_series[tkr] = float(cached_df.loc[tkr, "marketCap"])
        except Exception:
            pass
        
        # Recalculate what is still missing after checking cache
        missing = mcap_series[mcap_series.isna()].index.tolist()
    
    # ── LAYER 5: Baseline Scale Distribution (Absolute Last Resort) ──
    if missing:
        # If API fails and no cache exists yet, preserve structural proportions
        BASELINE_RELS = {
            "AAPL": 2.9e12, "ADBE": 220e9, "AMAT": 160e9, "AMZN": 1.9e12, 
            "ASML": 380e9, "CPRT": 50e9, "FICO": 30e9, "GOOGL": 2.1e12, 
            "LRCX": 130e9, "MA": 420e9, "META": 1.2e12, "MSCI": 40e9,
            "MSFT": 3.1e12, "NFLX": 260e9, "NVDA": 2.3e12, "TSM": 750e9, "V": 560e9,
        }
        for tkr in missing:
            mcap_series[tkr] = BASELINE_RELS.get(tkr, 1.0e10)
        
        st.sidebar.warning("⚠️ Market data endpoints throttled. Using cached/baseline distributions for cap-weights.")

    # ── CACHE UPDATE ON SUCCESS ──
    # If we have a complete, valid set of numbers, save them locally to protect future sessions
    if not mcap_series.isna().any():
        try:
            mcap_series.to_frame(name="marketCap").to_csv(CACHE_FILE)
        except Exception:
            pass

    weights = mcap_series / mcap_series.sum()
    return weights, consensus, recent_earnings

@st.cache_data(show_spinner="Loading risk-free rate from FRED…")
def load_rf():
    try:
        url  = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DGS1"
        df   = pd.read_csv(url, index_col="observation_date", parse_dates=True)
        df   = df[df["DGS1"] != "."]
        rate = float(df["DGS1"].iloc[-1]) / 100
        return rate
    except Exception:
        return 0.04


@st.cache_data(show_spinner="Running rolling backtests…")
def run_backtests(_tick_rets, _tick_capweights, estimation_window):
    """
    Runs all four rolling-window backtests and returns a returns DataFrame.
    Results are cached against the price data hash, so re-running only happens
    when new market data is fetched — not on every slider or input interaction.

    Underscore-prefixed args tell Streamlit to hash by object identity rather
    than value (DataFrames are not directly hashable).
    """
    ew_r  = erk.backtest_ws(
        _tick_rets, estimation_window=estimation_window,
        weighting=erk.weight_ew,
    )
    cw_r  = erk.backtest_ws(
        _tick_rets, estimation_window=estimation_window,
        weighting=erk.weight_cw, cap_weights=_tick_capweights,
    )
    gmv_r = erk.backtest_ws(
        _tick_rets, estimation_window=estimation_window,
        weighting=erk.weight_gmv,
        cov_estimator=erk.shrinkage_cov, delta=0.7,
    )
    erc_r = erk.backtest_ws(
        _tick_rets, estimation_window=estimation_window,
        weighting=erk.weight_erc,
        cov_estimator=erk.shrinkage_cov, delta=0.7,
    )
    return ew_r, cw_r, gmv_r, erc_r


# ─────────────────────────────────────────────────────────────────────────────
# BLACK-LITTERMAN HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def build_bl_inputs(tick_rets, tick_capweights, price_targets, confidence, delta, tau, rf):
    """
    Constructs all BL matrices and returns posterior mu, sigma, and the
    market-implied returns (pi) so we can show the decomposition table.

    Returns
    -------
    mu_bl      : pd.Series   -- posterior expected returns
    sigma_bl   : pd.DataFrame -- posterior covariance
    pi         : pd.Series   -- market-implied returns
    Q          : pd.Series   -- excess-return views fed into BL
    cw_w       : pd.Series   -- cap weights used as the market prior
    sigma      : pd.DataFrame -- annualised covariance (shrinkage)
    """
    # Total return view and excess return view Q
    # (price_targets already passed in as a Series indexed by ticker)
    total_return_views = pd.Series(price_targets)
    Q = total_return_views - rf

    # Identity P matrix -- one absolute view per stock
    P = pd.DataFrame(
        np.eye(len(tick_rets.columns)),
        columns=tick_rets.columns,
        index=tick_rets.columns,
    )

    sigma_daily = erk.shrinkage_cov(tick_rets, delta=0.5)
    sigma       = sigma_daily * 252               # annualised covariance

    cap_aligned = tick_capweights.reindex(columns=tick_rets.columns)
    if cap_aligned.empty:
        raise RuntimeError(
            "Cap-weight DataFrame is empty after alignment. "
            "This usually means all price data failed to download due to a "
            "Yahoo Finance rate-limit. Please wait a moment and refresh the page."
        )
    cw_w = cap_aligned.iloc[-1]
    cw_w = cw_w / cw_w.sum()

    pi = erk.implied_returns(delta=delta, sigma=sigma, w=cw_w)

    conf_series    = pd.Series(confidence).reindex(tick_rets.columns)
    omega_idzorek  = erk.idzorek_omega(
        tau=tau, sigma=sigma, p=P, q=Q, pi=pi, delta=delta, confidences=conf_series
    )

    mu_bl, sigma_bl = erk.bl(
        w_prior=cw_w, sigma_prior=sigma,
        p=P, q=Q, delta=delta, tau=tau, omega=omega_idzorek,
    )
    return mu_bl, sigma_bl, pi, Q, cw_w, sigma


def bl_msr_longonly(sigma, mu, riskfree_rate, min_weight=0.0, max_weight=1.0):
    """
    Max-Sharpe optimisation with long-only and position-size constraints.

    Parameters
    ----------
    min_weight : float
        Floor on any single position (e.g. 0.02 = 2%).
        Set to 0 to allow zero-weight (unconstrained floor).
    max_weight : float
        Ceiling on any single position (e.g. 0.15 = 15%).
    """
    n = mu.shape[0]

    # Feasibility check: n * min_weight must not exceed 1
    # If the floor is too tight for the universe size, relax it silently.
    if n * min_weight > 1.0:
        min_weight = 1.0 / n

    bounds      = ((min_weight, max_weight),) * n
    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}

    def neg_sharpe(w):
        ret = w @ mu
        vol = np.sqrt(w @ sigma.values @ w)
        return -(ret - riskfree_rate) / vol

    result = minimize(
        neg_sharpe,
        np.repeat(1 / n, n),   # equal-weight warm start (always feasible)
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )
    return pd.Series(result.x, index=mu.index)


def run_correlated_gbm(tick_rets, mu_bl, bl_w_series, n_scenarios=500, n_years=1, steps=252):
    """
    Runs a correlated GBM Monte Carlo using BL posterior returns as drift.
    Returns all_paths (steps+1, n_scenarios, n_stocks) and portfolio paths.
    """
    n_stocks  = len(tick_rets.columns)
    mu_vec    = mu_bl.reindex(tick_rets.columns).values / 252
    sigma_sim = erk.sample_cov(tick_rets)
    sigma_vec = np.sqrt(np.diag(sigma_sim.values))
    L         = np.linalg.cholesky(sigma_sim.values)

    all_paths = np.ones((steps + 1, n_scenarios, n_stocks))
    for t in range(1, steps + 1):
        Z                 = np.random.normal(0, 1, size=(n_stocks, n_scenarios))
        correlated_shocks = L @ Z
        drift             = mu_vec - 0.5 * sigma_vec ** 2
        diffusion         = correlated_shocks.T * sigma_vec
        all_paths[t]      = all_paths[t - 1] * np.exp(drift + diffusion)

    w         = bl_w_series.reindex(tick_rets.columns).fillna(0).values
    port_paths = (all_paths * w).sum(axis=2)
    return all_paths, port_paths


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR -- User Inputs
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Model Assumptions")

    # --- Data Range Selection ---
    _FLOOR = date(2012, 6, 1)
    _MAX_START      = (datetime.today() - pd.Timedelta(days=365 * 3)).date()
    
    st.subheader("1. Data Range")
    data_range_help = (
        "**Minimum: 2012-06-01** – one month after META's IPO (the most recent in the universe).\n\n"
        "**Recommended Default: 2015-01-01** – captures multiple market regimes "
        "(2015 volatility spike, 2018 correction, COVID crash, 2022 rate hikes, 2023-25 AI bull) "
        "without anchoring the covariance to the post-GFC zero-rate anomaly (2012-2014).\n\n"
        "Going shorter than 5 years risks an under-identified covariance matrix for 17 stocks."
    )
    data_start_date = st.sidebar.date_input(
        "Historical data start date",
        value=date(2015, 1, 1),
        min_value=_FLOOR,
        max_value=_MAX_START,
        key="data_start_date",
        help=data_range_help
    )
    st.sidebar.divider()
    
    price_data, tick_rets = load_market_data(TICKERS, start=data_start_date.strftime("%Y-%m-%d"))
    
    # Fetch the cached weights series and metadata
    weights_series, consensus_data, recent_earnings = load_ticker_metadata(TICKERS)
    
    # Build the cap-weight DataFrame dynamically with the fresh index
    idx = price_data.index
    tick_capweights = pd.DataFrame(
        [weights_series.reindex(TICKERS).values] * len(idx),
        index=idx, columns=TICKERS,
    )
    
    # Hard stop: if every single ticker failed (total rate-limit), nothing works downstream.
    if tick_rets.empty:
        st.error(
            "❌ **All price data downloads failed.** "
            "Yahoo Finance is likely rate-limiting the Streamlit Cloud shared IP. "
            "Wait 60 seconds and refresh the page."
        )
        st.stop()


    # --- Position size constraints ---
    st.subheader("2. Position Size Constraints")
    
    max_w_pct = st.slider(
        "Max position size (%)",
        min_value=5.0, max_value=100.0, value=25.0, step=0.5,
        help=("Imposes a ceiling on the maximum weight the optimiser can allocate into a single stock. "
              "Default: 25% to enable some concentration risks while forcing some diversification."
              )
    )
    min_w_pct = st.slider(
        "Min position size (%)",
        min_value=0.0, max_value=10.0, value=0.0, step=0.5,
        help=("Imposes a floor on the minimum weight the optimiser can allocate into a single stock. "
              "Default: 0% to allow for zero-weight positions."
              )
    )
    min_weight = min_w_pct / 100
    max_weight = max_w_pct / 100

    st.divider()

    # --- Per-stock price targets ---
    st.subheader("3. Price Targets & Confidence")
    st.caption(
        "**How these are set:** Base targets are rough estimates in-line with the Street View. "
        "Consensus figures are sourced from Yahoo Finance analyst aggregates and "
        "may lag recent revisions - treat them as directional references only. "
        "**Confidence** (Idzorek method) weights your view vs. the market-implied "
        "equilibrium: 0 = ignore your view entirely, 1 = full conviction."
    )
    st.caption("🟡 Ticker flagged = earnings reported in the last 30 days — consensus may have been revised.")
    st.caption("**Last Updated: 15 May 2026**")
    user_targets    = {}
    user_confidence = {}

    for i in range(0, len(TICKERS), 2):
        col_a, col_b = st.columns(2)
        for col, ticker in zip([col_a, col_b], TICKERS[i:i + 2]):
            with col:
                # Ticker label + earnings recency flag
                flag  = " 🟡" if recent_earnings.get(ticker, False) else ""
                st.markdown(f"**{ticker}**{flag}")

                # Consensus reference line
                cons   = consensus_data.get(ticker, {})
                mean_t = cons.get("mean", None)
                n_ana  = cons.get("n_analysts", None)
                if mean_t:
                    n_str = f" | from {int(n_ana)} analysts" if n_ana else ""
                    st.caption(f"Consensus: ${mean_t:,.0f}{n_str}")
                else:
                    st.caption("Consensus: N/A")

                user_targets[ticker] = st.number_input(
                    "Price target ($)",
                    min_value=0.01,
                    value=float(BASE_TARGETS[ticker]),
                    step=1.0,
                    key=f"pt_{ticker}",
                )
                user_confidence[ticker] = st.slider(
                    "Confidence",
                    min_value=0.0,
                    max_value=1.0,
                    value=float(BASE_CONFIDENCE[ticker]),
                    step=0.05,
                    key=f"conf_{ticker}",
                )

    st.divider()

    # --- Other BL parameters ---
    st.subheader("4. Other BL Parameters")

    delta = st.slider(
        "δ  Risk Aversion",
        min_value=1.0, max_value=5.0, value=2.5, step=0.1,
        help=("Delta is the risk-aversion coefficient of the market portfolio. "
              "Standard Value is 2.5. Higher means market expects more return per unit of risk."
              )
    )

    tau = st.slider(
        "τ  Prior Uncertainty",
        min_value=0.01, max_value=0.10, value=0.02, step=0.01,
        help=("Tau is the uncertainty in the prior returns benchmark. For this model, the Cap-Weighted allocation returns are the benchmark"
              "Standard valus is 0.025 as used by He-Litterman. Smaller means you trust the market more."
              )
    )
    
    st.divider()

    # --- Backtest estimation window ---
    st.subheader("5. Backtest Estimation Window")

    estimation_window_yrs = st.slider(
        "Estimation window (years)",
        min_value=1, max_value=7, value=3, step=1,
        help=("Controls how many years of historical data are used to estimate covariance and weights at "
              "each rolling rebalance step to prevent look-ahead bias for backtests. "
              "Default: 3 years captures a full market cycle and is the recommended default.\n\n"
              "5 years is the most stable but comes at the cost of reducing the backtest period. \n\n"
              "Note: A longer estimation window delays the backtest start date by the same amount such "
              "that the first debalance can only happen once a full window of data is available."
              )
    )
    estimation_window = estimation_window_yrs * 252

    st.divider()
    st.caption(
        "🔮 **DCF integration coming soon** -- once the Wall Street Prep models "
        "are finalised, xlwings will pull targets directly from Excel into the "
        "views matrix above."
    )

# ─────────────────────────────────────────────────────────────────────────────
# LOAD DATA  (price data and metadata already fetched before the sidebar)
# ─────────────────────────────────────────────────────────────────────────────
RF = load_rf()

# Align date ranges
common_index    = tick_rets.index.intersection(tick_capweights.index)
tick_rets       = tick_rets.loc[common_index]
tick_capweights = tick_capweights.loc[common_index]

# Current prices for computing total-return views
current_prices = price_data.reindex(columns=tick_rets.columns).iloc[-1]

# Compute total-return views from user price targets
targets_series = pd.Series(user_targets).reindex(tick_rets.columns)
total_return_views = (targets_series / current_prices) - 1

# ─────────────────────────────────────────────────────────────────────────────
# RUN BLACK-LITTERMAN
# ─────────────────────────────────────────────────────────────────────────────
with st.spinner("Running Black-Litterman optimisation…"):
    mu_bl, sigma_bl, pi, Q, cw_w, sigma = build_bl_inputs(
        tick_rets        = tick_rets,
        tick_capweights  = tick_capweights,
        price_targets    = total_return_views,   # excess returns already; recalculated inside
        confidence       = user_confidence,
        delta            = delta,
        tau              = tau,
        rf               = RF,
    )
    bl_w = bl_msr_longonly(sigma=sigma_bl, mu=mu_bl, riskfree_rate=RF, min_weight=min_weight, max_weight=max_weight)
    bl_w_series = bl_w   # pd.Series indexed by ticker

# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.title("DCF × Black-Litterman Portfolio Optimiser")
st.caption(
    f"Risk-free rate (1Y T-bill): **{RF:.2%}** (retrieved from FRED) |  "
    f"Universe: **{len(TICKERS)} stocks** |  "
    f"Data range: **{tick_rets.index[0].date()} → {tick_rets.index[-1].date()}**"
)

# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────
tab0, tab1, tab2, tab3, = st.tabs([
    "📖 Introduction",
    "📋 Views, Returns & Weights",
    "📈 Simulation & Stress Tests",
    "🔀 Strategy Comparison",
])


# ══════════════════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════════════════
# TAB 0 -- Introduction
# ══════════════════════════════════════════════════════════════════════════════
with tab0:

    st.markdown("#### About this Project")
    st.markdown(
        """
        **I started this self-guided project to better understand investment management
        and to guide the sizing of my own portfolio allocations.** It implements the skills
        and knowledge that I have acquired over the last three months from EDHEC's Advanced
        Portfolio Construction and Analysis and Wall Street Prep's DCF Modelling Course - all
        with the aim of answering question: Having derived the 1Y price target of a stock
        through DCF analysis, how does one use this knowledge to size thier portfolios?

        **This app uses the Black Litterman (BL) Model to size portfolio allocations.** It uses this model
        because of its advantages over other strategies. Most backtested strategies like Global Minimum Variance (GMV) 
        or Risk Parity are purely backward-looking, optimising on historical data and assuming the past repeats. 
        The Black-Litterman (BL) Model is different. It takes a forward-looking view on what each stock is worth and 
        asks the investor what their confidence levels are for these views, relative to the market's implied returns. 
        Crucially, this method allows investors to incorporate their views to guide portfolio allocations and intergrate DCF analysis 
        into one cohesive framework. This app lets you build that allocation  based on some modelling assumptions from the side-bar. 
        It stress-test those allocations against real market crashes, and benchmark it against other strategies. 

        **A clear limitation of this model is that it assumes that the investors goal is to pursue wealth accumulation**
        rather than wealth presevation. Sovereign Funds with payout obligations would pursue a different objective entirely,
        adopting a liability driven investing as its basis and optimising for duration matching of bond coupon payouts.
        This strategy does not take thoese repayment schedules into account, but can be used for the "riskier" equity
        allocations in conjuction with Constant Proportion Portfolio Insurance (CPPI) strategies to enforce downside
        protections.
        
        This project is still a work in progress. I plan to incorporate DCF assumptions as inputs to this model to
        experiment with the resulting allocations from DCF models directly. I plan to also incorporate the
        CPPI mechanics into this application soon. Stay tuned!
        """
    )

    st.divider()

    st.markdown("#### Stock Universe & Selection Criteria")
    st.markdown(
        """
        The 17 stocks in this portfolio were selected through a systematic
        fundamental screen using four criterias: **(1) 5-year average ROIC above 15%**, **(2) debt-to-equity
        below 1**, **(3) consecutive revenue growth over 5 years**, and **(4) a minimum
        market cap of $10 billion**. ROIC was chosen as the primary quality
        filter because it measures how efficiently a company converts capital
        into profit — sustained high ROIC over multiple years is one of the
        most reliable indicators of a durable competitive advantage. Though FCF margin is also a robust way
        to screen for profitable generating companies, this method would screen out high quality companies like
        AMZN, MSFT and GOOG which are undergoing unprecedented Capex spending-cycles for data-centre build-outs.

        The resulting universe is concentrated in technology, semiconductors,
        payments infrastructure, and financial data — sectors where
        capital-light business models and high switching costs tend to produce
        the kind of high quality businesses with durable moats. Two names
        warrant a note: FICO carries negative book equity due to sustained
        buybacks rather than distress, which causes standard debt screens to
        misread it; ASML is the sole supplier of extreme ultraviolet
        lithography equipment to the global semiconductor industry, making it
        structurally irreplaceable within the AI infrastructure stack.
        """
    )

    st.divider()

    st.markdown("#### How Black-Litterman Works")
    st.markdown(
        """
        Most portfolio optimisers have a fundamental problem: feed them expected returns
        built purely from historical data or analyst targets, and they produce extreme,
        unstable allocations (i.e. prone to error maximisation where even small estimation errors
        result in highly concentrated porfolios where no sensible investor would put their money into).

        **Black-Litterman solves this problem by never letting a view stand alone.**
        Instead, it always asks: *relative to what the market collectively believes,
        how much should a view actually shift the allocation?*

        The model has two inputs:

        - **The market prior (π)** — what the market implies everyone should expect,
          derived by reverse-engineering the CAPM: if every investor holds the market
          portfolio, what expected returns would justify current prices and weights?
          This is the baseline the model starts from.

        - **Analyst views (Q)** — the excess return implied by each DCF price target
          (total return minus the risk-free rate). This is the forward-looking judgment layer.

        It then blends them using a precision-weighted average. *Precision* is inverse
        uncertainty: the higher the confidence, the more weight a view receives. The
        lower the confidence, the more the model falls back to the market equilibrium.
        """
    )

    st.latex(
        r"\mu_{BL} = \left[(\tau\Sigma)^{-1} + P^\top\Omega^{-1}P\right]^{-1}\left[(\tau\Sigma)^{-1}\pi + P^\top\Omega^{-1}Q\right]"
    )

    st.markdown(
        """
        **Every term, in plain English:**

        | Symbol | Name | What it means |
        |--------|------|---------------|
        | **μ_BL** | Posterior expected return | The model's final blended return estimate — what feeds into the optimiser |
        | **π** (pi) | Market-implied equilibrium return | What the market collectively expects, derived from cap weights and risk aversion |
        | **Q** | Analyst views | Excess return implied by each DCF price target (total return minus risk-free rate) |
        | **Σ** (Sigma) | Covariance matrix | How much each stock moves, and how they move together — captures correlation risk |
        | **τ** (tau) | Prior uncertainty scalar | How much to distrust the market prior; smaller = trust the market more |
        | **P** | View matrix | Maps each view to the stocks it applies to; here an identity matrix — one view per stock |
        | **Ω** (Omega) | View uncertainty matrix | How uncertain each analyst view is; computed from the confidence sliders via the Idzorek method |

        **The intuition:** The formula is a tug-of-war between π and Q, refereed by
        uncertainty. When confidence is high, Ω is small, its inverse is large, and Q
        pulls the posterior strongly away from π. When confidence is low, Ω is large,
        its inverse shrinks, and the posterior barely moves from equilibrium. The
        covariance Σ ensures that stocks with shared risk exposures influence each
        other — a high-conviction view on NVDA nudges the posterior for TSM too,
        because they co-move. The table in the next tab shows this blending in action.
        """
    )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 -- Views, Returns & Weights
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.html("<div style='height: 18px;'></div>")
    st.markdown(
        """
        **This section walks through the full Black-Litterman pipeline in sequence.**
        Starting from DCF price targets, it computes the excess return view (Q)
        for each stock, blends it with the market-implied equilibrium return (π) using
        confidence settings, and produces the BL posterior return that the
        optimiser uses. The final section shows how those posterior returns
        translate into optimal portfolio weights via a long-only (no shorting allowed) 
        Max Sharpe optimisation. You can see directly how a change in a price target or
        confidence slider flows through to a change in position size.
        """
    )

    st.divider()

    # ── Section 1: Decomposition of Returns ──────────────────────────────────────
    st.markdown("#### 1. Decomposition of Returns")
    st.caption(
        "Each column is one layer of the BL process. "
        "**DCF-Implied Return** is the raw DCF-implied return. "
        "**Q** subtracts the risk-free rate to get the excess return fed into the model. "
        "**π** is the market equilibrium baseline. "
        "**BL Posterior** is the blended output — the return the optimiser actually uses."
    )

    view_df = pd.DataFrame({
        "Current Price ($)":          current_prices,
        "Price Target ($)":           targets_series,
        "DCF-Implied Return":         total_return_views,
        "Risk-Free Rate (rf)":        RF,
        "Excess Return View (Q)":     total_return_views - RF,
        "Market-Implied Return (π)":  pi,
        "BL Posterior Return":        mu_bl,
        "Confidence":                 pd.Series(user_confidence).reindex(tick_rets.columns),
    }).sort_values("BL Posterior Return", ascending=False)

    pct_cols    = ["DCF-Implied Return", "Risk-Free Rate (rf)", "Excess Return View (Q)",
                   "Market-Implied Return (π)", "BL Posterior Return", "Confidence"]
    dollar_cols = ["Current Price ($)", "Price Target ($)"]

    styled = view_df.style \
        .format("{:.2%}", subset=pct_cols) \
        .format("${:,.2f}", subset=dollar_cols) \
        .background_gradient(subset=["BL Posterior Return"], cmap="YlGnBu")

    st.dataframe(styled, use_container_width=True)

    fig = go.Figure()
    sorted_tickers = view_df.index.tolist()
    fig.add_trace(go.Bar(
        name="Market-Implied (π)", x=sorted_tickers,
        y=pi.reindex(sorted_tickers).values,
        marker_color="#c7e9b4", opacity=0.7,
    ))
    fig.add_trace(go.Bar(
        name="View (Q)", x=sorted_tickers,
        y=(total_return_views - RF).reindex(sorted_tickers).values,
        marker_color="#41b6c4", opacity=0.7,
    ))
    fig.add_trace(go.Bar(
        name="BL Posterior", x=sorted_tickers,
        y=mu_bl.reindex(sorted_tickers).values,
        marker_color="#253494", opacity=0.9,
    ))
    fig.update_layout(
        barmode="group",
        title="π  vs  View (Q)  vs  BL Posterior Return",
        yaxis_tickformat=".1%",
        height=420,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # ── Section 2: Optimised Weights ──────────────────────────────────────────
    st.markdown("#### 2. Optimised Portfolio Weights")
    st.caption(
        "The BL posterior returns from Part 1 feed directly into a long-only (no shorting) "
        "Max Sharpe optimisation. Stocks with higher posterior returns and lower "
        "correlation to the rest of the portfolio receive higher weights. "
        "The table also shows alternative weighting schemes for reference (bounded by min and max weights in the side bar."
    )

    ew_w   = erk.weight_ew(tick_rets)
    cw_w_s = tick_capweights.reindex(columns=tick_rets.columns).iloc[-1]
    cw_w_s = cw_w_s / cw_w_s.sum()
    gmv_w  = erk.weight_gmv(tick_rets, cov_estimator=erk.shrinkage_cov, delta=0.7)
    erc_w  = erk.weight_erc(tick_rets, cov_estimator=erk.shrinkage_cov, delta=0.7)

    weight_df = pd.DataFrame({
        "BL Optimised":    bl_w_series,
        "Cap-Weighted":    cw_w_s,
        "Equal-Weighted":  ew_w,
        "Global Mean Variance": gmv_w,
        "Risk Parity":     erc_w,
    }).sort_values("BL Optimised", ascending=False)

    st.dataframe(
        weight_df.style
            .format("{:.2%}")
            .background_gradient(subset=["BL Optimised"], cmap="YlGnBu"),
        use_container_width=True,
    )

    nonzero = weight_df[weight_df["BL Optimised"] > 0.001].reset_index()
    nonzero.columns = ["Ticker"] + list(nonzero.columns[1:])
    fig_tree = px.treemap(
        nonzero, path=["Ticker"], values="BL Optimised",
        title="BL Weight Allocation",
        color="BL Optimised", color_continuous_scale="YlGnBu",
    )
    fig_tree.update_traces(textinfo="label+percent entry")
    st.plotly_chart(fig_tree, use_container_width=True)

    fig2 = go.Figure()
    fig2.add_trace(go.Bar(
        name="BL Optimised", x=weight_df.index, y=weight_df["BL Optimised"],
        marker_color="#253494",
    ))
    fig2.add_trace(go.Bar(
        name="Cap-Weighted", x=weight_df.index, y=weight_df["Cap-Weighted"],
        marker_color="#41b6c4",
    ))
    fig2.update_layout(
        barmode="group", title="BL vs Cap-Weighted Allocation",
        yaxis_tickformat=".1%", height=380,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig2, use_container_width=True)

    st.divider()
    st.caption(
        "⚙️ **Note on covariance estimation:** GMV and Risk Parity use the Elton-Gruber Constant "
        "Correlation shrinkage estimator (δ = 0.7), blending 70% weight on a structured "
        "prior — where all pairwise correlations are set to the cross-sectional average — "
        "with 30% on the sample covariance. A higher δ was chosen because with only 17 "
        "stocks the sample covariance matrix is prone to estimation noise and "
        "near-singularity, which causes unconstrained optimisers to produce extreme, "
        "unstable weights. Shrinking toward the structured prior regularises the matrix, "
        "reduces its condition number, and makes the optimisation numerically well-behaved "
        "without requiring a larger asset universe to stabilise the estimate."
    )

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 -- Simulation & Stress Tests
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.html("<div style='height: 18px;'></div>")
    st.markdown(
        """
        **This section stress tests the BL weights using (1) a correlated GBM monte carlo simulation, 
        and (2) a historic stress test to backtest against historical shocks.** A correlated GBM treats future returns as a random walk; 
        each day shock is drawn independently, scaled by historical volatility and the correlations between stocks.
        Correlations are preserved via Cholesky decomposition to better reflect the performance of correlated assets. 
        Since many of the stocks in the universe are in the tech field, this is the apporpriate approach.

        **What the correlated GBM simulations cannot capture is the clustering of extreme events.** In real markets, crashes are not randomly distributed. 
        Volatility spikes, correlations break down, and losses arrive in sequences that a Gaussian model systematically understates. 
        The GBM simulations are hence best read as a baseline of what *typical* uncertainty looks like and not as a statement about tail risk. 
        The historic stress tests ground that picture in what actually happened holding the BL portfolio. Together, they present a more complete
        picture of the risks associated from holding the BL weights.
        """
    )
    st.divider()

    st.markdown("#### 1. Correlated GBM Simulation")
    col1, col2 = st.columns(2)
    n_scenarios = col1.slider("Number of scenarios", 100, 1000, 500, step=100)
    seed        = col2.number_input("Random seed (for reproducibility)", value=42, step=1)

    np.random.seed(int(seed))
    with st.spinner("Running simulation…"):
        all_paths, port_paths = run_correlated_gbm(
            tick_rets, mu_bl, bl_w_series, n_scenarios=n_scenarios
        )

    final_values = port_paths[-1]

    # Summary stats
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Mean",          f"${np.mean(final_values):.3f}")
    m2.metric("Median",        f"${np.median(final_values):.3f}")
    m3.metric("5th pct",       f"${np.percentile(final_values, 5):.3f}")
    m4.metric("95th pct",      f"${np.percentile(final_values, 95):.3f}")
    m5.metric("95--5 Spread",   f"${np.percentile(final_values, 95) - np.percentile(final_values, 5):.3f}")

    # Fan chart -- portfolio paths
    paths_df = pd.DataFrame(port_paths)
    x_axis   = list(range(port_paths.shape[0]))

    fig_mc = go.Figure()
    for col in paths_df.columns[:200]:   # plot up to 200 paths to keep it fast
        fig_mc.add_trace(go.Scatter(
            x=x_axis, y=paths_df[col], mode="lines",
            line=dict(color="steelblue", width=0.4),
            opacity=0.15, showlegend=False,
        ))
    fig_mc.add_trace(go.Scatter(
        x=x_axis,
        y=np.percentile(port_paths, 50, axis=1),
        mode="lines", name="Median",
        line=dict(color="black", width=2),
    ))
    fig_mc.add_trace(go.Scatter(
        x=x_axis, y=np.percentile(port_paths, 5, axis=1),
        mode="lines", name="5th pct",
        line=dict(color="#C44E52", width=1.5, dash="dash"),
    ))
    fig_mc.add_trace(go.Scatter(
        x=x_axis, y=np.percentile(port_paths, 95, axis=1),
        mode="lines", name="95th pct",
        line=dict(color="#55A868", width=1.5, dash="dash"),
    ))
    fig_mc.update_layout(
        title="Correlated GBM -- Portfolio Value Paths (starting $1)",
        xaxis_title="Trading Day",
        yaxis_title="Portfolio Value ($)",
        height=420,
    )
    st.plotly_chart(fig_mc, use_container_width=True)

    # Final value histogram
    fig_hist = go.Figure()
    fig_hist.add_trace(go.Histogram(
        x=final_values, nbinsx=60,
        marker_color="steelblue", opacity=0.8, name="Final Value",
    ))
    for pct, colour, label in [
        (5,  "#C44E52", "5th pct"),
        (50, "#404040",  "Median"),
        (95, "#55A868",  "95th pct"),
    ]:
        fig_hist.add_vline(
            x=np.percentile(final_values, pct),
            line_color=colour, line_dash="dash",
            annotation_text=label, annotation_position="top right",
        )
    fig_hist.update_layout(
        title="Distribution of Final Portfolio Value ($10,000 invested)",
        xaxis_title="Portfolio Value",
        yaxis_title="Frequency",
        height=360,
    )
    st.plotly_chart(fig_hist, use_container_width=True)

    st.divider()

    # ── Historical Stress Tests ───────────────────────────────────────────────
    st.markdown("#### 2. Historical Stress Test")
    st.caption(
        "Applies the BL optimal weights to *actual* historical returns during "
        "known market shocks. This shows how this allocation *would* have "
        "performed and is not a forecast."
    )

    w_stress = bl_w_series.reindex(tick_rets.columns).fillna(0).values
    stress_rows = {}

    for name, (start, end) in STRESS_PERIODS.items():
        period_rets = tick_rets.loc[start:end]
        if period_rets.empty:
            continue
        port_rets   = (period_rets * w_stress).sum(axis=1)
        cumulative  = (1 + port_rets).prod() - 1
        cumprod     = (1 + port_rets).cumprod()
        max_dd      = (cumprod / cumprod.cummax() - 1).min()
        ann_vol     = port_rets.std() * np.sqrt(252)
        stress_rows[name] = {
            "Period Return":  cumulative,
            "Max Drawdown":   max_dd,
            "Annualised Vol": ann_vol,
            "Trading Days":   len(port_rets),
        }

    stress_df = pd.DataFrame(stress_rows).T
    st.dataframe(
        stress_df.style
            .format("{:.1%}", subset=["Period Return", "Max Drawdown", "Annualised Vol"])
            .format("{:.0f}", subset=["Trading Days"]),
        use_container_width=True,
    )

    # Bar chart
    fig_stress = go.Figure()
    fig_stress.add_trace(go.Bar(
        name="Period Return",
        x=list(stress_rows.keys()),
        y=[v["Period Return"] for v in stress_rows.values()],
        marker_color=["#55A868" if v["Period Return"] >= 0 else "#C44E52"
                      for v in stress_rows.values()],
    ))
    fig_stress.update_layout(
        title="BL Portfolio -- Return During Stress Periods",
        yaxis_tickformat=".1%",
        height=380,
        xaxis_tickangle=-20,
    )
    st.plotly_chart(fig_stress, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 -- Strategy Comparison
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.html("<div style='height: 18px;'></div>")
    st.markdown(
        """
        **This section compares the BL Weights against other portfolio allocation strategies.**
        It does this by using a rolling backtest using an estimation window (selected by the user) to rebalance weights at each step. 
        EW, Cap-Weighted, GMV, and Risk Parity are properly rolled and weights are re-estimated
        each period using only data available at that point in time, so there is no look-ahead bias.
        The Black-Litterman is shown as a static allocation using the current optimal weights applied to the full history. 
        This is a gross simplification as true BL weights would require a fresh set of views at every rebalance date. 
        The chart is therefore best read as 'how would this portfolio have held up' rather than a like-for-like backtest.

        I complement this analysis in part 2 with a forward-looking distribution using the correlated GBM
        I had used in the "Simulation and Stress-Tests" section. The limitation of this analysis is that it assumes that
        the portfolio weights follow a correlated GBM pathway, but more robust techniques employing Machine Learning
        exists. I plan to employ these techniques after taking more courses on Machine Learning for Asset Management. 
        """
    )
    st.divider()
        
    st.markdown("#### 1. Historical Wealth Index")

    ew_r, cw_r, gmv_r, erc_r = run_backtests(
        tick_rets, tick_capweights, estimation_window
    )

    valid_start_date = tick_rets.index[estimation_window]

    # BL: static weights applied to history, aligned to rolling start date
    bl_static_w = bl_w_series.reindex(tick_rets.columns).fillna(0)
    bl_r        = (tick_rets * bl_static_w.values).sum(axis=1)]

    btr = pd.DataFrame({
            "Equal-Weighted":          pd.Series(ew_r).squeeze(),
            "Cap-Weighted":            pd.Series(cw_r).squeeze(),
            "Global Mean Variance":    pd.Series(gmv_r).squeeze(),
            "Risk Parity":             pd.Series(erc_r).squeeze(),
            "BL (static)":             pd.Series(bl_r).squeeze(),
        }).loc[valid_start_date:].dropna()

    bl_estimation_start = tick_rets.index[0].date()
    bl_estimation_end   = tick_rets.index[-1].date()
    backtest_start      = btr.index[0].date()
    backtest_end        = btr.index[-1].date()

    c1, c2 = st.columns(2)
    c1.caption(
        f"**BL estimation basis:** {bl_estimation_start} → {bl_estimation_end}  \n"
        f"Covariance, market-implied returns, and posterior are estimated over this full window."
    )
    c2.caption(
        f"**Backtest period:** {backtest_start} → {backtest_end}  \n"
        f"Starts {estimation_window_yrs} year(s) after data begins — the minimum needed for the first rolling estimate."
    )
    
    # Wealth Index Plot with starting $10,000 invested
    wealth = (1 + btr).cumprod() * 10_000

    palette = {
        "Equal-Weighted":          ("#94A3B8",    1.2, "solid"),
        "Cap-Weighted":            ("#CBD5E1",    1.2, "solid"),
        "Global Mean Variance":         ("#0D9488",   1.2, "solid"),
        "Risk Parity":             ("#6366F1", 1.2, "solid"),
        "BL (static)":             ("#002060",  2.5, "solid"),
    }

    fig_wealth = go.Figure()
    for col, (colour, width, dash) in palette.items():
        if col not in wealth.columns:
            continue
        fig_wealth.add_trace(go.Scatter(
            x=wealth.index, y=wealth[col],
            mode="lines", name=col,
            line=dict(color=colour, width=width, dash=dash),
        ))
    fig_wealth.update_layout(
        title="Wealth Index - $10,000 invested at backtest start",
        xaxis_title="Date",
        yaxis_title="Portfolio Value ($)",
        height=460,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig_wealth, use_container_width=True)

    # Summary stats
    _ann_scale = np.sqrt(252)
    
    summary_rows = {}
    for col in btr.columns:
        r         = btr[col].dropna()
        ann_r     = erk.annualize_rets(r, periods_per_year=252)
        ann_v     = erk.annualize_vol(r, periods_per_year=252)
        skew      = erk.skewness(r)
        kurt      = erk.kurtosis(r)          # raw kurtosis; normal = 3
        # Annualise daily VaR/CVaR → multiply by √252
        cf_var    = erk.var_gaussian(r, level=5, modified=True) * _ann_scale
        cvar_hist = erk.cvar_historic(r, level=5)               * _ann_scale
        sharpe    = erk.sharpe_ratio(r, riskfree_rate=RF, periods_per_year=252)
        cp        = (1 + r).cumprod()
        mdd       = (cp / cp.cummax() - 1).min()
        summary_rows[col] = {
            "Ann. Return":           ann_r,
            "Ann. Vol":              ann_v,
            "Sharpe Ratio":          sharpe,
            "Max Drawdown":          mdd,
            "Skewness":              skew,
            "Kurtosis":              kurt,
            "Ann. CF VaR (5%)":      cf_var,
            "Ann. CVaR (5%)":        cvar_hist,
        }

    summary_df = pd.DataFrame(summary_rows).T
    st.dataframe(
        summary_df.style
            .format("{:.2%}", subset=["Ann. Return", "Ann. Vol", "Max Drawdown",
                                      "Ann. CF VaR (5%)", "Ann. CVaR (5%)"])
            .format("{:.2f}", subset=["Sharpe Ratio", "Kurtosis", "Skewness"]),
        use_container_width=True,
        column_config={
            "Skewness": st.column_config.Column(
                "Skewness",
                help=(
                    "Measures asymmetry of the return distribution. "
                    "0 = symmetric (normal). Negative skew means more frequent "
                    "large losses than large gains — bad for portfolios."
                ),
            ),
            "Kurtosis": st.column_config.Column(
                "Kurtosis",
                help=(
                    "Raw kurtosis of the return distribution. "
                    "3 = normal distribution. Values above 3 indicate fat tails — "
                    "extreme gains/losses occur more often than a normal model would predict."
                ),
            ),
            "Ann. CF VaR (5%)": st.column_config.Column(
                "Ann. CF VaR (5%)",
                help=(
                    "Cornish-Fisher Value at Risk at the 5% level, annualised (×√252). "
                    "Adjusts the standard Gaussian VaR for the observed skewness and kurtosis "
                    "of the return distribution. Represents the annualised threshold loss "
                    "that is exceeded only 5% of the time. "
                    "Higher = worse tail risk."
                ),
            ),
            "Ann. CVaR (5%)": st.column_config.Column(
                "Ann. CVaR (5%)",
                help=(
                    "Historic Conditional VaR (Expected Shortfall) at the 5% level. "
                    "Answers: given that we are in the worst 5% of outcomes, "
                    "what is the average loss? CVaR is always expressed as a "
                    "percentage of portfolio value — a higher number means deeper "
                    "average losses in bad tail scenarios."
                ),
            ),
        },
    )

    st.divider()

    # ── Section 2: 1-Year Monte Carlo Return Forecast ────────────────────────
    st.markdown("#### 2. 1-Year Monte Carlo Return Forecast")
    st.caption(
        "Uses the same correlated GBM paths from Tab 3 but applied to each "
        "strategy's weights. Lets you see whether BL adds value over simpler alternatives."
    )

    strategy_weights = {
        "Equal-Weighted":   pd.Series(ew_w, index=tick_rets.columns),
        "Cap-Weighted":     cw_w_s,
        "Global Mean Variance":  pd.Series(gmv_w, index=tick_rets.columns),
        "Risk Parity":      pd.Series(erc_w, index=tick_rets.columns),
        "Black-Litterman":      bl_w_series,
    }

    comparison = {}
    for strat_name, w_s in strategy_weights.items():
        port_p = (all_paths * w_s.reindex(tick_rets.columns).values).sum(axis=2)
        fv     = port_p[-1]
        comparison[strat_name] = {
            "Expected Return": np.mean(fv) - 1,
            "5th pct":         np.percentile(fv, 5) - 1,
            "95th pct":        np.percentile(fv, 95) - 1,
            "Spread (95--5)":   np.percentile(fv, 95) - np.percentile(fv, 5),
        }

    comp_df = pd.DataFrame(comparison).T
    comp_df = comp_df.sort_values("Expected Return", ascending=False)
    st.dataframe(
        comp_df.style
            .format("{:.2%}", subset=["Expected Return", "5th pct", "95th pct"])
            .format("{:.3f}", subset=["Spread (95--5)"])
            .background_gradient(subset=["Expected Return"], cmap="YlGnBu"),
        use_container_width=True,
    )

    # Dot-plot
    fig_comp = go.Figure()
    for strat_name, row in comparison.items():
        fig_comp.add_trace(go.Scatter(
            x=[strat_name],
            y=[row["Expected Return"]],
            mode="markers",
            marker=dict(size=14, symbol="circle"),
            name=strat_name,
            error_y=dict(
                type="data",
                symmetric=False,
                array     =[row["95th pct"] - row["Expected Return"]],
                arrayminus=[row["Expected Return"] - row["5th pct"]],
            ),
        ))
    fig_comp.update_layout(
        title="Expected 1Y Return with 95% Confidence Interval",
        yaxis_tickformat=".1%",
        yaxis_title="1-Year Return",
        height=420,
        showlegend=False,
    )
    st.plotly_chart(fig_comp, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# FOOTER  (covariance note lives inside tab1; disclaimer shown on every tab)
# ─────────────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "⚠️ **Disclaimer**: This app is for educational and personal research purposes only. "
    "Nothing here constitutes financial advice. All data is sourced from Yahoo Finance and FRED. "
    "Price targets are personal estimates -- not investment recommendations."
)
