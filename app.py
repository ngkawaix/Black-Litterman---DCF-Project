"""
DCF + Black-Litterman Portfolio Optimiser
==========================================
A Streamlit app that lets you set Bear / Base / Bull price targets per stock
(derived from your DCF work) and see how each scenario shifts the Black-Litterman
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
from datetime import datetime
from scipy.optimize import minimize

# Your custom EDHEC helper module -- keep edhec_risk_kit_final.py in the same folder
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
    "AAPL", "ADBE", "AMAT", "AMD",  "AMZN", "AVGO", "CLS",  "CPRT",
    "DUOL", "FICO", "GE",   "GOOGL","IBM",  "IONQ", "LLY",  "LRCX",
    "MA",   "MSCI", "MSFT", "MU",   "NFLX", "NOW",  "NVDA", "PANW",
    "PYPL", "SOFI", "TSM",  "UBER", "V",    "WDC",
])

# Your base (Base case) price targets from the notebook
BASE_TARGETS = {
    "AAPL": 315.00, "ADBE": 380.00, "AMAT": 380.00, "AMD":  357.00,
    "AMZN": 300.00, "AVGO": 467.00, "CLS":  447.00, "CPRT":  43.00,
    "DUOL": 120.00, "FICO":1655.00, "GE":   352.00, "GOOGL":460.00,
    "IBM":  260.00, "IONQ":  67.00, "LLY": 1254.00, "LRCX": 310.00,
    "MA":   650.00, "MSCI": 700.00, "MSFT": 600.00, "MU":   455.00,
    "NFLX": 117.00, "NOW":  140.00, "NVDA": 275.00, "PANW": 225.00,
    "PYPL":  50.00, "SOFI":  20.00, "TSM":  465.00, "UBER": 107.00,
    "V":    394.00, "WDC":  415.00,
}

# Bear = 20 % below base;  Bull = 25 % above base  (reserved for future DCF scenario toggle)
# BEAR_TARGETS and BULL_TARGETS will be reintroduced once per-stock DCF models are complete.

BASE_CONFIDENCE = {
    "AAPL": 0.55, "ADBE": 0.40, "AMAT": 0.45, "AMD":  0.60,
    "AMZN": 0.65, "AVGO": 0.65, "CLS":  0.60, "CPRT": 0.50,
    "DUOL": 0.35, "FICO": 0.35, "GE":   0.65, "GOOGL":0.60,
    "IBM":  0.45, "IONQ": 0.20, "LLY":  0.55, "LRCX": 0.60,
    "MA":   0.50, "MSCI": 0.55, "MSFT": 0.60, "MU":   0.40,
    "NFLX": 0.60, "NOW":  0.60, "NVDA": 0.65, "PANW": 0.45,
    "PYPL": 0.40, "SOFI": 0.35, "TSM":  0.65, "UBER": 0.65,
    "V":    0.65, "WDC":  0.45,
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
def load_market_data(tickers, start="2000-01-01"):
    data = yf.download(tickers, start=start, interval="1d", auto_adjust=True, progress=False)
    price_data = data["Close"]
    price_data.index = price_data.index.tz_localize(None)
    price_data = price_data.loc[~price_data.index.duplicated(keep="first")]
    tick_rets  = price_data.pct_change().dropna()
    return price_data, tick_rets


@st.cache_data(show_spinner="Fetching ticker metadata (mcap, consensus, earnings)…", ttl=86400)
def load_ticker_metadata(_price_data, tickers):
    """
    Single-pass fetch for all per-ticker metadata: market cap, analyst
    consensus target, analyst count, and recent-earnings flag.

    Replaces the previous separate load_mcap_weights and
    load_consensus_and_earnings functions, halving the number of API calls
    (one .info + one .calendar per ticker instead of up to three .info calls).
    A 0.2 s sleep between tickers keeps Yahoo Finance from rate-limiting
    the Streamlit Cloud shared IP.

    Returns
    -------
    mcap_weights_df : pd.DataFrame  -- cap-weight matrix matching price_data index
    consensus       : dict          -- {ticker: {"mean": float|None, "n_analysts": int|None}}
    recent_earnings : dict          -- {ticker: bool}
    """
    import time

    today  = pd.Timestamp.today().normalize()
    cutoff = today - pd.Timedelta(days=30)

    mcap            = {}
    consensus       = {}
    recent_earnings = {}

    for ticker in tickers:
        t = yf.Ticker(ticker)

        # ── Single .info call — covers mcap, consensus fallback, analyst count ─
        info = {}
        try:
            info = t.info
        except Exception:
            pass

        mcap[ticker] = info.get("marketCap", None)

        # Consensus: try dedicated property first (yfinance 0.2.x), fall back to info
        mean_t     = None
        n_analysts = info.get("numberOfAnalystOpinions", None)
        try:
            apt    = t.analyst_price_targets
            mean_t = apt.get("mean", None) if isinstance(apt, dict) else None
        except Exception:
            pass
        if mean_t is None:
            mean_t = info.get("targetMeanPrice", None)

        consensus[ticker] = {"mean": mean_t, "n_analysts": n_analysts}

        # ── Calendar call for earnings recency flag ───────────────────────────
        flagged = False
        try:
            cal   = t.calendar
            dates = []
            if isinstance(cal, dict):
                raw   = cal.get("Earnings Date", [])
                dates = raw if isinstance(raw, list) else [raw]
            elif isinstance(cal, pd.DataFrame) and not cal.empty:
                col_name = "Earnings Date" if "Earnings Date" in cal.columns else None
                dates    = cal[col_name].dropna().tolist() if col_name else cal.iloc[0].dropna().tolist()
            flagged = any(
                cutoff <= pd.Timestamp(d).normalize() <= today
                for d in dates if d is not None
            )
        except Exception:
            pass
        recent_earnings[ticker] = flagged

        time.sleep(0.2)   # stay well within Yahoo Finance's rate limit

    # Build cap-weight DataFrame
    mcap_series = pd.Series(mcap)
    missing     = mcap_series[mcap_series.isna()].index.tolist()
    if missing:
        avg = mcap_series.dropna().mean()
        mcap_series[missing] = avg if pd.notna(avg) and avg > 0 else 1.0
    weights = mcap_series / mcap_series.sum()
    idx     = _price_data.index
    mcap_weights_df = pd.DataFrame(
        [weights.reindex(tickers).values] * len(idx),
        index=idx, columns=tickers,
    )

    return mcap_weights_df, consensus, recent_earnings


@st.cache_data(show_spinner="Loading risk-free rate from FRED…")
def load_rf():
    try:
        url  = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DGS3MO"
        df   = pd.read_csv(url, parse_dates=["DATE"])
        df   = df[df["DGS3MO"] != "."]
        rate = float(df["DGS3MO"].iloc[-1]) / 100
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
    current_prices = price_targets.index.map(
        lambda t: tick_rets.shape   # placeholder -- resolved below
    )
    # This is cleaner:
    price_data_last = tick_rets   # we only need the index here; prices come from outside

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

    cw_w = tick_capweights.reindex(columns=tick_rets.columns).iloc[-1]
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


def bl_msr_longonly(sigma, mu, riskfree_rate):
    """Max-Sharpe optimisation with long-only constraint."""
    n = mu.shape[0]
    bounds      = ((0.0, 1.0),) * n
    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}

    def neg_sharpe(w):
        ret = w @ mu
        vol = np.sqrt(w @ sigma.values @ w)
        return -(ret - riskfree_rate) / vol

    result = minimize(
        neg_sharpe,
        np.repeat(1 / n, n),
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
# Load consensus data before entering the sidebar block so it's available
# when rendering per-ticker inputs.  TTL=24 h keeps it fresh without
# hammering the API on every widget interaction.
# Price data is loaded here (before the sidebar) so load_ticker_metadata
# can use it to build cap-weight DataFrames in the same pass.
price_data, tick_rets = load_market_data(TICKERS)
tick_capweights, consensus_data, recent_earnings = load_ticker_metadata(price_data, TICKERS)

with st.sidebar:
    st.title("⚙️ Model Assumptions")

    # --- BL parameters ---
    st.subheader("1. Black-Litterman Parameters")
    st.caption(
        "**δ (delta)** = risk-aversion coefficient of the market portfolio. "
        "Standard value is 2.5. Higher --> market expects more return per unit of risk."
    )
    delta = st.slider("δ  Risk Aversion", min_value=1.0, max_value=5.0, value=2.5, step=0.1)

    st.caption(
        "**τ (tau)** = uncertainty in the prior (equilibrium) returns. "
        "Smaller τ means you trust the market prior more over your own views."
    )
    tau = st.slider("τ  Prior Uncertainty", min_value=0.01, max_value=0.10, value=0.02, step=0.01)

    st.divider()

    # --- Per-stock price targets ---
    st.subheader("2. Price Targets & Confidence")
    st.caption(
        "**How these are set:** Base targets are personal estimates derived from "
        "DCF work (May 2025) and updated as new information arrives. "
        "Consensus figures are sourced from Yahoo Finance analyst aggregates and "
        "may lag recent revisions — treat as directional reference only. "
        "**Confidence** (Idzorek method) weights your view vs. the market-implied "
        "equilibrium: 0 = ignore your view entirely, 1 = full conviction."
    )
    st.caption("🟡 Ticker flagged = earnings reported in the last 30 days — consensus may have been revised.")

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
                    n_str = f" · {int(n_ana)} analysts" if n_ana else ""
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
    st.caption(
        "🔮 **DCF integration coming soon** -- once your Wall Street Prep models "
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
    bl_w = bl_msr_longonly(sigma=sigma_bl, mu=mu_bl, riskfree_rate=RF)
    bl_w_series = bl_w   # pd.Series indexed by ticker

# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.title("📊 DCF × Black-Litterman Portfolio Optimiser")
st.caption(
    f"Risk-free rate (3M T-bill): **{RF:.2%}**  |  "
    f"Universe: **{len(TICKERS)} stocks**  |  "
    f"Data range: **{tick_rets.index[0].date()} → {tick_rets.index[-1].date()}**"
)

# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📋 Views & Returns",
    "⚖️ BL Weights",
    "📈 Simulation & Stress Tests",
    "🔀 Strategy Comparison",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 -- Views & Returns Decomposition
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown(
        """
        I built this to answer a question that nagged me through two courses — EDHEC's Advanced
        Portfolio Construction and Wall Street Prep's DCF programme: once you have a view on what
        a stock is worth, how do you actually size the position? Most backtested strategies like
        Global Minimum Variance or Risk Parity are purely backward-looking — they optimise on
        historical data and assume the past repeats. Black-Litterman is different: it takes a
        forward-looking view on what each stock is worth and asks how much conviction you should
        actually act on, relative to what the market already implies. The app lets you build that
        allocation, stress-test it against real market crashes, and benchmark it against the simpler
        strategies. It's a work in progress, but one I've found genuinely useful — and hopefully
        others will too.
        """
    )

    st.divider()

    st.subheader("Return Decomposition")
    st.caption(
        "This table shows how each input layer stacks up. "
        "**Q** is the excess-return view you feed in via your price target. "
        "**π (pi)** is what the market *implies* given cap weights and risk aversion. "
        "**BL Posterior** is the blended result -- how much your view shifts the equilibrium."
    )

    view_df = pd.DataFrame({
        "Current Price ($)":          current_prices,
        "Price Target ($)":           targets_series,
        "Total Return View":          total_return_views,
        "Risk-Free Rate (rf)":        RF,
        "Excess Return View (Q)":     total_return_views - RF,
        "Market-Implied Return (π)":  pi,
        "BL Posterior Return":        mu_bl,
        "Confidence":                 pd.Series(user_confidence).reindex(tick_rets.columns),
    }).sort_values("Excess Return View (Q)", ascending=False)

    pct_cols = [
        "Total Return View", "Risk-Free Rate (rf)",
        "Excess Return View (Q)", "Market-Implied Return (π)", "BL Posterior Return",
        "Confidence",
    ]
    dollar_cols = ["Current Price ($)", "Price Target ($)"]

    styled = view_df.style \
        .format("{:.2%}", subset=pct_cols) \
        .format("${:,.2f}", subset=dollar_cols) \
        .background_gradient(subset=["BL Posterior Return"], cmap="RdYlGn") \
        .background_gradient(subset=["Excess Return View (Q)"], cmap="RdYlGn")

    st.dataframe(styled, use_container_width=True)

    # Bar chart: Q vs pi vs BL
    fig = go.Figure()
    sorted_tickers = view_df.index.tolist()

    fig.add_trace(go.Bar(
        name="Market-Implied (π)",
        x=sorted_tickers,
        y=pi.reindex(sorted_tickers).values,
        marker_color="steelblue",
        opacity=0.7,
    ))
    fig.add_trace(go.Bar(
        name="View (Q)",
        x=sorted_tickers,
        y=(total_return_views - RF).reindex(sorted_tickers).values,
        marker_color="orange",
        opacity=0.7,
    ))
    fig.add_trace(go.Bar(
        name="BL Posterior",
        x=sorted_tickers,
        y=mu_bl.reindex(sorted_tickers).values,
        marker_color="seagreen",
        opacity=0.9,
    ))
    fig.update_layout(
        barmode="group",
        title="π  vs  View (Q)  vs  BL Posterior Return",
        yaxis_tickformat=".1%",
        height=420,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 -- BL Portfolio Weights
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Black-Litterman Optimal Weights (Long-Only Max Sharpe)")
    st.caption(
        "Weights are from a long-only max-Sharpe optimisation using the BL "
        "posterior covariance and expected returns. Zero weights mean the "
        "optimiser found no risk-adjusted benefit given the posterior."
    )

    # Also compute other strategy weights for comparison
    ew_w   = erk.weight_ew(tick_rets)
    cw_w_s = tick_capweights.reindex(columns=tick_rets.columns).iloc[-1]
    cw_w_s = cw_w_s / cw_w_s.sum()
    gmv_w  = erk.weight_gmv(tick_rets, cov_estimator=erk.shrinkage_cov, delta=0.7)
    erc_w  = erk.weight_erc(tick_rets, cov_estimator=erk.shrinkage_cov, delta=0.7)

    weight_df = pd.DataFrame({
        "BL Optimised":        bl_w_series,
        "Equal-Weighted":      ew_w,
        "Cap-Weighted":        cw_w_s,
        "GMV (Shrinkage)":     gmv_w,
        "Risk Parity":         erc_w,
    }).sort_values("BL Optimised", ascending=False)

    st.dataframe(
        weight_df.style.format("{:.2%}").background_gradient(
            subset=["BL Optimised"], cmap="Blues"
        ),
        use_container_width=True,
    )

    # Treemap of BL weights
    nonzero = weight_df[weight_df["BL Optimised"] > 0.001].reset_index()
    nonzero.columns = ["Ticker"] + list(nonzero.columns[1:])
    fig_tree = px.treemap(
        nonzero,
        path=["Ticker"],
        values="BL Optimised",
        title=f"BL Weight Allocation",
        color="BL Optimised",
        color_continuous_scale="Blues",
    )
    fig_tree.update_traces(textinfo="label+percent entry")
    st.plotly_chart(fig_tree, use_container_width=True)

    # Grouped bar: BL vs cap-weighted
    fig2 = go.Figure()
    fig2.add_trace(go.Bar(
        name="BL Optimised",
        x=weight_df.index,
        y=weight_df["BL Optimised"],
        marker_color="steelblue",
    ))
    fig2.add_trace(go.Bar(
        name="Cap-Weighted",
        x=weight_df.index,
        y=weight_df["Cap-Weighted"],
        marker_color="lightcoral",
    ))
    fig2.update_layout(
        barmode="group",
        title="BL vs Cap-Weighted Allocation",
        yaxis_tickformat=".1%",
        height=380,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig2, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 -- Simulation & Stress Tests
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("Correlated GBM Monte Carlo (BL Drift)")
    st.caption(
        "Each path simulates one year of daily returns. "
        "Drift comes from the **BL posterior returns**. "
        "Correlations are preserved via Cholesky decomposition of the sample covariance. "
        "Starting value is normalised to $1."
    )

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
        title="Distribution of Final Portfolio Value ($1 invested)",
        xaxis_title="Portfolio Value",
        yaxis_title="Frequency",
        height=360,
    )
    st.plotly_chart(fig_hist, use_container_width=True)


    st.divider()

    st.markdown(
        """
        A correlated GBM treats future returns as a random walk — each day's shock is drawn
        independently, scaled by historical volatility and the correlations between stocks.
        What it cannot capture is the clustering of extreme events: in real markets, crashes
        are not randomly distributed. Volatility spikes, correlations break down, and losses
        arrive in sequences that a Gaussian model systematically understates. The simulation
        above is best read as a baseline of what *typical* uncertainty looks like — not as a
        statement about tail risk. The stress tests below ground that picture in what actually
        happened.
        """
    )

    st.divider()

    # ── Historical Stress Tests ───────────────────────────────────────────────
    st.subheader("Historical Stress Test — BL Weights")
    st.caption(
        "Applies your BL optimal weights to *actual* historical returns during "
        "known market shocks. This shows how this allocation *would* have "
        "performed -- not a forecast."
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
            .format("{:.0f}", subset=["Trading Days"])
            .background_gradient(subset=["Period Return"], cmap="RdYlGn")
            .background_gradient(subset=["Max Drawdown"], cmap="RdYlGn_r"),
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
# TAB 4 -- Strategy Comparison
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    # ── Section 1: Backtested Wealth Index ───────────────────────────────────
    st.subheader("Historical Wealth Index — Strategy Comparison")
    st.caption(
        "Rolling backtest using a 2-year estimation window to rebalance weights at each step. "
        "EW, Cap-Weighted, GMV, and Risk Parity are properly rolled — weights are re-estimated "
        "each period using only data available at that point in time, so there is no look-ahead. "
        "**Black-Litterman** is shown as a static allocation using the current optimal "
        "weights applied to the full history. This is a simplification: true BL weights would "
        "require a fresh set of views at every rebalance date. The chart is therefore best read "
        "as 'how would this portfolio have held up' rather than a like-for-like backtest."
    )

    estimation_window = 2 * 252

    ew_r, cw_r, gmv_r, erc_r = run_backtests(
        tick_rets, tick_capweights, estimation_window
    )

    # BL: static weights applied to full history, aligned to rolling start date
    bl_static_w = bl_w_series.reindex(tick_rets.columns).fillna(0)
    bl_r        = (tick_rets * bl_static_w.values).sum(axis=1).loc[ew_r.index[0]:]

    btr = pd.DataFrame({
        "Equal-Weighted":          ew_r,
        "Cap-Weighted":            cw_r,
        "GMV (Shrinkage)":         gmv_r,
        "Risk Parity":             erc_r,
        "BL (static)": bl_r,
    }).dropna()

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
        f"Starts 2 years after data begins — the minimum needed for the first rolling estimate."
    )

    wealth = (1 + btr).cumprod()

    # Colour palette: BL stands out, others are muted
    palette = {
        "Equal-Weighted":          ("steelblue",    1.2, "dot"),
        "Cap-Weighted":            ("slategray",    1.2, "dot"),
        "GMV (Shrinkage)":         ("darkorange",   1.2, "dot"),
        "Risk Parity":             ("mediumpurple", 1.2, "dot"),
        "BL (static)":             ("seagreen",  2.5, "solid"),
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
        title="Wealth Index — $1 invested at backtest start",
        xaxis_title="Date",
        yaxis_title="Portfolio Value ($)",
        height=460,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig_wealth, use_container_width=True)

    # Summary stats
    summary_rows = {}
    for col in btr.columns:
        r      = btr[col].dropna()
        ann_r  = erk.annualize_rets(r, periods_per_year=252)
        ann_v  = erk.annualize_vol(r, periods_per_year=252)
        sharpe = erk.sharpe_ratio(r, riskfree_rate=RF, periods_per_year=252)
        cp     = (1 + r).cumprod()
        mdd    = (cp / cp.cummax() - 1).min()
        summary_rows[col] = {
            "Ann. Return":  ann_r,
            "Ann. Vol":     ann_v,
            "Sharpe Ratio": sharpe,
            "Max Drawdown": mdd,
        }

    summary_df = pd.DataFrame(summary_rows).T
    st.dataframe(
        summary_df.style
            .format("{:.2%}", subset=["Ann. Return", "Ann. Vol", "Max Drawdown"])
            .format("{:.2f}", subset=["Sharpe Ratio"])
            .background_gradient(subset=["Ann. Return"],  cmap="RdYlGn")
            .background_gradient(subset=["Sharpe Ratio"], cmap="RdYlGn")
            .background_gradient(subset=["Max Drawdown"], cmap="RdYlGn_r"),
        use_container_width=True,
    )

    st.divider()

    # ── Section 2: Forward Distribution (Monte Carlo) ─────────────────────────
    st.subheader("Forward Distribution — Strategy Comparison (Monte Carlo)")
    st.caption(
        "Uses the same correlated GBM paths from Tab 3 but applied to each "
        "strategy's weights. Lets you see whether BL adds value over simpler alternatives."
    )

    strategy_weights = {
        "Equal-Weighted":   pd.Series(ew_w, index=tick_rets.columns),
        "Cap-Weighted":     cw_w_s,
        "GMV (Shrinkage)":  pd.Series(gmv_w, index=tick_rets.columns),
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
    st.dataframe(
        comp_df.style
            .format("{:.2%}", subset=["Expected Return", "5th pct", "95th pct"])
            .format("{:.3f}", subset=["Spread (95--5)"])
            .background_gradient(subset=["Expected Return"], cmap="RdYlGn"),
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
        title="Expected Return with 5th--95th pct Range",
        yaxis_tickformat=".1%",
        yaxis_title="1-Year Return",
        height=420,
        showlegend=False,
    )
    st.plotly_chart(fig_comp, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "⚠️ **Disclaimer**: This app is for educational and personal research purposes only. "
    "Nothing here constitutes financial advice. All data is sourced from Yahoo Finance and FRED. "
    "Price targets are personal estimates -- not investment recommendations."
)
