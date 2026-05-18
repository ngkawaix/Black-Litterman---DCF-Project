"""
DCF + Black-Litterman Portfolio Optimiser
==========================================
A Streamlit app for setting Bear / Base / Bull price targets per stock
(derived from DCF work) and seeing how each scenario shifts the Black-Litterman
optimal portfolio weights, Monte Carlo distribution, and stress-test results.

How to run locally:"**Views & Weights** to see how price targets feed through the model → "
    streamlit run app.py

Deployment note:
    edhec_risk_kit_final.py  <-- must sit in the same folder as this file
    requirements.txt         <-- must also be in the repo rootdot
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
    page_title="Portfolio Optimiser (DCF-BL)",
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
    "AAPL": 305.00, "ADBE": 328.00, "AMAT": 486.00,
    "AMZN": 312.00, "ASML": 1661.00, "CPRT":  43.00,
    "FICO": 1562.00, "GOOGL": 428.00, "LRCX": 310.00,
    "MA":   650.00, "META": 827.00, "MSCI": 685.00,
    "MSFT": 562.00, "NFLX": 115.00, "NVDA": 270.00,
    "TSM":  463.00, "V":    399.00,
}

# FOR FUTURE IMPLEMENTATION: Bear = 20 % below base;  Bull = 25 % above base  (reserved for future DCF scenario toggle)

BASE_CONFIDENCE = {
    "AAPL": 0.20, "ADBE": 0.25, "AMAT": 0.15,
    "AMZN": 0.25, "ASML": 0.10, "CPRT": 0.20,
    "FICO": 0.45, "GOOGL":0.15, "LRCX": 0.10,
    "MA":   0.55, "META": 0.65, "MSCI": 0.55,
    "MSFT": 0.60, "NFLX": 0.55, "NVDA": 0.10,
    "TSM":  0.25, "V":    0.50,
}

# ── DCF Override Dict ─────────────────────────────────────────────────────────
# Once individual DCF models are complete, populate this dict to override
# the analyst-narrative confidence levels above with model-derived values.
# Any ticker present here will display a ⚡ DCF Model badge in the Investment
# Theses tab, visually distinguishing it from the pre-filled analyst estimates.
# Example (do not uncomment until DCF work is finalised):
#   "NVDA": 0.80, "META": 0.72, "AMZN": 0.68,
DCF_OVERRIDES: dict[str, float] = {}

# ── Earnings Highlights ───────────────────────────────────────────────────────
# Two-sentence summary of the most recent earnings report for each stock.
EARNINGS_HIGHLIGHTS = {
    "AAPL": "Services hit a record high on App Store and subscription strength, partly offsetting softer iPhone units. India manufacturing ramp progressing; tariff-driven supply chain risk remains a near-term overhang.",
    "ADBE": "Net new ARR missed for a second consecutive quarter - Firefly AI yet to move the growth needle. Stock down ~42% from highs on fears generative AI is commoditising the creative suite.",
    "AMAT": "Revenue declined YoY; NAND and trailing-edge weakness outweighed advanced packaging gains. China export restrictions cap access to one of the largest semiconductor equipment markets.",
    "AMZN": "AWS re-accelerated to +28% YoY ($37.6B) on AI inference and model-training demand - fastest growth in 15 quarters. Operating margin hit a record 13.1% - clear inflection from investment phase to profitable scale.",
    "ASML": "Raised 2026 sales guidance in April on robust EUV backlog from TSMC and Samsung. Sole global EUV supplier; demand visibility extends into 2027 with no credible competitive threat.",
    "CPRT": "Revenue grew steadily on resilient salvage volumes and international expansion. Debt-free balance sheet; margins held despite elevated SG&A - rare quality among mid-cap industrials.",
    "FICO": "Scores revenue outpaced platform, driven by usage-based pricing gains in mortgage and auto. Monetisation ceiling expanding as lenders accept higher per-score royalty rates.",
    "GOOGL": "Search (+19% YoY) and YouTube outperformed; Google Cloud surged +63% YoY to $20B on AI infrastructure demand. Gemini monetisation showing early signals - GenAI cost-per-query is the key margin watch item.",
    "LRCX": "Revenue surged on NAND recovery and advanced logic spend from TSMC and Samsung. Customer Support Business Group (~35% of revenue) provides structural insulation from equipment cycles.",
    "MA":   "Revenue grew on cross-border volume recovery and a higher-margin value-added services mix. No material consumer credit deterioration visible in transaction data - payments outlook resilient.",
    "META": "Revenue beat consensus; AI ad-relevance improvements lifted CPMs across Facebook and Instagram. Family of Apps DAUs grew - Llama-driven Reels ranking cited as the key engagement driver.",
    "MSCI": "Recurring subscription revenue grew; index and analytics retention above 95% reflects high switching costs. ESG & Real Assets showing early recovery after several quarters of soft institutional demand.",
    "MSFT": "Azure accelerated to +40% YoY, beating consensus; AI business hit a $37B annualised run rate (+123% YoY). Operating margin held at 46% - AI infrastructure spend absorbed without material dilution.",
    "NFLX": "Subscriber additions beat; ad-supported tier now ~40% of new sign-ups in available markets. Full-year FCF guidance raised - content investment funded internally without balance sheet strain.",
    "NVDA": "Q4 FY2026 revenue +73% YoY on Blackwell GPU shipments to hyperscalers. Q1 FY2027 guided ~$78B - US H20 export restrictions to China are a multi-billion dollar annualised headwind.",
    "TSM":  "Advanced node (3nm/5nm) mix expanded on Apple, NVIDIA, and AMD demand. Arizona fab capex on track; Taiwan geopolitical risk remains the primary discount embedded in the stock.",
    "V":    "Payments volume and transactions grew in line with estimates on cross-border recovery. Value-added services growing faster than core volume - gradually shifting mix to higher margins.",
}

CONFIDENCE_RATIONALE = {
    "AAPL": (
        "Target sits 14% below the market-implied price, a deliberate mild bearish tilt on tariff overhang and supply chain transition risk. "
        "Confidence at 0.20 preserves the discount signal without aggressively underweighting the second-largest position in the universe."
    ),
    "ADBE": (
        "Target is 12% above the market-implied price, a bullish view against a structurally challenged backdrop. "
        "Confidence at 0.25 is the lowest among bullish names | Reflects two consecutive ARR misses and unproven Firefly AI monetisation."
    ),
    "AMAT": (
        "Target is 9% below the market-implied price on continued NAND and trailing-edge weakness. "
        "Confidence at 0.15 keeps the bearish tilt soft | Deutsche Bank's $550 PT raise and solid Q2 execution prevent a stronger discount."
    ),
    "AMZN": (
        "Target falls 1.5% below the market-implied price, a near-accidental bearish gap that does not reflect a genuine bearish view. "
        "Confidence at 0.25 reflects confidence | AWS +28% YoY and record 13.1% operating margin do not warrant a discount."
    ),
    "ASML": (
        "Target is 9% below the market-implied price because any consensus-anchored estimate will sit below a market already pricing in ASML's EUV monopoly and 2027 demand visibility. "
        "Confidence at 0.10 preserves equilibrium weight | Reflects a data limitation, not a bearish thesis."
    ),
    "CPRT": (
        "Target is 16% above the market-implied price, but the stock has declined ~50% from its 52-week high. "
        "Confidence at 0.20 reflects uncertainty following the sharp de-rating, despite a debt-free balance sheet and resilient margins."
    ),
    "FICO": (
        "Target is 20% above the market-implied price, supported by strong Q2 results and accelerating Scores revenue. "
        "Confidence at 0.45 reflects uncertainty following Fannie Mae's April 2026 approval of VantageScore 4.0 as a FICO alternative | Structural competitive risk warrants caution."
    ),
    "GOOGL": (
        "Target is 9% below the market-implied price, but this does not reflect a genuine bearish view — Q1 2026 revenue +19% and Cloud +63% YoY contradict underweighting. "
        "Confidence at 0.15 defers largely to the market given continued execution and a significant run-up over the past year."
    ),
    "LRCX": (
        "Target is 12% below the market-implied price, a stale estimate that does not reflect revenue +28% YoY or ~35% recurring service revenue providing cycle insulation. "
        "Confidence at 0.10 reflects confidence in LRCX (but need more research)."
    ),
    "MA": (
        "Target is 14% above the market-implied price, with the stock near its 52-week low despite Q1 revenue +12% YoY and a 14th consecutive dividend increase. "
        "Confidence at 0.55 reflects strong earnings clarity and a below-historical valuation entry point."
    ),
    "META": (
        "Target is 11% above the market-implied price, the strongest bullish signal in the portfolio. "
        "Confidence capped at 0.65 rather than higher given 2026 capex guidance raised to $125–145B and buybacks paused."
    ),
    "MSCI": (
        "Target is 5% above the market-implied price on 95%+ subscription retention and early ESG segment recovery. "
        "Confidence at 0.55; the zero BL allocation is a max-Sharpe correlation artefact, not a reflection of low conviction."
    ),
    "MSFT": (
        "Target is 13% above the market-implied price; the pullback from $555 to $420 has widened the gap despite Azure +40% YoY and AI revenue at a $37B annualised run rate. "
        "Confidence at 0.60 reflects conviction that the selloff overweights short-term macro noise relative to the underlying business trajectory."
    ),
    "NFLX": (
        "Target is 10% above the market, implied price on ad-supported tier momentum (~40% of new sign-ups) and raised FCF guidance. "
        "Confidence at 0.55 rather than higher; this remains a street estimate with no completed DCF model."
    ),
    "NVDA": (
        "Target is 8.5% below the market-implied price, an intentional neutralisation given NVDA carries the highest π in the universe (largest cap weight, beta 1.7) and the current target implies meaningful underperformance. "
        "Confidence at 0.10 to let equilibrium dominate; a DCF-derived target above the break-even of $295 is needed before meaningful bullish confidence is warranted."
    ),
    "TSM": (
        "Target is 4% below the market-implied price, a deliberate discount for persistent Taiwan geopolitical risk. "
        "Confidence at 0.25, halved from its previous level, preserves the signal without making TSM a dominant underweight."
    ),
    "V": (
        "Target is 7% above the market-implied price, a genuine bullish view on cross-border recovery and a value-added services mix shift toward higher margins. "
        "The zero BL allocation reflects optimiser concentration in the higher-posterior MA rather than low conviction; tightening the max position size to 12–15% would reinstate both."
    ),
}

STRESS_PERIODS = {
    "COVID Crash (Feb--Mar 2020)":        ("2020-02-19", "2020-03-23"),
    "Post-COVID Rate Hikes (2022)":      ("2022-01-01", "2022-12-31"),
    "Tech Selloff (Nov 2021--May 2022)":  ("2021-11-19", "2022-05-20"),
    "GFC Echo (Aug--Oct 2015)":           ("2015-08-18", "2015-10-01"),
    "Trump Tariffs (Full Year 2025)": ("2025-02-01", "2025-12-31"),
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
            "Yahoo Finance likely rate-limited pull request - refresh in a minute to retry.",
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

    today  = pd.Timestamp.today().normalize()
    cutoff = today - pd.Timedelta(days=30)

    mcap            = {}
    consensus       = {}
    recent_earnings = {}
    
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

        time.sleep(0.25)   

    mcap_series = pd.Series(mcap)
    missing     = mcap_series[mcap_series.isna()].index.tolist()

    # If any tickers failed all three API layers, fill with the median of known values
    # so they receive a roughly "average" cap weight rather than distorting the allocation.
    if missing:
        median_cap = mcap_series.median()
        mcap_series = mcap_series.fillna(median_cap)
        st.sidebar.warning(
            f"⚠️ Could not fetch market cap for: **{', '.join(missing)}**. "
            "Filled with median of available caps - cap-weight strategy may be slightly off. "
            "Yahoo Finance likely rate-limited the pull; will retry on next cache refresh (24h)."
        )

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


@st.cache_data(show_spinner="Fetching valuation metrics…", ttl=86400)
def load_valuation_metrics(tickers):
    """
    Fetches valuation ratios and earnings dates for each ticker via yfinance.
    Kept separate from load_ticker_metadata so a slow or rate-limited ratio
    pull does not block the sidebar from loading cap weights and consensus.

    Returns a DataFrame indexed by ticker with columns:
        P/E (TTM), Fwd P/E, Beta, EV/EBITDA, P/S, P/B,
        Last Earnings, Next Earnings
    """
    import time

    today = pd.Timestamp.today().normalize()
    rows  = {}

    for ticker in tickers:
        t    = yf.Ticker(ticker)
        info = {}
        try:
            info = t.info
        except Exception:
            pass

        # ── Earnings dates ────────────────────────────────────────────────────
        last_earnings = "N/A"
        next_earnings = "N/A"

        try:
            mrq = info.get("mostRecentQuarter", None)
            if mrq:
                last_earnings = pd.Timestamp(mrq, unit="s").strftime("%Y-%m-%d")
        except Exception:
            pass

        try:
            cal   = t.calendar
            dates = []
            if isinstance(cal, dict):
                raw   = cal.get("Earnings Date", [])
                dates = raw if isinstance(raw, list) else [raw]
            elif isinstance(cal, pd.DataFrame) and not cal.empty:
                col_name = "Earnings Date" if "Earnings Date" in cal.columns else None
                dates    = cal[col_name].dropna().tolist() if col_name else cal.iloc[0].dropna().tolist()
            future = [
                pd.Timestamp(d).normalize()
                for d in dates
                if d is not None and pd.Timestamp(d).normalize() >= today
            ]
            if future:
                next_earnings = min(future).strftime("%Y-%m-%d")
        except Exception:
            pass

        # ── Valuation ratios ──────────────────────────────────────────────────
        rows[ticker] = {
            "P/E (TTM)":   info.get("trailingPE",                   None),
            "Fwd P/E":     info.get("forwardPE",                    None),
            "Beta":        info.get("beta",                         None),
            "EV/EBITDA":   info.get("enterpriseToEbitda",           None),
            "P/S":         info.get("priceToSalesTrailing12Months", None),
            "P/B":         info.get("priceToBook",                  None),
            "Last Earnings": last_earnings,
            "Next Earnings": next_earnings,
        }
        time.sleep(0.25)

    return pd.DataFrame(rows).T


@st.cache_data(show_spinner=False, ttl=86400)
def load_benchmark_data():
    """
    Downloads SPY (S&P 500 ETF) daily returns from 2012 onward.
    SPY is used rather than ^GSPC so that auto_adjust=True captures
    dividend reinvestment, giving total return rather than price return.
    Cached for 24 hours - same cadence as ticker metadata.
    """
    spx = yf.download("SPY", start="2012-01-01", auto_adjust=True, progress=False)
    spx = spx["Close"].squeeze()
    spx.index = spx.index.tz_localize(None)
    return spx.pct_change().dropna()


@st.cache_data(show_spinner="Running rolling backtests…")
def run_backtests(_tick_rets, _tick_capweights, estimation_window):
    """
    Runs all four rolling-window backtests and returns a returns DataFrame.
    Results are cached against the price data hash, so re-running only happens
    when new market data is fetched - not on every slider or input interaction.

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
# CPPI HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def run_cppi(risky_r, rf, multiplier, max_drawdown, initial_value = 1, n_periods = 252):
    """
    CPPI sleeve with a high-water mark drawdown floor
    """
    
    safe_r = rf / n_periods
    dates = risky_r.index
    n = len(dates)
    
    account = np.zeros(n)
    floors = np.zeros(n) 
    alloc = np.zeros(n)

    port_v = hwm = initial_value

    for i in range(n):
        hwm = max(hwm, port_v) # checks if portfolio value exceeds the highwater mark
        floor = (1- max_drawdown) * hwm
        cushion = max(port_v - floor, 0.0)
        risky_exposure = min(multiplier * cushion, port_v)
        safe_exposure =  port_v - risky_exposure
        port_v = risky_exposure * (1 + risky_r.iloc[i]) + safe_exposure * (1 + safe_r)
        account[i] = port_v
        floors[i] = floor
        alloc[i] = risky_exposure / port_v if port_v > 0 else 0.0

    account_val = pd.Series(account, index = dates)
    floor_val = pd.Series(floors, index = dates)
    risky_alloc = pd.Series(alloc, index = dates)

    # Create an array of port_vals to derive cppi_r
    prev_vals = np.concatenate([[initial_value], account[:-1]]) 
    cppi_r    = pd.Series(account / prev_vals - 1, index=dates)

    return cppi_r, account_val, floor_val, risky_alloc

def cppi_gap_risk(port_paths, rf, multipliers, max_drawdown, initial_value = 1, n_periods = 252):
    daily_rf = rf / n_periods
    path_rets  = np.diff(port_paths, axis=0) / port_paths[:1] # derive port path returns from absolute path values
    n_steps, n_scen = path_rets.shape
    results = {}
    
    for m in multipliers:
        breaches = 0
        for j in range(n_scen):
            port_v = hwm = initial_value
            for i in range(n_steps):
                hwm = max(hwm, port_v)
                floor = (1 - max_drawdown) * hwm
                cushion = max(port_v - floor, 0.0)
                risky_exposure = min(m * cushion, port_v)
                safe_exposure = port_v - risky_exposure
                port_v = risky_exposure * (1 + path_rets[i, j]) + safe_exposure * (1 + daily_rf)
                if port_v < floor * 0.999:
                    breaches += 1
                    break
        results[m] = breaches / n_scen

    return results
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
        "**How these are set:** 1-Year price targets are rough estimates in-line with the Street View. "
        "Consensus figures are sourced from Yahoo Finance analyst aggregates and "
        "may lag recent revisions. Please read the Confidence tab to see how the confidences were set "
    )
    st.caption("🟡 Ticker flagged = earnings reported in the last 30 days - consensus may have been revised.")
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
        min_value=0.01, max_value=0.10, value=0.025, step=0.01,
        help=("Tau is the uncertainty in the prior returns benchmark. For this model, the Cap-Weighted allocation returns are the benchmark"
              "Standard value is 0.025 as used by He-Litterman. Smaller means you trust the market more."
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
spx_rets    = load_benchmark_data()
val_metrics = load_valuation_metrics(TICKERS)

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
        price_targets    = total_return_views,
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
st.title("Portfolio Optimiser (DCF-BL)")
st.caption(
    f"Risk-free rate (1Y T-bill): **{RF:.2%}** (retrieved from FRED) |  "
    f"Universe: **{len(TICKERS)} stocks** |  "
    f"Data range: **{tick_rets.index[0].date()} → {tick_rets.index[-1].date()}**"
)

# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────
tab0, tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📖 Introduction",
    "🧮 DCF Models",
    "💡 Confidence",
    "📋 Views, Returns & Weights",
    "📈 Simulation & Stress Tests",
    "🔀 Strategy Comparison",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 0 - Introduction
# ══════════════════════════════════════════════════════════════════════════════
with tab0:
    st.markdown("#### About this project")
    st.markdown(
        """
        **I started this self-guided project to better understand investment 
        management and to guide the sizing of my own portfolio allocations.** 
        It implements the skillsand knowledge that I have acquired over the last 
        three months from *EDHEC's Advanced Portfolio Construction and Analysis* 
        and *Wall Street Prep's DCF Modelling Course* - all with the aim of 
        answering one question: Having derived the 1Y price target of a stock through 
        DCF analysis, how does one use this knowledge to size their portfolios?

        **This app uses the Black Litterman (BL) Model to size portfolio allocations.** 
        It uses this model because of its advantages over other strategies. 
        Most backtested strategies like Global Minimum Variance (GMV) or Risk Parity are 
        purely backward-looking, optimising on historical data and assuming the past repeats. 
        The Black-Litterman (BL) Model is different. It takes a forward-looking view on 
        what each stock is worth and asks the investor what their confidence levels 
        are for these views, relative to the market's implied returns. Crucially, this 
        method allows investors to incorporate their views to guide portfolio allocations 
        and integrate DCF analysis into one cohesive framework. This app lets you build 
        that allocation based on some modelling assumptions from the side-bar, stress-test 
        those allocations against real market crashes, and benchmark it against other 
        strategies. 

        **One limitation of this app is that it optimises for wealth accumulation rather 
        than wealth preservation.** Institutions with fixed payment obligations 
        (i.e. sovereign wealth funds) would instead adopt a liability-driven approach, 
        matching asset duration to future cash outflows. The CPPI framework in the 
        Strategy Comparison tab is one bridge between the two: it wraps the BL equity 
        allocation inside a drawdown floor, allowing it to coexist with a capital 
        preservation mandate.

        *Last Updated: 16 May 2026*
        """
    )

    st.divider()

    st.markdown("#### How to navigate this app")
    _nav_cols = st.columns(5)
    _nav_cards = [
        ("🧮", "DCF Models",          "Where the price targets come from -- coming soon"),
        ("💡", "Confidence",          "How strongly each view is held relative to the market"),
        ("📋", "Views & Weights",     "How BL blends those views into a portfolio allocation"),
        ("📈", "Simulation & Stress", "Tail risk and historical drawdowns on the allocation"),
        ("🔀", "Strategy Comparison", "BL benchmarked against simpler allocation strategies"),
    ]
    for _col, (_emoji, _title, _desc) in zip(_nav_cols, _nav_cards):
        with _col:
            with st.container(border=True):
                st.markdown(f"**{_emoji} {_title}**")
                st.caption(_desc)
    
    with st.container(border=True):
        st.caption(
            "⚙️ **Sidebar (global)** -- price targets, confidence levels, position size constraints, "
            "BL parameters, and backtest estimation window are all adjustable from the sidebar and "
            "flow through every tab in real time."
        )

    st.divider()
        
    st.markdown("#### How Black-Litterman Works")
    st.markdown(
        """
        Most portfolio optimisers have a fundamental problem: feed them expected returns
        built purely from historical data or analyst targets, and they produce extreme,
        unstable allocations (i.e. prone to error maximisation where even small estimation errors
        result in highly concentrated portfolios where no sensible investor would put their money into).

        **Black-Litterman solves this problem by never letting a view stand alone.**
        Instead, it always asks: *relative to what the market collectively believes,
        how much should a view actually shift the allocation?*

        The model has two inputs:

        - **The market prior (π)** - what the market implies everyone should expect,
          derived by reverse-engineering the CAPM: if every investor holds the market
          portfolio, what expected returns would justify current prices and weights?
          This is the baseline the model starts from.

        - **Analyst views (Q)** - the excess return implied by each DCF price target
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
        | **μ_BL** | Posterior expected return | The model's final blended return estimate - what feeds into the optimiser |
        | **π** (pi) | Market-implied equilibrium return | What the market collectively expects, derived from cap weights and risk aversion |
        | **Q** | Analyst views | Excess return implied by each DCF price target (total return minus risk-free rate) |
        | **Σ** (Sigma) | Covariance matrix | How much each stock moves, and how they move together - captures correlation risk |
        | **τ** (tau) | Prior uncertainty scalar | How much to distrust the market prior; smaller = trust the market more |
        | **P** | View matrix | Maps each view to the stocks it applies to; here an identity matrix - one view per stock |
        | **Ω** (Omega) | View uncertainty matrix | How uncertain each analyst view is; computed from the confidence sliders via the Idzorek method |

        **The intuition:** The formula is a tug-of-war between π and Q, refereed by
        uncertainty. When confidence is high, Ω is small, its inverse is large, and Q
        pulls the posterior strongly away from π. When confidence is low, Ω is large,
        its inverse shrinks, and the posterior barely moves from equilibrium. The
        covariance Σ ensures that stocks with shared risk exposures influence each
        other - a high-conviction view on NVDA nudges the posterior for TSM too,
        because they co-move. The table in the next tab shows this blending in action.
        """
    )

# ══════════════════════════════════════════════════════════════════════════════
# TAB1 - Confidence
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    # ── Section 1: Stock Universe & Selection Criteria ───────────────────────────────────
    st.markdown("#### 1. Stock Universe & Selection Criteria")
    st.markdown(
        """
        The 17 stocks in this portfolio were selected through a systematic
        fundamental screen using four criteria: **(1) 5-year average ROIC above 15%**, **(2) debt-to-equity
        below 1**, **(3) consecutive revenue growth over 5 years**, and **(4) a minimum
        market cap of $10 billion**. ROIC was chosen as the primary quality
        filter because it measures how efficiently a company converts capital
        into profit - sustained high ROIC over multiple years is one of the
        most reliable indicators of a durable competitive advantage. Though FCF margin is also a robust way
        to screen for profitable generating companies, this method would screen out high quality companies like
        AMZN, MSFT and GOOG which are undergoing unprecedented Capex spending-cycles for data-centre build-outs.

        The resulting universe is concentrated in technology, semiconductors,
        payments infrastructure, and financial data - sectors where
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

    # ── Section 2: DCF Models (placeholder) ───────────────────────────────────
    st.markdown("#### 2. Individual DCF Models")
    st.info(
        "📐  **In progress.** Individual DCF models with bear / base / bull scenarios "
        "and WACC sensitivity tables are being built for **Amazon, Nvidia, Google, "
        "Netflix, and Meta** as part of the Wall Street Prep DCF programme. "
        "Once complete, football field diagrams and the investment narrative for each "
        "name will appear here, and their confidence levels in the table above will be "
        "updated to reflect model-derived conviction (marked ⚡).\n\n"
        "For the remaining 12 stocks, analyst consensus targets are used as the "
        "view input, with confidence set conservatively to reflect the lower "
        "specificity of street estimates versus a bottom-up model.",
        icon="🔬",
    )

# ══════════════════════════════════════════════════════════════════════════════
# TAB2 - Confidence
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.html("<div style='height: 18px;'></div>")
    st.markdown(
        """
        **This section explains the default confidence levels set in the sidebar 
        and how they interact with the model.**
        
        Confidence in the Black-Litterman Model is not absolute. It is always 
        relative to the market. A confidence of 0 means ignoring your view entirely 
        and defer to what the market implies. Conversely, a confidence of 1 means you
        trusts your view completely and let it override the market equilibrium.
        
        This matters because the direction of the effect depends on whether your 
        price target is above or below what the market is already pricing in. 
        Stocks on the right side of the Conviction Map below have analyst targets 
        that imply more upside than the market expects, so higher confidence pushes 
        the BL weight up. Stocks on the left have targets implying less upside than 
        the market, so higher confidence pulls the weight down. 
        The conviction map makes this visual.
        """
    )
    st.divider()

    # ── Shared data prep ──────────────────────────────────────────────────────
    last_close = price_data.reindex(columns=TICKERS).iloc[-1]

    today        = pd.Timestamp.today().normalize()
    soon_cutoff  = today + pd.Timedelta(days=30)

    def _earnings_status(tkr):
        """Green circle = reported < 30 days ago / Yellow = reporting within 30 days / dash otherwise."""
        if recent_earnings.get(tkr, False):
            return "🟢"
        next_e = val_metrics.loc[tkr, "Next Earnings"]
        if next_e != "N/A":
            try:
                if today <= pd.Timestamp(next_e).normalize() <= soon_cutoff:
                    return "🟡"
            except Exception:
                pass
        return "-"

    conf_vals   = {}
    conf_source = {}
    for tkr in TICKERS:
        if tkr in DCF_OVERRIDES:
            conf_vals[tkr]   = DCF_OVERRIDES[tkr]
            conf_source[tkr] = "⚡ DCF Model"
        else:
            conf_vals[tkr]   = BASE_CONFIDENCE[tkr]
            conf_source[tkr] = "Analyst est."

    # ── Section 1: Conviction Map ──────────────────────────────────────────────
    st.markdown("#### 1. Conviction Map")
    st.caption(
        "Adjusting the confidence sliders in the slide bar will result in this map changing "
        " dyanmically. Use this chart and the Confidence Levels and Rationale to make map "
        "your own convictions. "
    )

    # ── Build per-ticker conviction rows from live BL outputs ─────────────────
    _crows = []
    for _tkr in tick_rets.columns:
        _q    = float((total_return_views - RF).get(_tkr, 0)) * 100
        _pi   = float(pi.get(_tkr, 0)) * 100
        _gap  = _q - _pi
        _cc   = float(user_confidence.get(_tkr, BASE_CONFIDENCE.get(_tkr, 0.5))) * 100
        _cd   = float(BASE_CONFIDENCE.get(_tkr, 0.5)) * 100
        _post = float(mu_bl.get(_tkr, 0)) * 100
        _crows.append({
            "ticker": _tkr,  "gap": _gap,  "conf_c": _cc,  "conf_d": _cd,
            "bullish": _gap > 0,  "changed": abs(_cc - _cd) > 0.5,
            "q": _q,  "pi": _pi,  "posterior": _post,
        })
    _cdf = pd.DataFrame(_crows)

    _TEAL, _CORAL = "#1D9E75", "#D85A30"
    _xmin = min(float(_cdf["gap"].min()) - 5, -30)
    _xmax = max(float(_cdf["gap"].max()) + 5, 35)
    _ymin, _ymax = 0.0, 100.0

    # Label positions tuned for the default gap distribution; top/bottom alternated
    # for the conf=65% cluster on the bearish side to avoid overlap.
    _tpos = {
        "META": "top center",    "NFLX": "top center",    "FICO": "top center",
        "MSFT": "bottom center", "NVDA": "top center",    "ADBE": "bottom center",
        "MA":   "bottom center", "CPRT": "top center",    "MSCI": "top center",
        "V":    "top center",    "AMZN": "top center",    "AMAT": "bottom center",
        "AAPL": "bottom center", "LRCX": "bottom center", "TSM":  "bottom center",
        "ASML": "top center",   "GOOGL": "bottom center",
    }

    _fg = go.Figure()

    # Quadrant background fills
    for _x0, _x1, _y0, _y1, _fc, _op in [
        (0,     _xmax, 50,    _ymax, _TEAL,  0.07),   # top-right:  bullish + high conf
        (_xmin, 0,     50,    _ymax, _CORAL, 0.11),   # top-left:   bearish + high conf
        (0,     _xmax, _ymin, 50,   _TEAL,  0.02),   # bottom-right
        (_xmin, 0,     _ymin, 50,   _CORAL, 0.03),   # bottom-left
    ]:
        _fg.add_shape(type="rect", x0=_x0, x1=_x1, y0=_y0, y1=_y1,
                      fillcolor=_fc, opacity=_op, line_width=0, layer="below")

    # Reference lines at gap = 0 and confidence = 50 %
    _fg.add_shape(type="line", x0=0, x1=0, y0=_ymin, y1=_ymax,
                  line=dict(color="rgba(130,130,130,0.4)", width=1, dash="dot"))
    _fg.add_shape(type="line", x0=_xmin, x1=_xmax, y0=50, y1=50,
                  line=dict(color="rgba(130,130,130,0.4)", width=1, dash="dot"))

    # Dotted connectors + hollow default markers where confidence has been adjusted
    _chg = _cdf[_cdf["changed"]]
    if not _chg.empty:
        _lx, _ly = [], []
        for _, _rw in _chg.iterrows():
            _lx += [_rw["gap"], _rw["gap"], None]
            _ly += [_rw["conf_d"], _rw["conf_c"], None]
        _fg.add_trace(go.Scatter(
            x=_lx, y=_ly, mode="lines",
            line=dict(color="rgba(100,100,100,0.4)", width=1.5, dash="dot"),
            showlegend=False, hoverinfo="skip",
        ))
        for _sub, _c in [(_chg[_chg["bullish"]], _TEAL), (_chg[~_chg["bullish"]], _CORAL)]:
            if _sub.empty:
                continue
            _fg.add_trace(go.Scatter(
                x=_sub["gap"], y=_sub["conf_d"], mode="markers",
                marker=dict(size=10, color="rgba(0,0,0,0)", line=dict(color=_c, width=2)),
                showlegend=False,
                hovertemplate="<b>%{customdata}</b> (default)<br>Default conf: %{y:.0f}%<extra></extra>",
                customdata=_sub["ticker"].values,
            ))

    # Filled current markers with ticker labels
    for _bull, _c, _nm in [(True, _TEAL, "Bullish (Q > π)"), (False, _CORAL, "Bearish (Q < π)")]:
        _sub = _cdf[_cdf["bullish"] == _bull]
        _fg.add_trace(go.Scatter(
            x=_sub["gap"], y=_sub["conf_c"],
            mode="markers+text",
            text=_sub["ticker"],
            textposition=[_tpos.get(t, "top center") for t in _sub["ticker"]],
            textfont=dict(size=11, color=_c),
            marker=dict(size=10, color=_c),
            name=_nm,
            customdata=_sub[["ticker", "q", "pi", "posterior", "conf_d"]].values,
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "View Q: %{customdata[1]:.1f}%<br>"
                "Market π: %{customdata[2]:.1f}%<br>"
                "Gap Q − π: %{x:.1f}pp<br>"
                "BL Posterior: %{customdata[3]:.1f}%<br>"
                "Confidence: %{y:.0f}% (default: %{customdata[4]:.0f}%)"
                "<extra></extra>"
            ),
        ))

    # Quadrant corner labels
    for _ax, _ay, _at, _anc in [
        (_xmin + 0.5, _ymax - 1.5, "deliberate underweight ⚠", "left"),
        (_xmax - 0.5, _ymax - 1.5, "deliberate overweight",    "right"),
        (_xmin + 0.5, _ymin + 1.5, "mild underweight",          "left"),
        (_xmax - 0.5, _ymin + 1.5, "mild overweight",           "right"),
    ]:
        _fg.add_annotation(x=_ax, y=_ay, text=_at, showarrow=False,
                            xanchor=_anc, font=dict(size=10, color="rgba(120,120,120,0.7)"))

    _fg.update_layout(
        xaxis=dict(title="View gap: Q − π (%)", range=[_xmin, _xmax],
                   zeroline=False, ticksuffix="%", tickformat="+.0f"),
        yaxis=dict(title="Confidence (%)", range=[_ymin, _ymax], ticksuffix="%"),
        height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=30),
    )
    st.plotly_chart(_fg, use_container_width=True)

    st.divider()
 
    # ── Section 3: Confidence Levels & Rationale ─────────────────────────────
    st.markdown("#### 2. Confidence Levels & Rationale")
    st.caption(
        "Sorted by gap to break-even (most bullish first). "
        "The break-even price is the market-implied equilibrium price; "
        "a target above it is a bullish signal and below it is bearish. "
        "The confidence setting then determines how strongly that signal shifts the BL allocation."
        )
 
    # Build the per-ticker rows using live BL outputs
    _conf_rows = {}
    for _tkr in tick_rets.columns:
        _pi_val  = float(pi.get(_tkr, 0)) if _tkr in pi.index else 0.0
        _cp      = float(current_prices.get(_tkr, 0)) if _tkr in current_prices.index else 0.0
        _tgt     = float(user_targets.get(_tkr, 0))
        _be      = _cp * (1 + _pi_val + RF)
        _gap_pct = (_tgt - _be) / _be if _be > 0 else 0.0
        _q_val   = float((total_return_views - RF).get(_tkr, 0)) if _tkr in total_return_views.index else 0.0
        _q_pi    = _q_val - _pi_val
        _cv      = DCF_OVERRIDES.get(_tkr, BASE_CONFIDENCE.get(_tkr, 0.0))

        _conf_rows[_tkr] = {
            "Target ($)":            _tgt,
            "Break-even ($)":        _be,
            "BL Direction":          "▲ Bullish" if _q_pi > 0 else "▼ Bearish",
            "Gap to B/E":            _gap_pct,
            "Q − π":                 _q_pi,
            "Confidence":            _cv,
            "Confidence Rationale":  CONFIDENCE_RATIONALE.get(_tkr, "—"),
        }

    _conf_df = (
        pd.DataFrame(_conf_rows)
        .T
        .sort_values("Gap to B/E", ascending=False)
    )

    _styled_conf = (
        _conf_df.style
        .format("{:.2f%}",   subset=["Gap to B/E", "Q − π"])
        .format("${:,.2f}", subset=["Target ($)", "Break-even ($)"])
        .format("{:.2f)", subset=["Confidence"]
        .map(
            lambda v: "color: #1D9E75; font-weight: 500;" if "Bullish" in str(v) else
                      "color: #D85A30; font-weight: 500;" if "Bearish" in str(v) else "",
            subset=["BL Direction"],
        )
        .map(
            lambda v: "color: #1D9E75;" if isinstance(v, float) and v > 0 else
                      "color: #D85A30;" if isinstance(v, float) and v < 0 else "",
            subset=["Gap to B/E", "Q − π"],
        )
    )

    st.dataframe(
        _styled_conf,
        use_container_width=True,
        height=700,
        column_config={
            "BL Direction": st.column_config.TextColumn(
                "Direction",
                width="small",
                help="Whether your price target implies Q > π (Bullish) or Q < π (Bearish).",
            ),
            "Target ($)": st.column_config.NumberColumn(
                "Target ($)",
                width="small",
                help="Your 1-year price target for this stock.",
            ),
            "Break-even ($)": st.column_config.NumberColumn(
                "Break-even ($)",
                width="small",
                help=(
                    "The market-implied 1-year price target, derived from the market equilibrium "
                    "return (π) over your view horizon. "
                    "Computed as: Current Price x (1 + π + rf). "
                    "If your target is below this, raising confidence will underweight the stock."
                ),
            ),
            "Gap to B/E": st.column_config.NumberColumn(
                "Gap to B/E",
                width="small",
                help=(
                    "How far your target is above (+) or below (-) the market-implied price, "
                    "as a percentage. This is the x-axis of the Conviction Map above."
                ),
            ),
            "Q − π": st.column_config.NumberColumn(
                "Q - π",
                width="small",
                help=(
                    "Gap between your excess-return view (Q) and the market-implied equilibrium "
                    "return (π), expressed in return space. Positive = bullish signal. "
                    "Negative = bearish signal. Directly corresponds to the x-axis of the Conviction Map."
                ),
            ),
            "Confidence": st.column_config.NumberColumn(
                "Confidence",
                width="small",
                help=(
                    "Idzorek confidence (0-100%). Controls how strongly the price target "
                    "view overrides the market equilibrium in the BL model. "
                    "Direction matters: for Bullish stocks higher confidence overweights; "
                    "for Bearish stocks higher confidence underweights. "
                    "0% = ignore view entirely; 100% = full conviction over the prior."
                ),
            ),
            "Confidence Rationale": st.column_config.TextColumn(
                "Confidence Rationale",
                width="large",
                help=(
                    "BL-specific justification for the confidence level: explains the "
                    "gap to break-even and why this confidence is appropriate given "
                    "current earnings, business quality, and key risks."
                ),
            ),
        },
    )
    st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 - Views, Returns & Weights
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.html("<div style='height: 18px;'></div>")
    st.markdown(
        """
        **This tab walks through the full Black-Litterman pipeline in sequence.**
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
        "**BL Posterior** is the blended output - the return the optimiser actually uses."
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
        "Global Minimum Variance": gmv_w,
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

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 - Simulation & Stress Tests
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.html("<div style='height: 18px;'></div>")
    st.markdown(
        """
        **This tab evaluates Black-Litterman portfolio weights under market 
        stress using two distinct methodologies:** 
        1. **Correlated Geometric Brownian Motion (GBM) Monte Carlo Simulation**: 
        Projects future returns as a random walk, preserving asset interdependencies 
        via Cholesky decomposition. Given the heavy concentration of technology 
        equities in this portfolio universe, accounting for these embedded correlations 
        is essential for a realistic baseline.
        2. **Historical Stress Testing**: Backtests the portfolio against actual 
        historical market shocks to evaluate performance during systemic crises.

        **Critical Model Caveats**: While the correlated GBM captures *typical* 
        market uncertainty, it fundamentally assumes a Gaussian distribution. 
        It cannot model **volatility clustering** or the **breakdown of historical 
        correlations** that occur during severe market drawdowns. In a free-fall market, 
        diversification benefits often vanish-a tail-risk reality that standard GBM 
        systematically understates. Consequently, the Monte Carlo simulation represents 
        a baseline for normal market regimes, while the historical stress test provides 
        the necessary reality check for tail risk. A more robust approach would replace 
        the Gaussian assumption with a model that allows for fatter tails and volatility 
        clustering, such as a GARCH-based simulation, and is a natural extension of this work.
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

    # Summary stats - expressed as 1Y return (final value – 1) with $10k terminal context
    mean_ret   = float(np.mean(final_values))   - 1
    median_ret = float(np.median(final_values)) - 1
    p5_ret     = float(np.percentile(final_values,  5)) - 1
    p95_ret    = float(np.percentile(final_values, 95)) - 1
    spread     = p95_ret - p5_ret

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Expected Return",  f"{mean_ret:+.1%}",  delta=f"${(1 + mean_ret)*10_000:,.0f} on $10k",  delta_color="off")
    m2.metric("Median Return",    f"{median_ret:+.1%}", delta=f"${(1 + median_ret)*10_000:,.0f} on $10k", delta_color="off")
    m3.metric("5th Percentile",   f"{p5_ret:+.1%}",   delta=f"${(1 + p5_ret)*10_000:,.0f} on $10k",   delta_color="off")
    m4.metric("95th Percentile",  f"{p95_ret:+.1%}",  delta=f"${(1 + p95_ret)*10_000:,.0f} on $10k",  delta_color="off")
    m5.metric("Uncertainty Band", f"{spread:.1%}",     delta="95th − 5th pct width",                   delta_color="off")

    # Fan chart and histogram - side by side
    col_fan, col_hist = st.columns([3, 2])

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
        title="GBM Portfolio Paths (starting $1)",
        xaxis_title="Trading Day",
        yaxis_title="Portfolio Value ($)",
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    col_fan.plotly_chart(fig_mc, use_container_width=True)

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
        title="Distribution of 1Y Final Value",
        xaxis_title="Portfolio Value ($)",
        yaxis_title="Frequency",
        height=400,
    )
    col_hist.plotly_chart(fig_hist, use_container_width=True)

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
    
    # Summary Stats for Stress Periods
    for name, (start, end) in STRESS_PERIODS.items():
        period_rets = tick_rets.loc[start:end]
        if period_rets.empty:
            continue
        bl_period_rets = (period_rets * w_stress).sum(axis=1)
        
        # Calculate the single compounded return for the period
        total_period_return = (1 + bl_period_rets).prod() - 1 
        
        ann_rets_bl_period = erk.annualize_rets(bl_period_rets, periods_per_year=252)
        ann_vol_bl_period = erk.annualize_vol(bl_period_rets, periods_per_year=252)
        skew_bl_period = erk.skewness(bl_period_rets) # Left here but not useful acutally for short period stress tests
        kurt_bl_period = erk.kurtosis(bl_period_rets) # Left here but not useful acutally for short period stress tests
        cf_var_bl_period = erk.var_gaussian(bl_period_rets, level=5, modified=True) * np.sqrt(252) # Left here but not useful acutally for short period stress tests
        cvar_bl_period = erk.cvar_historic(bl_period_rets, level=5) * np.sqrt(252) # Left here but not useful acutally for short period stress tests
        sharpe_bl_period = erk.sharpe_ratio(bl_period_rets, riskfree_rate=RF, periods_per_year=252)
        cumprod_bl = (1 + bl_period_rets).cumprod()
        max_dd_bl = (cumprod_bl / cumprod_bl.cummax() - 1).min()

        # SPY benchmark for the same period
        spx_period = spx_rets.loc[start:end]
        spx_period_return = float((1 + spx_period).prod() - 1)
        excess_return = (total_period_return - spx_period_return)
        cumprod_spx = (1 + spx_period).cumprod()
        max_dd_spx = float((cumprod_spx / cumprod_spx.cummax() - 1).min())

        stress_rows[name] = {
            "Period Return":    total_period_return,
            "S&P 500 (SPY) Return":   spx_period_return,
            "Excess Return":    excess_return,
            "Trading Days":     len(bl_period_rets),
            "Ann. Return":      ann_rets_bl_period,
            "Ann. Vol":         ann_vol_bl_period,
            "Sharpe Ratio":     sharpe_bl_period,
            "Max Drawdown":     max_dd_bl,
            "SPY Max Drawdown": max_dd_spx 
        }

    stress_df = pd.DataFrame(stress_rows).T
    st.dataframe(
        stress_df.sort_values("Max Drawdown", ascending=True)
            .style
            .format("{:.2%}", subset=["Period Return", "S&P 500 (SPY) Return", "Excess Return",
                                      "Ann. Return", "Ann. Vol", "Max Drawdown", "SPY Max Drawdown"])
            .format("{:.0f}", subset=["Trading Days"])
            .format("{:.2f}", subset=["Sharpe Ratio"])
            .background_gradient(subset=["Max Drawdown"], cmap="Reds_r", vmax=0.0)
            .background_gradient(subset=["SPY Max Drawdown"], cmap="Reds_r", vmax=0.0),
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
# TAB 5 - Strategy Comparison
# ══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.html("<div style='height: 18px;'></div>")
    st.markdown(
        """
        **This tab benchmarks the Black-Litterman portfolio against four alternative strategies 
        and the S&P 500 across three lenses: a historical wealth index, a 1-year Monte Carlo 
        return forecast, and (uniquely) a CPPI drawdown protection analysis.** The wealth index 
        and Monte Carlo answer *how does BL compare to simpler approaches?* The CPPI analysis asks 
        a different question: *what does it cost to protect capital, and how does the BL portfolio 
        behave once a drawdown floor is imposed?* This matters because BL, like all Sharpe-maximising 
        strategies, is built for wealth accumulation. However, most real investors have a loss tolerance, 
        and cannot afford to absorb drawdowns like the 34% COVID crash and simply wait for recovery. 
        CPPI wraps a dynamic floor around BL, creating a version of the portfolio that a 
        capital-constrained investor could hold. Comparing the two reveals the cost of that 
        protection in terms of forgone upside.

        The historical wealth index uses a rolling backtest with the estimation window set in the 
        sidebar. Equal-Weighted, Cap-Weighted, Global Minimum Variance, and Risk Parity are all 
        properly rolled, with weights re-estimated at each step using only data available at that 
        point, so there is no look-ahead bias. Black-Litterman is shown differently, as a static 
        allocation applying the current optimal weights to the full history. This is an intentional 
        design choice rather than an oversight. Because BL weights are derived from a forward-looking 
        view on price targets rather than purely from historical patterns, applying them statically 
        is the more natural representation of what the model is actually doing.

        **Model Caveats**: The historical comparison is not a like-for-like backtest. 
        The rolling strategies adapt to new data over time while BL holds fixed weights throughout. 
        This comparison is best read as an illustration of the strategies' structural differences 
        rather than a performance horse race. The 1-Year Monte Carlo Return Forecast in Section 2 
        provides a more comparable forward-looking view, though as noted in the Simulation & Stress 
        Tests tab, GBM paths assume Gaussian returns and should be treated as a baseline rather than 
        a tail-risk estimate.
        """
    )
    st.divider()

    # ── Session state defaults for CPPI parameters ────────────────────────────
    # Must be initialised before CPPI computation so values are available for
    # the wealth index in Section 1. Sliders rendered in Section 3 write back to these keys
    if "m_cppi" not in st.session_state:
        st.session_state["m_cppi"] = 3.0
    if "mdd_cppi_pct" not in st.session_state:
        st.session_state["mdd_cppi_pct"] = 15.0

    m_cppi   = st.session_state["m_cppi"]
    mdd_cppi = st.session_state["mdd_cppi_pct"] / 100

    # ── Pre-compute returns for wealth index ──────────────────────────────────
    ew_r, cw_r, gmv_r, erc_r = run_backtests(
        tick_rets, tick_capweights, estimation_window
    )

    #Align Start Date for Wealth Index
    valid_start_date = tick_rets.index[estimation_window]
    
    bl_static_w = bl_w_series.reindex(tick_rets.columns).fillna(0)
    bl_r        = (tick_rets * bl_static_w.values).sum(axis=1)

    # Run CPPI
    cppi_r, cppi_account, cppi_floor, cppi_alloc = run_cppi(
        risky_r=bl_r, rf=RF, multiplier=m_cppi, max_drawdown=mdd_cppi
    )

    btr = pd.DataFrame({
        "Equal-Weighted":          pd.Series(ew_r).squeeze(),
        "Cap-Weighted":            pd.Series(cw_r).squeeze(),
        "Global Minimum Variance": pd.Series(gmv_r).squeeze(),
        "Risk Parity":             pd.Series(erc_r).squeeze(),
        "BL (static)":             pd.Series(bl_r).squeeze(),
        "BL (CPPI)":               pd.Series(cppi_r).squeeze(),
        "S&P 500 (SPY)":           spx_rets.reindex(tick_rets.index),
    }).loc[valid_start_date:].dropna()

    # ── Section 1: Historical Wealth Index ────────────────────────────────────
    st.markdown("#### 1. Historical Wealth Index")

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
        f"Starts {estimation_window_yrs} year(s) after data begins - the minimum needed for the first rolling estimate."
    )

    wealth = (1 + btr).cumprod() * 10_000

    palette = {
        "Equal-Weighted":          ("#fecc5c",  1.4, "solid"),
        "Cap-Weighted":            ("#a1dab4",  1.4, "solid"),
        "Global Minimum Variance": ("#41b6c4",  1.4, "solid"),
        "Risk Parity":             ("#2c7fb8",  1.4, "solid"),
        "BL (static)":             ("#253494",  2.5, "solid"),
        "BL (CPPI)":               ("#7B2D8B",  2.0, "solid"),
        "S&P 500 (SPY)":           ("#888888",  1.4, "solid"),
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

    #Summary Statistics for Wealth Index
    _ann_scale = np.sqrt(252)
    summary_rows = {}
    for col in btr.columns:
        r         = btr[col].dropna()
        ann_r     = erk.annualize_rets(r, periods_per_year=252)
        ann_v     = erk.annualize_vol(r, periods_per_year=252)
        skew      = erk.skewness(r)
        kurt      = erk.kurtosis(r)
        cf_var    = erk.var_gaussian(r, level=5, modified=True) * _ann_scale
        cvar_hist = erk.cvar_historic(r, level=5)               * _ann_scale
        sharpe    = erk.sharpe_ratio(r, riskfree_rate=RF, periods_per_year=252)
        cp        = (1 + r).cumprod()
        mdd       = (cp / cp.cummax() - 1).min()
        summary_rows[col] = {
            "Ann. Return":      ann_r,
            "Ann. Vol":         ann_v,
            "Sharpe Ratio":     sharpe,
            "Max Drawdown":     mdd,
            "Skewness":         skew,
            "Kurtosis":         kurt,
            "Ann. CF VaR (5%)": cf_var,
            "Ann. CVaR (5%)":   cvar_hist,
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
                    "large losses than large gains - bad for portfolios."
                ),
            ),
            "Kurtosis": st.column_config.Column(
                "Kurtosis",
                help=(
                    "Raw kurtosis of the return distribution. "
                    "3 = normal distribution. Values above 3 indicate fat tails - "
                    "extreme gains/losses occur more often than a normal model would predict."
                ),
            ),
            "Ann. CF VaR (5%)": st.column_config.Column(
                "Ann. CF VaR (5%)",
                help=(
                    "Cornish-Fisher Value at Risk at the 5% level, annualised (×√252). "
                    "Adjusts the standard Gaussian VaR for the observed skewness and kurtosis "
                    "of the return distribution. Represents the annualised threshold loss "
                    "that is exceeded only 5% of the time. Higher = worse tail risk."
                ),
            ),
            "Ann. CVaR (5%)": st.column_config.Column(
                "Ann. CVaR (5%)",
                help=(
                    "Historic Conditional VaR (Expected Shortfall) at the 5% level. "
                    "Answers: given that we are in the worst 5% of outcomes, "
                    "what is the average loss? CVaR is always expressed as a "
                    "percentage of portfolio value - a higher number means deeper "
                    "average losses in bad tail scenarios."
                ),
            ),
        },
    )

    st.caption(
        "**Note on returns series**: Wealth Index and Summary Stats assume all strategies "
        "have Direct Reinvestment Plan (DRIP) enabled."
    )
    st.caption(
        "⚙️ **Note on covariance estimation:** GMV and Risk Parity use the Elton-Gruber Constant "
        "Correlation shrinkage estimator (δ = 0.7), blending 70% weight on a structured "
        "prior - where all pairwise correlations are set to the cross-sectional average - "
        "with 30% on the sample covariance. A higher δ was chosen because with only 17 "
        "stocks the sample covariance matrix is prone to estimation noise and "
        "near-singularity, which causes unconstrained optimisers to produce extreme, "
        "unstable weights. Shrinking toward the structured prior regularises the matrix, "
        "reduces its condition number, and makes the optimisation numerically well-behaved "
        "without requiring a larger asset universe to stabilise the estimate."
    )

    st.divider()

    # ── Section 2: 1-Year Monte Carlo Return Forecast ─────────────────────────
    st.markdown("#### 2. 1-Year Monte Carlo Return Forecast")
    st.caption(
        "Uses the same correlated GBM paths from the Simulation & Stress Tests tab but applied "
        "to each strategy's weights. Lets you see whether BL adds value over simpler alternatives."
    )

    strategy_weights = {
        "Equal-Weighted":          pd.Series(ew_w, index=tick_rets.columns),
        "Cap-Weighted":            cw_w_s,
        "Global Minimum Variance": pd.Series(gmv_w, index=tick_rets.columns),
        "Risk Parity":             pd.Series(erc_w, index=tick_rets.columns),
        "Black-Litterman":         bl_w_series,
    }

    comparison = {}
    for strat_name, w_s in strategy_weights.items():
        port_p = (all_paths * w_s.reindex(tick_rets.columns).values).sum(axis=2)
        fv     = port_p[-1]
        comparison[strat_name] = {
            "Expected Return": np.mean(fv) - 1,
            "5th pct":         np.percentile(fv, 5) - 1,
            "95th pct":        np.percentile(fv, 95) - 1,
            "Spread (95--5)":  np.percentile(fv, 95) - np.percentile(fv, 5),
        }

    comp_df = pd.DataFrame(comparison).T.sort_values("Expected Return", ascending=False)
    st.dataframe(
        comp_df.style
            .format("{:.2%}", subset=["Expected Return", "5th pct", "95th pct"])
            .format("{:.3f}", subset=["Spread (95--5)"])
            .background_gradient(subset=["Expected Return"], cmap="YlGnBu"),
        use_container_width=True,
    )

    _dotplot_colours = {
        "Equal-Weighted":          "#fecc5c",
        "Cap-Weighted":            "#a1dab4",
        "Global Minimum Variance": "#41b6c4",
        "Risk Parity":             "#2c7fb8",
        "Black-Litterman":         "#253494",
    }

    fig_comp = go.Figure()
    for strat_name, row in comparison.items():
        _c = _dotplot_colours.get(strat_name, "#888888")
        fig_comp.add_trace(go.Scatter(
            x=[strat_name],
            y=[row["Expected Return"]],
            mode="markers",
            marker=dict(size=14, symbol="circle", color=_c),
            error_y=dict(
                type="data", symmetric=False,
                array     =[row["95th pct"] - row["Expected Return"]],
                arrayminus=[row["Expected Return"] - row["5th pct"]],
                color=_c,
            ),
            name=strat_name,
        ))

    fig_comp.update_layout(
        title="Expected 1Y Return with 95% Confidence Interval",
        yaxis_tickformat=".1%",
        yaxis_title="1-Year Return",
        height=420,
        showlegend=False,
    )
    st.plotly_chart(fig_comp, use_container_width=True)

    st.divider()

    # ── Section 3: CPPI Drawdown Protection Analysis ──────────────────────────
    st.markdown("#### 3. CPPI Drawdown Protection Analysis")

    with st.expander("⚙️ CPPI Parameters", expanded=True):
        st.caption(
            "At each step, CPPI allocates equity exposure equal to the multiplier times the cushion "
            "(portfolio value minus floor), with the remainder in a safe asset. "
            "The two parameters below set how aggressively equity is sized and where the floor sits."
        )
        _col_m, _col_mdd = st.columns(2)
        _col_m.slider(
            "Multiplier (m)",
            min_value=1.0, max_value=5.0, step=0.5,
            key="m_cppi",
            help=(
                "Scales equity exposure relative to the available cushion. "
                "Higher m = more upside participation but more gap risk. "
                "At m=3, the risky sleeve can drawdown at most 33.33% (1/m) between rebalances "
                "before the cushion is exhausted and the HWM floor is breached. The multiplier "
                "implicitly assumes no single-period drop exceeds this threshold."
            ),
        )
        _col_mdd.slider(
            "Max Drawdown Floor (%)",
            min_value=5.0, max_value=40.0, step=1.0,
            key="mdd_cppi_pct",
            help=(
                "Maximum tolerated drawdown from the portfolio's all-time high. "
                "Floor = (1 − this value) × high-water mark. "
                "A tighter floor provides more protection but triggers cash lock-in more frequently."
            ),
        )

    # ── CPPI vs BL wealth index with floor overlay ────────────────────────────
    _init      = 10_000
    _cppi_w    = cppi_account * _init
    _cppi_fl_w = cppi_floor   * _init
    _bl_w      = (1 + btr["BL (static)"]).cumprod() * _init

    fig_cppi = go.Figure()
    fig_cppi.add_trace(go.Scatter(
        x=_bl_w.index, y=_bl_w.values,
        mode="lines", name="BL (static)",
        line=dict(color="#253494", width=1.5),
        opacity=0.55,
    ))
    fig_cppi.add_trace(go.Scatter(
        x=_cppi_fl_w.index, y=_cppi_fl_w.values,
        mode="lines", name="HWM Floor",
        line=dict(color="#C44E52", width=1.2, dash="dash"),
        fill="tozeroy",
        fillcolor="rgba(196, 78, 82, 0.04)",
    ))
    fig_cppi.add_trace(go.Scatter(
        x=_cppi_w.index, y=_cppi_w.values,
        mode="lines", name="BL (CPPI)",
        line=dict(color="#7B2D8B", width=2.2),
    ))
    fig_cppi.update_layout(
        title=f"BL (CPPI) vs BL (Unprotected) - $10,000 invested (m = {m_cppi:.1f}, floor = {mdd_cppi:.0%} MDD)",
        xaxis_title="Date",
        yaxis_title="Portfolio Value ($)",
        height=420,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig_cppi, use_container_width=True)

    # ── Equity allocation over time ───────────────────────────────────────────
    _eq_pct = (cppi_alloc * 100).clip(0, 100)
    fig_alloc = go.Figure()
    fig_alloc.add_trace(go.Scatter(
        x=_eq_pct.index, y=_eq_pct.values,
        mode="lines", name="Equity %",
        line=dict(color="#7B2D8B", width=1.5),
        fill="tozeroy",
        fillcolor="rgba(123, 45, 139, 0.12)",
    ))
    fig_alloc.add_hline(
        y=100, line_dash="dot",
        line_color="rgba(130,130,130,0.4)",
        annotation_text="100% equity",
        annotation_position="bottom right",
    )
    fig_alloc.update_layout(
        title="Equity Allocation Over Time (% of portfolio)",
        xaxis_title="Date",
        yaxis_title="Equity Allocation (%)",
        yaxis_range=[0, 112],
        height=280,
        showlegend=False,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig_alloc, use_container_width=True)

    st.divider()

    # ── CPPI Performance During Stress Periods ────────────────────────────────
    st.markdown("##### CPPI Protection During Historical Stress Periods")
    st.caption(
        "The table below checks how the BL with CPPI sleeve would have performed "
        "relative to the unprotected BL allocation."
    )

    _stress_cppi_rows = {}
    for _name, (_start, _end) in STRESS_PERIODS.items():
        _period_rets = btr["BL (static)"].loc[_start:_end].dropna()
        if len(_period_rets) < 5:
            continue

        _c_r, _c_acct, _c_floor, _c_alloc = run_cppi(
            risky_r=_period_rets, rf=RF,
            multiplier=m_cppi, max_drawdown=mdd_cppi,
        )

        _bl_cp    = (1 + _period_rets).cumprod()
        _cppi_cp  = (1 + _c_r).cumprod()

        _bl_total   = float(_bl_cp.iloc[-1] - 1)
        _cppi_total = float(_cppi_cp.iloc[-1] - 1)
        _bl_mdd     = float((_bl_cp   / _bl_cp.cummax()   - 1).min())
        _cppi_mdd   = float((_cppi_cp / _cppi_cp.cummax() - 1).min())

        _floor_breached = bool((_c_acct < _c_floor * 0.999).any())
        _cash_locked    = float(_c_alloc.min()) < 0.001

        _stress_cppi_rows[_name] = {
            "BL Return":         _bl_total,
            "CPPI Return":       _cppi_total,
            "Return Difference": _cppi_total - _bl_total,
            "BL Max Drawdown":   _bl_mdd,
            "CPPI Max Drawdown": _cppi_mdd,
            "DD Reduction":      _cppi_mdd - _bl_mdd,
            "Floor Breached":    "⚠️ Yes" if _floor_breached else "✅ No",
            "Cash Lock-in":      "🔒 Yes" if _cash_locked    else "No",
        }

    _stress_cppi_df = pd.DataFrame(_stress_cppi_rows).T
    st.dataframe(
        _stress_cppi_df.style
            .format("{:.2%}", subset=["BL Return", "CPPI Return", "Return Difference",
                                      "BL Max Drawdown", "CPPI Max Drawdown", "DD Reduction"])
            .background_gradient(subset=["CPPI Max Drawdown"], cmap="Reds_r",  vmax=0.0)
            .background_gradient(subset=["BL Max Drawdown"],   cmap="Reds_r",  vmax=0.0)
            .background_gradient(subset=["DD Reduction"],      cmap="RdYlGn", vmin=-0.1, vmax=0.0),
        use_container_width=True,
        column_config={
            "DD Reduction": st.column_config.Column(
                "DD Reduction",
                help="CPPI Max Drawdown minus BL Max Drawdown. Negative = CPPI suffered less. "
                     "In a genuine gap-down event the floor may still breach if the single-day "
                     "move exceeds the cushion - this is the residual gap risk CPPI cannot eliminate.",
            ),
            "Floor Breached": st.column_config.Column(
                "Floor Breached",
                help=f"Whether the CPPI portfolio fell below its HWM floor ((1 − {mdd_cppi:.0%}) × peak) "
                     "at any point during the stress window.",
            ),
            "Cash Lock-in": st.column_config.Column(
                "Cash Lock-in",
                help="Whether equity exposure reached zero during the period. Once locked into cash, "
                     "the CPPI strategy stays there for the remainder of that window.",
            ),
        },
    )

    st.caption(
        f"**Note on risk-free rate:** the safe sleeve earns a fixed {RF:.2%} (current 1Y T-bill) across "
        "all historical windows. In practice this varied significantly: near zero during the GFC Echo (2015) "
        "and COVID Crash (2020), rising to ~4.5% during the 2022 rate hike cycle. A time-varying rate would "
        "improve return attribution precision, but the primary outputs here (drawdown reduction, floor breach, "
        "and equity allocation dynamics) are determined entirely by the equity sleeve and are unaffected by "
        "this assumption."
    )
# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "⚠️ **Disclaimer**: This app is for educational and personal research purposes only. "
    "Nothing here constitutes financial advice. All data is sourced from Yahoo Finance and FRED. "
    "Price targets are personal estimates -- not investment recommendations."
)
