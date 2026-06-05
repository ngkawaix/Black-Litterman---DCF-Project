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
    "MSFT": 562.00, "NFLX": 115.00, "NVDA": 242.47,
    "TSM":  463.00, "V":    399.00,
}

BASE_CONFIDENCE = {
    "AAPL": 0.20, "ADBE": 0.15, "AMAT": 0.15,
    "AMZN": 0.25, "ASML": 0.10, "CPRT": 0.20,
    "FICO": 0.25, "GOOGL":0.15, "LRCX": 0.10,
    "MA":   0.25, "META": 0.25, "MSCI": 0.25,
    "MSFT": 0.60, "NFLX": 0.25, "NVDA": 0.45,
    "TSM":  0.25, "V":    0.25,
}

# ── DCF Override Dict ─────────────────────────────────────────────────────────
# Tickers with a completed DCF model. Displays ⚡ DCF Model badge in the
# Confidence tab, distinguishing model-derived confidence from analyst estimates.
DCF_OVERRIDES: dict[str, float] = {
    "NVDA": 0.45,
}

# ── Research Tiers ────────────────────────────────────────────────────────────
# Confidence levels are grouped into three tiers based on depth of research.
# This avoids fabricating strong views where none exist and keeps rationales
# genuinely defensible.
#
#   FULL_DCF   – Completed bottom-up DCF model. Confidence is model-derived.
#   TACTICAL   – Personal holding or high-level thesis. Single-point rationale.
#   SYSTEMATIC – No active thesis. Confidence defers to market-prior signal.
#
RESEARCH_TIERS: dict = {

    # ── Tier 1: Research-Led (DCF-backed) ─────────────────────────────────────
    "FULL_DCF": {
        "NVDA": {
            "rationale": (
                "Completed DCF (Q1 FY2027, May 2026) yields a merged fair value of $242.47, the simple "
                "average of the perpetuity approach ($190.15) and the exit-EBITDA approach "
                "($294.80). Base-case WACC of 11.52% (Rf=4.50%, \u03b2=1.634 industry-adjusted, "
                "Damodaran ERP=4.24% May 2026 update). "
                "Confidence of 0.45 reflects an acknowledgment of uncertainty based on the wide spread of the two valuation methods"
            ),
            "date": "May 2026",
        },
    },

    # ── Tier 2: Tactical / Thematic (personal thesis, no full model) ──────────
    "TACTICAL": {
        "MSFT": (
            "Strongest enterprise moat in the universe with deep AI integration across cloud and productivity. "
            "Confidence at 0.60 reflects strong conviction in the long-run compounding thesis."
        ),
        "AMZN": (
            "AWS re-accelerating to +28% YoY ($37.6B) and record 13.1% "
            "operating margin represent an inflection from investment phase to "
            "profitable scale. Bullish thesis, but confidence is kept at 0.25 because the "
            "target is near market-implied."
        ),
        "GOOGL": (
            "Search +19% YoY and Cloud +63% YoY demonstrate durable "
            "compounding across both core and AI-driven revenue as per Q1 FY2026 earnings announced on 29 April 2026. "
            "Confidence at 0.15 as the target sits close to market-implied, "
            "reflecting broad agreement with the market rather than a strong override."
        ),
        "TSM": (
            "Dominant advanced-node manufacturer underpinning the global "
            "AI infrastructure stack. Confidence at 0.25 preserves a discount for Taiwan's country risk "
            "without making TSM a dominant underweight."
        ),
        "ASML": (
            "ASML is the sole global EUV supplier with no credible competitive threat  "
            "and demand visibility into 2027. Confidence at 0.10 reflects caution with valuation "
            "with the cost run up eroding the Margin of Safety (MoS). Premium is priced in at current valuations. "
        ),
    },
}


def render_rationale(ticker: str) -> str:
    """
    Returns a tier-labelled rationale string for a given ticker.

    Tier 1 (Full DCF)  – model-calibrated, with last-calibrated date.
    Tier 2 (Tactical)  – personal thesis, single bullet.
    Tier 3 (Systematic)– no active thesis; exposure driven by market prior.
    """
    if ticker in RESEARCH_TIERS["FULL_DCF"]:
        data = RESEARCH_TIERS["FULL_DCF"][ticker]
        return (
            f"[Full DCF] {data['rationale']}  "
            f"(Last calibrated: {data['date']})"
        )
    if ticker in RESEARCH_TIERS["TACTICAL"]:
        return f"[Tactical] {RESEARCH_TIERS['TACTICAL'][ticker]}"
    return (
        "[Systematic] No active thesis; "
        "confidence calibrated to defer to the universe-implied equilibrium return  (π) ."
    )

STRESS_PERIODS = {
    "COVID Crash (Feb--Mar 2020)":        ("2020-02-19", "2020-03-23"),
    "Post-COVID Rate Hikes (2022)":      ("2022-01-01", "2022-12-31"),
    "Tech Selloff (Nov 2021--May 2022)":  ("2021-11-19", "2022-05-20"),
    "GFC Echo (Aug--Oct 2015)":           ("2015-08-18", "2015-10-01"),
    "Trump Tariffs (Full Year 2025)": ("2025-02-01", "2025-12-31"),
}

# ── High-Conviction DCF Universe ─────────────────────────────────────────────
# The five companies with completed or in-progress DCF models.
# Only tickers present in both INVESTMENT_THESES and DCF_OUTPUTS are rendered
# with full content; all others show a "coming soon" placeholder.
HIGH_CONVICTION = ["NVDA", "META"]

INVESTMENT_THESES = {
    "NVDA": {
        "latest_earnings": [
            "NVDA reported GAAP revenue of $81.6B (+85% YoY, +20% QoQ) and gross margin expansion from 71.1% (FY2026) "
            "to 74.9% (Q1 FY2027), signalling a re-acceleration of growth for the third consecutive quarter and "
            "indicating strong adoption of NVIDIA's Grace-Blackwell (GB) platforms.",
        ],
        "theses": [
            "NVDA adopts a full stack approach and remains at the forefront of AI hardware and software innovation. The release of its Vera-Rubin "
            "architecture at GTC in March 2026 further extends its lead ahead of AMD, despite growing threats from "
            "custom ASICs by hyperscalers (e.g. Google's Ironwood TPU, Amazon's AWS Trainium-3).",
            "CUDA remains a durable software moat, making switching costs structurally prohibitive for the roughly "
            "4M developers trained on the ecosystem.",
            "NVDA's net cash position of 53.88B and strong FCF generation provide significant reserves to innovate "
            "and defend its moat.",
        ],
        "guidance": [
            "NVIDIA guided Q2 FY2027 revenue of 91.0B, more than $4B (+2%) above consensus. While NVIDIA does not "
            "provide full-year revenue guidance, this implies a full-year trajectory of roughly 390B, equivalent to "
            "approximately 81.3% YoY revenue growth, up from 65.5% for full FY2026.",
            "NVIDIA guided a full-year effective tax rate of 16% to 18%.",
            "NVDA announced an 80B buyback authorisation and a 25x dividend increase, with payout on 26 June 2026 "
            "(ex-dividend: 4 June 2026), signalling management's confidence in sustained near-term cash flow generation.",
        ],
        "growth_drivers": [
        (       "Networking",
                "Data Centre networking revenue hit 14.8B in Q1 FY2027, up 199% YoY and 35% QoQ, "
                "outpacing compute growth (+77% YoY) for the third consecutive quarter. InfiniBand, "
                "Spectrum-X Ethernet, and NVLink are now mandatory infrastructure for GB200 NVL72 rack "
                "deployments."
            ),
            (
                "Inference Inflection via Agentic AI",
                "Inference accounts for roughly two-thirds of all AI compute in 2026, up from one-third in 2023 "
                "(Gartner), with McKinsey projecting 35% CAGR through 2030 vs. 22% for training. Agentic "
                "AI is expected to compound demand with Huang noting that agentic systems will require more compute "
                "steps per query compared to traditional chatbots."
            ),
            (
                "Vera-Rubin Shipments",
                "Vera-Rubin begins initial shipments in Q3 FY2027 with volume ramp targeted for Q4, "
                "overlapping with GB300 demand. The Vera CPU is the higher-conviction near-term "
                "catalyst with CFO Kress guiding for ~20B visibility for FY2027 thus far, entering an additional 200B TAM. "
            ),
        ],
    },
}

DCF_OUTPUTS = {
    "NVDA": {
        "current_price": 215.33,         # Latest closing price, 22 May 2026
        "DCF Fair Value": 242.47,         # Merged perpetuity / EBITDA average
        "wacc":                    0.1152,
        "terminal_growth":         0.03,
        "terminal_ebitda_multiple": 19,
        "model_link": "https://docs.google.com/spreadsheets/d/1f6yXc8R1_Faq8n0NhSWwO7gs0zVDx9zuOzJmo4LYVJc/edit?usp=sharing",

        "football_field": [
            {"label": "DCF: Perpetuity (2.0–4.0% g, 11.52% WACC)",        "low": 179.11, "high": 204.09, "base": 190.15},
            {"label": "DCF: Exit EBITDA Multiple (17–21x at 11.52% WACC)",  "low": 274.00, "high": 316.00, "base": 294.80},
            {"label": "Analyst Consensus",                                  "low": 180.00, "high": 500.00, "base": 294.00},
            {"label": "52-Week Range",                                      "low": 132.92, "high": 236.54, "base": None},
        ],

        "wacc_assumptions": [
            {"Cost of Debt": "3.20%",
             "Tax Rate": "17.00%",
             "Cost of Debt (After Tax)": "2.66%",
             "Risk Free Rate": "4.50%",
             "Observed Beta (Bloomberg)": "2.240",
             "Industry Beta (Adjusted)": "1.634",
             "Market Risk Premium": "4.24%",
             "Cost of Equity": "11.43%",
             "WACC": "11.52%"
            },
        ],

        "is_assumptions": [
            {
                "name":   "Revenue Growth – Stage 1 (FY2027)",
                "actual": "+65.5%",
                "bear":   "+71.1%",
                "base":   "+81.3%",
                "bull":   "+81.3%",
                "note":   "+81.3% in line with management's Q1 FY2027 revenue trajectory. "
                          "Driven by continued Blackwell architecture ramp and accelerating hyperscaler AI capex commitments.",
            },
            {
                "name":   "Revenue Growth – Stage 1 (FY2028–31)",
                "actual": "N/A",
                "bear":   "~14%",
                "base":   "~20%",
                "bull":   "~26%",
                "note":   "Decelerating from the FY2027 peak as hyperscaler capex comps tighten. "
                          "Sustained by inference infrastructure build-out and sovereign AI demand from 35+ national programmes.",
            },
            {
                "name":   "Revenue Growth – Stage 2 (FY2032–36)",
                "actual": "N/A",
                "bear":   "~5%",
                "base":   "~9%",
                "bull":   "~11%",
                "note":   "Gradual convergence toward long-run growth. Assumes NVIDIA retains platform leadership "
                          "but faces rising custom silicon competition (Google TPUs, AMD MI-series) on inference costs.",
            },
            {
                "name":   "Gross Margin",
                "actual": "71.1%",
                "bear":   "73.5%",
                "base":   "74.5%",
                "bull":   "75–77%",
                "note":   "74.5% is the midpoint of management's Q2 FY2027 gross margin guidance. "
                          "Stage 2 modest compression assumes inference pricing pressure as hyperscalers compete with NVDA on pricing.",
            },
            {
                "name":   "R&D % of Revenue",
                "actual": "8.6%",
                "bear":   "12.0%",
                "base":   "8.6%",
                "bull":   "7.0%",
                "note":   "Straight-lined at 8.6% through Stage 1, consistent with recent actuals. "
                          "Stage 2 stepped up modestly as NVIDIA matures.",
            },
            {
                "name":   "SG&A % of Revenue",
                "actual": "2.1%",
                "bear":   "2.2%",
                "base":   "1.9%",
                "bull":   "1.7%",
                "note":   "Structurally low due to NVIDIA's direct hyperscaler sales model. "
                          "Stage 2 gradually increases to 3.0% as NVIDIA matures.",
            },
            {
                "name":   "Tax Rate",
                "actual": "15.1%",
                "bear":   "17.0%",
                "base":   "17.0%",
                "bull":   "17.0%",
                "note":   "Straight-lined at 17% midpoint of Q1 full-year FY2027 guidance. "
            },
        ],
        # DCF and terminal value assumptions: no FY2026 Actual column for these rows.
        "dcf_assumptions": [
            {
                "name": "WACC",
                "bear": "11.52%",
                "base": "11.52%",
                "bull": "11.52%",
                "note": "11.52% via CAPM: Rf = 4.50% (10Y UST as of 23 May 2026, elevated by Moody's US downgrade and fiscal deficit concerns), "
                        "industry-adjusted β = 1.634 (de-levered peer group: AMD, AVGO, TSM, QCOM, ASML; re-levered at NVDA's capital structure), "
                        "Damodaran ERP = 4.24% (May 2026 update, down from 4.66% in January). "
                        "NVIDIA's large net cash position (net debt = -$54.1B) pushes equity weight above 1; WACC sits marginally above cost of equity of 11.43%. "
            },
            {
                "name": "Terminal Growth Rate (g)",
                "bear": "2.5%",
                "base": "3.0%",
                "bull": "4.0%",
                "note": "Base 3.0% set marginally above long-run nominal GDP growth (~2.5%). "
                        "Reflects NVIDIA's CUDA ecosystem durability. "
                        "Conservative relative to consensus-implied growth of 7.3% in the exit-multiple model. "
            },
            {
                "name": "Terminal EBITDA Multiple",
                "bear": "16x",
                "base": "19x",
                "bull": "22x",
                "note": "19x represents meaningful mean reversion from the current NTM EV/EBITDA of ~27x, "
                        "as top-line growth decelerates from ~80% toward single digits by FY2036. "
                        "Peer benchmarks: AVGO 20x, ASML 22.5x, AMD 25x, QCOM 22.5x (sector median ~21.5x). "
                        "A 12.5% discount applied to the peer median reflects anticipated growth deceleration "
                        "and rising custom silicon competition by the terminal year.",
            },
            {
                "name": "Implied Price: Perpetuity",
                "bear": "~$179",
                "base": "$190.15",
                "bull": "~$204",
                "note": "The perpetuity approach implies a modest downside to the current price ($215.33), driven by "
                        "a conservative 3% terminal growth rate at a WACC of 11.52%. "
            },
            {
                "name": "Implied Price: Exit Multiple",
                "bear": "~$274",
                "base": "$294.80",
                "bull": "~$316",
                "note": "Exit multiple approach implies an upside to the current price driven by a 19x exit EV/EBITDA "
                        "multiple implying a 7.3% terminal growth rate. The merged fair value of $242.47 is the simple "
                        "average of both the exit multiple and perpetuity approaches."
            },
        ],
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING  (cached so Streamlit doesn't re-download on every interaction)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Fetching market data from Yahoo Finance…")
def load_market_data(tickers, start="2012-01-01"):
    import time

    data = yf.download(tickers, start=start, interval="1d", auto_adjust=True, progress=False)
    price_data = data["Close"]
    price_data.index = price_data.index.tz_localize(None)
    price_data = price_data.loc[~price_data.index.duplicated(keep="first")]

    # Identify tickers that returned all-NaN from the batch download.
    # Yahoo Finance occasionally silently drops one ticker from a batch response
    # without raising an error. Retry each failed ticker individually before giving up.
    failed_initial = price_data.columns[price_data.isna().all()].tolist()
    still_failed = []

    for ticker in failed_initial:
        recovered = False
        for attempt in range(3):
            time.sleep(1.5 + attempt)   # back off: 1.5s → 2.5s → 3.5s
            try:
                retry = yf.download(
                    ticker, start=start, interval="1d",
                    auto_adjust=True, progress=False,
                )
                if retry.empty:
                    continue
                retry_close = retry["Close"].squeeze()
                retry_close.index = retry_close.index.tz_localize(None)
                if not retry_close.isna().all():
                    price_data[ticker] = retry_close.reindex(price_data.index)
                    recovered = True
                    break
            except Exception:
                pass
        if not recovered:
            still_failed.append(ticker)

    if still_failed:
        st.warning(
            f"⚠️ Could not download price data for: **{', '.join(still_failed)}**. "
            "These tickers have been excluded from this run. "
            "Yahoo Finance likely rate-limited the pull — refresh in a minute to retry.",
        )
        price_data = price_data.drop(columns=still_failed)

    # Forward-fill intra-series gaps, then use dropna(how="all") so a single
    # missing ticker on one day does not wipe out the entire returns history.
    price_data = price_data.ffill()
    tick_rets  = price_data.pct_change().dropna(how="all")
    return price_data, tick_rets
    

# Retrieval of Market Cap, Consensus Estimates and Earnings Data
@st.cache_data(show_spinner="Fetching ticker metadata (market cap, price data, consensus, earnings)…", ttl=86400)
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
        mean_t = None
        n_analysts = None

        for _attempt in range(3):
            try:
                info = t.info
                n_analysts = (
                    info.get("numberOfAnalystOpinions")
                    or info.get("numAnalystOpinions")
                )
                break
            except Exception:
                if _attempt < 2:
                    time.sleep(2.0 + _attempt)

        for _attempt in range(3):
            try:
                apt = t.analyst_price_targets
                mean_t = apt.get("mean", None) if isinstance(apt, dict) else None
                if mean_t is not None:
                    break
            except Exception:
                if _attempt < 2:
                    time.sleep(2.0 + _attempt)

        if mean_t is None:
            mean_t = info.get("targetMeanPrice", None)

        if mean_t is not None and n_analysts is None:
            try:
                time.sleep(1.5)
                _info_retry = t.info
                n_analysts = (
                    _info_retry.get("numberOfAnalystOpinions")
                    or _info_retry.get("numAnalystOpinions")
                )
            except Exception:
                pass

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

        time.sleep(0.35)   

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
            "Debt/Equity": info.get("debtToEquity",                 None),
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

# ─────────────────────────────────────────────────────────────────────────────
# SIMULATION HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def run_correlated_gbm(tick_rets, mu_bl, bl_w_series, n_scenarios=500, n_years=1, steps=252):
    """
    Runs a  Monte Carlo using BL posterior returns as drift.
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

# Even better than 
def run_copula_simulation(tick_rets, bl_w_series, theta=2.0, n_scenarios=3_000, n_steps=252, seed=42):
    """
    Clayton Copula + Empirical Inverse Transform Monte Carlo.

    Replaces GBM's two core assumptions:
      - Gaussian margins  →  empirical historical return distribution per asset
      - Symmetric Gaussian correlation  →  Clayton copula with lower-tail dependence

    Each of the n_steps daily draws is i.i.d. (no GARCH / volatility clustering).
    theta > 0 controls crash co-movement: higher = assets lock together more in the tails.

    Returns port_paths (n_steps+1, n_scenarios), same shape as run_correlated_gbm.
    """
    rng      = np.random.default_rng(seed)
    n_assets = len(tick_rets.columns)
    w        = bl_w_series.reindex(tick_rets.columns).fillna(0).values

    # Drop any rows where ANY ticker has a NaN, custom tickers with shorter
    # listing history cause per-column lengths to differ, which breaks np.stack.
    tick_rets_aligned = tick_rets.dropna()

    # Pre-sort each asset's historical daily returns once (non-parametric CDF lookup table)
    sorted_hists = np.stack(
        [np.sort(tick_rets_aligned[col].values) for col in tick_rets_aligned.columns],
        axis=1,
    )  # shape: (n_hist, n_assets)
    n_hist = sorted_hists.shape[0]

    port_paths = np.ones((n_steps + 1, n_scenarios))

    for t in range(1, n_steps + 1):
        # ── Step 1: Clayton copula via Frailty (Marshall-Olkin) method ──────────
        # Draw a shared Gamma random variable W that introduces the tail dependence.
        # When W is small (tail event), all assets simultaneously draw extreme quantiles.
        # Frailty parameter: W ~ Gamma(1/theta, 1)
        W       = rng.gamma(1.0 / theta, 1.0, size=n_scenarios)                       # (n_scenarios,)
        indep_u = rng.uniform(0.0, 1.0, size=(n_scenarios, n_assets))                 # (n_scenarios, n_assets)

        # Transform independent uniforms into Clayton-correlated uniforms U
        # U_i = (1 - log(V_i) / W)^(-1/theta)  where V_i are independent Uniform(0,1)
        U = (1.0 - np.log(indep_u) / W[:, None]) ** (-1.0 / theta)
        U = np.clip(U, 1e-6, 1.0 - 1e-6)                                              # numerical safety

        # ── Step 2: Empirical inverse transform ──────────────────────────────────
        # Map each uniform U[s, a] to the empirical quantile of asset a's return distribution.
        # This replaces the Gaussian draw in GBM with the actual historical return shape.
        idx        = np.clip((U * n_hist).astype(int), 0, n_hist - 1)                 # (n_scenarios, n_assets)
        daily_rets = sorted_hists[idx, np.arange(n_assets)]                           # (n_scenarios, n_assets)

        # ── Step 3: Compound portfolio value ─────────────────────────────────────
        port_paths[t] = port_paths[t - 1] * (1.0 + daily_rets @ w)

    return port_paths

@st.cache_data(show_spinner="Fitting GARCH(1,1) to each asset - first run only, cached thereafter…")
def fit_garch_params(_tick_rets, tickers_key: tuple = ()):
    """
    Fits GARCH(1,1) to each asset's daily return series.

    The underscore prefix on _tick_rets tells Streamlit to skip hashing the
    DataFrame (using object identity instead), which avoids serialisation issues
    on large DataFrames while still correctly invalidating the cache when the
    underlying data object changes.

    Returns
    -------
    garch_params : dict  : {ticker: {omega, alpha, beta, mu}}
    std_resids   : dict  : {ticker: array of standardised residuals z_t = ε_t / σ_t}
    last_state   : dict  : {ticker: {sigma2, epsilon}} in scaled units (returns × 100)
    cond_vol_df  : DataFrame : annualised conditional volatility per asset over time
    """
    try:
        from arch import arch_model
    except ImportError:
        return None, None, None, None, None

    import warnings

    garch_params = {}
    std_resids   = {}
    last_state   = {}
    cond_vol     = {}

    #GARCH fitting for each individual stock''s return series
    for col in _tick_rets.columns:
        # Scale returns to percentage points (e.g. 0.02 → 2.0) for numerical stability
        # GARCH optimisers work better when the series has variance ~1–5 rather than ~0.0001
        r = _tick_rets[col].dropna() * 100

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = arch_model(
                r, vol='Garch', p=1, q=1, dist='normal', mean='Constant'
            ).fit(disp='off', show_warning=False)

        # The constant mean parameter name differs across arch versions:
        # older versions use 'Const', newer use 'mu'. Try both, fall back to 0.
        _mu_val = 0.0
        for _mu_key in ('mu', 'Const', 'constant'):
            if _mu_key in res.params.index:
                _mu_val = float(res.params[_mu_key])
                break

        garch_params[col] = {
            'omega': float(res.params['omega']),      # ω: long-run baseline variance
            'alpha': float(res.params['alpha[1]']),   # α: sensitivity to last shock
            'beta':  float(res.params['beta[1]']),    # β: persistence of last variance
            'mu':    _mu_val,                          # μ: mean return (scaled)
        }

        # Standardised residuals: z_t = ε_t / σ_t  ≈ i.i.d. once vol is stripped out
        std_resids[col] = (res.resid / res.conditional_volatility).values

        # Last observed state, seeds the forward GARCH recursion
        last_state[col] = {
            'sigma2':  float(res.conditional_volatility.iloc[-1] ** 2),  # σ²_T  (scaled²)
            'epsilon': float(res.resid.iloc[-1]),                          # ε_T   (scaled)
        }

        # Annualised conditional vol for the visualisation chart
        # conditional_volatility is in scaled units → divide by 100 to get decimal,
        # then multiply by √252 to annualise
        cond_vol[col] = (res.conditional_volatility / 100 * np.sqrt(252)).values

    min_len     = min(len(v) for v in cond_vol.values())
    cond_vol_df = pd.DataFrame(
        {col: cond_vol[col][-min_len:] for col in _tick_rets.columns},
        index=_tick_rets.index[-min_len:],
    )

    # ── Fit Clayton θ via MLE on GARCH standardised residuals ────────────────
    # Fitting to standardised residuals (not raw returns) is the correct approach:
    # the GARCH layer has already stripped out the time-varying volatility envelope,
    # so the copula captures only the pure cross-asset dependency structure of shocks.
    #
    # Steps:
    #   1. Stack residuals into (n_obs, n_assets)
    #   2. Convert each column to a uniform margin via empirical rank
    #      (Hazen plotting position: rank / (n+1) keeps values strictly in (0,1))
    #   3. Maximise the d-dimensional Clayton log-likelihood over θ
    #      Optimising over log(θ) keeps the search unconstrained (θ must be > 0)
    from scipy.stats    import rankdata
    from scipy.optimize import minimize_scalar

    _resid_arr = np.stack(
        [std_resids[col][-min_len:] for col in _tick_rets.columns], axis=1
    )  # (n_obs, n_assets)

    _n_obs = _resid_arr.shape[0]
    _u = np.stack(
        [rankdata(_resid_arr[:, i]) / (_n_obs + 1)
         for i in range(_resid_arr.shape[1])],
        axis=1,
    )  # (n_obs, n_assets), uniform margins strictly in (0, 1)

    def _clayton_neg_ll(log_theta):
        """
        Negative log-likelihood of the d-dimensional Clayton copula.
        Parameterised as log(θ) so the optimiser searches over all reals
        while θ stays positive.

        Clayton log-density (d dimensions):
          log c = (d−1)·log(θ+1)
                  + (−θ−1) · Σ_i log(u_i)           ← margin contribution
                  + (−1/θ − d) · log(Σ_i u_i^(−θ) − d + 1)  ← generator term
        """
        theta     = np.exp(log_theta)
        d         = _u.shape[1]
        generator = (_u ** (-theta)).sum(axis=1) - d + 1  # (n_obs,)
        if np.any(generator <= 0):
            return 1e10  # outside the feasible region
        log_density = (
            np.log(theta + 1) * (d - 1)
            + (-theta - 1) * np.log(_u).sum(axis=1)
            + (-1.0 / theta - d) * np.log(generator)
        )
        return -log_density.sum()

    # bounds=(-3, 3) → θ ∈ (e^-3 ≈ 0.05,  e^3 ≈ 20)
    _opt         = minimize_scalar(_clayton_neg_ll, bounds=(-3.0, 3.0), method='bounded')
    fitted_theta = float(np.exp(_opt.x))

    # Align std_resids to min_len so run_garch_copula_simulation receives
    # equal-length arrays for every ticker, custom tickers with shorter history
    # produce shorter residual series, which breaks np.stack downstream.
    std_resids_aligned = {col: std_resids[col][-min_len:] for col in _tick_rets.columns}

    return garch_params, std_resids_aligned, last_state, cond_vol_df, fitted_theta


def run_garch_copula_simulation(
    tick_rets, bl_w_series, garch_params, std_resids, last_state,
    theta=2.0, n_scenarios=2_000, n_steps=252, seed=42,
):
    """
    GARCH(1,1) + Clayton Copula + Empirical Inverse Transform Monte Carlo.

    Three layers stacked:
      1. GARCH(1,1): per-asset time-varying volatility
             σ²_t = ω + α · ε²_{t-1} + β · σ²_{t-1}
         A large shock on day t raises σ² for day t+1, volatility clustering.

      2. Clayton Copula (Frailty method): correlated draws of standardised
         residuals z_t across all assets, with lower-tail dependence controlled by θ.

      3. Empirical margins: each z_t is mapped back through the empirical CDF of
         the fitted standardised residuals, preserving fat tails and skew.

    Reconstruction: r_t = (μ + σ_t × z_t) / 100  (unscale back to decimal returns)

    All sigma / epsilon tracking is kept in scaled units (returns × 100) to match
    the units the GARCH parameters were estimated in.
    """
    rng      = np.random.default_rng(seed)
    n_assets = len(tick_rets.columns)
    cols     = tick_rets.columns.tolist()
    w        = bl_w_series.reindex(tick_rets.columns).fillna(0).values

    # Pre-sort standardised residuals for empirical inverse transform
    sorted_std = np.stack(
        [np.sort(std_resids[col]) for col in cols], axis=1
    )  # shape: (n_hist, n_assets)
    n_hist = sorted_std.shape[0]

    # GARCH parameters vectorised across assets
    omegas = np.array([garch_params[col]['omega'] for col in cols])  # (n_assets,)
    alphas = np.array([garch_params[col]['alpha'] for col in cols])
    betas  = np.array([garch_params[col]['beta']  for col in cols])
    mus    = np.array([garch_params[col]['mu']     for col in cols])

    # Each scenario starts from the last observed σ² and ε, path-dependent from here
    # Broadcasting from (n_assets,) to (n_scenarios, n_assets)
    sigma2_s  = np.tile([last_state[col]['sigma2']  for col in cols], (n_scenarios, 1))
    epsilon_s = np.tile([last_state[col]['epsilon'] for col in cols], (n_scenarios, 1))

    port_paths = np.ones((n_steps + 1, n_scenarios))

    for t in range(1, n_steps + 1):

        # ── Step 1: GARCH variance update ────────────────────────────────────
        # σ²_t = ω + α · ε²_{t-1} + β · σ²_{t-1}
        # Each scenario has its own σ² path because last period's ε differs
        sigma2_s = omegas + alphas * epsilon_s ** 2 + betas * sigma2_s   # (n_scenarios, n_assets)
        sigma_s  = np.sqrt(np.maximum(sigma2_s, 1e-8))                    # σ_t (scaled units)

        # ── Step 2: Clayton copula draw ───────────────────────────────────────
        W       = rng.gamma(1.0 / theta, 1.0, size=n_scenarios)
        indep_u = rng.uniform(0.0, 1.0, size=(n_scenarios, n_assets))
        U       = (1.0 - np.log(indep_u) / W[:, None]) ** (-1.0 / theta)
        U       = np.clip(U, 1e-6, 1.0 - 1e-6)

        # ── Step 3: Empirical inverse transform on standardised residuals ─────
        idx  = np.clip((U * n_hist).astype(int), 0, n_hist - 1)
        z_t  = sorted_std[idx, np.arange(n_assets)]                      # (n_scenarios, n_assets)

        # ── Step 4: Reconstruct return and update GARCH state ─────────────────
        # r_t (scaled) = μ + σ_t × z_t
        r_scaled  = mus + sigma_s * z_t                                   # (n_scenarios, n_assets)
        r_actual  = r_scaled / 100.0                                      # decimal return

        # ε_t = r_scaled - μ  → feeds back into the GARCH recursion next step
        # This is what creates path-dependency: a bad draw here raises σ² tomorrow
        epsilon_s = r_scaled - mus

        # ── Step 5: Compound ──────────────────────────────────────────────────
        port_paths[t] = port_paths[t - 1] * (1.0 + r_actual @ w)

    return port_paths


@st.cache_data(show_spinner="Running GARCH-Copula for all strategies…")
def run_garch_copula_all_strategies(_tick_rets, _garch_params, _std_resids, _last_state, strategy_names, _strategy_weights_arr, theta=2.0, n_scenarios=2_000, n_steps=252, seed=42, tickers_key: tuple = ()):
    """
    Runs one set of GARCH-Copula asset paths and applies all strategy weight
    vectors in a single pass, the expensive GARCH + copula sampling happens
    once; weight application is a cheap matmul per step.

    strategy_names        : tuple[str]  : names, same order as weight rows
    _strategy_weights_arr : np.ndarray (n_strategies, n_assets)

    Returns dict {strategy_name: terminal_portfolio_values (n_scenarios,)}
    """
    rng      = np.random.default_rng(seed)
    n_assets = len(_tick_rets.columns)
    cols     = _tick_rets.columns.tolist()

    sorted_std = np.stack(
        [np.sort(_std_resids[col]) for col in cols], axis=1
    )
    n_hist = sorted_std.shape[0]

    omegas = np.array([_garch_params[col]['omega'] for col in cols])
    alphas = np.array([_garch_params[col]['alpha'] for col in cols])
    betas  = np.array([_garch_params[col]['beta']  for col in cols])
    mus    = np.array([_garch_params[col]['mu']     for col in cols])

    sigma2_s  = np.tile([_last_state[col]['sigma2']  for col in cols], (n_scenarios, 1))
    epsilon_s = np.tile([_last_state[col]['epsilon'] for col in cols], (n_scenarios, 1))

    port_vals = np.ones((len(strategy_names), n_scenarios))

    for t in range(1, n_steps + 1):
        sigma2_s = omegas + alphas * epsilon_s ** 2 + betas * sigma2_s
        sigma_s  = np.sqrt(np.maximum(sigma2_s, 1e-8))

        W        = rng.gamma(1.0 / theta, 1.0, size=n_scenarios)
        indep_u  = rng.uniform(0.0, 1.0, size=(n_scenarios, n_assets))
        U        = (1.0 - np.log(indep_u) / W[:, None]) ** (-1.0 / theta)
        U        = np.clip(U, 1e-6, 1.0 - 1e-6)

        idx  = np.clip((U * n_hist).astype(int), 0, n_hist - 1)
        z_t  = sorted_std[idx, np.arange(n_assets)]

        r_scaled  = mus + sigma_s * z_t
        r_actual  = r_scaled / 100.0
        epsilon_s = r_scaled - mus

        # Apply all strategies simultaneously:
        # (n_strategies, n_assets) @ (n_assets, n_scenarios) → (n_strategies, n_scenarios)
        port_vals *= (1.0 + _strategy_weights_arr @ r_actual.T)

    return {name: port_vals[k] for k, name in enumerate(strategy_names)}


@st.cache_data(show_spinner=False)
def run_spy_garch_simulation(_spx_rets_aligned, n_scenarios=2_000, n_steps=252, seed=42):
    """
    Single-asset GARCH(1,1) + empirical inverse-transform simulation for SPY.

    No copula is needed, as SPY is already a diversified index with no cross-asset
    dependency to model. Uses the same GARCH + empirical residual approach as
    the multi-asset simulation for methodological consistency.

    Returns terminal portfolio values shape (n_scenarios,).
    """
    from arch import arch_model
    import warnings

    r = _spx_rets_aligned.dropna() * 100
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = arch_model(
            r, vol='Garch', p=1, q=1, dist='normal', mean='Constant'
        ).fit(disp='off', show_warning=False)

    _mu = 0.0
    for _k in ('mu', 'Const', 'constant'):
        if _k in res.params.index:
            _mu = float(res.params[_k])
            break

    omega = float(res.params['omega'])
    alpha = float(res.params['alpha[1]'])
    beta  = float(res.params['beta[1]'])

    sorted_z    = np.sort((res.resid / res.conditional_volatility).values)
    n_hist      = len(sorted_z)
    last_sigma2 = float(res.conditional_volatility.iloc[-1] ** 2)
    last_eps    = float(res.resid.iloc[-1])

    rng      = np.random.default_rng(seed)
    sigma2_s = np.full(n_scenarios, last_sigma2)
    eps_s    = np.full(n_scenarios, last_eps)
    port_v   = np.ones(n_scenarios)

    for t in range(n_steps):
        sigma2_s = omega + alpha * eps_s ** 2 + beta * sigma2_s
        sigma_s  = np.sqrt(np.maximum(sigma2_s, 1e-8))

        u   = rng.uniform(0, 1, n_scenarios)
        idx = np.clip((u * n_hist).astype(int), 0, n_hist - 1)
        z_t = sorted_z[idx]

        r_scaled = _mu + sigma_s * z_t
        r_actual = r_scaled / 100.0
        eps_s    = r_scaled - _mu
        port_v  *= (1.0 + r_actual)

    return port_v

    
# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM TICKER VALIDATION
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False, ttl=3600)
def validate_custom_ticker(ticker: str, start_date_str: str):
    """
    Validates a user-supplied ticker before it is admitted into the universe.
    Cached for 1 hour so repeated attempts on the same ticker don't re-hit the API.

    Returns
    -------
    is_valid   : bool
    message    : str  – human-readable pass/fail reason
    warnings   : list[str] – soft warnings (non-blocking)
    """
    import time
    t = yf.Ticker(ticker)

    # ── Hard check 1: ticker exists ───────────────────────────────────────────
    try:
        last_price = t.fast_info.last_price
        if last_price is None or last_price <= 0:
            return False, (
                f"**{ticker}** does not appear to be a valid ticker symbol. "
                "Double-check the symbol on Yahoo Finance."
            ), []
    except Exception:
        return False, (
            f"Could not retrieve data for **{ticker}**. "
            "Check the ticker symbol and try again."
        ), []

    # ── Hard check 2: sufficient price history from the chosen start date ─────
    try:
        raw = yf.download(ticker, start=start_date_str, auto_adjust=True, progress=False)
        if raw.empty:
            return False, (
                f"**{ticker}** returned no price data from {start_date_str}. "
                "The stock may have listed after that date."
            ), []
        prices = raw["Close"].squeeze()
        n_days = int(prices.dropna().shape[0])
        
        # Minimum is set to 1 008 trading days (~4 years) to make sure data isnt cut
        _MIN_DAYS = 1_008
        if n_days < _MIN_DAYS:
            return False, (
                f"**{ticker}** only has **{n_days} trading days** of data from {start_date_str} "
                f"(minimum {_MIN_DAYS} required, roughly 4 years). "
                "A short-history ticker truncates the covariance matrix and backtest period "
                "for the entire universe, not just itself. "
                "Try moving the data start date **earlier** (further back in time) to give this ticker more history, "
                "or choose a stock with a longer listing history."
            ), []
    except Exception as exc:
        return False, f"Failed to download price data for **{ticker}**: {exc}", []

    # ── Hard check 3: data completeness ───────────────────────────────────────
    missing_pct = float(prices.isna().mean())
    if missing_pct > 0.05:
        return False, (
            f"**{ticker}** has {missing_pct:.0%} missing price data from {start_date_str}. "
            "This is too high for a reliable covariance estimate, as the data is likely incomplete."
        ), []

    # ── Soft warnings (non-blocking) ──────────────────────────────────────────
    soft_warnings = []

    # Warn if the ticker would noticeably shorten the shared history
    _PREFERRED_DAYS = 1_260   # ~5 years
    if n_days < _PREFERRED_DAYS:
        soft_warnings.append(
            f"Only {n_days} trading days available (preferred ≥ {_PREFERRED_DAYS}, ~5 years). "
            "Adding this ticker will shorten the effective analysis window for all other stocks. "
            "Consider using a longer data start date or a stock with more history."
        )

    try:
        mcap = t.fast_info.market_cap
        if mcap is not None and mcap < 10e9:
            soft_warnings.append(
                f"Market cap is ~${mcap / 1e9:.1f}B, below the $10B minimum "
                "used to screen the core universe."
            )
    except Exception:
        pass

    return True, (
        f"✅ **{ticker}** added ({n_days} trading days available from {start_date_str})."
    ), soft_warnings


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR -- User Inputs
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Model Assumptions")

    # ── Session state: custom tickers list ────────────────────────────────────
    if "custom_tickers" not in st.session_state:
        st.session_state.custom_tickers = []
    if "ticker_add_msg" not in st.session_state:
        st.session_state.ticker_add_msg = None   # (kind, text) tuple or None
    if "ticker_add_warnings" not in st.session_state:
        st.session_state.ticker_add_warnings = []

    # Active universe = core 17 + any user-added tickers (both sorted)
    active_tickers = sorted(TICKERS + st.session_state.custom_tickers)

    # --- Data Range Selection ---
    _FLOOR = date(2012, 6, 1)
    _MAX_START      = (datetime.today() - pd.Timedelta(days=365 * 3)).date()
    
    st.subheader("1. Data Range")
    data_range_help = (
        "**Minimum: 2012-06-01** – one month after META's IPO (the most recent in the universe).\n\n"
        "**Recommended Default: 2015-01-01** – captures multiple market regimes "
        "(2015 volatility spike, 2018 correction, COVID crash, 2022 rate hikes, 2023-25 AI bull).\n\n"
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
    
    price_data, tick_rets = load_market_data(active_tickers, start=data_start_date.strftime("%Y-%m-%d"))
    
    # Fetch the cached weights series and metadata
    weights_series, consensus_data, recent_earnings = load_ticker_metadata(active_tickers)
    
    # Build the cap-weight DataFrame dynamically with the fresh index
    idx = price_data.index
    tick_capweights = pd.DataFrame(
        [weights_series.reindex(active_tickers).values] * len(idx),
        index=idx, columns=active_tickers,
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
        min_value=5.0, max_value=100.0, value=15.0, step=0.5,
        help=("Imposes a ceiling on the maximum weight the optimiser can allocate into a single stock. "
              "Default: 15% to enable some concentration risks while forcing some diversification."
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
        "1-Year price targets are the aggregate price estimates of analysts "
        "sourced from Yahoo Finance and are automatically ported in. "
        "Calibrate the Confidence levels with the Confidence tab."
    )
    st.caption("🟡 = earnings reported in the last 30 days")
    user_targets    = {}
    user_confidence = {}

    for i in range(0, len(active_tickers), 2):
        col_a, col_b = st.columns(2)
        for col, ticker in zip([col_a, col_b], active_tickers[i:i + 2]):
            with col:
                # Ticker label + earnings recency flag
                earnings_flag  = " 🟡" if recent_earnings.get(ticker, False) else ""
                _is_dcf = ticker in DCF_OVERRIDES
                _dcf_tag = "⚡" if _is_dcf else ""
                if ticker in TICKERS:
                    label = f"**{ticker}**{_dcf_tag}{earnings_flag}"
                else:
                    label = f"**{ticker}**{earnings_flag}" if ticker in TICKERS else f"**{ticker}** ✦{earnings_flag}"
                
                st.markdown(label)

                # Consensus reference line
                cons   = consensus_data.get(ticker, {})
                mean_t = cons.get("mean", None)
                n_ana  = cons.get("n_analysts", None)
                if mean_t:
                    n_str = f" | from {int(n_ana)} analysts" if n_ana else ""
                    st.caption(f"Consensus: ${mean_t:,.0f}{n_str}")
                else:
                    st.caption("Consensus: N/A")

                # Default target: consensus first (live Yahoo Finance), then BASE_TARGETS as fallback,then a generic placeholder.
                _cons_default = float(mean_t) if mean_t else None
                if _is_dcf: 
                    _target_default =  float(BASE_TARGETS.get (ticker, 100.0))
                else:
                    _target_default = (
                        _cons_default
                        if _cons_default is not None
                        else float(BASE_TARGETS.get(ticker, 100.0))
                    )
                user_targets[ticker] = st.number_input(
                    "Price target ($)",
                    min_value=0.01,
                    value=_target_default,
                    step=1.0,
                    key=f"pt_{ticker}",
                )
                user_confidence[ticker] = st.slider(
                    "Confidence",
                    min_value=0.0,
                    max_value=1.0,
                    value=float(BASE_CONFIDENCE.get(ticker, 0.20)),
                    step=0.05,
                    key=f"conf_{ticker}",
                )

    st.divider()

    # ── Section 4: Add Custom Tickers ─────────────────────────────────────────
    st.subheader("4. Custom Tickers")

    _inp_col, _btn_col = st.columns([3, 1])
    with _inp_col:
        _new_ticker_raw = st.text_input(
            "Ticker symbol",
            placeholder="e.g. CRM",
            key="new_ticker_input",
            label_visibility="collapsed",
            help = ("Add any valid Yahoo Finance ticker. "
                    "Tickers must have at least **1 008 trading days (~4 years)** of data from your chosen "
                    "start date and less than 5% missing values before being admitted. "
                    "A shorter-history ticker would truncate the covariance and backtest window for every "
                    "other stock in the universe."
                   ),
        )
    with _btn_col:
        _add_clicked = st.button("Add", use_container_width=True, key="add_ticker_btn")

    if _add_clicked and _new_ticker_raw.strip():
        _t = _new_ticker_raw.strip().upper()
        if _t in active_tickers:
            st.session_state.ticker_add_msg = ("warning", f"**{_t}** is already in the universe.")
            st.session_state.ticker_add_warnings = []
        else:
            with st.spinner(f"Validating {_t}…"):
                _valid, _msg, _warns = validate_custom_ticker(
                    _t, data_start_date.strftime("%Y-%m-%d")
                )
            if _valid:
                st.session_state.custom_tickers.append(_t)
                st.session_state.ticker_add_msg = ("success", _msg)
                st.session_state.ticker_add_warnings = _warns
                st.rerun()
            else:
                st.session_state.ticker_add_msg = ("error", _msg)
                st.session_state.ticker_add_warnings = []

    # Render the last add attempt message
    if st.session_state.ticker_add_msg:
        _kind, _text = st.session_state.ticker_add_msg
        if _kind == "success":
            st.success(_text)
        elif _kind == "error":
            st.error(_text)
        elif _kind == "warning":
            st.warning(_text)
        for _w in st.session_state.ticker_add_warnings:
            st.warning(f"⚠️ {_w}")

    # List of currently active custom tickers with remove buttons
    if st.session_state.custom_tickers:
        st.markdown("**Custom tickers in universe:**")
        for _ct in list(st.session_state.custom_tickers):
            _name_col, _rem_col = st.columns([4, 1])
            with _name_col:
                st.markdown(f"`{_ct}` ✦")
            with _rem_col:
                if st.button("✕", key=f"remove_{_ct}", help=f"Remove {_ct} from universe"):
                    st.session_state.custom_tickers.remove(_ct)
                    st.session_state.ticker_add_msg = None
                    st.session_state.ticker_add_warnings = []
                    st.rerun()

    st.divider()

    # --- Other BL parameters ---
    st.subheader("5. Other BL Parameters")

    delta = st.slider(
        "δ  Risk Aversion",
        min_value=1.0, max_value=5.0, value=2.5, step=0.1,
        help=("Delta is the risk-aversion coefficient of the market portfolio. "
              "Standard Value is 2.5. Higher means market expects more return per unit of risk."
              )
    )

    tau = st.slider(
        "τ  Prior Uncertainty",
        min_value=0.01, max_value=0.10, value=0.025, step=0.005,
        help=("Tau is the uncertainty in the prior returns benchmark. For this model, the Cap-Weighted allocation returns are the benchmark"
              "Standard value is 0.025 as used by He-Litterman. Smaller means you trust the market more."
              )
    )
    
    st.divider()

    # --- Backtest estimation window ---
    st.subheader("6. Backtest Estimation Window")

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

# ─────────────────────────────────────────────────────────────────────────────
# LOAD DATA  (price data and metadata already fetched before the sidebar)
# ─────────────────────────────────────────────────────────────────────────────
RF = load_rf()
spx_rets    = load_benchmark_data()
val_metrics = load_valuation_metrics(active_tickers)

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
    f"Universe: **{len(active_tickers)} stocks** |  "
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
        It is built on what I have learned over the last 
        few months from *EDHEC's Advanced Portfolio Construction and Analysis* 
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
        than wealth preservation.** Institutions with fixed payment obligations, 
        such as sovereign wealth funds, would instead adopt a liability-driven approach, 
        duration-matching asset cash flows to known liability schedules. The CPPI framework in the 
        **Strategy Comparison** tab is one bridge between the two: it wraps the BL equity 
        allocation inside a drawdown floor, allowing it to coexist with a capital 
        preservation mandate.

        """
    )

    st.divider()

    st.markdown("#### How to navigate this app")
    _nav_cols = st.columns(5)
    _nav_cards = [
        ("🧮", "DCF Models",          "Where the price targets come from"),
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
            "⚙️ **Sidebar (global)** Price targets, confidence levels, position size constraints, "
            "BL parameters, and backtest estimation window are all adjustable from the sidebar and "
            "flow through every tab in real time."
        )

    st.divider()
        
    st.markdown("#### How Black-Litterman Works")
    st.markdown(
        """
        Most portfolio optimisers have a well-known flaw. Feed them expected returns
        built purely from historical data or analyst targets, and they produce extreme,
        unstable allocations (i.e. prone to error maximisation where even small estimation errors
        result in highly concentrated portfolios where no sensible investor would hold).

        **Black-Litterman solves this problem by never letting a view stand alone.**
        Instead, it always asks: *relative to what the market collectively believes,
        how much should a view actually shift the allocation?*

        The model has two inputs:

        - **The market prior (π)**: what the market implies everyone should expect,
          derived by reverse-engineering the CAPM: if every investor holds the market
          portfolio, what expected returns would justify current prices and weights?
          This is the baseline the model starts from. Note that the market prior for this app
          uses the stock universe capweights and not capweights as derived from the S&P 500.

        - **Analyst views (Q)**: the excess return implied by each DCF price target
          (total return minus the risk-free rate). This is the forward-looking view of the investor.

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
        | **μ_BL** | Posterior expected return | The model's final blended return estimate which feeds into the optimiser |
        | **π** (pi) | Market-implied equilibrium return | What the market collectively expects, derived from cap weights and risk aversion |
        | **Q** | Analyst views | Excess return implied by each DCF price target (total return minus risk-free rate) |
        | **Σ** (Sigma) | Covariance matrix | How much each stock moves, and how they move together, capturing correlation risk |
        | **τ** (tau) | Prior uncertainty scalar | How much to distrust the market prior; smaller means more trust in the market |
        | **P** | View matrix | Maps each view to the stocks it applies to; here an identity matrix for one view per stock |
        | **Ω** (Omega) | View uncertainty matrix | How uncertain each analyst view is; computed from the confidence sliders via the Idzorek method |

        **Intuition:** The formula is a tug-of-war between π and Q, refereed by
        uncertainty. When confidence is high, Ω is small, its inverse is large, and Q
        pulls the posterior strongly away from π. When confidence is low, Ω is large,
        its inverse shrinks, and the posterior barely moves from equilibrium. The covariance Σ 
        ensures that stocks with shared risk exposures influence each other. 
        The table in the **Views, Returns & Weights** tab show this blending in action.
        """
    )

# ══════════════════════════════════════════════════════════════════════════════
# TAB1 - DCF Models
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.html("<div style='height: 18px;'></div>")
    st.markdown(
        """
        **This tab explains the selection of the stock universe, and the derivation of the price
        targets used for the BL model.** For most stocks which do not have a
        manual DCF analysis, I applied aggregate price targets pulled automatically from Yahoo Finance.
        Stock tickers with DCF analysis that I have conducted override the price targets by analysts and reflect
        my personal view on the valuation of those companies (NVDA available, META coming soon).
        You may override the price targets in the sidebar, adjust the confidence levels under the **Confidence** tab, 
        and see the resulting portfolio allocation displayed in the **Views, Returns & Weights** tab.
        """
    )
    
    st.divider()

    # ── Section 1: Stock Universe & Selection Criteria ────────────────────────
    st.markdown("#### 1. Stock Universe & Selection Criteria")
    st.markdown(
        """
        The 17 stocks were selected through a combination of quantitative screening and qualitative 
        judgment. The initial screen filtered for **(1) 5-year average ROIC above 15%**, 
        **(2) debt-to-equity below 1**, **(3) 5-year revenue growth above 15%**, and 
        **(4) minimum market cap of $10 billion**, producing a candidate list of roughly 42 stocks. 
        From this, I selected businesses with identifiable competitive moats and a reasonable 
        spread across sectors. Most positions don't carry an active thesis; the allocation defers 
        to the market prior, with views reserved for names where I've done deeper research.
        
        ROIC was chosen as the primary filter because it measures capital efficiency independent of 
        leverage. Sustained high ROIC is one of the most reliable indicators of a durable moat. 
        FCF margin was deliberately excluded as a screen to avoid penalising compounders like AMZN, 
        MSFT, and GOOGL that are in a heavy capex cycle for AI infrastructure.
        
        """
    )
    
    st.divider()

    # ── Section 2: Individual DCF Models ──────────────────────────────────────
    st.markdown("#### 2. Individual DCF Models")
    st.caption(
        "✅ NVDA model is complete; ❌ META coming soon. "
        "All other stocks use analyst consensus targets from Yahoo Finance in the BL model."
    )

    _dcf_tabs = st.tabs(
        [f"{'✅ ' if t in DCF_OUTPUTS else '❌ '}{t}" for t in HIGH_CONVICTION]
    )

    for _ctab, _ticker in zip(_dcf_tabs, HIGH_CONVICTION):
        with _ctab:

            # ── Placeholder for companies without a completed model ────────────
            if _ticker not in INVESTMENT_THESES or _ticker not in DCF_OUTPUTS:
                st.info(
                    f"**{_ticker}**: DCF model in progress.",
                    icon="⚡",
                )
                continue

            _thesis = INVESTMENT_THESES[_ticker]
            _dcf    = DCF_OUTPUTS[_ticker]

            # Live current price, uses latest close from the already-loaded price data
            _cp = (
                float(price_data[_ticker].iloc[-1])
                if _ticker in price_data.columns
                else float(_dcf["current_price"])
            )
            _fair_value = float(_dcf["DCF Fair Value"])
            _upside     = (_fair_value / _cp) - 1

            # ── Key Valuation Summary ──────────────────────────────────────
            _ff = _dcf["football_field"]
            _perp_base  = next((r["base"] for r in _ff if "Perpetuity"    in r["label"] and r["base"] is not None), None)
            _ebitda_base= next((r["base"] for r in _ff if "EBITDA"        in r["label"] and r["base"] is not None), None)

            _vm1, _vm2, _vm3, _vm4, _vm5 = st.columns(5)
            _vm1.metric(
                "Current Price",
                f"${_cp:,.2f}",
                help="Latest closing price, live from Yahoo Finance",
            )
            _vm2.metric(
                "**DCF Fair Value**",
                f"${_fair_value:,.2f}",
                f"{_upside:+.1%}",
                delta_color="normal",
                help="Simple average of the perpetuity and exit-EBITDA approaches.",
            )
            if _perp_base:
                _vm3.metric(
                    "Perpetuity Approach",
                    f"${_perp_base:,.2f}",
                    f"{(_perp_base / _cp) - 1:+.1%}",
                    delta_color="normal",
                    help="Gordon Growth Model perpetuity value at base WACC and terminal g.",
                )
            if _ebitda_base:
                _vm4.metric(
                    "Exit Multiple",
                    f"${_ebitda_base:,.2f}",
                    f"{(_ebitda_base / _cp) - 1:+.1%}",
                    delta_color="normal",
                    help="Terminal-year EBITDA × exit multiple, discounted at WACC.",
                )
            _vm5.metric(
                "WACC",
                f"{_dcf['wacc']:.2%}",
                help="Weighted Average Cost of Capital used to discount cash flows.",
            )

            st.caption(
                "**Updated from NVDA's Q1 FY2027 Report: 20 May 2026** "
                "DCF Fair Value is the simple average between the perpetuity and exit multiple DCF valuations"
            )
            st.divider()


            # ── Football Field Valuation ────────────────────────────────────
            st.markdown("##### DCF Valuation Football Field")
            st.caption(
                "Bars show the valuation range for each methodology. "
                "◆ marks the base-case point estimate with % upside / downside to the current price. "
                f"The dashed line is the current share price (${_cp:,.2f}, live from Yahoo Finance)."
            )

            _fig_ff = go.Figure()

            # Range bars
            for _row in _ff:
                _fig_ff.add_trace(go.Bar(
                    x=[_row["high"] - _row["low"]],
                    y=[_row["label"]],
                    base=[_row["low"]],
                    orientation="h",
                    marker=dict(
                        color="rgba(29, 158, 117, 0.18)",
                        line=dict(color="rgba(29, 158, 117, 0.65)", width=1.5),
                    ),
                    showlegend=False,
                    hovertemplate=(
                        f"<b>{_row['label']}</b><br>"
                        f"Range: ${_row['low']:,} – ${_row['high']:,}<extra></extra>"
                    ),
                ))

            # Base-case diamond markers + % upside annotations
            _base_x = [r["base"] for r in _ff if r["base"] is not None]
            _base_y = [r["label"] for r in _ff if r["base"] is not None]
            if _base_x:
                _fig_ff.add_trace(go.Scatter(
                    x=_base_x, y=_base_y,
                    mode="markers",
                    marker=dict(size=12, color="#1D9E75", symbol="diamond"),
                    name="Base Case",
                    hovertemplate="<b>%{y}</b><br>Base Case: $%{x:,.2f}<extra></extra>",
                ))
                # Percentage labels next to each diamond
                for _bx, _by in zip(_base_x, _base_y):
                    _pct = (_bx / _cp) - 1
                    _pct_txt = f" {_pct:+.1%}"
                    _fig_ff.add_annotation(
                        x=_bx, y=_by,
                        text=_pct_txt,
                        showarrow=False,
                        xanchor="left",
                        yanchor="middle",
                        font=dict(size=10, color="#1D9E75" if _pct >= 0 else "#D85A30"),
                    )

            # Current price line
            _fig_ff.add_vline(
                x=_cp,
                line_dash="dash", line_color="#D85A30", line_width=2,
                annotation_text=f"Current ${_cp:,.2f}",
                annotation_position="top right",
                annotation_font=dict(color="#D85A30", size=11),
            )

            _all_vals = [r["low"] for r in _ff] + [r["high"] for r in _ff] + [_cp]
            _fig_ff.update_layout(
                height=280,
                xaxis=dict(
                    title="Equity Value per Share ($)",
                    range=[min(_all_vals) * 0.85, max(_all_vals) * 1.12],
                    tickprefix="$",
                    zeroline=False,
                ),
                yaxis=dict(autorange="reversed"),
                barmode="overlay",
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                margin=dict(t=20, b=40, l=10, r=40),
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
            )
            st.plotly_chart(_fig_ff, use_container_width=True)

            st.divider()

            # ── Investment Thesis ──────────────────────────────────────────
            # Rendered as plain markdown -  edit text directly in INVESTMENT_THESES above.

            st.markdown("##### Investment Thesis")
            for _pt in _thesis["theses"]:
                st.markdown(f"- {_pt}")

            st.markdown("##### Latest Earnings")
            for _pe in _thesis.get("latest_earnings", []):
                st.markdown(f"- {_pe}")

            _gc1, _gc2 = st.columns(2)
            with _gc1:
                st.markdown("##### Forward Guidance")
                for _pg in _thesis.get("guidance", []):
                    st.markdown(f"- {_pg}")
            with _gc2:
                st.markdown("##### Key Growth Drivers")
                for _gname, _gdesc in _thesis["growth_drivers"]:
                    st.markdown(f"- **{_gname}**: {_gdesc}")

            st.divider()

            # ── Summary of DCF Assumptions (pandas DataFrames) ─────────────
            st.markdown("##### Summary of DCF Assumptions")

            # Part A: Income Statement Drivers
            st.markdown("###### Income Statement Drivers")
            st.caption(
                "**FY2026 Actual** anchors each assumption to the last reported fiscal year. "
            )
            _is_rows = _dcf["is_assumptions"]
            _is_df = pd.DataFrame([
                {
                    "Metric":         r["name"],
                    "FY2026 Actual":  r["actual"],
                    "Base Case":      r["base"],
                    "Note":           r["note"],
                }
                for r in _is_rows
            ]).set_index("Metric")
            st.dataframe(_is_df, use_container_width=True,
                column_config={
                    "Note":          st.column_config.TextColumn("Note",          width="large"),
                    "FY2026 Actual": st.column_config.TextColumn("FY2026 Actual", width="small"),
                    "Base Case":     st.column_config.TextColumn("Base Case",     width="small"),
                }
            )

            # Part B: DCF & Terminal Value Assumptions
            st.markdown("###### DCF & Terminal Value Assumptions")
            _dcf_rows = _dcf["dcf_assumptions"]
            _dcf_df = pd.DataFrame([
                {
                    "Assumption": r["name"],
                    "Base Case":  r["base"],
                    "Note":       r["note"],
                }
                for r in _dcf_rows
            ]).set_index("Assumption")
            st.dataframe(_dcf_df, use_container_width=True,
                column_config={
                    "Note":      st.column_config.TextColumn("Note",      width="large"),
                    "Base Case": st.column_config.TextColumn("Base Case", width="small"),
                }
            )
            
            st.divider()

            # ── Full Model Download ─────────────────────────────────────────
            st.markdown("##### Full DCF Model Download")
            _link = _dcf.get("model_link")
            if _link:
                st.markdown(
                    f"📥 **[Download the full 3-Statement Model & DCF for {_ticker}]({_link})**  \n"
                )
            else:
                st.info(
                    f"Model link not yet configured. Populate `DCF_OUTPUTS['{_ticker}']['model_link']` "
                    "in `app.py` with a Google Drive share link or GitHub raw-download URL.",
                    icon="📎",
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
        trust your view completely and let it override the market equilibrium.
        
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
    last_close = price_data.reindex(columns=active_tickers).iloc[-1]

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
    for tkr in active_tickers:
        if tkr in DCF_OVERRIDES:
            conf_vals[tkr]   = DCF_OVERRIDES[tkr]
            conf_source[tkr] = "⚡ DCF Model"
        else:
            conf_vals[tkr]   = BASE_CONFIDENCE.get(tkr, 0.20)
            conf_source[tkr] = "Analyst est." if tkr in BASE_CONFIDENCE else "Custom (default 0.20)"

    # ── Section 1: Conviction Map ──────────────────────────────────────────────
    st.markdown("#### 1. Conviction Map")
    st.caption(
        "Adjusting the confidence sliders in the slide bar will result in this map changing "
        "dyanmically. Use this chart and the Confidence Levels to make map "
        "price target confidences relative to the market. "
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
    st.plotly_chart(_fg, width="stretch")

    st.divider()
 
    # ── Section 3: Confidence Levels & Rationale ─────────────────────────────
    st.markdown("#### 2. Confidence Levels & Rationale")
    st.caption(
        "Sorted by gap to break-even (most bullish first). "
        "The break-even price is the market-implied equilibrium price: "
        "a target above it is a bullish signal and below it is bearish relative to the market."
        "The confidence setting then determines how strongly that signal shifts the BL allocation."
        "Gap to B/E expresses this signal as a percentage difference between price targets "
        "and the market-implied price. Q − π is the same signal translated into **return space**, "
        "and is the figure the BL formula actually uses."
        )
 
    # ── Tier legend ───────────────────────────────────────────────────────────
    _t1, _t2, _t3 = st.columns(3)
    with _t1:
        with st.container(border=True):
            st.markdown("**⚡ Full DCF**")
            st.caption(
                "Completed bottom-up model. Confidence is derived directly "
                "from DCF output and terminal value uncertainty."
            )
    with _t2:
        with st.container(border=True):
            st.markdown("**🎯 Tactical**")
            st.caption(
                "Personal holding or high-level thesis. Single-sentence "
                "rationale; no full model."
            )
    with _t3:
        with st.container(border=True):
            st.markdown("**📊 Systematic**")
            st.caption(
                "No active thesis. Confidence is calibrated to stay close to "
                "market equilibrium; the model does not choose a view."
            )

    st.html("<div style='height: 6px;'></div>")
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

        if _tkr in RESEARCH_TIERS["FULL_DCF"]:
            _tier = "⚡ Full DCF"
        elif _tkr in RESEARCH_TIERS["TACTICAL"]:
            _tier = "🎯 Tactical"
        else:
            _tier = "📊 Systematic"

        _conf_rows[_tkr] = {
            "Target ($)":            _tgt,
            "Break-even ($)":        _be,
            "BL Direction":          "▲ Bullish" if _q_pi > 0 else "▼ Bearish",
            "Gap to B/E":            _gap_pct,
            "Q − π":                 _q_pi,
            "Confidence":            _cv,
            "Confidence Rationale":  render_rationale(_tkr),
        }

    _conf_df = (
        pd.DataFrame(_conf_rows)
        .T
        .sort_values("Gap to B/E", ascending=False)
    )

    _styled_conf = (
        _conf_df.style
        .format("{:.2%}",   subset=["Gap to B/E", "Q − π"])
        .format("${:,.2f}", subset=["Target ($)", "Break-even ($)"])
        .format("{:.2f}", subset=["Confidence"])
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
        width="stretch",
        height=700,
        column_config={
            "Tier": st.column_config.TextColumn(
                "Tier",
                width="small",
                help=(
                    "⚡ Full DCF: confidence derived from a completed bottom-up model. "
                    "🎯 Tactical: personal holding or monitored thesis, single-bullet rationale. "
                    "📊 Systematic: no active thesis, confidence defers to market equilibrium."
                ),
            ),
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

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 - Views, Returns & Weights
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.html("<div style='height: 18px;'></div>")
    st.markdown(
        """
        **This tab walks through the full Black-Litterman pipeline in sequence.**
        Starting from DCF price targets, the model computes the excess return view (Q)
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

    st.dataframe(styled, width="stretch")

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
    st.plotly_chart(fig, width="stretch")

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
        width="stretch",
    )

    nonzero = weight_df[weight_df["BL Optimised"] > 0.001].reset_index()
    nonzero.columns = ["Ticker"] + list(nonzero.columns[1:])
    fig_tree = px.treemap(
        nonzero, path=["Ticker"], values="BL Optimised",
        title="BL Weight Allocation",
        color="BL Optimised", color_continuous_scale="YlGnBu",
    )
    fig_tree.update_traces(textinfo="label+percent entry")
    st.plotly_chart(fig_tree, width="stretch")

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
    st.plotly_chart(fig2, width="stretch")

# ── GARCH fitting (cached), shared by Tab 4 stress context and Tab 5 GARCH-Copula ──
garch_params, std_resids_g, last_state, cond_vol_df, fitted_theta = fit_garch_params(
    tick_rets, tuple(tick_rets.columns)
)
_theta_default = (
    float(np.clip(round((fitted_theta or 2.0) / 0.1) * 0.1, 0.1, 10.0))
    if fitted_theta is not None else 2.0
)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 - Simulation & Stress Tests
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.html("<div style='height: 18px;'></div>")
    st.markdown(
        """
        **This tab evaluates Black-Litterman portfolio weights under market 
        stress using two methodologies:** 
        1. **Correlated Geometric Brownian Motion (GBM) Monte Carlo Simulation**: 
        Projects future returns as a random walk, preserving asset interdependencies 
        via Cholesky decomposition. Given the heavy concentration of technology 
        equities in this portfolio universe, accounting for these embedded correlations 
        is essential for a realistic baseline.
        2. **Historical Stress Testing**: Backtests the portfolio against actual 
        historical market shocks to evaluate performance during systemic crises.

        **Model Caveats**: While GBM captures *typical* market uncertainty, it assumes that
        market returns follow a normal Gaussian distribution. 
        It cannot model **volatility clustering** or the **breakdown of historical 
        correlations** that occur during severe market drawdowns. In a free-fall market, 
        stocks that are otherwise not correlated all fall together, which a standard GBM will 
        understate. The Monte Carlo simulation therefore represents 
        a baseline for normal market regimes, while the historical stress test provides 
        the necessary reality check for tail risk.
        """
    )
    st.divider()
    
    # ── Section 1: Correlated GBM Simulation ───────────────────────────────────────────────
    st.markdown("#### 1. Correlated GBM Simulation")
    col1, col2 = st.columns(2)
    n_scenarios = col1.slider("Scenarios", min_value=500, max_value=10000, value=2000, step=500)
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
    col_fan.plotly_chart(fig_mc, width="stretch")

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
    col_hist.plotly_chart(fig_hist, width="stretch")

    st.divider()

    # ── Section 2: Historical Stress Tests ───────────────────────────────────────────────
    st.markdown("#### 2. Historical Stress Test")
    st.caption(
        "Applies the BL optimal weights to *actual* historical returns during "
        "known market shocks. This shows how this allocation *would* have "
        "performed and is not a forecast."
    )

    w_stress = bl_w_series.reindex(tick_rets.columns).fillna(0).values
    stress_rows = {}
    
    # Summary Stats for Stress Periods (including SPY)
    for name, (start, end) in STRESS_PERIODS.items():
        period_rets = tick_rets.loc[start:end]
        if period_rets.empty:
            continue
        bl_period_rets = (period_rets * w_stress).sum(axis=1)
        
        # Calculate the single compounded return for the period
        total_period_return = (1 + bl_period_rets).prod() - 1 
        
        ann_rets_bl_period = erk.annualize_rets(bl_period_rets, periods_per_year=252)
        ann_vol_bl_period  = erk.annualize_vol(bl_period_rets, periods_per_year=252)
        sharpe_bl_period   = erk.sharpe_ratio(bl_period_rets, riskfree_rate=RF, periods_per_year=252)
        cumprod_bl = (1 + bl_period_rets).cumprod()
        max_dd_bl  = (cumprod_bl / cumprod_bl.cummax() - 1).min()

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
    
    #Creation of Stress Test Data Frame
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
        width="stretch",
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
    st.plotly_chart(fig_stress, width="stretch")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 - Strategy Comparison
# ══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.html("<div style='height: 18px;'></div>")
    st.markdown(
        """
        **This tab benchmarks the Black-Litterman portfolio against four alternative strategies 
        and the S&P 500**, presenting a historical wealth index, a 1-year GBM return 
        forecast and a CPPI drawdown protection analysis. While the wealth index 
        and GBM forecast answer the question of "how does BL compare to alternative strategies?", the CPPI analysis asks 
        a different question: *what does it cost to protect capital, and how does the BL portfolio 
        behave once a drawdown floor is imposed?* This matters because BL is built for wealth accumulation. However, most real investors have a loss tolerance, 
        and cannot afford to absorb drawdowns like the 34% COVID crash and simply wait for recovery. 
        CPPI wraps a dynamic floor around BL, creating a version of the portfolio that an 
        investor could hold.

        The historical wealth index uses a rolling backtest with the estimation window set in the 
        sidebar. Equal-Weighted, Cap-Weighted, Global Minimum Variance, and Risk Parity are all 
        properly rolled, with weights re-estimated at each step using only data available at that 
        point, so there is no look-ahead bias. Black-Litterman is shown differently, as a static 
        allocation applying the current optimal weights to the full history. This is an intentional 
        design choice. As BL weights are derived from a forward-looking 
        view on price targets rather than purely from historical patterns, applying them statically 
        must be read more simply as "how would this portfolio have done retrospectively?"

        **Model Caveats**: The historical comparison is not a like-for-like backtest. 
        The rolling strategies adapt to new data over time while BL holds fixed weights throughout. 
        This comparison is best read as an illustration of the structural differences between BL and
        the other strategies. The 1-Year Monte Carlo Return Forecast in Section 2 
        provides a more comparable forward-looking view, though as noted in the **Simulation & Stress 
        Tests** tab, GBM paths assume Gaussian returns and should be treated as a baseline rather than 
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

    # ── Guard: cap estimation_window to available data ───────────────────────
    # estimation_window (in trading days) can exceed len(tick_rets) when the
    # user sets data_start_date too recently relative to the window slider.
    # e.g. start_date = _MAX_START (~3yr ago, ~756 td) + window = 7yr (1764 td)
    # -> tick_rets.index[1764] raises IndexError and crashes the entire tab.
    # Cap silently and surface a clear warning so the user knows what happened.
    _n_available_td = len(tick_rets)
    _max_safe_ew    = max(_n_available_td - 2, 1)   # leave at least 1 row after the window
    if estimation_window > _max_safe_ew:
        _safe_yrs = max(_max_safe_ew // 252, 1)
        st.warning(
            "**Estimation window reduced automatically.**  \n"
            f"The selected window ({estimation_window_yrs}yr = {estimation_window} trading days) "
            f"exceeds the available price history ({_n_available_td} trading days from "
            f"{data_start_date}).  \n"
            f"Window capped at **{_safe_yrs}yr ({_safe_yrs * 252} trading days)** for this run.  \n"
            "To use a longer window, move the **data start date** earlier in the sidebar.",
        )
        estimation_window = _safe_yrs * 252

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

    # Create Wealth Index
    btr = pd.DataFrame({
        "Equal-Weighted":          pd.Series(ew_r).squeeze(),
        "Cap-Weighted":            pd.Series(cw_r).squeeze(),
        "Global Minimum Variance": pd.Series(gmv_r).squeeze(),
        "Risk Parity":             pd.Series(erc_r).squeeze(),
        "BL (static)":             pd.Series(bl_r).squeeze(),
        "BL (CPPI)":               pd.Series(cppi_r).squeeze(),
        "S&P 500 (SPY)":           spx_rets.reindex(tick_rets.index),
    }).loc[valid_start_date:].dropna()

    # ── Guard: abort Tab 5 cleanly if btr is empty ───────────────────────────
    # btr can be empty when valid_start_date falls at or past the last row of
    # tick_rets (e.g. extremely short data range or estimation_window == n-1).
    # Every downstream operation indexes btr.index[0], so we stop here with a
    # clear message rather than propagating an opaque KeyError/IndexError.
    if btr.empty:
        st.error(
            "**Not enough data to build the wealth index.**  \n"
            "After reserving the estimation window, no rows remain for the backtest period.  \n"
            "**Fix:** move the **data start date** earlier in the sidebar, "
            "or reduce the **estimation window** slider."
        )
        st.stop()

    # ── Section 1: Historical Wealth Index ────────────────────────────────────
    st.markdown("#### 1. Historical Wealth Index")

    #Define start and end dates of BL and backtest data
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
    st.plotly_chart(fig_wealth, width="stretch")

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
        width="stretch",
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
        "**Note**: Historic wealth index and summary statistics assume all strategies "
        "have Direct Reinvestment Plan (DRIP) enabled. "
        "GMV and Risk Parity use the Elton-Gruber Constant Correlation shrinkage estimator "
        "(δ = 0.7), which blends the sample covariance matrix with a structured prior where "
        "all pairwise correlations are equal. With only 17 assets, the sample matrix is noisy "
        "and unreliable, so shrinkage stabilises it. δ = 0.7 reflects that optimal shrinkage "
        "intensity rises as the asset universe shrinks; fewer assets means less trustworthy "
        "sample estimates, so a stronger pull toward the prior is warranted.")

    st.divider()

    # ── Section 2: 1Y Correlated GBM Forecast ─────────────────────────
    st.markdown("#### 2. Correlated GBM 1Y Return Forecast")
    st.caption(
        "Uses the same paths from the Simulation & Stress Tests tab but applied "
        "to each strategy's weights. Lets you see whether BL adds value over simpler alternatives "
        "but should be read as a naive estimation of actual expected returns since it does not model "
        "correlated crashes nd volitlity clustering. "
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
        width="stretch",
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
        title="Correlated GBM: Expected 1Y Return with 95% CI",
        yaxis_tickformat=".1%",
        yaxis_title="1-Year Return",
        height=420,
        showlegend=False,
    )
    st.plotly_chart(fig_comp, width="stretch")

    # ── GARCH-Copula 1Y Return Forecast ───────────────────────────────────
    if garch_params is not None:
        with st.expander("📐 Exploratory Extension: GARCH-Copula 1Y Return Forecast", expanded=False):
            st.info(
                "**Note on this section:** The GARCH-Copula simulation was added as an exploratory "
                "extension to understand how GBM understates tail risk, and it is not part of the core "
                "EDHEC coursework. The two methodological improvements over GBM are: "
                "(1) **GARCH(1,1)** replaces the constant-volatility assumption with volatility clustering"
                "in different period windows."
                "(2) **Clayton copula** replaces the Gaussian correlation structure with one that "
                "allows correlations to spike in the lower tail such that assets crash together more "
                "while rallying independently. The copula parameter θ is fitted via Maximum Likelihood "
                "Estimation on the standardised residuals from the GARCH step. "
                "The practical result is a fatter left tail relative to GBM, visible in the 5th "
                "percentile column.",
            )
            st.caption(
                "Compare the **5th percentile column** directly with the GBM table above. "
                "The gap is the tail risk that GBM systematically misses. "
                f"MLE-fitted θ = {_theta_default:.2f}, 2,000 scenarios."
            )

            _gc_strat_names = tuple(strategy_weights.keys())
            _gc_weights_arr = np.stack([
                w.reindex(tick_rets.columns).fillna(0).values
                for w in strategy_weights.values()
            ])

            with st.spinner("Running GARCH-Copula for all strategies…"):
                _gc_strat_results = run_garch_copula_all_strategies(
                    tick_rets, garch_params, std_resids_g, last_state,
                    _gc_strat_names, _gc_weights_arr,
                    theta=_theta_default, n_scenarios=2_000, n_steps=252, seed=int(seed),
                    tickers_key=tuple(tick_rets.columns),
                )

            _gc_comparison = {}
            for _sn, _fv in _gc_strat_results.items():
                _gc_comparison[_sn] = {
                    "Expected Return": float(np.mean(_fv))            - 1,
                    "5th pct":         float(np.percentile(_fv,  5))  - 1,
                    "95th pct":        float(np.percentile(_fv, 95))  - 1,
                    "Spread (95--5)":  float(np.percentile(_fv, 95) - np.percentile(_fv, 5)),
                }

            _gc_comp_df = pd.DataFrame(_gc_comparison).T.sort_values("Expected Return", ascending=False)
            st.dataframe(
                _gc_comp_df.style
                    .format("{:.2%}", subset=["Expected Return", "5th pct", "95th pct"])
                    .format("{:.3f}", subset=["Spread (95--5)"])
                    .background_gradient(subset=["Expected Return"], cmap="YlGnBu"),
                width="stretch",
            )

            _fig_gc_comp = go.Figure()
            for _sn, _row in _gc_comparison.items():
                _c = _dotplot_colours.get(_sn, "#888888")
                _fig_gc_comp.add_trace(go.Scatter(
                    x=[_sn],
                    y=[_row["Expected Return"]],
                    mode="markers",
                    marker=dict(size=14, symbol="circle", color=_c),
                    error_y=dict(
                        type="data", symmetric=False,
                        array     =[_row["95th pct"] - _row["Expected Return"]],
                        arrayminus=[_row["Expected Return"] - _row["5th pct"]],
                        color=_c,
                    ),
                    name=_sn,
                ))
            _fig_gc_comp.update_layout(
                title=f"GARCH-Copula: Expected 1Y Return with 95% CI  (θ = {_theta_default:.1f})",
                yaxis_tickformat=".1%",
                yaxis_title="1-Year Return",
                height=420,
                showlegend=False,
            )
            st.plotly_chart(_fig_gc_comp, width="stretch")

    st.divider()

    # ── Section 4: CPPI Drawdown Protection Analysis ──────────────────────────
    st.markdown("#### 4. CPPI Drawdown Protection Analysis")

    with st.expander("⚙️ CPPI Parameters", expanded=True):
        st.caption(
            "At each step, the Constant Portfolio Protection Insurance (CPPI) allocates equity exposure equal to the multiplier times the cushion "
            "(portfolio value minus floor), with the remainder in a safe asset. "
            "The two parameters below set how aggressively equity is sized and where the floor sits."
        )
        _col_m, _col_mdd = st.columns(2)
        _col_m.slider(
            "Multiplier (m)",
            min_value=1.0, max_value=10.0, step=0.5, 
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
    _bl_w      = (1 + btr["BL (static)"]).cumprod() * _init
    _cppi_w    = (1 + btr["BL (CPPI)"]).cumprod()   * _init
    
    # Rebase floor to the aligned CPPI wealth series so all three lines
    # share the same start date and scale (floor-to-portfolio ratio is preserved)
    _floor_raw = cppi_floor.reindex(btr.index)
    _acct_raw  = cppi_account.reindex(btr.index)
    _cppi_fl_w = (_floor_raw / _acct_raw) * _cppi_w
    
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
    st.plotly_chart(fig_cppi, width="stretch")

    # ── Equity allocation over time ───────────────────────────────────────────
    #Reindexed to backtest start date with rolling window constraint, not out of necessity, but consistency.
    #BL are static weights with allocations that go back even to the start date.
    _eq_pct = (cppi_alloc.reindex(btr.index) * 100).clip(0, 100) 
    
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
    st.plotly_chart(fig_alloc, width="stretch")

    # ── Quarterly Allocation Snapshots ───────────────────────────────────────
    st.markdown("###### Quarterly Equity Allocation Snapshots")
    st.caption(
        "Quarter-end equity allocation (% of portfolio) derived from the CPPI strategy. "
        )

    _alloc_aligned  = cppi_alloc.reindex(btr.index).clip(0, 1)
    _quarterly_alloc = _alloc_aligned.resample("QE").last() * 100
    _q_labels        = _quarterly_alloc.index.to_period("Q").astype(str)

    _bar_colours = [
        "#55A868" if v >= 80 else "#fecc5c" if v >= 30 else "#C44E52"
        for v in _quarterly_alloc.values
    ]

    fig_q = go.Figure()
    fig_q.add_trace(go.Bar(
        x=_q_labels,
        y=_quarterly_alloc.values,
        marker_color=_bar_colours,
        text=[f"{v:.0f}%" for v in _quarterly_alloc.values],
        textposition="outside",
        textfont=dict(size=9),
        hovertemplate="<b>%{x}</b><br>Equity Allocation: %{y:.1f}%<extra></extra>",
    ))
    fig_q.add_hline(
        y=80, line_dash="dot",
        line_color="rgba(85, 168, 104, 0.45)",
        annotation_text="80% threshold",
        annotation_position="bottom right",
        annotation_font_size=10,
    )
    fig_q.add_hline(
        y=30, line_dash="dot",
        line_color="rgba(196, 78, 82, 0.45)",
        annotation_text="30% threshold",
        annotation_position="bottom right",
        annotation_font_size=10,
    )
    fig_q.update_layout(
        xaxis=dict(tickangle=-45, tickfont=dict(size=9)),
        yaxis=dict(title="Equity Allocation (%)", range=[0, 115]),
        height=380,
        showlegend=False,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=20, b=60),
    )
    st.plotly_chart(fig_q, width="stretch")

    # Summary counts
    _n_green  = int((_quarterly_alloc >= 80).sum())
    _n_amber  = int(((_quarterly_alloc >= 30) & (_quarterly_alloc < 80)).sum())
    _n_red    = int((_quarterly_alloc < 30).sum())
    _n_total  = len(_quarterly_alloc)

    # ── CPPI Performance During Stress Periods ────────────────────────────────
    st.markdown("###### CPPI Protection During Historical Stress Periods")
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
            "Cash Lock-in":      "🔒 Yes" if _cash_locked    else "✅ No",
        }

    _stress_cppi_df = pd.DataFrame(_stress_cppi_rows).T
    st.dataframe(
        _stress_cppi_df.style
            .format("{:.2%}", subset=["BL Return", "CPPI Return", "Return Difference",
                                      "BL Max Drawdown", "CPPI Max Drawdown", "DD Reduction"])
            .background_gradient(subset=["CPPI Max Drawdown"], cmap="Reds_r",  vmax=0.0)
            .background_gradient(subset=["BL Max Drawdown"],   cmap="Reds_r",  vmax=0.0)
            .background_gradient(subset=["DD Reduction"],      cmap="RdYlGn", vmin=-0.1, vmax=0.0),
        width="stretch",
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
        "all historical windows and is time invariant. In practice this varied significantly, going near zero during the GFC Echo (2015) "
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
    "Price targets are personal estimates and are not investment recommendations."
)
