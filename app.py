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
# AI CONFIDENCE SUGGESTIONS  (Anthropic API + web search)
# ─────────────────────────────────────────────────────────────────────────────
def fetch_ai_confidence_suggestions(tickers: list[str]) -> dict:
    """
    Calls the Anthropic API to get AI-suggested confidence scores for every
    ticker in one round-trip, using the built-in web_search tool so the model
    can ground its reasoning in current news.

    HOW THIS WORKS — a short primer for future reference
    ─────────────────────────────────────────────────────
    1. We POST to /v1/messages with:
         • model        — the Claude model to use
         • messages     — the conversation so far (just one user turn here)
         • tools        — a list of tools Claude may call; we enable web_search
         • system       — a system prompt telling Claude its role and output format

    2. The API may return a response with stop_reason = "tool_use".
       That means Claude decided to search before answering.  The response
       content is a list of blocks — some are {"type": "text"}, others are
       {"type": "tool_use"}.  We do NOT need to manually handle the tool
       results; the web_search tool is server-side, so Anthropic runs the
       searches and feeds the results back to the model automatically in a
       single HTTP response.  By the time we receive the final response
       (stop_reason = "end_turn") the model has already read the search
       results and synthesised them.

    3. We extract the text block from the final response and parse it as JSON.
       The system prompt instructs Claude to return ONLY a JSON array, which
       makes parsing straightforward.

    4. Error handling: any network hiccup, malformed JSON, or unexpected
       response shape is caught and returns an empty dict so the app degrades
       gracefully — the sliders just keep their current values.

    Returns
    -------
    dict : {ticker: {"confidence": float, "rationale": str}, ...}
           Empty dict if the call fails.
    """
    import json as _json

    # ── Prompt design ──────────────────────────────────────────────────────────
    # The system prompt defines the task and — critically — the output schema.
    # Asking for JSON-only output and specifying the schema tightly reduces
    # the chance of the model returning prose that breaks the JSON parser.
    system_prompt = """You are a quantitative equity analyst helping calibrate
Black-Litterman confidence parameters.

For each ticker the user provides, search for the most recent analyst
narratives, earnings results, and macro risks (May 2026). Then assign a
confidence score representing how certain a DCF price-target view is likely
to be correct over a 12-month horizon.

Confidence scale:
  0.10–0.29  Very low — material uncertainty (regulatory, AI disruption,
             supply chain issues, unclear competitive position)
  0.30–0.49  Low-moderate — real headwinds or mixed signals
  0.50–0.64  Moderate — clear business but meaningful risks remain
  0.65–0.79  High — strong recent results, clear narrative, manageable risks
  0.80–1.00  Very high — dominant position, exceptional visibility (rare)

IMPORTANT: Return ONLY a valid JSON array. No preamble, no markdown fences,
no trailing text. Each element must have exactly these keys:
  "ticker"     : string  — the ticker symbol
  "confidence" : float   — rounded to nearest 0.05, in [0.10, 0.90]
  "rationale"  : string  — one sentence, max 20 words, stating the key reason

Example of valid output (do not copy these values):
[
  {"ticker": "AAPL", "confidence": 0.45, "rationale": "Tariff uncertainty and mid-transition supply chain weigh on margin visibility."},
  {"ticker": "NVDA", "confidence": 0.75, "rationale": "Record revenue and strong guidance provide exceptional near-term visibility."}
]"""

    user_message = (
        f"Provide confidence scores for these tickers: {', '.join(tickers)}. "
        "Search for recent news on each before responding."
    )

    # ── API call ───────────────────────────────────────────────────────────────
    # The web_search tool type is a first-party Anthropic tool — no extra
    # configuration needed beyond declaring it in the tools list.
    # The API key is injected by the Streamlit Cloud environment; we never
    # hard-code it in source.
    try:
        resp = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "Content-Type":         "application/json",
                "anthropic-version":    "2023-06-01",
                # The beta header is required to enable built-in tools like
                # web_search on this endpoint.
                "anthropic-beta":       "interleaved-thinking-2025-05-14",
            },
            json={
                "model":      "claude-sonnet-4-20250514",
                "max_tokens": 2048,
                "system":     system_prompt,
                "tools": [
                    # Declaring the web_search tool tells Claude it may search.
                    # Anthropic runs the actual searches server-side — we never
                    # see the raw search results, only the final synthesised text.
                    {
                        "type": "web_search_20250305",
                        "name": "web_search",
                    }
                ],
                "messages": [
                    {"role": "user", "content": user_message}
                ],
            },
            timeout=60,   # web searches add latency; 60 s is generous but safe
        )
        resp.raise_for_status()
        data = resp.json()

        # ── Response parsing ───────────────────────────────────────────────────
        # The response content is a list of blocks.  We want the final text
        # block, which contains the JSON array we asked for.
        # Block types we might see:
        #   {"type": "thinking", ...}   — chain-of-thought (if beta enabled)
        #   {"type": "tool_use", ...}   — Claude is calling web_search
        #   {"type": "tool_result", ...}— search results fed back (server-side)
        #   {"type": "text", ...}       — the actual answer we want
        text_blocks = [
            b["text"] for b in data.get("content", [])
            if b.get("type") == "text"
        ]
        if not text_blocks:
            return {}

        raw_text = text_blocks[-1].strip()

        # Strip markdown code fences if the model wrapped the JSON anyway
        if raw_text.startswith("```"):
            raw_text = raw_text.split("```")[1]
            if raw_text.startswith("json"):
                raw_text = raw_text[4:]
            raw_text = raw_text.strip()

        parsed = _json.loads(raw_text)

        # Normalise into {ticker: {confidence, rationale}}
        return {
            item["ticker"]: {
                "confidence": float(item["confidence"]),
                "rationale":  str(item["rationale"]),
            }
            for item in parsed
            if "ticker" in item and "confidence" in item
        }

    except Exception as e:
        # Surface the error in session state so the UI can show it, but don't
        # crash the app — the sliders keep their current values.
        st.session_state["ai_suggestion_error"] = str(e)
        return {}

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
    "AAPL": 315.00, "ADBE": 380.00, "AMAT": 380.00,
    "AMZN": 300.00, "ASML":1791.00, "CPRT":  43.00,
    "FICO":1655.00, "GOOGL":460.00, "LRCX": 310.00,
    "MA":   650.00, "META": 810.00, "MSCI": 700.00,
    "MSFT": 600.00, "NFLX": 117.00, "NVDA": 275.00,
    "TSM":  465.00, "V":    394.00,
}

# Bear = 20 % below base;  Bull = 25 % above base  (reserved for future DCF scenario toggle)

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
def load_market_data(tickers, start="2000-01-01"):
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
            "This is usually a Yahoo Finance rate-limit — refresh in a minute to retry.",
        )
        price_data = price_data.drop(columns=failed)

    # Forward-fill intra-series gaps, then use dropna(how="all") so a single
    # missing ticker on one day does not wipe out the entire returns history.
    price_data = price_data.ffill()
    tick_rets  = price_data.pct_change().dropna(how="all")
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
# Load consensus data before entering the sidebar block so it's available
# when rendering per-ticker inputs.  TTL=24 h keeps it fresh without
# hammering the API on every widget interaction.
# Price data is loaded here (before the sidebar) so load_ticker_metadata
# can use it to build cap-weight DataFrames in the same pass.
price_data, tick_rets = load_market_data(TICKERS)
tick_capweights, consensus_data, recent_earnings = load_ticker_metadata(price_data, TICKERS)

# Hard stop: if every single ticker failed (total rate-limit), nothing works downstream.
if tick_rets.empty:
    st.error(
        "❌ **All price data downloads failed.** "
        "Yahoo Finance is likely rate-limiting the Streamlit Cloud shared IP. "
        "Wait 60 seconds and refresh the page."
    )
    st.stop()

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

    # --- Position size constraints ---
    st.subheader("2. Position Size Constraints")
    st.caption(
        "**Max position** caps the optimiser from piling into a single stock — "
        "a common failure mode of unconstrained mean-variance. "
        "Industry standard for concentrated active funds is **5–15%**; "
        "diversified funds typically use **3–5%**. "
        "**Min position** prevents the optimiser assigning noise-level weights "
        "that would be uneconomic to trade — **1–2%** is typical. "
        "Set min to 0 to allow zero-weight positions."
    )
    min_w_pct = st.slider(
        "Min position size (%)",
        min_value=0, max_value=10, value=1, step=1,
        help="Floor on any single stock weight. 0 = allow the optimiser to zero out a stock.",
    )
    max_w_pct = st.slider(
        "Max position size (%)",
        min_value=5, max_value=100, value=15, step=1,
        help="Ceiling on any single stock weight. 15% is a reasonable starting point for a 17-stock portfolio.",
    )
    min_weight = min_w_pct / 100
    max_weight = max_w_pct / 100

    st.divider()

    st.divider()

    # --- Per-stock price targets ---
    st.subheader("3. Price Targets & Confidence")
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

    # ── AI Confidence Suggestions ─────────────────────────────────────────────
    st.divider()
    st.subheader("🤖 AI Confidence Review")
    st.caption(
        "Calls Claude with web search to scan the latest analyst narratives "
        "and earnings for each stock, then suggests updated confidence scores. "
        "Review the rationale before applying — this is a starting point, "
        "not a replacement for judgement."
    )

    # HOW SESSION STATE WORKS HERE
    # ────────────────────────────
    # Streamlit reruns the entire script on every interaction.  To persist
    # data across reruns (like API results), we use st.session_state — a
    # dict that survives reruns within a session.
    #
    # Pattern used here:
    #   1. Button click → store API results in st.session_state["ai_suggestions"]
    #   2. On the next rerun, the results are still there → display them
    #   3. "Apply" button → write each confidence value into
    #      st.session_state[f"conf_{ticker}"]  (the slider's key)
    #   4. Next rerun → sliders initialise from session_state → values updated

    if st.button("🔍 Fetch AI Confidence Suggestions", use_container_width=True):
        # Clear any previous error before a fresh attempt
        st.session_state.pop("ai_suggestion_error", None)
        with st.spinner("Searching latest narratives for all tickers…"):
            results = fetch_ai_confidence_suggestions(TICKERS)
            st.session_state["ai_suggestions"] = results

    # Show any API error that was caught inside the function
    if "ai_suggestion_error" in st.session_state:
        st.error(
            f"API call failed: {st.session_state['ai_suggestion_error']}  \n"
            "Check that ANTHROPIC_API_KEY is set in Streamlit Cloud secrets."
        )

    # Display results table if suggestions exist
    if st.session_state.get("ai_suggestions"):
        sug = st.session_state["ai_suggestions"]

        # Build a comparison DataFrame: current vs suggested
        rows = []
        for t in TICKERS:
            if t in sug:
                rows.append({
                    "Ticker":    t,
                    "Current":   st.session_state.get(f"conf_{t}", BASE_CONFIDENCE.get(t, 0.5)),
                    "Suggested": sug[t]["confidence"],
                    "Rationale": sug[t]["rationale"],
                })
        if rows:
            sug_df = pd.DataFrame(rows).set_index("Ticker")

            # Colour the Suggested column: green if higher than current,
            # red if lower, grey if same — gives instant visual signal
            def colour_delta(row):
                delta = row["Suggested"] - row["Current"]
                if delta > 0.01:
                    colour = "#d4edda"   # light green
                elif delta < -0.01:
                    colour = "#f8d7da"   # light red
                else:
                    colour = ""
                return ["", f"background-color: {colour}" if colour else "", f"background-color: {colour}" if colour else "", ""]

            st.dataframe(
                sug_df.style
                    .apply(colour_delta, axis=1)
                    .format("{:.2f}", subset=["Current", "Suggested"]),
                use_container_width=True,
                height=min(50 + 35 * len(rows), 400),
            )

            # Apply button: writes each suggested value into the slider's
            # session_state key so sliders update on the next rerun
            if st.button("✅ Apply AI Suggestions to Sliders", use_container_width=True):
                for t in TICKERS:
                    if t in sug:
                        # Round to nearest 0.05 to match slider step
                        rounded = round(sug[t]["confidence"] / 0.05) * 0.05
                        st.session_state[f"conf_{t}"] = float(rounded)
                st.success("Confidence sliders updated. Scroll up to review.")
                st.rerun()

    st.divider()

    # --- Backtest estimation window ---
    st.subheader("4. Backtest Estimation Window")
    st.caption(
        "Controls how many years of historical data are used to estimate "
        "covariance and weights at each rolling rebalance step. "
        "**2 years** is responsive but noisy. "
        "**3 years** captures a full market cycle and is the recommended default. "
        "**5 years** is most stable but may be stale for a sector that re-priced "
        "structurally in 2023. Note: a longer window delays the backtest start date "
        "by the same amount — the first rebalance can only happen once a full "
        "window of data is available."
    )
    estimation_window_yrs = st.slider(
        "Estimation window (years)",
        min_value=1, max_value=7, value=3, step=1,
        help="1 year = 252 trading days. Feeds into EW, CW, GMV, and Risk Parity rolling backtests.",
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
    f"Risk-free rate (3M T-bill): **{RF:.2%}**  |  "
    f"Universe: **{len(TICKERS)} stocks**  |  "
    f"Data range: **{tick_rets.index[0].date()} → {tick_rets.index[-1].date()}**"
)

# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────
tab0, tab1, tab2, tab3, tab4 = st.tabs([
    "📖 Introduction",
    "📋 Views & Returns",
    "⚖️ BL Weights",
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
        I built this app to implement the skills I acquired from EDHEC's Advanced
        Portfolio Construction and Wall Street Prep's DCF modelling course. It answers the
        following question: After calculating the fair value of a stock price through DCF
        analysis, how does one implement this knowledge to size their portfolios?

        Most backtested strategies like Global Minimum Variance (GMV) or Risk Parity
        are purely backward-looking, optimising on historical data and assuming the past repeats.
        Black-Litterman is different: it takes a forward-looking view on what each stock is worth
        and asks how much conviction you should act on, relative to the market implied returns.
        The app lets you build that allocation, stress-test it against real market crashes,
        and benchmark it against the simpler strategies.

        This project is still a work in progress and I plan to incorporate more
        robust DCF assumptions as inputs to experiment with the resulting allocations.
        """
    )

    st.divider()

    st.markdown("#### Stock Universe & Selection Criteria")
    st.markdown(
        """
        The 17 stocks in this portfolio were selected through a systematic
        fundamental screen: 5-year average ROIC above 15%, debt-to-equity
        below 1, consecutive revenue growth over 5 years, and a minimum
        market cap of $10 billion. ROIC was chosen as the primary quality
        filter because it measures how efficiently a company converts capital
        into profit — sustained high ROIC over multiple years is one of the
        most reliable indicators of a durable competitive advantage.

        The resulting universe is concentrated in technology, semiconductors,
        payments infrastructure, and financial data — sectors where
        capital-light business models and high switching costs tend to produce
        the kind of persistent ROIC that justifies long-term holding. Two names
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
        unstable allocations — 80% in one stock, zero in everything else. They treat
        every input as gospel and optimise aggressively on noise.

        **Black-Litterman solves this by never letting a view stand alone.**
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
# TAB 1 -- Views & Returns Decomposition
# ══════════════════════════════════════════════════════════════════════════════
with tab1:

    st.markdown("#### Return Decomposition")
    st.caption(
        "Each column is one layer of the BL process. "
        "**Total Return View** is the raw DCF-implied return. "
        "**Q** subtracts the risk-free rate to get the excess return fed into the model. "
        "**π** is the market equilibrium baseline. "
        "**BL Posterior** is the blended output — the return the optimiser actually uses."
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
        .background_gradient(subset=["BL Posterior Return"], cmap="RdYlGn")

    st.dataframe(styled, use_container_width=True)

    # Bar chart: Q vs pi vs BL posterior
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
    st.markdown("#### Black-Litterman Optimal Weights (Long-Only Max Sharpe)")
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
    st.markdown("#### Correlated GBM Monte Carlo (BL Drift)")
    st.caption(
        "This section stress tests the BL weights using (1) a correlated GBM monte carlo simulation, and (2) a historic stress test to backtest against historical shocks. "
        "For the GBM, each path simulates one year of daily returns. "
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
        above is best read as a baseline of what *typical* uncertainty looks like and not as a
        statement about tail risk. The historic stress tests below ground that picture in what actually
        happened holding the BL portfolio.
        """
    )

    st.divider()

    # ── Historical Stress Tests ───────────────────────────────────────────────
    st.markdown("#### Historical Stress Test — BL Weights")
    st.caption(
        "Applies the BL optimal weights to *actual* historical returns during "
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
    st.markdown("#### Historical Wealth Index — Strategy Comparison")
    st.caption(
        f"Rolling backtest using a **{estimation_window_yrs}-year** estimation window to rebalance weights at each step. "
        "EW, Cap-Weighted, GMV, and Risk Parity are properly rolled — weights are re-estimated "
        "each period using only data available at that point in time, so there is no look-ahead. "
        "**Black-Litterman** is shown as a static allocation using the current optimal "
        "weights applied to the full history. This is a simplification: true BL weights would "
        "require a fresh set of views at every rebalance date. The chart is therefore best read "
        "as 'how would this portfolio have held up' rather than a like-for-like backtest."
    )

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
        f"Starts {estimation_window_yrs} year(s) after data begins — the minimum needed for the first rolling estimate."
    )

    wealth = (1 + btr).cumprod() * 10_000

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
        title="Wealth Index — $10,000 invested at backtest start",
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
            .format("{:.2f}", subset=["Sharpe Ratio"]),
        use_container_width=True,
    )

    st.divider()

    # ── Section 2: Forward Distribution (Monte Carlo) ─────────────────────────
    st.markdown("#### Forward Distribution — Strategy Comparison (Monte Carlo)")
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
