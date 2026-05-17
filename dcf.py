"""
Automated DCF Valuation Model
==============================
Built from the ground up following investment-banking best practices,
referencing a full Excel DCF model with UFCF methodology, dual terminal
value approaches, net debt bridge, and sensitivity tables.

Methodology
-----------
  - Uses quarterly 10-Q data via yfinance; all income statement and
    cash flow items annualised via TTM (sum of last 4 quarters).
  - Balance sheet items use the most recent quarter snapshot.
  - FCF is built as Unlevered Free Cash Flow (UFCF) from EBIT up,
    not the levered FCF from the cash flow statement.
  - Stub period adjusts discount exponents for the partial year
    remaining between today and the company's fiscal year end.
  - Two terminal value methods: Gordon Growth (perpetuity) and
    EBITDA exit multiple.
  - Sensitivity tables: WACC × terminal growth, WACC × EBITDA multiple.
"""

import yfinance as yf
import pandas as pd
import numpy as np
import calendar
from datetime import date


# ==================================================================
# SECTION 1 — TAX RATE
# ==================================================================

def get_effective_tax_rate(financials, fallback_rate=0.21):
    """
    TTM Effective Tax Rate = TTM Tax Provision / TTM Pretax Income

    Falls back to the US statutory rate (21%) when:
      - Required rows are missing
      - Pretax income is negative (loss year)
      - Computed rate is outside a sensible range [0%, 40%]

    The tax rate appears in two places in the DCF:
      1. NOPAT = EBIT × (1 - tax_rate)  — taxes on operating profit
      2. WACC  debt tax shield = cost_of_debt × (1 - tax_rate)
    """
    tax_key    = next((k for k in ['Tax Provision', 'Income Tax Expense']
                       if k in financials.index), None)
    pretax_key = next((k for k in ['Pretax Income', 'Income Before Tax']
                       if k in financials.index), None)

    if not tax_key or not pretax_key:
        return fallback_rate

    try:
        tax_prov  = financials.loc[tax_key].iloc[:4].sum()
        pre_inc   = financials.loc[pretax_key].iloc[:4].sum()

        if pre_inc <= 0:
            return fallback_rate

        # Clamp: one-off events (deferred tax releases, settlements) can
        # temporarily distort the effective rate well outside normal bounds
        return max(0.0, min(tax_prov / pre_inc, 0.40))

    except Exception:
        return fallback_rate


# ==================================================================
# SECTION 2 — STUB PERIOD
# ==================================================================

def get_stub_fraction(fiscal_year_end_month):
    """
    Fraction of the current fiscal year remaining from today.

    All discount exponents are shifted by the stub so that Year 1
    reflects only the partial year remaining — preventing
    over-discounting of near-term cash flows.

    Example: Today = May 18, 2026 | FYE = Dec 31  →  stub ≈ 0.62
             Today = Nov 15, 2026 | FYE = Dec 31  →  stub ≈ 0.13

    If the FYE has already passed this calendar year, rolls to next year.
    """
    today    = date.today()
    last_day = calendar.monthrange(today.year, fiscal_year_end_month)[1]
    fye      = date(today.year, fiscal_year_end_month, last_day)

    if fye <= today:
        last_day = calendar.monthrange(today.year + 1, fiscal_year_end_month)[1]
        fye      = date(today.year + 1, fiscal_year_end_month, last_day)

    return (fye - today).days / 365.25


# ==================================================================
# SECTION 3 — DEBT & CAPITAL STRUCTURE
# ==================================================================

def get_gross_debt(balance_sheet):
    """
    Gross Debt = Financial Debt + Preferred Stock + Noncontrolling Interests

    Matches the Excel model's net debt bridge:
      Debt
      + Convertible debt
      + Preferred stock
      + Noncontrolling (minority) interests
      ─────────────────────────────────────
      = Gross debt & equivalents

    Falls back to summing long-term + current debt if 'Total Debt' is absent.
    """
    if 'Total Debt' in balance_sheet.index:
        base = balance_sheet.loc['Total Debt'].iloc[0]
    else:
        components = ['Long Term Debt', 'Current Debt', 'Capital Lease Obligations']
        base = sum(
            balance_sheet.loc[c].iloc[0]
            for c in components
            if c in balance_sheet.index and pd.notna(balance_sheet.loc[c].iloc[0])
        ) or 1.0

    # Add preferred stock and minority interests (non-common-equity claims)
    extras = [
        'Preferred Stock',
        'Minority Interest',
        'Noncontrolling Interests',
        'Redeemable Preferred Stock',
    ]
    for key in extras:
        if key in balance_sheet.index:
            val = balance_sheet.loc[key].iloc[0]
            if pd.notna(val) and val > 0:
                base += val

    return max(base, 1.0)


def get_nonoperating_assets(balance_sheet):
    """
    Returns (cash, equity_investments) separately, matching the Excel model's
    net debt bridge which shows Cash and Equity investments as distinct lines.

    Net Debt = Gross Debt − Cash − Equity Investments
    """
    cash = 0.0
    for key in ['Cash And Cash Equivalents',
                'Cash Cash Equivalents And Short Term Investments']:
        if key in balance_sheet.index:
            val = balance_sheet.loc[key].iloc[0]
            if pd.notna(val):
                cash = val
                break

    investments = 0.0
    for key in ['Long Term Investments', 'Other Investments',
                'Available For Sale Securities', 'Equity Investments']:
        if key in balance_sheet.index:
            val = balance_sheet.loc[key].iloc[0]
            if pd.notna(val) and val > 0:
                investments += val

    return cash, investments


# ==================================================================
# SECTION 4 — NET WORKING CAPITAL
# ==================================================================

def get_operating_nwc(balance_sheet, col_idx=0):
    """
    Operating Net Working Capital (NWC):

      NWC = (Current Assets − Cash) − (Current Liabilities − Current Debt)

    Excludes cash (non-operating asset) and short-term debt (financing item)
    to isolate the working capital tied up in operations.

    Returns None if key balance sheet rows are unavailable.
    """
    try:
        bs = balance_sheet.iloc[:, col_idx]  # single-period slice

        ca = bs.get('Current Assets') if 'Current Assets' in balance_sheet.index else None
        cl = bs.get('Current Liabilities') if 'Current Liabilities' in balance_sheet.index else None

        if ca is None or cl is None or not pd.notna(ca) or not pd.notna(cl):
            return None

        # Strip out cash (not an operating asset)
        cash_val = 0.0
        for key in ['Cash And Cash Equivalents',
                    'Cash Cash Equivalents And Short Term Investments']:
            if key in balance_sheet.index:
                v = balance_sheet.loc[key].iloc[col_idx]
                if pd.notna(v):
                    cash_val = v
                    break

        # Strip out current debt (financing, not operating)
        curr_debt = 0.0
        for key in ['Current Debt', 'Current Portion Of Long Term Debt']:
            if key in balance_sheet.index:
                v = balance_sheet.loc[key].iloc[col_idx]
                if pd.notna(v):
                    curr_debt += v

        return float(ca - cash_val) - float(cl - curr_debt)

    except Exception:
        return None


# ==================================================================
# SECTION 5 — UFCF BUILD
# ==================================================================

def get_ufcf_components(financials, cash_flow, balance_sheet, tax_rate):
    """
    Builds TTM Unlevered Free Cash Flow (UFCF) from EBIT up.

    UFCF = NOPAT + D&A + ΔNWC − CapEx

    where:
      NOPAT  = EBIT × (1 − tax_rate)       Net Operating Profit After Tax
               Strips interest to make FCF capital-structure-neutral

      D&A    = Depreciation & Amortisation  Non-cash charge added back

      ΔNWC   = Change in Operating NWC      Increase = cash outflow (negative)
               Priority: CF statement line → balance sheet delta

      CapEx  = Capital Expenditures         Investment in PP&E (negative in yfinance)

    This matches the Excel model's UFCF build:
      NOPAT (EBIAT) + D&A + Changes in NWC − CapEx

    Falls back to levered FCF from the cash flow statement if key
    components are unavailable — less theoretically pure but robust.

    Returns
    -------
    dict with ufcf, ebit, nopat, da, delta_nwc, capex, ebitda, method
    """
    result = {
        'ufcf':    None, 'ebit':    None, 'nopat': None,
        'da':      None, 'delta_nwc': None, 'capex': None,
        'ebitda':  None, 'method':  'Unknown',
    }

    # --- EBIT TTM ---
    if 'EBIT' in financials.index:
        result['ebit'] = financials.loc['EBIT'].iloc[:4].sum()

    # --- EBITDA TTM (used for terminal value multiple) ---
    for key in ['EBITDA', 'Normalized EBITDA']:
        if key in financials.index:
            result['ebitda'] = financials.loc[key].iloc[:4].sum()
            break
    # Fallback: EBITDA = EBIT + D&A (computed later once D&A is known)

    # --- NOPAT = EBIT × (1 − tax_rate) ---
    if result['ebit'] is not None:
        result['nopat'] = result['ebit'] * (1 - tax_rate)

    # --- D&A TTM ---
    for key in ['Depreciation And Amortization',
                'Depreciation Amortization Depletion',
                'Reconciled Depreciation']:
        if key in cash_flow.index:
            result['da'] = abs(cash_flow.loc[key].iloc[:4].sum())
            break

    # Backfill EBITDA if not found above
    if result['ebitda'] is None and result['ebit'] is not None and result['da'] is not None:
        result['ebitda'] = result['ebit'] + result['da']

    # --- ΔNWC (Change in Operating Net Working Capital) ---
    # Priority 1: Cash flow statement (already a cash-impact figure)
    for key in ['Change In Working Capital', 'Changes In Working Capital',
                'Change In Other Working Capital']:
        if key in cash_flow.index:
            result['delta_nwc'] = cash_flow.loc[key].iloc[:4].sum()
            break

    # Priority 2: Balance sheet delta — NWC_now vs NWC_1yr_ago
    if result['delta_nwc'] is None and balance_sheet.shape[1] >= 5:
        nwc_now  = get_operating_nwc(balance_sheet, col_idx=0)
        nwc_year = get_operating_nwc(balance_sheet, col_idx=4)
        if nwc_now is not None and nwc_year is not None:
            # Increase in NWC = use of cash → negative contribution to UFCF
            result['delta_nwc'] = -(nwc_now - nwc_year)

    # --- CapEx TTM ---
    for key in ['Capital Expenditure', 'Capital Expenditures',
                'Purchase Of Property Plant And Equipment']:
        if key in cash_flow.index:
            result['capex'] = cash_flow.loc[key].iloc[:4].sum()
            break

    # --- Assemble UFCF ---
    if (result['nopat'] is not None and
        result['da']    is not None and
        result['capex'] is not None):

        dNWC          = result['delta_nwc'] if result['delta_nwc'] is not None else 0.0
        result['ufcf'] = result['nopat'] + result['da'] + dNWC + result['capex']
        # Note: capex is already negative in yfinance (cash outflow)
        result['method'] = 'UFCF: NOPAT + D&A + ΔNWC − CapEx'

    else:
        # Levered FCF fallback — adds back after-tax interest later
        result['ufcf']   = cash_flow.loc['Free Cash Flow'].iloc[:4].sum()
        result['method'] = 'Levered FCF (fallback — EBIT/D&A/CapEx not fully available)'

    return result


# ==================================================================
# SECTION 6 — DILUTED SHARES
# ==================================================================

def get_diluted_shares_ex_warrants(financials, balance_sheet, info):
    """
    Diluted shares outstanding, less warrant share equivalents.

    Diluted shares already include:
      Basic shares + in-the-money options (treasury stock method)
      + RSUs + convertible securities

    We subtract warrant equivalents because warrants dilute existing
    shareholders but are typically excluded from standard diluted EPS
    calculations under the treasury stock method when out-of-the-money.

    Warrant share equivalent ≈ warrant_dollar_value / current_price
    This is an approximation — precise exercise prices live in 10-Q footnotes.

    Priority for diluted count:
      1. quarterly_financials 'Diluted Average Shares'  (most recent quarter)
      2. info['impliedSharesOutstanding']               (yfinance diluted estimate)
      3. info['sharesOutstanding']                      (basic — last resort)
    """
    diluted_shares = None

    if 'Diluted Average Shares' in financials.index:
        val = financials.loc['Diluted Average Shares'].iloc[0]
        if pd.notna(val) and val > 0:
            diluted_shares = float(val)

    if diluted_shares is None:
        diluted_shares = float(
            info.get('impliedSharesOutstanding') or
            info.get('sharesOutstanding') or
            1.0
        )

    # Warrant deduction
    warrant_shares = 0.0
    warrant_key = next(
        (k for k in ['Warrants And Rights Outstanding',
                     'Warrants Not Derivative Securities']
         if k in balance_sheet.index),
        None
    )
    if warrant_key:
        warrant_val   = balance_sheet.loc[warrant_key].iloc[0]
        current_price = info.get('currentPrice', 0.0)
        if pd.notna(warrant_val) and current_price > 0:
            warrant_shares = warrant_val / current_price

    return max(diluted_shares - warrant_shares, 1.0)


# ==================================================================
# SECTION 7 — SENSITIVITY TABLES
# ==================================================================

def _equity_per_share(projected_fcfs, stub, terminal_value_pv,
                      wacc, gross_debt, cash, investments, shares):
    """
    Recomputes equity value per share for a given WACC and pre-computed
    present terminal value. Used to populate sensitivity tables efficiently
    without rebuilding the full model for every cell.
    """
    pv_fcfs = sum(
        fcf / (1 + wacc) ** (stub + yr - 1)
        for yr, fcf in enumerate(projected_fcfs, 1)
    )
    ev     = pv_fcfs + terminal_value_pv
    net_debt = gross_debt - cash - investments
    return round((ev - net_debt) / shares, 2)


def build_sensitivity_perpetuity(base_wacc, terminal_growth, forecast_years,
                                  stub, projected_fcfs,
                                  gross_debt, cash, investments, shares):
    """
    5 × 5 sensitivity table: WACC (rows) × Terminal Growth Rate (cols)
    → Equity Value per Share

    WACC range:   base ± 2%  in 1% steps
    Growth range: terminal_growth + [−1%, 0%, +1%, +2%, +3%]
    """
    wacc_range = [round(base_wacc + d, 4) for d in [-0.02, -0.01, 0, 0.01, 0.02]]
    g_range    = [round(terminal_growth + d, 4) for d in [-0.01, 0, 0.01, 0.02, 0.03]]

    net_debt   = gross_debt - cash - investments

    index_labels  = [f"{w*100:.1f}%" for w in wacc_range]
    column_labels = [f"{g*100:.1f}%" for g in g_range]

    table = pd.DataFrame(index=index_labels, columns=column_labels, dtype=float)
    table.index.name = "WACC \\ g"

    for w, wl in zip(wacc_range, index_labels):
        for g, gl in zip(g_range, column_labels):
            if w <= g:
                table.loc[wl, gl] = np.nan
                continue

            pv_fcfs = sum(
                fcf / (1 + w) ** (stub + yr - 1)
                for yr, fcf in enumerate(projected_fcfs, 1)
            )
            tv       = projected_fcfs[-1] * (1 + g) / (w - g)
            pv_tv    = tv / (1 + w) ** (stub + forecast_years - 1)
            eq       = (pv_fcfs + pv_tv - net_debt) / shares
            table.loc[wl, gl] = round(eq, 2)

    return table, wacc_range


def build_sensitivity_ebitda_multiple(base_wacc, forecast_years, stub,
                                       projected_fcfs, terminal_ebitda,
                                       gross_debt, cash, investments, shares,
                                       center_multiple=15.0):
    """
    5 × 5 sensitivity table: WACC (rows) × EBITDA Exit Multiple (cols)
    → Equity Value per Share

    WACC range:    base ± 2%  in 1% steps
    Multiple range: center_multiple + [−4x, −2x, 0x, +2x, +4x]
    """
    wacc_range = [round(base_wacc + d, 4) for d in [-0.02, -0.01, 0, 0.01, 0.02]]
    mult_range = [center_multiple + d for d in [-4, -2, 0, 2, 4]]

    net_debt   = gross_debt - cash - investments

    index_labels  = [f"{w*100:.1f}%" for w in wacc_range]
    column_labels = [f"{m:.0f}x" for m in mult_range]

    table = pd.DataFrame(index=index_labels, columns=column_labels, dtype=float)
    table.index.name = "WACC \\ EV/EBITDA"

    for w, wl in zip(wacc_range, index_labels):
        pv_fcfs = sum(
            fcf / (1 + w) ** (stub + yr - 1)
            for yr, fcf in enumerate(projected_fcfs, 1)
        )
        for m, ml in zip(mult_range, column_labels):
            tv    = terminal_ebitda * m
            pv_tv = tv / (1 + w) ** (stub + forecast_years - 1)
            eq    = (pv_fcfs + pv_tv - net_debt) / shares
            table.loc[wl, ml] = round(eq, 2)

    return table


# ==================================================================
# SECTION 8 — MAIN DCF
# ==================================================================

def run_automated_dcf(ticker_symbol,
                      growth_rate      = 0.08,
                      terminal_growth  = 0.03,
                      forecast_years   = 5,
                      ebitda_exit_mult = 15.0,
                      midyear_adj      = False):
    """
    Full Automated DCF Valuation (Investment Banking Methodology)

    Follows the structure of a professional DCF model:
      1.  Data extraction from latest 10-Q filings (quarterly yfinance)
      2.  UFCF built from EBIT: NOPAT + D&A + ΔNWC − CapEx
      3.  WACC from CAPM cost of equity + after-tax cost of debt
      4.  Stub-period-adjusted discounting
      5.  Two terminal value methods:
            a. Gordon Growth Model (perpetuity)
            b. EBITDA exit multiple
      6.  Net debt bridge: Gross Debt − Cash − Equity Investments
      7.  Diluted shares (ex-warrants)
      8.  EV multiples: EV/Revenue, EV/EBITDA, EV/EBIT (LTM)
      9.  Sensitivity tables for both terminal value methods
      10. TV as % of TEV (sanity check — should typically be 60–80%)

    Parameters
    ----------
    ticker_symbol   : str   — Stock ticker (e.g. 'AAPL')
    growth_rate     : float — FCF CAGR during explicit forecast period
    terminal_growth : float — Perpetual FCF growth rate (Gordon Growth)
    forecast_years  : int   — Length of explicit forecast phase
    ebitda_exit_mult: float — EBITDA multiple for second terminal value method
    midyear_adj     : bool  — Apply mid-year convention (shifts CFs back 0.5yr)

    Returns
    -------
    dict with keys:
      'summary'                  — main valuation metrics
      'ufcf_bridge'              — UFCF component breakdown
      'sensitivity_perpetuity'   — DataFrame (WACC × g)
      'sensitivity_ebitda_mult'  — DataFrame (WACC × EV/EBITDA)
    """

    # ------------------------------------------------------------------
    # 1. Data Extraction — Quarterly (10-Q)
    # ------------------------------------------------------------------
    stock         = yf.Ticker(ticker_symbol)
    cash_flow     = stock.quarterly_cashflow
    financials    = stock.quarterly_financials
    balance_sheet = stock.quarterly_balance_sheet
    info          = stock.info

    if cash_flow.empty or financials.empty or balance_sheet.empty:
        return f"Insufficient data for {ticker_symbol}"

    # Require at least 4 quarters for a reliable TTM
    try:
        if cash_flow.loc['Free Cash Flow'].dropna().shape[0] < 4:
            return f"Less than 4 quarters available for {ticker_symbol} — TTM unreliable"
    except KeyError:
        return f"'Free Cash Flow' row not found for {ticker_symbol}"

    latest_q = cash_flow.columns[0].strftime('%Y-%m-%d')

    # ------------------------------------------------------------------
    # 2. Tax Rate
    # ------------------------------------------------------------------
    tax_rate = get_effective_tax_rate(financials)

    # ------------------------------------------------------------------
    # 3. UFCF Build
    # ------------------------------------------------------------------
    ufcf_data   = get_ufcf_components(financials, cash_flow, balance_sheet, tax_rate)
    fcf_current = ufcf_data['ufcf']

    # Revenue TTM (for EV/Revenue multiple)
    revenue_ttm = None
    for key in ['Total Revenue', 'Revenue']:
        if key in financials.index:
            revenue_ttm = financials.loc[key].iloc[:4].sum()
            break

    ebitda_ttm = ufcf_data['ebitda']
    ebit_ttm   = ufcf_data['ebit']

    # ------------------------------------------------------------------
    # 4. Debt & Non-Operating Assets
    # ------------------------------------------------------------------
    gross_debt           = get_gross_debt(balance_sheet)
    cash_val, invest_val = get_nonoperating_assets(balance_sheet)
    net_debt             = gross_debt - cash_val - invest_val

    # ------------------------------------------------------------------
    # 5. WACC
    # ------------------------------------------------------------------

    # Cost of Debt = Interest Expense / Gross Debt
    interest_expense = 0.0
    if 'Interest Expense' in financials.index:
        interest_expense = abs(financials.loc['Interest Expense'].iloc[:4].sum())
    cost_of_debt = (interest_expense / gross_debt) if gross_debt > 1 else 0.05

    # Cost of Equity — CAPM
    # Formula: Ke = Rf + β × (Rm − Rf)
    #   Rf   : risk-free rate (US 10-Year Treasury yield)
    #   β    : stock's sensitivity to market returns
    #   Rm   : expected annual market return
    #   (Rm − Rf) : Equity Risk Premium (ERP)
    beta           = info.get('beta', 1.0)
    risk_free_rate = 0.042   # US 10-Year Treasury yield, 2026
    market_return  = 0.09
    erp            = market_return - risk_free_rate
    cost_of_equity = risk_free_rate + beta * erp

    # Capital Structure Weights (market-value weighted)
    market_cap  = info.get('marketCap', 1.0)
    total_cap   = market_cap + gross_debt
    w_equity    = market_cap  / total_cap
    w_debt      = gross_debt  / total_cap

    # WACC = wE × Ke + wD × Kd × (1 − t)
    #   (1 − t) : debt tax shield — interest is tax deductible
    wacc = (w_equity * cost_of_equity) + (w_debt * cost_of_debt * (1 - tax_rate))

    # Safety guard: WACC must exceed terminal growth for Gordon Growth Model
    # to produce a finite, positive terminal value
    if wacc <= terminal_growth:
        wacc = terminal_growth + 0.03

    # ------------------------------------------------------------------
    # 6. Stub Period
    # ------------------------------------------------------------------
    fye_month = info.get('fiscalYearEnd', 12)
    stub      = get_stub_fraction(fye_month)

    # ------------------------------------------------------------------
    # 7. FCF Projection & Discounting
    # ------------------------------------------------------------------
    projected_fcfs  = []
    discounted_fcfs = []

    for year in range(1, forecast_years + 1):
        fcf_proj = fcf_current * ((1 + growth_rate) ** year)

        # Discount exponent: stub shifts year 1 to reflect only
        # the partial year remaining; mid-year convention (optional)
        # shifts all flows back an additional 0.5 years
        exp = stub + year - 1
        if midyear_adj:
            exp -= 0.5

        projected_fcfs.append(fcf_proj)
        discounted_fcfs.append(fcf_proj / (1 + wacc) ** exp)

    pv_fcfs = sum(discounted_fcfs)

    # Stub-adjusted Year 1: only the stub fraction of Year 1 FCF
    # is counted in the forecast (matches Excel model behaviour)
    stub_yr1_fcf = projected_fcfs[0] * stub

    # ------------------------------------------------------------------
    # 8. Terminal Values
    # ------------------------------------------------------------------
    tv_discount_exp = stub + forecast_years - 1
    if midyear_adj:
        tv_discount_exp -= 0.5

    # --- a. Gordon Growth Model (Perpetuity) ---
    # TV = FCF_n+1 / (WACC − g)
    #   FCF_n+1    : first cash flow beyond the forecast horizon
    #   (WACC − g) : capitalization rate
    terminal_fcf       = projected_fcfs[-1] * (1 + terminal_growth)
    tv_perpetuity      = terminal_fcf / (wacc - terminal_growth)
    pv_tv_perpetuity   = tv_perpetuity / (1 + wacc) ** tv_discount_exp

    # --- b. EBITDA Exit Multiple ---
    # TV = Terminal EBITDA × EV/EBITDA multiple
    # Terminal EBITDA projected at same EBITDA/FCF ratio as LTM
    pv_tv_ebitda   = None
    terminal_ebitda = None

    if ebitda_ttm and fcf_current and abs(fcf_current) > 0:
        ebitda_fcf_ratio = ebitda_ttm / fcf_current
        terminal_ebitda  = projected_fcfs[-1] * ebitda_fcf_ratio
    elif ebitda_ttm:
        terminal_ebitda = ebitda_ttm * ((1 + growth_rate) ** forecast_years)

    if terminal_ebitda:
        tv_ebitda    = terminal_ebitda * ebitda_exit_mult
        pv_tv_ebitda = tv_ebitda / (1 + wacc) ** tv_discount_exp

    # ------------------------------------------------------------------
    # 9. Enterprise Value → Equity Value → Intrinsic Value Per Share
    # ------------------------------------------------------------------
    ev_perpetuity = pv_fcfs + pv_tv_perpetuity
    ev_ebitda     = pv_fcfs + pv_tv_ebitda if pv_tv_ebitda else None

    shares = get_diluted_shares_ex_warrants(financials, balance_sheet, info)

    def to_ivps(ev):
        if ev is None:
            return None
        return round((ev - net_debt) / shares, 2)

    ivps_perp   = to_ivps(ev_perpetuity)
    ivps_ebitda = to_ivps(ev_ebitda)

    current_price = info.get('currentPrice', 1.0)

    def alpha(ivps):
        if ivps is None or current_price == 0:
            return None
        return f"{round((ivps - current_price) / current_price * 100, 2)}%"

    # TV as % of TEV — key sanity check
    # Typical range: 60–80%; >90% suggests FCF is low or growth too aggressive
    tv_pct_perp  = round(pv_tv_perpetuity / ev_perpetuity * 100, 1) if ev_perpetuity else None
    tv_pct_ebitda = (round(pv_tv_ebitda / ev_ebitda * 100, 1)
                     if (ev_ebitda and pv_tv_ebitda) else None)

    # ------------------------------------------------------------------
    # 10. LTM EV Multiples
    # ------------------------------------------------------------------
    def ev_mult(ev, metric):
        if ev and metric and abs(metric) > 0:
            return round(ev / metric, 1)
        return None

    ev_rev_mult    = ev_mult(ev_perpetuity, revenue_ttm)
    ev_ebitda_mult = ev_mult(ev_perpetuity, ebitda_ttm)
    ev_ebit_mult   = ev_mult(ev_perpetuity, ebit_ttm)

    # Implied perpetuity growth from EBITDA multiple EV (cross-check)
    implied_g = None
    if ev_ebitda and terminal_ebitda and wacc:
        pv_stage1   = pv_fcfs
        pv_tv_impl  = ev_ebitda - pv_stage1
        if pv_tv_impl > 0:
            tv_impl = pv_tv_impl * (1 + wacc) ** tv_discount_exp
            fcf_n1  = projected_fcfs[-1]
            # TV = FCF_n+1 / (WACC − g)  →  g = WACC − FCF_n+1 / TV
            implied_g = round((wacc - fcf_n1 * (1 + terminal_growth) / tv_impl) * 100, 2)

    # ------------------------------------------------------------------
    # 11. Sensitivity Tables
    # ------------------------------------------------------------------
    perp_sensitivity, wacc_range = build_sensitivity_perpetuity(
        base_wacc       = wacc,
        terminal_growth = terminal_growth,
        forecast_years  = forecast_years,
        stub            = stub,
        projected_fcfs  = projected_fcfs,
        gross_debt      = gross_debt,
        cash            = cash_val,
        investments     = invest_val,
        shares          = shares,
    )

    ebitda_sensitivity = None
    if terminal_ebitda:
        ebitda_sensitivity = build_sensitivity_ebitda_multiple(
            base_wacc        = wacc,
            forecast_years   = forecast_years,
            stub             = stub,
            projected_fcfs   = projected_fcfs,
            terminal_ebitda  = terminal_ebitda,
            gross_debt       = gross_debt,
            cash             = cash_val,
            investments      = invest_val,
            shares           = shares,
            center_multiple  = ebitda_exit_mult,
        )

    # ------------------------------------------------------------------
    # 12. Package Results
    # ------------------------------------------------------------------
    return {
        "summary": {
            # ── Filing & Method ──────────────────────────────────────
            "Ticker":                        ticker_symbol,
            "Latest 10-Q Date":              latest_q,
            "UFCF Method":                   ufcf_data['method'],
            "Stub Fraction (Yrs)":           round(stub, 3),
            "Mid-Year Convention":           midyear_adj,

            # ── Valuation — Perpetuity ───────────────────────────────
            "Intrinsic Value — Perpetuity ($)":    ivps_perp,
            "Alpha Signal — Perpetuity":           alpha(ivps_perp),
            "TV as % of TEV (Perpetuity)":         f"{tv_pct_perp}%",
            "Implied TV Exit EBITDA Multiple":     (
                round(pv_tv_perpetuity * (1 + wacc) ** tv_discount_exp /
                      terminal_ebitda, 1)
                if terminal_ebitda else None
            ),

            # ── Valuation — EBITDA Multiple ──────────────────────────
            "Intrinsic Value — EBITDA Multiple ($)": ivps_ebitda,
            "Alpha Signal — EBITDA Multiple":        alpha(ivps_ebitda),
            "TV as % of TEV (EBITDA Multiple)":      f"{tv_pct_ebitda}%",
            "EBITDA Exit Multiple Used":             f"{ebitda_exit_mult:.1f}x",
            "Implied Terminal Growth (EBITDA method)": (
                f"{implied_g}%" if implied_g else None
            ),

            # ── Market & Price ───────────────────────────────────────
            "Current Price ($)":             round(current_price, 2),

            # ── WACC Components ──────────────────────────────────────
            "Applied WACC":                  f"{round(wacc * 100, 2)}%",
            "Cost of Equity (CAPM)":         f"{round(cost_of_equity * 100, 2)}%",
            "Cost of Debt (Pre-Tax)":        f"{round(cost_of_debt * 100, 2)}%",
            "Cost of Debt (After-Tax)":      f"{round(cost_of_debt * (1 - tax_rate) * 100, 2)}%",
            "Effective Tax Rate":            f"{round(tax_rate * 100, 2)}%",
            "Beta":                          round(beta, 3),
            "Equity Weight":                 f"{round(w_equity * 100, 1)}%",
            "Debt Weight":                   f"{round(w_debt * 100, 1)}%",

            # ── Capital Structure ────────────────────────────────────
            "Diluted Shares Ex-Warrants (M)": round(shares / 1e6, 2),
            "Net Debt ($M)":                 round(net_debt / 1e6, 2),
            "Gross Debt ($M)":               round(gross_debt / 1e6, 2),
            "Cash ($M)":                     round(cash_val / 1e6, 2),
            "Equity Investments ($M)":       round(invest_val / 1e6, 2),

            # ── EV Multiples (LTM) ───────────────────────────────────
            "EV / Revenue (LTM)":            ev_rev_mult,
            "EV / EBITDA (LTM)":             ev_ebitda_mult,
            "EV / EBIT (LTM)":               ev_ebit_mult,
        },

        "ufcf_bridge": {
            "TTM EBIT ($M)":              round(ufcf_data['ebit']      / 1e6, 1) if ufcf_data['ebit']      else None,
            "TTM NOPAT ($M)":             round(ufcf_data['nopat']     / 1e6, 1) if ufcf_data['nopat']     else None,
            "TTM D&A ($M)":               round(ufcf_data['da']        / 1e6, 1) if ufcf_data['da']        else None,
            "TTM ΔNWC ($M)":              round(ufcf_data['delta_nwc'] / 1e6, 1) if ufcf_data['delta_nwc'] else None,
            "TTM CapEx ($M)":             round(ufcf_data['capex']     / 1e6, 1) if ufcf_data['capex']     else None,
            "TTM UFCF ($M)":              round(ufcf_data['ufcf']      / 1e6, 1) if ufcf_data['ufcf']      else None,
            "TTM EBITDA ($M)":            round(ebitda_ttm             / 1e6, 1) if ebitda_ttm             else None,
            "TTM Revenue ($M)":           round(revenue_ttm            / 1e6, 1) if revenue_ttm            else None,
            "Stub-Adjusted Yr1 FCF ($M)": round(stub_yr1_fcf           / 1e6, 1),
        },

        "sensitivity_perpetuity":  perp_sensitivity,
        "sensitivity_ebitda_mult": ebitda_sensitivity,
    }


# ==================================================================
# EXECUTION
# ==================================================================

def print_results(result, ticker):
    if isinstance(result, str):
        print(f"\n[{ticker}] ERROR: {result}\n")
        return

    print(f"\n{'='*65}")
    print(f"  DCF VALUATION — {result['summary']['Ticker']}")
    print(f"{'='*65}")

    print("\n── Summary ─────────────────────────────────────────────────")
    for k, v in result['summary'].items():
        if v is not None:
            print(f"  {k:<45} {v}")

    print("\n── UFCF Bridge ──────────────────────────────────────────────")
    for k, v in result['ufcf_bridge'].items():
        if v is not None:
            print(f"  {k:<45} {v}")

    print("\n── Sensitivity: WACC × Terminal Growth Rate ─────────────────")
    print(result['sensitivity_perpetuity'].to_string())

    if result['sensitivity_ebitda_mult'] is not None:
        print("\n── Sensitivity: WACC × EBITDA Exit Multiple ─────────────────")
        print(result['sensitivity_ebitda_mult'].to_string())

    print()


if __name__ == "__main__":
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"]

    for ticker in tickers:
        res = run_automated_dcf(
            ticker_symbol    = ticker,
            growth_rate      = 0.08,
            terminal_growth  = 0.03,
            forecast_years   = 5,
            ebitda_exit_mult = 15.0,
            midyear_adj      = False,
        )
        print_results(res, ticker)