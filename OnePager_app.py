
"""
Updates:
- Compact header; English UI
- Removed custom dates → only years slider
- Charts smaller & finer (global typography + subtle grid)
- Two charts per row (Price left, Income right)
- Income chart redesigned as **stylized bar (waterfall-like)** similar to reference
- KPI block (Trailing P/E). Forward P/E only in Valuation Measures
- Robust dividend yield (scale fixes + TTM fallback)
- P/E donut (Market Cap vs Earnings) with compact PE badge
- Valuation/Profitability and Balance Sheet/CF sections

Run:
    pip install streamlit yfinance pandas numpy matplotlib
    streamlit run bmps_onepager_streamlit.py
"""
from __future__ import annotations
import math
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib as mpl

st.set_page_config(page_title="BMPS – One-Pager", layout="wide")

# ---------- Global style (compact, professional) ----------
mpl.rcParams.update({
    "font.size": 9,
    "axes.titlesize": 11,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": 120,
})

st.markdown(
    """
    <style>
    .block-container {padding-top:0.6rem; padding-bottom:0.6rem;}
    [data-testid="stMetricValue"] {font-size:1.12rem;}
    [data-testid="stMetricLabel"] {font-size:0.74rem; color:#666;}
    h1, h2, h3 {letter-spacing:0.2px}
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------
# Helpers
# ----------------------------------

def safe_get(d: Dict, key: str, default=np.nan):
    try:
        v = d.get(key, default)
        return np.nan if v is None else v
    except Exception:
        return default


def currency_symbol(code: str) -> str:
    m = {
        "EUR": "€", "USD": "$", "GBP": "£", "JPY": "¥", "CHF": "CHF", "CAD": "$",
        "AUD": "$", "SEK": "kr", "NOK": "kr", "DKK": "kr", "PLN": "zł", "HKD": "$",
    }
    return m.get(str(code).upper(), str(code))


def bn(x: float) -> float:
    return float(x) / 1e9 if pd.notna(x) else np.nan


def normalize_percent_robust(x: float) -> float:
    """Normalize percent/fraction reliably to 0..1, fixing 10.58/1058 style inputs."""
    if pd.isna(x):
        return np.nan
    try:
        y = float(x)
    except Exception:
        return np.nan
    while y > 1.0:
        y /= 100.0
    return np.nan if y < 0 else y


def first_notna(*vals):
    for v in vals:
        if v is not None and not (isinstance(v, float) and math.isnan(v)):
            return v
    return np.nan


def get_income_value(df: pd.DataFrame, candidates: List[str]) -> float:
    if not isinstance(df, pd.DataFrame) or df.empty:
        return np.nan
    idx_map = {i.lower(): i for i in df.index}
    for c in candidates:
        key = c.lower()
        if key in idx_map:
            ser = df.loc[idx_map[key]]
            try:
                return float(ser.dropna().iloc[0])
            except Exception:
                continue
    return np.nan


def load_ticker_info(ticker: str) -> Dict:
    t = yf.Ticker(ticker)
    try:
        info = t.info or {}
    except Exception:
        info = {}
    fast = getattr(t, "fast_info", {}) or {}
    return {"ticker": t, "info": info, "fast": fast}


def compute_dividend_yield(tkr: yf.Ticker, price: float, info: Dict) -> float:
    """Dividend yield (0..1): info → rate/price → TTM fallback."""
    y0 = normalize_percent_robust(safe_get(info, "dividendYield", np.nan))
    y1 = normalize_percent_robust(safe_get(info, "trailingAnnualDividendYield", np.nan))
    rate = safe_get(info, "trailingAnnualDividendRate", np.nan)
    y2 = np.nan
    if pd.notna(rate) and pd.notna(price) and price > 0:
        y2 = float(rate) / float(price)
    y = first_notna(*(c for c in (y0, y1, y2) if pd.notna(c) and 0 < c < 1.0))
    if (pd.isna(y) or y == 0.0) and pd.notna(price) and price > 0:
        try:
            div = tkr.dividends
            if isinstance(div, pd.Series) and not div.empty:
                last_date = div.index.max()
                ttm = div[div.index >= last_date - pd.DateOffset(years=1)].sum()
                if ttm > 0:
                    y = float(ttm / price)
        except Exception:
            pass
    return np.nan if (pd.notna(y) and y > 0.5) else y


def fmt_pct(x: float) -> str:
    return "n/a" if pd.isna(x) else f"{x*100:.2f}%"


def fmt_money_bn(x: float, code: str) -> str:
    sym = currency_symbol(code)
    return "n/a" if pd.isna(x) else f"{sym}{bn(x):.2f}b ({code})"


# ---------- Compact axis styling ----------

def style_axes(ax):
    ax.grid(True, linestyle=":", alpha=0.22)
    for spine in ax.spines.values():
        spine.set_alpha(0.22)
    ax.tick_params(labelsize=8)


# ---------- Sidebar Inputs ----------
left, right = st.columns([2, 1])
with right:
    st.header("Parameters")
    ticker = st.text_input("Ticker", value="BMPS.MI").strip()
    years_window = st.slider("Price window (years)", min_value=1, max_value=10, value=3, step=1)

with left:
    st.title("SHI Management – STOCK PROFILE: One‑Pager")


# ---------- Load core data ----------
if ticker:
    data = load_ticker_info(ticker)
    tkr: yf.Ticker = data["ticker"]
    info: Dict = data["info"]
    fast: Dict = data["fast"]

    # Basics
    long_name = safe_get(info, "longName", ticker)
    website = safe_get(info, "website", "")
    industry = safe_get(info, "industry", "")
    sector = safe_get(info, "sector", "")
    country = safe_get(info, "country", "")
    employees = safe_get(info, "fullTimeEmployees", np.nan)
    exch = first_notna(safe_get(info, "fullExchangeName", None), safe_get(info, "exchange", None), "")
    mktcap = first_notna(safe_get(info, "marketCap", None), safe_get(fast, "market_cap", None))
    shares = first_notna(safe_get(info, "sharesOutstanding", None), safe_get(fast, "shares_outstanding", None))
    price = first_notna(safe_get(info, "currentPrice", None), safe_get(fast, "last_price", None))
    currency = safe_get(info, "currency", "EUR")
    sym = currency_symbol(currency)
    # Robust chart label derived from live ticker info (prevents stale labels like 'LULU')
    label_tkr = (safe_get(info, "symbol", None) or ticker).upper()

    # KPIs
    trailing_pe = safe_get(info, "trailingPE", np.nan)
    forward_pe = safe_get(info, "forwardPE", np.nan)
    ps_ttm = safe_get(info, "priceToSalesTrailing12Months", np.nan)
    pb = first_notna(safe_get(info, "priceToBook", None), np.nan)

    # Dividend
    dividend_yield = compute_dividend_yield(tkr, price, info)
    payout_ratio = normalize_percent_robust(safe_get(info, "payoutRatio", np.nan))

    # Earnings date
    ed = safe_get(info, "earningsDate", [])
    next_earnings = None
    if isinstance(ed, (list, tuple)) and len(ed) > 0:
        try:
            next_earnings = pd.to_datetime(ed[0]).date().isoformat()
        except Exception:
            next_earnings = None

    # Financials (annual → quarterly fallback)
    try:
        fin_a = tkr.financials
    except Exception:
        fin_a = pd.DataFrame()
    try:
        fin_q = tkr.quarterly_financials
    except Exception:
        fin_q = pd.DataFrame()
    fin = fin_a if isinstance(fin_a, pd.DataFrame) and not fin_a.empty else fin_q

    revenue = get_income_value(fin, ["Total Revenue", "Revenue"])  # absolute
    cost_rev = get_income_value(fin, ["Cost Of Revenue", "Cost of Revenue", "Cost of revenue"])
    gross_profit = get_income_value(fin, ["Gross Profit", "Gross profit"])  # falls NaN → später herleiten
    op_ex = get_income_value(fin, ["Total Operating Expenses", "Operating Expense", "Operating Expenses"])
    net_income = get_income_value(fin, ["Net Income", "Net income", "Net Income Common Stockholders"])

    if pd.isna(gross_profit) and pd.notna(revenue) and pd.notna(cost_rev):
        gross_profit = revenue - cost_rev
    other_expenses = np.nan
    if pd.notna(op_ex) and pd.notna(cost_rev):
        other_expenses = max(0.0, op_ex - cost_rev)
    else:
        other_expenses = op_ex

    # ---------- Header & Meta ----------
    meta_col1, meta_col2, meta_col3 = st.columns([1.4, 1.1, 1.2])
    with meta_col1:
        st.subheader(long_name)
        st.markdown(f"**Ticker:** {ticker}  ")
        st.markdown(f"**Exchange:** {exch}  ")
        st.markdown(f"**Country:** {country}  ")
        st.markdown(f"**Web:** [{website}]({website})")
    with meta_col2:
        st.markdown(f"**Industry:** {industry}")
        st.markdown(f"**Sector:** {sector}")
        st.markdown(f"**Employees:** {int(employees) if pd.notna(employees) else 'n/a'}")
        st.markdown(f"**Shares Outstanding (bn):** {bn(shares):.3f}")
    with meta_col3:
        st.markdown(f"**Market Cap (bn):** {bn(mktcap):.2f} {currency}")
        st.markdown(f"**Current Price:** {price if pd.notna(price) else 'n/a'} {currency}")
        st.markdown(f"**Next earnings:** {next_earnings or 'n/a'}")

    if safe_get(info, "longBusinessSummary", ""):
        st.caption(safe_get(info, "longBusinessSummary", ""))

    st.markdown("---")

    # ---------- KPI Block (compact) ----------
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("P/E (Trailing)", f"{trailing_pe:.2f}" if pd.notna(trailing_pe) else "n/a")
    c2.metric("P/S (TTM)", f"{ps_ttm:.2f}" if pd.notna(ps_ttm) else "n/a")
    c3.metric("P/B", f"{pb:.2f}" if pd.notna(pb) else "n/a")
    c4.metric("Dividend Yield", fmt_pct(dividend_yield))
    c5.metric("Payout Ratio", fmt_pct(payout_ratio))

    # ---------- P/E Donut (Market Cap vs Earnings) ----------
    st.markdown("---")
    donut_l, donut_r = st.columns([1, 1])
    with donut_l:
        earnings = net_income  # ttm
        mc = mktcap
        if pd.notna(earnings) and pd.notna(mc) and earnings > 0 and mc > 0:
            figd, axd = plt.subplots(figsize=(4.6, 2.2))
            sizes = [earnings, max(mc - earnings, 0.0)]
            axd.pie(sizes, startangle=90, wedgeprops=dict(width=0.28))
            axd.set_aspect('equal')
            # FIX: f-Strings mit '
' statt Zeilenbruch im Stringliteral
            axd.text(-1.05, 0.62, f"Earnings
{sym}{bn(earnings):.2f}b", fontsize=8.5, ha='left', va='center')
            axd.text(0, -0.06, f"Market Cap
{sym}{bn(mc):.2f}b", fontsize=8.5, ha='center', va='center')
            st.pyplot(figd, clear_figure=True)
        else:
            st.caption("P/E donut unavailable (missing market cap or earnings).")
    with donut_r:
        if pd.notna(trailing_pe):
            st.markdown(f"<div style='font-size:36px;font-weight:700;line-height:1'> {trailing_pe:.1f}x </div>", unsafe_allow_html=True)
            st.markdown("<div style='font-size:12px;color:#666'>PE Ratio</div>", unsafe_allow_html=True)
        else:
            st.caption("PE Ratio: n/a")

    st.markdown("---")

    # ---------- Two charts side by side ----------
    ch_left, ch_right = st.columns(2)

    with ch_left:
        st.caption("Price (Close)")
        try:
            hist = yf.download(ticker, period=f"{years_window}y", interval="1d", progress=False)
            if isinstance(hist, pd.DataFrame) and not hist.empty:
                series = hist["Close"].dropna()
                figp, axp = plt.subplots(figsize=(4.8, 2.2))
                axp.plot(series.index, series.values, linewidth=1.0)
                axp.set_title(f"{label_tkr} – {years_window}y", fontsize=10)
                axp.set_xlabel("Date", fontsize=8)
                axp.set_ylabel(f"Price ({currency})", fontsize=8)
                style_axes(axp)
                st.pyplot(figp, clear_figure=True)
            else:
                st.info("No price history available.")
        except Exception as e:
            st.warning(f"Could not load price data: {e}")

    with ch_right:
        st.caption("Earnings & Revenue (last period)")
        # Stylized bars (waterfall-like): Revenue, Cost of Revenue, Gross Profit, Other Expenses, Earnings
        names = ["Revenue", "Cost of Revenue", "Gross Profit", "Other Expenses", "Earnings"]
        gross_val = gross_profit if pd.notna(gross_profit) else (revenue - cost_rev if pd.notna(revenue) and pd.notna(cost_rev) else np.nan)
        vals_abs = [bn(revenue), bn(cost_rev), bn(gross_val), bn(other_expenses), bn(net_income)]
        colors = ["#1f77b4", "#b04a4a", "#2ca02c", "#b04a4a", "#17becf"]  # blue, red, green, red, teal
        fig, ax = plt.subplots(figsize=(4.8, 2.2))
        x = np.arange(len(names))
        bars = ax.bar(x, vals_abs, color=colors, width=0.6)
        ax.set_ylabel(f"{sym} bn ({currency})", fontsize=8)
        ax.set_title(f"{label_tkr} – {currency}", fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=10)
        style_axes(ax)
        # Value labels on top
        for rect, v in zip(bars, vals_abs):
            if pd.notna(v):
                ax.text(rect.get_x() + rect.get_width()/2, rect.get_height(), f"{sym}{v:.2f}b", ha='center', va='bottom', fontsize=8)
        st.pyplot(fig, clear_figure=True)

    st.markdown("---")

    # ---------- Valuation & Profitability ----------
    st.subheader("Valuation & Profitability")

    ev = safe_get(info, "enterpriseValue", np.nan)
    peg = safe_get(info, "pegRatio", np.nan)
    ev_rev = first_notna(safe_get(info, "enterpriseToRevenue", None), safe_get(info, "enterpriseToRev", None), np.nan)
    ev_ebitda = safe_get(info, "enterpriseToEbitda", np.nan)
    profit_margin = safe_get(info, "profitMargins", np.nan)
    roa = safe_get(info, "returnOnAssets", np.nan)
    roe = safe_get(info, "returnOnEquity", np.nan)
    revenue_ttm = safe_get(info, "totalRevenue", np.nan)
    nic_ttm = safe_get(info, "netIncomeToCommon", np.nan)

    val_rows: List[Tuple[str, str]] = [
        ("Enterprise Value", fmt_money_bn(ev, currency)),
        ("Trailing P/E", "n/a" if pd.isna(trailing_pe) else f"{trailing_pe:.2f}×"),
        ("Forward P/E", "n/a" if pd.isna(forward_pe) else f"{forward_pe:.2f}×"),
        ("PEG (5y exp)", "n/a" if pd.isna(peg) else f"{peg:.2f}"),
        ("EV/Revenue", "n/a" if pd.isna(ev_rev) else f"{ev_rev:.2f}×"),
        ("EV/EBITDA", "n/a" if pd.isna(ev_ebitda) else f"{ev_ebitda:.2f}×"),
        ("Profit Margin", fmt_pct(profit_margin)),
        ("ROA (ttm)", fmt_pct(roa)),
        ("ROE (ttm)", fmt_pct(roe)),
        ("Revenue (ttm)", fmt_money_bn(revenue_ttm, currency)),
        ("Net Income to Common (ttm)", fmt_money_bn(nic_ttm, currency)),
    ]

    val_df = pd.DataFrame(val_rows, columns=["Metric", "Value"])
    st.dataframe(val_df, use_container_width=True)

    st.markdown("---")

    # ---------- Balance Sheet & Cash Flow ----------
    st.subheader("Balance Sheet & Cash Flow")

    total_cash = safe_get(info, "totalCash", np.nan)  # mrq
    d_to_e = safe_get(info, "debtToEquity", np.nan)   # mrq
    lfcf = safe_get(info, "leveredFreeCashflow", np.nan)

    d_to_e_disp = "n/a"
    if pd.notna(d_to_e):
        if d_to_e > 5:
            d_to_e_disp = f"{d_to_e:.1f}% (~{d_to_e/100:.2f}×)"
        else:
            d_to_e_disp = f"{d_to_e:.2f}×"

    bs_rows: List[Tuple[str, str]] = [
        ("Total Cash (mrq)", fmt_money_bn(total_cash, currency)),
        ("Total Debt/Equity (mrq)", d_to_e_disp),
        ("Levered Free Cash Flow", fmt_money_bn(lfcf, currency)),
    ]

    bs_df = pd.DataFrame(bs_rows, columns=["Metric", "Value"])
    st.dataframe(bs_df, use_container_width=True)

    st.markdown("---")

    # ---------- Raw data & export ----------
    st.subheader("Raw data & export")
    meta_table = pd.DataFrame({
        "Field": ["Name","Ticker","Exchange","Country","Industry","Sector","Employees","Currency","MarketCap (bn)","Shares (bn)","Price"],
        "Value": [long_name,ticker,exch,country,industry,sector,employees,currency,bn(mktcap),bn(shares),price]
    })
    st.dataframe(meta_table, use_container_width=True)

    st.download_button("Download CSV (meta)", data=meta_table.to_csv(index=False).encode("utf-8"), file_name=f"{ticker}_meta.csv", mime="text/csv")

else:
    st.info("Enter a Yahoo ticker, e.g., BMPS.MI.")
