
"""
Korrekturen & Add-ons (aktuell):
- Charts kleiner & dezenter (kompakt, feine Gitterlinien, enges Layout)
- Peer-Vergleich & Eigentümer entfernt
- Aktuelles P/E (trailing) im KPI-Block; Forward P/E in den **Valuation Measures** (auf Wunsch)
- Dividend Yield robust normalisiert + TTM-Fallback
- Preis-Chart: Zeitraum via Jahre **oder** Start/End-Datum (kleines Format)
- Income-Grafik mit **korrekter Währung** (Symbol + Code)
- **Neue Kennzahlen** integriert:
  Valuation Measures: EV, Trailing P/E, Forward P/E, PEG (5y exp), EV/Revenue, EV/EBITDA,
  Profitability: Profit Margin, ROA (ttm), ROE (ttm), Revenue (ttm), Net Income to Common (ttm),
  Balance Sheet & CF: Total Cash (mrq), Debt/Equity (mrq), Levered Free Cash Flow.

Start:
    pip install streamlit yfinance pandas numpy matplotlib
    streamlit run bmps_onepager_streamlit.py
"""
from __future__ import annotations
import math
from datetime import date, timedelta
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="BMPS – One-Pager", layout="wide")

# ----------------------------------
# Helpers
# ----------------------------------

def safe_get(d: Dict, key: str, default=np.nan):
    try:
        v = d.get(key, default)
        return np.nan if v is None else v
    except Exception:
        return default


def bn(x: float) -> float:
    return float(x) / 1e9 if pd.notna(x) else np.nan


def mm(x: float) -> float:
    return float(x) / 1e6 if pd.notna(x) else np.nan


def normalize_percent_robust(x: float) -> float:
    """Normalisiert Prozent/Fraction zuverlässig auf Fraktion (0..1).
    Korrigiert Werte wie 10.58 oder 1058 iterativ durch /100.
    """
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
    """Dividendenrendite (0..1): info → Rate/Preis → TTM-Fallback."""
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


def fmt_ratio(x: float) -> str:
    return "n/a" if pd.isna(x) else f"{x:.2f}×"


def fmt_money_bn(x: float, code: str, symbol: str) -> str:
    return "n/a" if pd.isna(x) else f"{symbol} {bn(x):.2f} bn ({code})"

# ----------------------------------
# Sidebar Inputs
# ----------------------------------
left, right = st.columns([2, 1])
with right:
    st.header("⚙️ Parameter")
    ticker = st.text_input("Ticker ", value="BMPS.MI").strip()
    currency_hint = st.text_input("Währungssymbol (Anzeige)", value="€")

    period_mode = st.radio("Zeitraum wählen", ["Jahre (Slider)", "Start/End Datum"], index=0)
    years_window = st.slider("Kurs‑Zeitraum (Jahre)", min_value=1, max_value=10, value=3, step=1)
    start_date = st.date_input("Start‑Datum", value=date.today() - timedelta(days=365*3))
    end_date = st.date_input("End‑Datum", value=date.today())

with left:
    st.title("SHI Management – STOCK PROFILE: One‑Pager")
    st.caption("Quick Check")

# ----------------------------------
# Load core data
# ----------------------------------
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

    # KPIs (aktuelles P/E etc.)
    trailing_pe = safe_get(info, "trailingPE", np.nan)
    forward_pe = safe_get(info, "forwardPE", np.nan)
    ps_ttm = safe_get(info, "priceToSalesTrailing12Months", np.nan)
    pb = first_notna(safe_get(info, "priceToBook", None), np.nan)

    # Dividende – korrigiert
    dividend_yield = compute_dividend_yield(tkr, price, info)  # Fraktion 0..1
    payout_ratio = normalize_percent_robust(safe_get(info, "payoutRatio", np.nan))

    # Earnings
    ed = safe_get(info, "earningsDate", [])
    next_earnings = None
    if isinstance(ed, (list, tuple)) and len(ed) > 0:
        try:
            next_earnings = pd.to_datetime(ed[0]).date().isoformat()
        except Exception:
            next_earnings = None

    # Financial statements (annual preferred → quarterly)
    try:
        fin_a = tkr.financials
    except Exception:
        fin_a = pd.DataFrame()
    try:
        fin_q = tkr.quarterly_financials
    except Exception:
        fin_q = pd.DataFrame()
    fin = fin_a if isinstance(fin_a, pd.DataFrame) and not fin_a.empty else fin_q

    revenue = get_income_value(fin, ["Total Revenue", "Revenue"])  # absolute Währung
    cost_rev = get_income_value(fin, ["Cost Of Revenue", "Cost of Revenue", "Cost of revenue"])
    gross_profit = get_income_value(fin, ["Gross Profit", "Gross profit"])
    op_ex = get_income_value(fin, ["Total Operating Expenses", "Operating Expense", "Operating Expenses"])
    net_income = get_income_value(fin, ["Net Income", "Net income", "Net Income Common Stockholders"])

    other_expenses = np.nan
    if pd.notna(op_ex) and pd.notna(cost_rev):
        other_expenses = max(0.0, op_ex - cost_rev)
    else:
        other_expenses = op_ex

    # ----------------------------------
    # Header & Meta
    # ----------------------------------
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

    # ----------------------------------
    # KPI Block (kompakt, dezent)
    # ----------------------------------
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("P/E (Trailing)", f"{trailing_pe:.2f}" if pd.notna(trailing_pe) else "n/a")
    c2.metric("P/S (TTM)", f"{ps_ttm:.2f}" if pd.notna(ps_ttm) else "n/a")
    c3.metric("P/B", f"{pb:.2f}" if pd.notna(pb) else "n/a")
    c4.metric("Dividend Yield", fmt_pct(dividend_yield))
    c5.metric("Payout Ratio", fmt_pct(payout_ratio))

    st.markdown("---")

    # ----------------------------------
    # Price Chart – klein & dezent
    # ----------------------------------
    st.subheader("Preisverlauf (Close)")
    try:
        if period_mode == "Start/End Datum":
            hist = yf.download(ticker, start=pd.to_datetime(start_date), end=pd.to_datetime(end_date) + pd.Timedelta(days=1), interval="1d", progress=False)
        else:
            hist = yf.download(ticker, period=f"{years_window}y", interval="1d", progress=False)
        if isinstance(hist, pd.DataFrame) and not hist.empty:
            series = hist["Close"].dropna()
            figp, axp = plt.subplots(figsize=(6, 2.6))
            axp.plot(series.index, series.values, linewidth=1.0)
            title_range = f"{years_window}J" if period_mode != "Start/End Datum" else f"{pd.to_datetime(start_date).date()} → {pd.to_datetime(end_date).date()}"
            axp.set_title(f"{ticker} – {title_range} Close", fontsize=11)
            axp.set_xlabel("Datum", fontsize=9)
            axp.set_ylabel(f"Preis ({currency})", fontsize=9)
            axp.grid(True, linestyle=":", alpha=0.3)
            axp.tick_params(labelsize=8)
            for spine in axp.spines.values():
                spine.set_alpha(0.3)
            st.pyplot(figp, clear_figure=True)
        else:
            st.info("Kein Kursverlauf verfügbar.")
    except Exception as e:
        st.warning(f"Kursdaten konnten nicht geladen werden: {e}")

    st.markdown("---")

    # ----------------------------------
    # Income Statement Bars – mit Währung
    # ----------------------------------
    st.subheader("Income (zuletzt verfügbar)")
    names = ["Revenue", "Cost of Revenue", "Gross Profit", "Other Expenses", "Net Income"]
    values = [bn(revenue), bn(cost_rev), bn(gross_profit), bn(other_expenses), bn(net_income)]
    df_plot = pd.DataFrame({"Item": names, "Value": values})
    fig, ax = plt.subplots(figsize=(5.8, 3.2))
    ax.bar(df_plot["Item"], df_plot["Value"]) 
    ax.set_ylabel(f"{currency_hint} bn ({currency})", fontsize=9)
    ax.set_title(f"BMPS – Ergebnisblöcke (letzte Periode) – {currency}", fontsize=11)
    for i, v in enumerate(values):
        if pd.notna(v):
            ax.text(i, v if v >= 0 else 0, f"{v:.2f}", ha='center', va='bottom', fontsize=8)
    plt.xticks(rotation=12)
    ax.grid(axis='y', linestyle=":", alpha=0.25)
    ax.tick_params(labelsize=8)
    for spine in ax.spines.values():
        spine.set_alpha(0.3)
    st.pyplot(fig, clear_figure=True)

    st.markdown("---")

    # ----------------------------------
    # Valuation Measures & Profitability
    # ----------------------------------
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
        ("Enterprise Value", fmt_money_bn(ev, currency, currency_hint)),
        ("Trailing P/E", "n/a" if pd.isna(trailing_pe) else f"{trailing_pe:.2f}×"),
        ("Forward P/E", "n/a" if pd.isna(forward_pe) else f"{forward_pe:.2f}×"),
        ("PEG (5y exp)", "n/a" if pd.isna(peg) else f"{peg:.2f}"),
        ("EV/Revenue", "n/a" if pd.isna(ev_rev) else f"{ev_rev:.2f}×"),
        ("EV/EBITDA", "n/a" if pd.isna(ev_ebitda) else f"{ev_ebitda:.2f}×"),
        ("Profit Margin", fmt_pct(profit_margin)),
        ("ROA (ttm)", fmt_pct(roa)),
        ("ROE (ttm)", fmt_pct(roe)),
        ("Revenue (ttm)", fmt_money_bn(revenue_ttm, currency, currency_hint)),
        ("Net Income to Common (ttm)", fmt_money_bn(nic_ttm, currency, currency_hint)),
    ]

    val_df = pd.DataFrame(val_rows, columns=["Metric", "Value"])
    st.dataframe(val_df, use_container_width=True)

    st.markdown("---")

    # ----------------------------------
    # Balance Sheet & Cash Flow
    # ----------------------------------
    st.subheader("Balance Sheet & Cash Flow")

    total_cash = safe_get(info, "totalCash", np.nan)  # mrq
    d_to_e = safe_get(info, "debtToEquity", np.nan)   # mrq (typisch in %/ratio)
    lfcf = safe_get(info, "leveredFreeCashflow", np.nan)

    # Debt/Equity kann als Zahl (z.B. 120) geliefert werden – Prozent normalisieren? Wir zeigen als Ratio/Prozent verständlich an.
    d_to_e_disp = "n/a"
    if pd.notna(d_to_e):
        # Heuristik: >5 → wahrscheinlich Prozentangabe (z.B. 120 -> 120%). Wir zeigen beides.
        if d_to_e > 5:
            d_to_e_disp = f"{d_to_e:.1f}% (~{d_to_e/100:.2f}×)"
        else:
            d_to_e_disp = f"{d_to_e:.2f}×"

    bs_rows: List[Tuple[str, str]] = [
        ("Total Cash (mrq)", fmt_money_bn(total_cash, currency, currency_hint)),
        ("Total Debt/Equity (mrq)", d_to_e_disp),
        ("Levered Free Cash Flow", fmt_money_bn(lfcf, currency, currency_hint)),
    ]

    bs_df = pd.DataFrame(bs_rows, columns=["Metric", "Value"])
    st.dataframe(bs_df, use_container_width=True)

    st.markdown("---")

    # ----------------------------------
    # Rohdaten & Export
    # ----------------------------------
    st.subheader("Rohdaten & Export")
    meta_table = pd.DataFrame({
        "Field": ["Name","Ticker","Exchange","Country","Industry","Sector","Employees","Currency","MarketCap (bn)","Shares (bn)","Price"],
        "Value": [long_name,ticker,exch,country,industry,sector,employees,currency,bn(mktcap),bn(shares),price]
    })
    st.dataframe(meta_table, use_container_width=True)

    st.download_button("CSV – Meta exportieren", data=meta_table.to_csv(index=False).encode("utf-8"), file_name=f"{ticker}_meta.csv", mime="text/csv")

else:
    st.info("Ticker (z. B. BMPS.MI).")
