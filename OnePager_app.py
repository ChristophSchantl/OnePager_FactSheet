# bmps_onepager_streamlit.py
# -*- coding: utf-8 -*-
"""
Banca Monte dei Paschi di Siena (BMPS) – One-Pager (Streamlit)

Korrekturen & Add-ons:
- Peer-Vergleich entfernt
- "Forward P/E" entfernt; Anzeige nur aktuelles P/E (trailing)
- Dividend Yield robust normalisiert (Skalenfehler -> automatisch korrigiert) + Fallbacks (Rate/Preis, TTM)
- Kurschart verkleinert (7×3) und Zeitraum wahlweise per Jahre-Slider **oder** Start/End-Datum
- Income-Grafik und alle Labels mit korrekter Währung (Symbol + Code)
- EU-Holder-Sektion entfernt

Start:
    pip install streamlit yfinance pandas numpy matplotlib
    streamlit run bmps_onepager_streamlit.py
"""
from __future__ import annotations
import math
from datetime import date, timedelta
from typing import Dict, List

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


def normalize_percent_robust(x: float) -> float:
    """Normalisiert Prozent/Fraction zuverlässig auf eine Fraktion (0..1).
    Korrigiert übergroße Skalen wie 10.58 (10,58%) oder 1058 (1.058%) durch wiederholtes /100.
    """
    if pd.isna(x):
        return np.nan
    try:
        y = float(x)
    except Exception:
        return np.nan
    while y > 1.0:
        y /= 100.0
    if y < 0:
        return np.nan
    return y


def first_notna(*vals):
    for v in vals:
        if v is not None and not (isinstance(v, float) and math.isnan(v)):
            return v
    return np.nan


def get_income_value(df: pd.DataFrame, candidates: List[str]) -> float:
    if not isinstance(df, pd.DataFrame) or df.empty:
        return np.nan
    df_idx = {i.lower(): i for i in df.index}
    for c in candidates:
        key = c.lower()
        if key in df_idx:
            ser = df.loc[df_idx[key]]
            try:
                return float(ser.dropna().iloc[0])  # letzte verfügbare Periode
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
    """Robuste Ermittlung der Dividendenrendite (Fraktion 0..1).
    Reihenfolge:
      1) dividendYield / trailingAnnualDividendYield (robust normalisiert)
      2) trailingAnnualDividendRate / Preis (falls vorhanden)
      3) TTM‑Fallback: Summe der letzten 12 Monate Dividenden / Preis
    """
    y0 = normalize_percent_robust(safe_get(info, "dividendYield", np.nan))
    y1 = normalize_percent_robust(safe_get(info, "trailingAnnualDividendYield", np.nan))
    rate = safe_get(info, "trailingAnnualDividendRate", np.nan)
    y2 = np.nan
    if pd.notna(rate) and pd.notna(price) and price > 0:
        y2 = float(rate) / float(price)

    for cand in (y0, y1, y2):
        if pd.notna(cand) and 0 < cand < 1.0:
            y = cand
            break
    else:
        y = np.nan

    if (pd.isna(y) or y == 0.0) and pd.notna(price) and price > 0:
        try:
            div = tkr.dividends
            if isinstance(div, pd.Series) and not div.empty:
                last_date = div.index.max()
                ttm_sum = div[div.index >= last_date - pd.DateOffset(years=1)].sum()
                if ttm_sum > 0:
                    y = float(ttm_sum / price)
        except Exception:
            pass

    if pd.notna(y) and y > 0.5:
        y = np.nan
    return y

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

    st.caption("Peers & Eigentümer ausgeblendet (auf Wunsch wieder aktivierbar).")

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

    # Valuation ratios (nur aktuelles P/E)
    trailing_pe = safe_get(info, "trailingPE", np.nan)
    ps_ttm = safe_get(info, "priceToSalesTrailing12Months", np.nan)
    pb = first_notna(safe_get(info, "priceToBook", None), np.nan)

    # Dividende – korrigiert
    dividend_yield = compute_dividend_yield(tkr, price, info)  # Fraktion 0..1
    payout_ratio = normalize_percent_robust(safe_get(info, "payoutRatio", np.nan))

    # Earnings dates
    ed = safe_get(info, "earningsDate", [])
    next_earnings = None
    if isinstance(ed, (list, tuple)) and len(ed) > 0:
        try:
            next_earnings = pd.to_datetime(ed[0]).date().isoformat()
        except Exception:
            next_earnings = None

    # Business summary
    long_summary = safe_get(info, "longBusinessSummary", "")

    # Financial statements (annual preferred; fallback quarterly)
    try:
        fin_a = tkr.financials  # annual
    except Exception:
        fin_a = pd.DataFrame()
    try:
        fin_q = tkr.quarterly_financials
    except Exception:
        fin_q = pd.DataFrame()
    fin = fin_a if isinstance(fin_a, pd.DataFrame) and not fin_a.empty else fin_q

    revenue = get_income_value(fin, ["Total Revenue", "Revenue"])
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
        st.write("")
    with meta_col2:
        st.markdown(f"**Industry:** {industry}")
        st.markdown(f"**Sector:** {sector}")
        st.markdown(f"**Employees:** {int(employees) if pd.notna(employees) else 'n/a'}")
        st.markdown(f"**Shares Outstanding (bn):** {bn(shares):.3f} ")
    with meta_col3:
        st.markdown(f"**Market Cap (bn):** {bn(mktcap):.2f} {currency}")
        st.markdown(f"**Current Price:** {price if pd.notna(price) else 'n/a'} {currency}")
        st.markdown(f"**Next earnings:** {next_earnings or 'n/a'}")

    if long_summary:
        st.caption(long_summary)

    st.markdown("---")

    # ----------------------------------
    # Valuation block (aktuelles P/E)
    # ----------------------------------
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Price‑Earnings (Current)", f"{trailing_pe:.2f}" if pd.notna(trailing_pe) else "n/a")
    c2.metric("Price‑Sales (TTM)", f"{ps_ttm:.2f}" if pd.notna(ps_ttm) else "n/a")
    c3.metric("Price‑Book", f"{pb:.2f}" if pd.notna(pb) else "n/a")
    c4.metric("Dividend Yield", f"{dividend_yield*100:.2f}%" if pd.notna(dividend_yield) else "n/a")
    c5.metric("Payout Ratio", f"{payout_ratio*100:.1f}%" if pd.notna(payout_ratio) else "n/a")

    st.markdown("---")

    # ----------------------------------
    # Price Chart – Close, klein & mit Datumsauswahl
    # ----------------------------------
    st.subheader("Preisverlauf (Close)")
    try:
        if period_mode == "Start/End Datum":
            hist = yf.download(ticker, start=pd.to_datetime(start_date), end=pd.to_datetime(end_date) + pd.Timedelta(days=1), interval="1d", progress=False)
        else:
            hist = yf.download(ticker, period=f"{years_window}y", interval="1d", progress=False)
        if isinstance(hist, pd.DataFrame) and not hist.empty:
            series = hist["Close"].dropna()
            figp, axp = plt.subplots(figsize=(7, 3))
            axp.plot(series.index, series.values)
            title_range = f"{years_window}J" if period_mode != "Start/End Datum" else f"{pd.to_datetime(start_date).date()} → {pd.to_datetime(end_date).date()}"
            axp.set_title(f"{ticker} – {title_range} Close")
            axp.set_xlabel("Datum")
            axp.set_ylabel(f"Preis ({currency})")
            axp.grid(True, linestyle=":", alpha=0.4)
            st.pyplot(figp, clear_figure=True)
        else:
            st.info("Kein Kursverlauf verfügbar.")
    except Exception as e:
        st.warning(f"Kursdaten konnten nicht geladen werden: {e}")

    st.markdown("---")

    # ----------------------------------
    # Income Statement Bars (mit Währungsangabe)
    # ----------------------------------
    st.subheader("Income (zuletzt verfügbar)")
    names = ["Revenue", "Cost of Revenue", "Gross Profit", "Other Expenses", "Net Income"]
    values = [bn(revenue), bn(cost_rev), bn(gross_profit), bn(other_expenses), bn(net_income)]
    df_plot = pd.DataFrame({"Item": names, "Value": values})
    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    ax.bar(df_plot["Item"], df_plot["Value"]) 
    ax.set_ylabel(f"{currency_hint} bn ({currency})")
    ax.set_title(f"BMPS – Ergebnisblöcke (letzte Periode) – Werte in {currency}")
    for i, v in enumerate(values):
        if pd.notna(v):
            ax.text(i, v if v >= 0 else 0, f"{v:.2f}", ha='center', va='bottom')
    plt.xticks(rotation=15)
    st.pyplot(fig, clear_figure=True)

    st.markdown("---")

    # ----------------------------------
    # Raw tables for export
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
