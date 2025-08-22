# bmps_onepager_streamlit.py
# -*- coding: utf-8 -*-
"""
Banca Monte dei Paschi di Siena (BMPS) – One-Pager (Streamlit)


Korrekturen & Add-ons:
- Feld "Founded" entfernt
- Dividend Yield robust normalisiert (0–1 oder 0–100) + TTM-Fallback aus Dividendenhistorie
- 3‑Jahres Kurschart (flexibel einstellbare Jahre, 1–10) mit Adjusted Close
- Saubere Anzeige/Abfang fehlender EU‑Holderdaten

Start:
    pip install streamlit yfinance pandas numpy matplotlib
    streamlit run bmps_onepager_streamlit.py
"""
from __future__ import annotations
import math
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


def normalize_percent(x: float) -> float:
    """Bringt gemischte Prozent-/Fraktionswerte in eine Fraktion (0..1)."""
    if pd.isna(x):
        return np.nan
    if 1.0 < x <= 100.0:
        return float(x) / 100.0
    return float(x)


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
    """Robuste Ermittlung der Dividendenrendite (Fraktion). Reihenfolge:
    1) info['dividendYield'] (normalisiert), 2) trailingAnnualDividendYield (normalisiert),
    3) TTM-Fallback: Summe der letzten 4 Quartalsdividenden / Preis
    """
    y0 = normalize_percent(safe_get(info, "dividendYield", np.nan))
    y1 = normalize_percent(safe_get(info, "trailingAnnualDividendYield", np.nan))
    y = first_notna(y0, y1, np.nan)

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
    return y

# ----------------------------------
# Sidebar Inputs
# ----------------------------------
left, right = st.columns([2, 1])
with right:
    st.header("⚙️ Parameter")
    ticker = st.text_input("Ticker ", value="BMPS.MI").strip()
    currency_hint = st.text_input("Währungssymbol (Anzeige)", value="€")
    peers_default = "FBK.MI, MB.MI, BPE.MI, BAMI.MI"
    peers_text = st.text_input("Peer‑Ticker (kommagetrennt)", value=peers_default)
    years_window = st.slider("Kurs-Zeitraum (Jahre)", min_value=1, max_value=10, value=3, step=1)
    reload_btn = st.button("Neu laden")

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

    # Valuation ratios
    trailing_pe = safe_get(info, "trailingPE", np.nan)
    forward_pe = safe_get(info, "forwardPE", np.nan)
    ps_ttm = safe_get(info, "priceToSalesTrailing12Months", np.nan)
    pb = first_notna(safe_get(info, "priceToBook", None), np.nan)

    # Dividende
    dividend_yield = compute_dividend_yield(tkr, price, info)  # Fraktion 0..1
    payout_ratio = normalize_percent(safe_get(info, "payoutRatio", np.nan))

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
    # Header & Meta (ohne Founded)
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

    # Kurzbeschreibung
    if long_summary:
        st.caption(long_summary)

    st.markdown("---")

    # ----------------------------------
    # Valuation block
    # ----------------------------------
    v1, v2, v3, v4, v5, v6 = st.columns(6)
    v1.metric("Price‑Earning (Trailing)", f"{trailing_pe:.2f}" if pd.notna(trailing_pe) else "n/a")
    v2.metric("Price‑Earning (Forward)", f"{forward_pe:.2f}" if pd.notna(forward_pe) else "n/a")
    v3.metric("Price‑Sales (TTM)", f"{ps_ttm:.2f}" if pd.notna(ps_ttm) else "n/a")
    v4.metric("Price‑Book", f"{pb:.2f}" if pd.notna(pb) else "n/a")
    v5.metric("Dividend Yield", f"{dividend_yield*100:.2f}%" if pd.notna(dividend_yield) else "n/a")
    v6.metric("Payout Ratio", f"{payout_ratio*100:.1f}%" if pd.notna(payout_ratio) else "n/a")

    st.markdown("---")

    # ----------------------------------
    # Price Chart – flexibel (Jahre 1..10)
    # ----------------------------------
    st.subheader("Preisverlauf (Adjusted Close)")
    try:
        hist = yf.download(ticker, period=f"{years_window}y", interval="1d", progress=False)
        if isinstance(hist, pd.DataFrame) and not hist.empty:
            series = hist["Adj Close"].dropna()
            figp, axp = plt.subplots(figsize=(10, 4))
            axp.plot(series.index, series.values)
            axp.set_title(f"{ticker} – {years_window}J Adjusted Close")
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
    # Income Statement Bars & Peers
    # ----------------------------------
    cols = st.columns([1.2, 1.4])
    with cols[0]:
        st.subheader("Income (zuletzt verfügbar, Mrd)")
        names = ["Revenue", "Cost of Revenue", "Gross Profit", "Other Expenses", "Net Income"]
        values = [bn(revenue), bn(cost_rev), bn(gross_profit), bn(other_expenses), bn(net_income)]
        df_plot = pd.DataFrame({"Item": names, "Value": values})
        fig, ax = plt.subplots(figsize=(6.5, 4.0))
        ax.bar(df_plot["Item"], df_plot["Value"]) 
        ax.set_ylabel("€ bn")
        ax.set_title("BMPS – Ergebnisblöcke (letzte Periode)")
        for i, v in enumerate(values):
            if pd.notna(v):
                ax.text(i, v if v >= 0 else 0, f"{v:.2f}", ha='center', va='bottom')
        plt.xticks(rotation=15)
        st.pyplot(fig, clear_figure=True)

    with cols[1]:
        st.subheader("Peers – Forward P/E & Market Cap")
        peers = [p.strip() for p in peers_text.split(',') if p.strip()]
        peer_rows = []
        for p in peers:
            try:
                pi = load_ticker_info(p)
                pinf = pi["info"]
                fpe = safe_get(pinf, "forwardPE", np.nan)
                cap = safe_get(pinf, "marketCap", np.nan)
                lname = safe_get(pinf, "longName", p)
                if pd.notna(fpe):
                    peer_rows.append((lname, p, float(fpe), bn(cap)))
            except Exception:
                continue
        peer_df = pd.DataFrame(peer_rows, columns=["Company", "Ticker", "Forward PE", "MktCap (bn)"]).sort_values("Forward PE")
        if not peer_df.empty:
            fig2, ax2 = plt.subplots(figsize=(6.8, 4.2))
            ax2.barh(peer_df["Company"], peer_df["Forward PE"])  
            for i, (pe, cap) in enumerate(zip(peer_df["Forward PE"], peer_df["MktCap (bn)"])):
                ax2.text(pe, i, f"  PE {pe:.1f} | MCap {cap:.1f}bn", va='center')
            ax2.set_xlabel("Forward P/E")
            ax2.set_title("Peer‑Vergleich")
            st.pyplot(fig2, clear_figure=True)
            st.dataframe(peer_df, use_container_width=True)
        else:
            st.info("Keine Peer‑Daten verfügbar.")

    st.markdown("---")

    # ----------------------------------
    # Ownership – falls verfügbar
    # ----------------------------------
    own1, own2 = st.columns([1.2, 1.0])

    with own1:
        st.subheader("Top Holder (falls vorhanden)")
        holders_df = None
        try:
            inst = tkr.institutional_holders
            if isinstance(inst, pd.DataFrame) and not inst.empty:
                holders_df = inst.copy()
                holders_df.rename(columns={"% Out": "% Out"}, inplace=True)
        except Exception:
            holders_df = None
        if holders_df is not None:
            st.dataframe(holders_df.head(10), use_container_width=True)
        else:
            st.caption("")

    with own2:
        st.subheader("Eigentümer‑Kreise (Skizze)")
        labels = []
        sizes = []
        try:
            mh = tkr.major_holders
            if isinstance(mh, pd.DataFrame) and not mh.empty:
                for row in mh.iloc[:, 0].astype(str).tolist():
                    labels.append(row)
                for val in mh.iloc[:, 1].astype(str).tolist():
                    try:
                        sizes.append(float(val.strip('%')))
                    except Exception:
                        sizes.append(np.nan)
        except Exception:
            pass
        if sizes and all(pd.notna(s) for s in sizes):
            fig3, ax3 = plt.subplots(figsize=(5.0, 5.0))
            ax3.pie(sizes, labels=labels, autopct="%1.1f%%")
            ax3.set_title("Major Holders")
            st.pyplot(fig3, clear_figure=True)
        else:
            st.caption("Major‑Holder‑Anteile nicht verfügbar.")

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
