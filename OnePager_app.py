# app.py
from __future__ import annotations
import math
from typing import Dict, List, Tuple, Optional

import requests
import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import matplotlib.pyplot as plt

# ──────────────────────────────────────────────────────────────────────────────
# Seite & Stil
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="SHI – STOCK CHECK", layout="wide")

st.markdown(
    """
    <style>
      .main .block-container { padding-top: 3.2rem !important; padding-bottom: 0.6rem; }
      @media (max-width: 768px) { .main .block-container { padding-top: 3.8rem !important; } }
      .page-title { font-size:1.8rem; font-weight:800; margin:0.2rem 0 .6rem 0; line-height:1.25; }
      .kpi-wrap { margin:0.15rem 0; }
      .kpi-label { font-size:.78rem; color:#444; text-transform:uppercase; letter-spacing:.02em; }
      .kpi-value { font-size:1.38rem; font-weight:700; color:#111; }
      .kpi-green { color:#16a34a !important; }
      .smallnote { color:#666; font-size:.8rem; }
      [data-testid="stMetricValue"] { font-size:1.2rem; }
      [data-testid="stMetricLabel"] { font-size:.75rem; color:#666; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
def safe_get(d: Dict, key: str, default=np.nan):
    try:
        v = d.get(key, default)
        return np.nan if v is None else v
    except Exception:
        return default

def currency_symbol(code: str) -> str:
    m = {"EUR":"€","USD":"$","GBP":"£","JPY":"¥","CHF":"CHF","CAD":"$","AUD":"$","SEK":"kr","NOK":"kr","DKK":"kr","PLN":"zł","HKD":"$"}
    return m.get(str(code).upper(), str(code))

def bn(x: float) -> float:
    return float(x) / 1e9 if pd.notna(x) else np.nan

def normalize_percent_robust(x: float) -> float:
    if pd.isna(x): return np.nan
    try: y = float(x)
    except Exception: return np.nan
    while y > 1.0:
        y /= 100.0
    return np.nan if y < 0 else y

def first_notna(*vals):
    for v in vals:
        if v is not None and not (isinstance(v, float) and math.isnan(v)):
            return v
    return np.nan

def get_income_value(df: pd.DataFrame, candidates: List[str]) -> float:
    if not isinstance(df, pd.DataFrame) or df.empty: return np.nan
    idx_map = {str(i).lower(): i for i in df.index}
    for c in candidates:
        key = c.lower()
        if key in idx_map:
            ser = df.loc[idx_map[key]]
            try: return float(ser.dropna().iloc[0])
            except Exception: continue
    return np.nan

def compute_dividend_yield(tkr: yf.Ticker, price: float, info: Dict) -> float:
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
                if ttm > 0: y = float(ttm / price)
        except Exception:
            pass
    if pd.notna(y) and y > 0.5:
        return np.nan
    return y

def kpi(col, label: str, value_str: str, highlight: bool = False):
    klass = "kpi-value kpi-green" if highlight else "kpi-value"
    col.markdown(
        f"<div class='kpi-wrap'><div class='kpi-label'>{label}</div>"
        f"<div class='{klass}'>{value_str}</div></div>",
        unsafe_allow_html=True,
    )

# Put/Call Ratio (nächste Fälligkeit)
def compute_put_call_ratio(tkr: yf.Ticker) -> Tuple[float, float, Optional[str]]:
    try:
        exps = getattr(tkr, "options", []) or []
        if not exps:
            return np.nan, np.nan, None
        today = pd.Timestamp.today().normalize()
        dates = [pd.to_datetime(x, errors="coerce") for x in exps]
        future = [d for d in dates if pd.notna(d) and d >= today]
        exp_dt = min(future) if future else min([d for d in dates if pd.notna(d)])
        exp = exp_dt.strftime("%Y-%m-%d")
        oc = tkr.option_chain(exp)
        calls = getattr(oc, "calls", pd.DataFrame())
        puts  = getattr(oc, "puts",  pd.DataFrame())
        c_vol = float(pd.to_numeric(calls.get("volume"), errors="coerce").fillna(0).sum()) if not calls.empty else 0.0
        p_vol = float(pd.to_numeric(puts.get("volume"),  errors="coerce").fillna(0).sum()) if not puts.empty  else 0.0
        pcr_vol = p_vol / c_vol if c_vol > 0 else np.nan
        c_oi = float(pd.to_numeric(calls.get("openInterest"), errors="coerce").fillna(0).sum()) if not calls.empty else 0.0
        p_oi = float(pd.to_numeric(puts.get("openInterest"),  errors="coerce").fillna(0).sum()) if not puts.empty  else 0.0
        pcr_oi = p_oi / c_oi if c_oi > 0 else np.nan
        return pcr_vol, pcr_oi, exp
    except Exception:
        return np.nan, np.nan, None

# VOI-Ratios (nächste Fälligkeit)
def compute_voi_ratios(tkr: yf.Ticker) -> Tuple[float, float, float, Optional[str]]:
    try:
        exps = getattr(tkr, "options", []) or []
        if not exps:
            return np.nan, np.nan, np.nan, None
        today = pd.Timestamp.today().normalize()
        dates = [pd.to_datetime(x, errors="coerce") for x in exps]
        future = [d for d in dates if pd.notna(d) and d >= today]
        exp_dt = min(future) if future else min([d for d in dates if pd.notna(d)])
        exp = exp_dt.strftime("%Y-%m-%d")
        oc = tkr.option_chain(exp)
        calls = getattr(oc, "calls", pd.DataFrame())
        puts  = getattr(oc, "puts",  pd.DataFrame())
        c_vol = float(pd.to_numeric(calls.get("volume"),       errors="coerce").fillna(0).sum()) if not calls.empty else 0.0
        p_vol = float(pd.to_numeric(puts.get("volume"),        errors="coerce").fillna(0).sum()) if not puts.empty  else 0.0
        c_oi  = float(pd.to_numeric(calls.get("openInterest"), errors="coerce").fillna(0).sum()) if not calls.empty else 0.0
        p_oi  = float(pd.to_numeric(puts.get("openInterest"),  errors="coerce").fillna(0).sum()) if not puts.empty  else 0.0
        voi_total = (p_vol + c_vol) / (p_oi + c_oi) if (p_oi + c_oi) > 0 else np.nan
        voi_call  =  c_vol / c_oi if c_oi > 0 else np.nan
        voi_put   =  p_vol / p_oi if p_oi > 0 else np.nan
        return voi_total, voi_call, voi_put, exp
    except Exception:
        return np.nan, np.nan, np.nan, None

# Yahoo-Suche
def yahoo_symbol_search(query: str, limit: int = 12) -> List[Dict]:
    if not query:
        return []
    headers = {"User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                               "AppleWebKit/537.36 (KHTML, like Gecko) "
                               "Chrome/124.0 Safari/537.36")}
    try:
        url = "https://query2.finance.yahoo.com/v1/finance/search"
        params = {"q": query, "quotesCount": limit, "newsCount": 0, "listsCount": 0}
        r = requests.get(url, params=params, headers=headers, timeout=8)
        r.raise_for_status()
        quotes = (r.json() or {}).get("quotes", []) or []
        out = []
        for q in quotes:
            sym = q.get("symbol")
            name = q.get("shortname") or q.get("longname") or q.get("name") or ""
            exch = q.get("exchDisp") or q.get("exchange") or ""
            if sym and name:
                out.append({"symbol": sym, "name": name, "exchange": exch})
        if out:
            return out[:limit]
    except Exception:
        pass
    try:
        url = "https://autoc.finance.yahoo.com/autoc"
        params = {"query": query, "region": 1, "lang": "en"}
        r = requests.get(url, params=params, headers=headers, timeout=8)
        r.raise_for_status()
        results = ((r.json() or {}).get("ResultSet", {}) or {}).get("Result", []) or []
        out = []
        for e in results:
            sym = e.get("symbol")
            name = e.get("name")
            exch = e.get("exchDisp") or e.get("exch") or ""
            if sym and name:
                out.append({"symbol": sym, "name": name, "exchange": exch})
        return out[:limit]
    except Exception:
        return []

# CSV-Ticker-Parsing
def parse_tickers(s: str, max_n: int = 10) -> List[str]:
    if not s:
        return []
    for sep in [";", "\n", "\t", " "]:
        s = s.replace(sep, ",")
    toks = [t.strip().upper() for t in s.split(",") if t.strip()]
    return list(dict.fromkeys(toks))[:max_n]

def parse_csv_tickers(file, max_n: int = 10) -> List[str]:
    try:
        df = pd.read_csv(file, dtype=str)
        if df.empty:
            return []
        cols = [c for c in df.columns if str(c).lower() in ("ticker","symbol","code")]
        if not cols:
            cols = [df.columns[0]]
        values: List[str] = []
        for c in cols:
            series = df[c].dropna().astype(str).tolist()
            for cell in series:
                for sep in [",",";"," ","\t","\n"]:
                    cell = cell.replace(sep, ",")
                values.extend([t.strip().upper() for t in cell.split(",") if t.strip()])
        return list(dict.fromkeys(values))[:max_n]
    except Exception:
        return []

# Kennzahlen-DF (für Einzeltitel-Export)
def _as_num(x):
    try:
        if pd.isna(x): return np.nan
        return float(x)
    except Exception:
        return np.nan

def build_metrics_df(meta: Dict, info: Dict, computed: Dict) -> pd.DataFrame:
    cur = meta.get("currency", "EUR")
    rows: List[Tuple[str, float, str]] = []

    rows += [
        ("Ticker", np.nan, meta.get("label_tkr", "")),
        ("Name", np.nan, meta.get("long_name", "")),
        ("Exchange", np.nan, meta.get("exchange", "")),
        ("Country", np.nan, meta.get("country", "")),
        ("Currency", np.nan, cur),
        ("Employees", _as_num(meta.get("employees")), "count"),
        ("Shares Outstanding (bn)", _as_num(bn(meta.get("shares"))), "bn"),
        ("Market Cap (bn)", _as_num(bn(meta.get("mktcap"))), f"bn {cur}"),
        ("Price", _as_num(meta.get("price")), cur),
    ]

    rows += [
        ("Trailing P/E", _as_num(info.get("trailingPE")), "x"),
        ("P/S (TTM)", _as_num(info.get("priceToSalesTrailing12Months")), "x"),
        ("P/B", _as_num(info.get("priceToBook")), "x"),
        ("Dividend Yield", _as_num(computed.get("dividend_yield") * 100 if pd.notna(computed.get("dividend_yield")) else np.nan), "%"),
        ("Payout Ratio", _as_num(computed.get("payout_ratio") * 100 if pd.notna(computed.get("payout_ratio")) else np.nan), "%"),
        ("Put/Call Ratio (vol)", _as_num(computed.get("pcr_vol")), "x"),
        ("Put/Call Ratio (OI)",  _as_num(computed.get("pcr_oi")),  "x"),
        ("VOI (Total)", _as_num(computed.get("voi_total")), "x"),
        ("VOI (Call)",  _as_num(computed.get("voi_call")),  "x"),
        ("VOI (Put)",   _as_num(computed.get("voi_put")),   "x"),
    ]

    rows += [
        ("Enterprise Value (bn)", _as_num(bn(info.get("enterpriseValue"))), f"bn {cur}"),
        ("EV/Revenue", _as_num(first_notna(info.get("enterpriseToRevenue"), info.get("enterpriseToRev"))), "x"),
        ("EV/EBITDA", _as_num(info.get("enterpriseToEbitda")), "x"),
        ("Profit Margin", _as_num(info.get("profitMargins") * 100 if pd.notna(info.get("profitMargins")) else np.nan), "%"),
        ("ROA (ttm)", _as_num(info.get("returnOnAssets") * 100 if pd.notna(info.get("returnOnAssets")) else np.nan), "%"),
        ("ROE (ttm)", _as_num(info.get("returnOnEquity") * 100 if pd.notna(info.get("returnOnEquity")) else np.nan), "%"),
        ("Revenue (ttm) (bn)", _as_num(bn(info.get("totalRevenue"))), f"bn {cur}"),
        ("Net Income to Common (ttm) (bn)", _as_num(bn(info.get("netIncomeToCommon"))), f"bn {cur}"),
    ]
    return pd.DataFrame(rows, columns=["Metric", "Value", "Unit"])

# ──────────────────────────────────────────────────────────────────────────────
# Rendering-Funktion für einen Ticker
# ──────────────────────────────────────────────────────────────────────────────
def render_ticker(ticker: str, years_window: int) -> Tuple[pd.DataFrame, Dict[str, float], Dict[str, float]]:
    try:
        tkr = yf.Ticker(ticker)
        info: Dict = tkr.info or {}

        # Basics
        long_name = safe_get(info, "longName", ticker)
        website = safe_get(info, "website", "")
        industry = safe_get(info, "industry", "")
        sector = safe_get(info, "sector", "")
        country = safe_get(info, "country", "")
        employees = safe_get(info, "fullTimeEmployees", np.nan)
        exch = first_notna(safe_get(info, "fullExchangeName", None), safe_get(info, "exchange", None), "")
        mktcap = first_notna(safe_get(info, "marketCap", None), safe_get(getattr(tkr,"fast_info",{}) or {}, "market_cap", None))
        shares = first_notna(safe_get(info, "sharesOutstanding", None), safe_get(getattr(tkr,"fast_info",{}) or {}, "shares_outstanding", None))
        price = first_notna(safe_get(info, "currentPrice", None), safe_get(getattr(tkr,"fast_info",{}) or {}, "last_price", None))
        currency = safe_get(info, "currency", "EUR")
        sym = currency_symbol(currency)
        label_tkr = (safe_get(info, "symbol", None) or ticker).upper()

        # KPIs
        trailing_pe = safe_get(info, "trailingPE", np.nan)
        ps_ttm = safe_get(info, "priceToSalesTrailing12Months", np.nan)
        pb = first_notna(safe_get(info, "priceToBook", None), np.nan)

        # Dividende
        dividend_yield = compute_dividend_yield(tkr, price, info)
        payout_ratio = normalize_percent_robust(safe_get(info, "payoutRatio", np.nan))

        # Optionen
        pcr_vol, pcr_oi, pcr_exp = compute_put_call_ratio(tkr)
        voi_total, voi_call, voi_put, voi_exp = compute_voi_ratios(tkr)

        # Financials
        fin_a = tkr.financials if isinstance(getattr(tkr, "financials", None), pd.DataFrame) else pd.DataFrame()
        fin_q = tkr.quarterly_financials if isinstance(getattr(tkr, "quarterly_financials", None), pd.DataFrame) else pd.DataFrame()
        fin = fin_a if not fin_a.empty else fin_q

        revenue      = get_income_value(fin, ["Total Revenue", "Revenue"])
        cost_rev     = get_income_value(fin, ["Cost Of Revenue", "Cost of Revenue", "Cost of revenue"])
        gross_profit = get_income_value(fin, ["Gross Profit", "Gross profit"])
        net_income   = get_income_value(fin, ["Net Income", "Net income", "Net Income Common Stockholders"])
        if pd.isna(gross_profit) and pd.notna(revenue) and pd.notna(cost_rev):
            gross_profit = revenue - cost_rev

        # Header
        st.markdown("---")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.subheader(long_name)
            st.write("**Ticker:**", label_tkr)
            st.write("**Exchange:**", exch)
            st.write("**Country:**", country)
            if website: st.write("**Web:**", website)
        with c2:
            st.write("**Industry:**", industry or "n/a")
            st.write("**Sector:**", sector or "n/a")
            st.write("**Employees:**", int(employees) if pd.notna(employees) else "n/a")
            st.write("**Shares Outstanding (bn):**", f"{bn(shares):.3f}" if pd.notna(shares) else "n/a")
        with c3:
            st.write("**Market Cap (bn):**", f"{bn(mktcap):.2f} {currency}" if pd.notna(mktcap) else "n/a")
            st.write("**Current Price:**", f"{price:.3f} {currency}" if pd.notna(price) else "n/a")
            ed = safe_get(info, "earningsDate", [])
            nxt = None
            if isinstance(ed, (list, tuple)) and ed:
                try: nxt = pd.to_datetime(ed[0]).date().isoformat()
                except Exception: pass
            st.write("**Next earnings:**", nxt or "n/a")
        if safe_get(info, "longBusinessSummary", ""):
            st.caption(safe_get(info, "longBusinessSummary", ""))

        # KPI-Zeile
        st.markdown("---")
        k1, k2, k3, k4, k5, k6, k7 = st.columns(7)
        pe_ok  = pd.notna(trailing_pe) and trailing_pe < 10
        pb_ok  = pd.notna(pb)          and pb < 1
        dy_ok  = pd.notna(dividend_yield) and dividend_yield > 0.05
        kpi(k1, "P/E RATIO", f"{trailing_pe:.2f}x" if pd.notna(trailing_pe) else "n/a", pe_ok)
        kpi(k2, "P/S (TTM)", f"{ps_ttm:.2f}" if pd.notna(ps_ttm) else "n/a", False)
        kpi(k3, "P/B", f"{pb:.2f}" if pd.notna(pb) else "n/a", pb_ok)
        kpi(k4, "DIVIDEND YIELD", "n/a" if pd.isna(dividend_yield) else f"{dividend_yield*100:.2f}%", dy_ok)
        kpi(k5, "PAYOUT RATIO", "n/a" if pd.isna(payout_ratio) else f"{payout_ratio*100:.2f}%", False)
        pcr_str = " | ".join([s for s in [f"vol {pcr_vol:.2f}" if pd.notna(pcr_vol) else None,
                                          f"OI {pcr_oi:.2f}"  if pd.notna(pcr_oi)  else None] if s]) or "n/a"
        kpi(k6, "PUT/CALL RATIO", pcr_str, False)
        voi_ok = bool(pd.notna(voi_total) and voi_total > 1)
        kpi(k7, "VOI (VOL/OI)", "n/a" if pd.isna(voi_total) else f"{voi_total:.2f}", voi_ok)
        if (pcr_exp or voi_exp):
            st.caption(f"Optionen-Metriken auf nächster Fälligkeit: {voi_exp or pcr_exp}.")

        # Charts
        st.markdown("---")
        ch_left, ch_right = st.columns(2)
        with ch_left:
            st.caption("Price (Close)")
            try:
                hist = yf.download(ticker, period=f"{years_window}y", interval="1d", progress=False)
                if isinstance(hist, pd.DataFrame) and not hist.empty:
                    series = hist["Close"].dropna()
                    figp, axp = plt.subplots(figsize=(4.8, 2.0))
                    axp.plot(series.index, series.values, linewidth=0.7)
                    axp.set_title(f"{label_tkr} – {years_window}y", fontsize=9)
                    axp.set_xlabel("Date", fontsize=6)
                    axp.set_ylabel(f"Price ({currency})", fontsize=6)
                    axp.grid(True, linestyle=":", alpha=0.22)
                    axp.tick_params(axis="both", labelsize=6)
                    st.pyplot(figp, clear_figure=True)
                else:
                    st.info("No price history available.")
            except Exception as e:
                st.warning(f"Could not load price data: {e}")
        with ch_right:
            st.caption("Earnings & Revenue (last period)")
            names = ["Revenue", "Cost of Revenue", "Gross Profit", "Earnings"]
            gross_val = gross_profit if pd.notna(gross_profit) else (
                revenue - cost_rev if pd.notna(revenue) and pd.notna(cost_rev) else np.nan
            )
            vals_abs = [bn(revenue), bn(cost_rev), bn(gross_val), bn(net_income)]
            fig, ax = plt.subplots(figsize=(4.8, 2.0))
            x = np.arange(len(names))
            bars = ax.bar(x, vals_abs, width=0.8)
            ax.set_ylabel(f"{sym} bn ({currency})", fontsize=7)
            ax.set_title(f"{label_tkr} – {currency}", fontsize=7)
            ax.set_xticks(x)
            ax.set_xticklabels(names, rotation=10, fontsize=7)
            ax.grid(True, linestyle=":", alpha=0.22, axis="y")
            for rect, v in zip(bars, vals_abs):
                if pd.notna(v):
                    ax.text(rect.get_x() + rect.get_width()/2, rect.get_height(),
                            f"{sym}{v:.2f}b", ha="center", va="bottom", fontsize=6)
            st.pyplot(fig, clear_figure=True)

        # Valuation & Profitability (Anzeige)
        st.markdown("---")
        st.subheader("Valuation & Profitability")
        ev         = safe_get(info, "enterpriseValue", np.nan)
        ev_rev     = first_notna(safe_get(info, "enterpriseToRevenue", None),
                                 safe_get(info, "enterpriseToRev", None), np.nan)
        ev_ebitda  = safe_get(info, "enterpriseToEbitda", np.nan)
        profit_m   = safe_get(info, "profitMargins", np.nan)
        roa        = safe_get(info, "returnOnAssets", np.nan)
        roe        = safe_get(info, "returnOnEquity", np.nan)
        revenue_ttm= safe_get(info, "totalRevenue", np.nan)
        nic_ttm    = safe_get(info, "netIncomeToCommon", np.nan)

        GREEN = 'color: #16a34a; font-weight:700;'
        def style_val_rows(row):
            m = row["Metric"]
            if m == "EV/EBITDA" and pd.notna(ev_ebitda) and ev_ebitda < 10: return [GREEN, GREEN]
            if m == "Profit Margin" and pd.notna(profit_m) and profit_m > 0.20: return [GREEN, GREEN]
            return ["", ""]
        val_rows: List[Tuple[str, str]] = [
            ("Enterprise Value",            "n/a" if pd.isna(ev)          else f"{currency_symbol(currency)}{bn(ev):.2f}b ({currency})"),
            ("Trailing P/E",                "n/a" if pd.isna(trailing_pe) else f"{trailing_pe:.2f}×"),
            ("EV/Revenue",                  "n/a" if pd.isna(ev_rev)      else f"{ev_rev:.2f}×"),
            ("EV/EBITDA",                   "n/a" if pd.isna(ev_ebitda)   else f"{ev_ebitda:.2f}×"),
            ("Profit Margin",               "n/a" if pd.isna(profit_m)    else f"{profit_m*100:.2f}%"),
            ("ROA (ttm)",                   "n/a" if pd.isna(roa)         else f"{roa*100:.2f}%"),
            ("ROE (ttm)",                   "n/a" if pd.isna(roe)         else f"{roe*100:.2f}%"),
            ("Revenue (ttm)",               "n/a" if pd.isna(revenue_ttm) else f"{currency_symbol(currency)}{bn(revenue_ttm):.2f}b ({currency})"),
            ("Net Income to Common (ttm)",  "n/a" if pd.isna(nic_ttm)     else f"{currency_symbol(currency)}{bn(nic_ttm):.2f}b ({currency})"),
        ]
        val_df = pd.DataFrame(val_rows, columns=["Metric", "Value"])
        st.table(val_df.style.apply(style_val_rows, axis=1))

        # Export Einzeltitel
        st.markdown("---")
        st.subheader("Export")
        meta_dict = {"currency": currency, "label_tkr": label_tkr, "long_name": long_name,
                     "exchange": exch, "country": country, "employees": employees,
                     "shares": shares, "mktcap": mktcap, "price": price}
        computed = {"dividend_yield": dividend_yield, "payout_ratio": payout_ratio,
                    "pcr_vol": pcr_vol, "pcr_oi": pcr_oi, "pcr_exp": pcr_exp,
                    "voi_total": voi_total, "voi_call": voi_call, "voi_put": voi_put}
        metrics_df = build_metrics_df(meta_dict, info, computed)
        col_us, col_eu = st.columns(2)
        with col_us:
            st.caption("CSV (US-Format: , & .)")
            st.download_button("⬇️ CSV (US)", data=metrics_df.to_csv(index=False).encode("utf-8"),
                               file_name=f"{label_tkr}_metrics.csv", mime="text/csv", use_container_width=True)
        with col_eu:
            st.caption("CSV (EU-Format: ; & ,)")
            st.download_button("⬇️ CSV (EU)", data=metrics_df.to_csv(index=False, sep=";", decimal=",", float_format="%.4f").encode("utf-8-sig"),
                               file_name=f"{label_tkr}_metrics_eu.csv", mime="text/csv", use_container_width=True)

        # Vergleichszeilen zurückgeben
        core_row = {
            "Ticker": label_tkr, "Name": long_name, "Währung": currency,
            "Preis": float(price) if pd.notna(price) else np.nan,
            "MktCap_bn": float(bn(mktcap)) if pd.notna(mktcap) else np.nan,
            "P/E": float(trailing_pe) if pd.notna(trailing_pe) else np.nan,
            "P/B": float(pb) if pd.notna(pb) else np.nan,
            "DivYield_%": float(dividend_yield*100) if pd.notna(dividend_yield) else np.nan,
            "PCR_vol": float(pcr_vol) if pd.notna(pcr_vol) else np.nan,
            "PCR_oi": float(pcr_oi) if pd.notna(pcr_oi) else np.nan,
            "VOI_total": float(voi_total) if pd.notna(voi_total) else np.nan,
        }
        val_row = {
            "Ticker": label_tkr,
            "EV_bn": float(bn(ev)) if pd.notna(ev) else np.nan,
            "PE": float(trailing_pe) if pd.notna(trailing_pe) else np.nan,
            "EV_Rev": float(ev_rev) if pd.notna(ev_rev) else np.nan,
            "EV_EBITDA": float(ev_ebitda) if pd.notna(ev_ebitda) else np.nan,
            "ProfitMargin_%": float(profit_m*100) if pd.notna(profit_m) else np.nan,
            "ROA_%": float(roa*100) if pd.notna(roa) else np.nan,
            "ROE_%": float(roe*100) if pd.notna(roe) else np.nan,
            "Revenue_bn": float(bn(revenue_ttm)) if pd.notna(revenue_ttm) else np.nan,
            "NetIncome_bn": float(bn(nic_ttm)) if pd.notna(nic_ttm) else np.nan,
        }
        return metrics_df, core_row, val_row

    except Exception as e:
        st.error(f"Could not load data for '{ticker}': {e}")
        return pd.DataFrame(), {}, {}

# ──────────────────────────────────────────────────────────────────────────────
# Session defaults
# ──────────────────────────────────────────────────────────────────────────────
if "ticker" not in st.session_state:
    st.session_state.ticker = "BMPS.MI"
if "tickers_raw" not in st.session_state:
    st.session_state.tickers_raw = st.session_state.ticker
if "search_results" not in st.session_state:
    st.session_state.search_results = []
if "search_query" not in st.session_state:
    st.session_state.search_query = ""

# ──────────────────────────────────────────────────────────────────────────────
# Layout: Suche links, Parameter rechts
# ──────────────────────────────────────────────────────────────────────────────
left, right = st.columns([2, 1])

with left:
    st.markdown("<div class='page-title'>SHI - STOCK CHECK</div>", unsafe_allow_html=True)
    st.text_input("Search company or ticker", key="search_query", placeholder="e.g. Microsoft or MSFT")
    if st.button("Search"):
        st.session_state.search_results = yahoo_symbol_search(st.session_state.search_query)

    results = st.session_state.search_results
    if results:
        st.caption("Search results")
        df_res = pd.DataFrame(results)
        st.dataframe(df_res[["symbol", "name", "exchange"]], use_container_width=True, hide_index=True)
        labels = [f"{r['symbol']} — {r['name']} ({r['exchange']})" for r in results]
        pick = st.selectbox("Pick one", options=list(range(len(results))),
                            format_func=lambda i: labels[i], key="search_pick")
        if st.button("Use selection"):
            chosen = results[pick]
            st.session_state.ticker = chosen["symbol"]
            st.session_state.tickers_raw = chosen["symbol"]
            st.success(f"Using {chosen['symbol']}")
            st.rerun()

with right:
    st.header("Parameters")
    st.text_input("Ticker(s) – bis zu 10, getrennt durch , ; Leerzeichen oder Zeilen",
                  key="tickers_raw", value=st.session_state.tickers_raw)
    upload = st.file_uploader("CSV mit Tickern (Spalte: ticker/symbol/code oder erste Spalte)", type=["csv"])
    years_window = st.slider("Price window (years)", 1, 10, 3, 1)

# ──────────────────────────────────────────────────────────────────────────────
# Core – Multi-Ticker Rendering
# ──────────────────────────────────────────────────────────────────────────────
tickers_text = parse_tickers(st.session_state.get("tickers_raw", ""), max_n=10)
tickers_csv  = parse_csv_tickers(upload, max_n=10) if upload else []
tickers = list(dict.fromkeys([*tickers_text, *tickers_csv]))[:10]

if not tickers:
    st.info("Mindestens einen Ticker eingeben oder CSV hochladen.")
    st.stop()

tabs = st.tabs(tickers)
core_rows: List[Dict[str, float]] = []
val_rows_all: List[Dict[str, float]] = []

for tab, t in zip(tabs, tickers):
    with tab:
        st.markdown(f"---\n### {t}")
        _, core_row, val_row = render_ticker(t, years_window)
        if core_row: core_rows.append(core_row)
        if val_row:  val_rows_all.append(val_row)

# Vergleiche
st.markdown("---")
st.subheader("Vergleich – Kerndaten")
if core_rows:
    comp_core = pd.DataFrame(core_rows).set_index("Ticker")
    st.dataframe(comp_core, use_container_width=True)
    c1, c2 = st.columns(2)
    with c1:
        st.download_button("⬇️ Vergleich Kerndaten (US CSV)",
                           data=comp_core.reset_index().to_csv(index=False).encode("utf-8"),
                           file_name="compare_core.csv", mime="text/csv", use_container_width=True)
    with c2:
        st.download_button("⬇️ Vergleich Kerndaten (EU CSV)",
                           data=comp_core.reset_index().to_csv(index=False, sep=";", decimal=",", float_format="%.4f").encode("utf-8-sig"),
                           file_name="compare_core_eu.csv", mime="text/csv", use_container_width=True)
else:
    st.info("Keine Kerndaten verfügbar.")

st.markdown("---")
st.subheader("Vergleich – Valuation & Profitability")
if val_rows_all:
    comp_val = pd.DataFrame(val_rows_all).set_index("Ticker")
    st.dataframe(comp_val, use_container_width=True)
    c3, c4 = st.columns(2)
    with c3:
        st.download_button("⬇️ Vergleich Valuation (US CSV)",
                           data=comp_val.reset_index().to_csv(index=False).encode("utf-8"),
                           file_name="compare_valuation.csv", mime="text/csv", use_container_width=True)
    with c4:
        st.download_button("⬇️ Vergleich Valuation (EU CSV)",
                           data=comp_val.reset_index().to_csv(index=False, sep=";", decimal=",", float_format="%.4f").encode("utf-8-sig"),
                           file_name="compare_valuation_eu.csv", mime="text/csv", use_container_width=True)
else:
    st.info("Keine Valuation-Daten verfügbar.")
