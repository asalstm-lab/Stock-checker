"""
Stock Setup Checker - a Streamlit web app.

Type a company name or ticker and the app pulls live (about 15 min delayed) data
from Yahoo Finance and checks: catalysts, valuation vs peers, momentum vs the
S&P 500, news, and links to filings.

Not financial advice. Data is unofficial and can be incomplete or wrong.
"""
import datetime as dt
import time

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

BENCH = "^GSPC"

SECTOR_DRIVERS = {
    "Energy": [("Crude oil (WTI)", "CL=F"), ("Crude oil (Brent)", "BZ=F"),
               ("Gasoline (RBOB)", "RB=F"), ("Heating oil / diesel", "HO=F"),
               ("Natural gas (US)", "NG=F")],
    "Basic Materials": [("Gold", "GC=F"), ("Silver", "SI=F"), ("Copper", "HG=F")],
    "Financial Services": [("US 10-year yield", "^TNX"), ("US 13-week yield", "^IRX")],
    "Real Estate": [("US 10-year yield", "^TNX")],
    "Utilities": [("US 10-year yield", "^TNX")],
    "Technology": [("Nasdaq Composite", "^IXIC"), ("Semiconductor index", "^SOX")],
    "Consumer Cyclical": [("Crude oil (WTI)", "CL=F"), ("S&P 500", "^GSPC")],
    "Industrials": [("Crude oil (WTI)", "CL=F"), ("Copper", "HG=F")],
}
INDUSTRY_DRIVERS = {
    "airline": [("Crude oil (WTI)", "CL=F"), ("Jet fuel proxy (heating oil)", "HO=F")],
    "solar": [("US 10-year yield", "^TNX"), ("Natural gas (US)", "NG=F")],
    "semiconductor": [("Semiconductor index", "^SOX")],
}


# ------------------------------------------------------------------ data layer
def _retry(fn, tries=3, delay=1.5):
    for i in range(tries):
        try:
            return fn()
        except Exception:
            time.sleep(delay * (i + 1))
    return None


@st.cache_data(ttl=3600, show_spinner=False)
def search_symbols(query):
    res = _retry(lambda: yf.Search(query, max_results=10).quotes) or []
    out = []
    for r in res:
        if r.get("quoteType") == "EQUITY" and r.get("symbol"):
            out.append({"symbol": r["symbol"],
                        "name": r.get("longname") or r.get("shortname") or r["symbol"],
                        "exch": r.get("exchDisp", "")})
    return out


@st.cache_data(ttl=900, show_spinner=False)
def get_info(sym):
    info = _retry(lambda: yf.Ticker(sym).info)
    return info if isinstance(info, dict) else {}


@st.cache_data(ttl=900, show_spinner=False)
def get_prices(sym, period="2y"):
    df = _retry(lambda: yf.Ticker(sym).history(period=period, auto_adjust=True))
    if df is None or len(df) == 0 or "Close" not in df:
        return pd.Series(dtype=float)
    s = df["Close"].dropna()
    s.index = pd.to_datetime(s.index).tz_localize(None).normalize()
    return s[~s.index.duplicated(keep="last")]


@st.cache_data(ttl=3600, show_spinner=False)
def get_peers(industry_key, sym, n=6):
    try:
        df = yf.Industry(industry_key).top_companies
        return [s for s in df.index if s != sym][:n]
    except Exception:
        return []


@st.cache_data(ttl=3600, show_spinner=False)
def get_calendar(sym):
    try:
        cal = yf.Ticker(sym).calendar
    except Exception:
        return {}
    if isinstance(cal, dict):
        return cal
    if isinstance(cal, pd.DataFrame) and not cal.empty:
        return {k: list(cal.loc[k]) for k in cal.index}
    return {}


@st.cache_data(ttl=3600, show_spinner=False)
def get_earnings_history(sym):
    df = _retry(lambda: yf.Ticker(sym).get_earnings_dates(limit=12))
    return df if isinstance(df, pd.DataFrame) else pd.DataFrame()


@st.cache_data(ttl=600, show_spinner=False)
def get_news(sym):
    raw = _retry(lambda: yf.Ticker(sym).news) or []
    out = []
    for n in raw:
        c = n.get("content", n) if isinstance(n, dict) else {}
        title = c.get("title")
        url = ((c.get("canonicalUrl") or {}).get("url")
               or (c.get("clickThroughUrl") or {}).get("url") or c.get("link"))
        pub = (c.get("provider") or {}).get("displayName") or c.get("publisher")
        date = c.get("pubDate") or c.get("displayTime")
        if title:
            out.append({"title": title, "url": url, "publisher": pub, "date": date})
    return out


@st.cache_data(ttl=3600, show_spinner=False)
def get_isin(sym):
    try:
        v = yf.Ticker(sym).isin
        return v if v and v != "-" else None
    except Exception:
        return None


# ------------------------------------------------------------------ calculations
def pct(s, days):
    if s is None or len(s) <= days:
        return None
    return float(s.iloc[-1] / s.iloc[-days - 1] - 1) * 100


def rsi(s, n=14):
    if s is None or len(s) < n + 2:
        return None
    d = s.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    last_up, last_dn = up.iloc[-1], dn.iloc[-1]
    if last_dn == 0:
        return 100.0
    return float(100 - 100 / (1 + last_up / last_dn))


def num(x):
    try:
        v = float(x)
        return v if np.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def valuation_row(sym):
    i = get_info(sym)
    mc, fcf = num(i.get("marketCap")), num(i.get("freeCashflow"))
    return {"Ticker": sym,
            "P/E": num(i.get("trailingPE")),
            "Fwd P/E": num(i.get("forwardPE")),
            "EV/EBITDA": num(i.get("enterpriseToEbitda")),
            "P/B": num(i.get("priceToBook")),
            "FCF yield %": (fcf / mc * 100) if (fcf is not None and mc) else None}


def crack_spread():
    cl, rb, ho = get_prices("CL=F"), get_prices("RB=F"), get_prices("HO=F")
    df = pd.concat([cl, rb, ho], axis=1, keys=["cl", "rb", "ho"]).dropna()
    if len(df) < 100:
        return pd.Series(dtype=float)
    return ((2 * df["rb"] * 42 + df["ho"] * 42 - 3 * df["cl"]) / 3).rename("3-2-1 crack spread ($/bbl)")


def fmt(x, d=2, suffix="", plus=False):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "n/a"
    return f"{x:+,.{d}f}{suffix}" if plus else f"{x:,.{d}f}{suffix}"


# ------------------------------------------------------------------ UI sections
def section_links(sym, name):
    st.subheader("Screener and filing links")
    links = [
        ("Yahoo Finance", f"https://finance.yahoo.com/quote/{sym}/"),
        ("Yahoo: analysis", f"https://finance.yahoo.com/quote/{sym}/analysis/"),
        ("Finviz", f"https://finviz.com/quote.ashx?t={sym}"),
        ("TradingView", f"https://www.tradingview.com/symbols/{sym.replace('.', '-')}/"),
        ("SEC filings (US)", f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={sym}&type=10-&dateb=&owner=include&count=40"),
        ("Investor relations", f"https://www.google.com/search?q={name.replace(' ', '+')}+investor+relations"),
    ]
    cols = st.columns(len(links))
    for c, (label, url) in zip(cols, links):
        c.link_button(label, url, width="stretch")
    isin = get_isin(sym)
    if isin:
        st.caption(f"ISIN: **{isin}** (search this in Trade Republic to confirm you can trade it)")


def build_scorecard(sym, info, s, bench, peers_df, cal, hist, crack):
    flags = []
    price = num(info.get("currentPrice") or info.get("regularMarketPrice")) or (float(s.iloc[-1]) if len(s) else None)

    if peers_df is not None and len(peers_df) > 1:
        med = peers_df.iloc[1:].median(numeric_only=True)
        me = peers_df.iloc[0]
        for col, low_good, label in [("Fwd P/E", True, "Forward P/E cheaper than peers"),
                                     ("EV/EBITDA", True, "EV/EBITDA cheaper than peers"),
                                     ("FCF yield %", False, "Free cash flow yield higher than peers")]:
            a, b = num(me.get(col)), num(med.get(col))
            if a is None or b is None or b <= 0:
                flags.append((label, None, "not enough data"))
            else:
                ok = a < 0.85 * b if low_good else a > 1.15 * b
                flags.append((label, ok, f"{a:.2f} vs peer median {b:.2f}"))
    else:
        flags.append(("Valuation vs peers", None, "no peers found; add some above"))

    if len(s) > 60:
        ma50 = float(s.rolling(50).mean().iloc[-1])
        flags.append(("Price above 50-day average", float(s.iloc[-1]) > ma50, f"{s.iloc[-1]:.2f} vs {ma50:.2f}"))
        if len(s) >= 200:
            ma200 = float(s.rolling(200).mean().iloc[-1])
            flags.append(("Price above 200-day average", float(s.iloc[-1]) > ma200, f"{s.iloc[-1]:.2f} vs {ma200:.2f}"))
        r3, b3 = pct(s, 63), pct(bench, 63)
        if r3 is not None and b3 is not None:
            flags.append(("Beating the S&P 500 over 3 months", r3 > b3, f"{r3:+.1f}% vs {b3:+.1f}%"))
        r = rsi(s)
        if r is not None:
            flags.append(("Not overheated (RSI under 70)", r < 70, f"RSI {r:.0f}"))

    target = num(info.get("targetMeanPrice"))
    if target and price:
        up = (target / price - 1) * 100
        flags.append(("Analyst average target >10% above price", up > 10, f"{up:+.1f}% upside"))

    if not hist.empty and "Surprise(%)" in hist:
        past = hist.dropna(subset=["Reported EPS"]).head(4)
        if len(past):
            beats = int((past["Surprise(%)"] > 0).sum())
            flags.append(("Beat EPS estimates in most of the last 4 quarters", beats >= 3, f"{beats}/{len(past)} beats"))

    nxt = next_earnings_date(cal, hist)
    if nxt is not None:
        days = (nxt - dt.date.today()).days
        flags.append(("Earnings (a dated catalyst) within 90 days", 0 <= days <= 90, f"{nxt} ({days} days)"))

    if crack is not None and len(crack) > 252:
        flags.append(("Refining margin above its 1-year average", float(crack.iloc[-1]) > float(crack.tail(252).mean()),
                      f"{crack.iloc[-1]:.1f} vs {crack.tail(252).mean():.1f} $/bbl"))
    return flags


def next_earnings_date(cal, hist):
    ed = cal.get("Earnings Date") if cal else None
    dates = []
    for d in (ed or []):
        try:
            dates.append(pd.to_datetime(d).date())
        except Exception:
            pass
    today = dt.date.today()
    future = sorted(d for d in dates if d >= today)
    if future:
        return future[0]
    if not hist.empty:
        fut = [pd.to_datetime(i).date() for i in hist.index if pd.to_datetime(i).date() >= today]
        if fut:
            return min(fut)
    return None


def render_scorecard(flags, s):
    scored = [f for f in flags if f[1] is not None]
    good = sum(1 for f in scored if f[1])
    st.metric("Flags positive", f"{good} / {len(scored)}")
    for label, ok, detail in flags:
        icon = "⚪" if ok is None else ("✅" if ok else "❌")
        st.markdown(f"{icon} **{label}** &nbsp; <span style='opacity:.7'>{detail}</span>", unsafe_allow_html=True)
    r3 = pct(s, 63)
    if r3 is not None and r3 > 50:
        st.warning("Up more than 50% in 3 months: much of the move may already be priced in.")
    st.caption("These are prompts to investigate, not predictions or advice. A low P/E can mean cheap OR temporarily high profits (common in cyclical sectors like refining and chips). "
               "The three things that matter most for big moves are not in this list: a real external shock, estimates rising faster than the price, and a dated catalyst. Judge those yourself from the news.")


def render_valuation(peers_df):
    st.subheader("Valuation vs peers")
    if peers_df is None or len(peers_df) < 2:
        st.info("No peers found automatically. Type comparison tickers in the box above (comma-separated).")
    if peers_df is not None:
        df = peers_df.copy()
        if len(df) > 1:
            med = df.iloc[1:].median(numeric_only=True)
            df.loc[len(df)] = {"Ticker": "PEER MEDIAN", **med.to_dict()}
        st.dataframe(df.style.format({c: "{:,.2f}" for c in df.columns if c != "Ticker"}, na_rep="n/a"),
                     hide_index=True, width="stretch")
    st.caption("P/E = price relative to profits. Forward P/E uses expected profits. EV/EBITDA includes debt. P/B = price to book value. "
               "FCF yield = free cash flow as % of market value (higher = cheaper). Compare with the company's own history on a site like Finviz or Macrotrends too.")


def render_momentum(sym, s, bench):
    st.subheader("Momentum vs the S&P 500")
    if len(s) < 60:
        st.warning("Not enough price history returned. Yahoo may be rate-limiting; try again in a minute.")
        return
    rows = []
    for label, d in [("1 month", 21), ("3 months", 63), ("6 months", 126), ("12 months", 252)]:
        r, b = pct(s, d), pct(bench, d)
        rows.append({"Period": label, "Stock %": r, "S&P 500 %": b, "Difference (pp)": (r - b) if (r is not None and b is not None) else None})
    st.dataframe(pd.DataFrame(rows).style.format({"Stock %": "{:+.1f}", "S&P 500 %": "{:+.1f}", "Difference (pp)": "{:+.1f}"}, na_rep="n/a"),
                 hide_index=True, width="stretch")
    c1, c2, c3, c4 = st.columns(4)
    ma50 = float(s.rolling(50).mean().iloc[-1])
    ma200 = float(s.rolling(200).mean().iloc[-1]) if len(s) >= 200 else None
    r = rsi(s)
    tail = s.tail(252)
    pos = (s.iloc[-1] - tail.min()) / (tail.max() - tail.min()) * 100 if tail.max() > tail.min() else None
    c1.metric("50-day average", fmt(ma50))
    c2.metric("200-day average", fmt(ma200))
    c3.metric("RSI (14)", fmt(r, 0))
    c4.metric("Position in 52-week range", fmt(pos, 0, "%"))
    if r is not None and r >= 70:
        st.warning("RSI above 70: the stock is stretched, and pullbacks are more likely.")
    both = pd.concat([s, bench], axis=1, keys=[sym, "S&P 500"]).dropna().tail(252)
    if len(both) > 10:
        st.caption("Last 12 months, both rebased to 100")
        st.line_chart(both / both.iloc[0] * 100)


def render_catalysts(sym, info, cal, hist, crack):
    st.subheader("Catalysts")
    nxt = next_earnings_date(cal, hist)
    c1, c2, c3 = st.columns(3)
    c1.metric("Next earnings date", str(nxt) if nxt else "n/a")
    eps_avg = num(cal.get("Earnings Average")) if cal else None
    c2.metric("Expected EPS", fmt(eps_avg))
    c3.metric("Analyst opinion", str(info.get("recommendationKey") or "n/a").replace("_", " "))

    price = num(info.get("currentPrice") or info.get("regularMarketPrice"))
    t_mean, t_hi, t_lo = num(info.get("targetMeanPrice")), num(info.get("targetHighPrice")), num(info.get("targetLowPrice"))
    st.markdown("**Analyst price targets**")
    a, b, c, d = st.columns(4)
    a.metric("Average", fmt(t_mean), fmt((t_mean / price - 1) * 100, 1, "%", True) if (t_mean and price) else None)
    b.metric("High", fmt(t_hi))
    c.metric("Low", fmt(t_lo))
    d.metric("Analysts", fmt(num(info.get("numberOfAnalystOpinions")), 0))

    st.markdown("**Earnings surprise history**")
    if not hist.empty and "Surprise(%)" in hist:
        past = hist.dropna(subset=["Reported EPS"]).head(8).copy()
        past.index = pd.to_datetime(past.index).strftime("%Y-%m-%d")
        st.dataframe(past[["EPS Estimate", "Reported EPS", "Surprise(%)"]].style.format("{:,.2f}", na_rep="n/a"), width="stretch")
        st.bar_chart(past["Surprise(%)"].iloc[::-1])
    else:
        st.info("No earnings history returned for this ticker.")

    sector, industry = info.get("sector"), (info.get("industry") or "")
    drivers = list(SECTOR_DRIVERS.get(sector, []))
    for key, extra in INDUSTRY_DRIVERS.items():
        if key in industry.lower():
            drivers += [d for d in extra if d not in drivers]
    if drivers:
        st.markdown(f"**Sector drivers** ({sector or industry})")
        rows = []
        for label, tick in drivers:
            ds = get_prices(tick)
            rows.append({"Driver": label, "Latest": float(ds.iloc[-1]) if len(ds) else None,
                         "1m %": pct(ds, 21), "3m %": pct(ds, 63), "12m %": pct(ds, 252)})
        st.dataframe(pd.DataFrame(rows).style.format({"Latest": "{:,.2f}", "1m %": "{:+.1f}", "3m %": "{:+.1f}", "12m %": "{:+.1f}"}, na_rep="n/a"),
                     hide_index=True, width="stretch")
        st.caption("Yields (^TNX, ^IRX) are in percent; the % columns show relative change.")

    if crack is not None and len(crack) > 100:
        st.markdown("**Refining margin: 3-2-1 crack spread ($/barrel)**")
        k1, k2, k3 = st.columns(3)
        k1.metric("Now", fmt(float(crack.iloc[-1]), 1))
        k2.metric("3 months ago", fmt(float(crack.iloc[-64]), 1))
        k3.metric("1-year average", fmt(float(crack.tail(252).mean()), 1))
        st.line_chart(crack.tail(252))
        st.caption("Approximation: (2 x gasoline + 1 x diesel - 3 x crude) / 3 from futures prices. Refiners earn more when this rises.")


def render_news(sym, name):
    st.subheader("Latest headlines")
    news = get_news(sym)
    if not news:
        st.info("No headlines returned. Check the Google News link below.")
    for n in news[:10]:
        meta = " - ".join(x for x in [n.get("publisher"), (n.get("date") or "")[:10]] if x)
        st.markdown(f"- [{n['title']}]({n['url']}) &nbsp; <span style='opacity:.6'>{meta}</span>" if n.get("url") else f"- {n['title']}",
                    unsafe_allow_html=True)
    st.markdown(f"[More on Google News](https://www.google.com/search?q={name.replace(' ', '+')}+stock&tbm=nws)")
    st.caption("Read the actual earnings reports and guidance (SEC and investor-relations links above), not just headlines or social media.")


# ------------------------------------------------------------------ main
def analyse(sym, manual_peers):
    info = get_info(sym)
    s = get_prices(sym)
    bench = get_prices(BENCH)
    if len(s) == 0:
        st.error("No price data came back for this ticker. Yahoo may be busy or the ticker may be wrong (European stocks need a suffix such as .DE). Try again in a minute.")
        return
    name = info.get("longName") or info.get("shortName") or sym
    price = num(info.get("currentPrice") or info.get("regularMarketPrice")) or float(s.iloc[-1])

    st.header(f"{name} ({sym})")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Price", f"{fmt(price)} {info.get('currency', '')}")
    m2.metric("Market cap", fmt((num(info.get("marketCap")) or 0) / 1e9, 1, "B") if info.get("marketCap") else "n/a")
    m3.metric("Sector", info.get("sector") or "n/a")
    m4.metric("Industry", info.get("industry") or "n/a")
    st.caption(f"Last price date in data: {s.index[-1].date()}. Yahoo data is typically delayed ~15 minutes.")

    peers = manual_peers or (get_peers(info.get("industryKey"), sym) if info.get("industryKey") else [])
    with st.spinner("Comparing with peers..."):
        peers_df = pd.DataFrame([valuation_row(sym)] + [valuation_row(p) for p in peers])
    cal, hist = get_calendar(sym), get_earnings_history(sym)
    crack = crack_spread() if "refin" in (info.get("industry") or "").lower() else None

    flags = build_scorecard(sym, info, s, bench, peers_df, cal, hist, crack)
    tabs = st.tabs(["Scorecard", "Valuation", "Momentum", "Catalysts", "News & filings"])
    with tabs[0]:
        render_scorecard(flags, s)
    with tabs[1]:
        render_valuation(peers_df)
    with tabs[2]:
        render_momentum(sym, s, bench)
    with tabs[3]:
        render_catalysts(sym, info, cal, hist, crack)
    with tabs[4]:
        render_news(sym, name)
        section_links(sym, name)


def main():
    st.set_page_config(page_title="Stock Setup Checker", page_icon="📈", layout="wide")
    st.title("📈 Stock Setup Checker")
    st.caption("Type a company name or ticker. Data from Yahoo Finance (unofficial, ~15 min delayed). Research aid only, not financial advice.")

    q = st.text_input("Company name or ticker", placeholder="e.g. Marathon Petroleum, MPC, First Solar, NDX1.DE")
    q = q.strip()
    if not q:
        st.info("Enter a name or ticker to start.")
        return

    matches = search_symbols(q)
    options = [m["symbol"] for m in matches]
    labels = {m["symbol"]: f'{m["symbol"]} - {m["name"]} ({m["exch"]})' for m in matches}
    direct = q.upper()
    if " " not in q and direct not in options:
        options.insert(0, direct)
        labels[direct] = f"{direct} (as typed)"
    if not options:
        st.error("Nothing found. Try the ticker symbol.")
        return
    sym = st.selectbox("Which listing?", options, format_func=lambda s: labels.get(s, s))
    peers_in = st.text_input("Optional: your own comparison tickers (comma-separated)", placeholder="e.g. VLO, PSX, DINO")
    manual = [p.strip().upper() for p in peers_in.split(",") if p.strip()] or None
    analyse(sym, manual)

    st.divider()
    st.caption("Data: Yahoo Finance via the open-source yfinance library. Numbers can be missing or wrong; verify anything important on the company's filings. "
               "Past performance does not predict future returns. You can lose money you invest.")


if __name__ == "__main__":
    main()
