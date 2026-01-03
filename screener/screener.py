"""
Forward-Vol Mispricing Screener (NO Tkinter) — Long ATM CALL Calendar

What it finds:
  1) Liquid tickers with HIGH near-term (front) ATM call IV
  2) For MANY expiry pairs (front/back), compute forward vol + a "forward factor"
  3) Show only opportunities where forward_factor >= threshold (default 20%)

Key formulas:
  - ATM strike: closest strike to spot (from FRONT call chain)
  - Forward vol between expiries:
        fwd^2 = (iv_back^2 * T_back - iv_front^2 * T_front) / (T_back - T_front)
  - Forward factor (mispricing score for long calendar):
        forward_factor = (iv_front - forward_vol) / iv_front

Strategy pricing (Long ATM call calendar = Buy back, Sell front):
  - strategy_mid_debit    = mid(back) - mid(front)
  - strategy_natural_debit= ask(back) - bid(front)

Universe:
  - Top N "most active" tickers from Yahoo Finance page (with fallback list)
  - OR parse tickers from a string

Install:
  pip install yfinance pandas numpy lxml html5lib
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List, Dict, Optional

import numpy as np
import pandas as pd
import yfinance as yf


# -----------------------------
# Universe: Yahoo Finance Most Active
# -----------------------------
YF_MOST_ACTIVE_URL = "https://finance.yahoo.com/markets/stocks/most-active/"


def fetch_top_most_active_tickers(n: int = 100) -> List[str]:
    """
    Fetch most-active tickers from Yahoo Finance page via pandas.read_html.
    If Yahoo only returns a short table, fallback list guarantees up to n symbols.
    """
    try:
        tables = pd.read_html(YF_MOST_ACTIVE_URL)
        syms: List[str] = []
        for tbl in tables:
            cols = [c.strip() if isinstance(c, str) else c for c in tbl.columns]
            tbl.columns = cols
            if "Symbol" in tbl.columns:
                part = tbl["Symbol"].astype(str).str.upper().tolist()
                part = [s.replace(".", "-") for s in part]
                part = [s for s in part if s and s != "NAN"]
                syms.extend(part)

        # de-dupe preserving order
        seen = set()
        out = []
        for s in syms:
            if s not in seen:
                seen.add(s)
                out.append(s)
            if len(out) >= n:
                break

        if len(out) >= min(n, 25):  # use whatever Yahoo gave us if it’s not empty
            return out[:n]
    except Exception:
        pass

    # Fallback 100-ish liquid names (guarantees length even if Yahoo scraping fails)
    fallback = [
        "SPY","QQQ","IWM","DIA","AAPL","MSFT","NVDA","TSLA","AMZN","META","GOOGL","AMD","INTC","NFLX","AVGO",
        "BAC","JPM","WFC","C","GS","XOM","CVX","COP","SLB","KO","PEP","DIS","NKE","WMT","COST","TGT",
        "PFE","MRNA","JNJ","UNH","LLY","ABBV","CAT","BA","GE","GM","F","UBER","LYFT","PLTR","SOFI",
        "COIN","PYPL","SQ","SHOP","ADBE","CRM","ORCL","CSCO","QCOM","MU","AMAT","INTU","PANW","CRWD",
        "NET","SNOW","DDOG","MDB","ARM","SMCI","TSM","ASML","NOW","V","MA","AXP","HD","LOW","BKNG",
        "DAL","AAL","UAL","RCL","CCL","MCD","SBUX","CMG","CVS","T","VZ","TMUS","RIVN","LCID","SNAP",
        "ROKU","GME","AMC","BABA","NIO","PDD","JD","SE","DKNG","HOOD","MRVL","TXN"
    ]
    seen = set()
    out = []
    for s in fallback:
        if s not in seen:
            seen.add(s)
            out.append(s)
        if len(out) >= n:
            break
    return out[:n]


def parse_manual_tickers(text: str) -> List[str]:
    """
    Accepts comma/space/newline separated tickers.
    """
    if not text:
        return []
    raw = text.replace(",", " ").split()
    syms = [s.strip().upper() for s in raw if s.strip()]
    syms = [s.replace(".", "-") for s in syms]  # yfinance style for dots
    seen = set()
    out = []
    for s in syms:
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


# -----------------------------
# Option helpers
# -----------------------------
def safe_float(x) -> float:
    try:
        if x is None:
            return float("nan")
        return float(x)
    except Exception:
        return float("nan")


def dte(expiry_str: str) -> int:
    now = datetime.now(timezone.utc)
    exp = datetime.strptime(expiry_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return (exp - now).days


def years_from_dte(dte_days: int) -> float:
    return max(dte_days, 0) / 365.25


def mid_price(bid: float, ask: float, last: float) -> float:
    if np.isfinite(bid) and np.isfinite(ask) and ask > 0 and bid >= 0 and ask >= bid:
        return float((bid + ask) / 2.0)
    if np.isfinite(last) and last > 0:
        return float(last)
    return float("nan")


def spread_pct(bid: float, ask: float, mid: float) -> float:
    if not (np.isfinite(bid) and np.isfinite(ask) and np.isfinite(mid)) or mid <= 0:
        return float("nan")
    return float((ask - bid) / mid)


def pick_atm_strike(calls_df: pd.DataFrame, spot: float) -> Optional[float]:
    if calls_df is None or calls_df.empty or not np.isfinite(spot):
        return None
    strikes = calls_df["strike"].values
    if strikes.size == 0:
        return None
    return float(strikes[int(np.argmin(np.abs(strikes - spot)))])


def row_at_strike(df: pd.DataFrame, strike: float) -> Optional[pd.Series]:
    if df is None or df.empty:
        return None
    r = df.loc[df["strike"] == strike]
    if r.empty:
        return None
    return r.iloc[0]


def forward_vol(iv_front: float, iv_back: float, t1: float, t2: float) -> float:
    """
    Forward vol between t1 and t2:
      fwd^2 = (iv_back^2 * t2 - iv_front^2 * t1) / (t2 - t1)
    """
    if not (np.isfinite(iv_front) and np.isfinite(iv_back) and np.isfinite(t1) and np.isfinite(t2)):
        return float("nan")
    if iv_front <= 0 or iv_back <= 0:
        return float("nan")
    if t2 <= t1 or t1 <= 0:
        return float("nan")

    num = (iv_back ** 2) * t2 - (iv_front ** 2) * t1
    den = (t2 - t1)
    if den <= 0 or num <= 0:
        return float("nan")
    return float(np.sqrt(num / den))


@dataclass
class Pair:
    front_exp: str
    back_exp: str
    front_dte: int
    back_dte: int


def generate_pairs(expiries: List[str],
                   min_front_dte: int,
                   max_front_dte: int,
                   min_gap_dte: int,
                   max_back_dte: int,
                   max_pairs_per_ticker: int = 120) -> List[Pair]:
    exps = [(e, dte(e)) for e in expiries]
    exps = [(e, dt) for (e, dt) in exps if dt > 0]
    exps.sort(key=lambda x: x[1])

    fronts = [(e, dt) for (e, dt) in exps if min_front_dte <= dt <= max_front_dte]
    if not fronts:
        return []

    pairs: List[Pair] = []
    for f_exp, f_dte in fronts:
        backs = [(e, dt) for (e, dt) in exps if (dt >= f_dte + min_gap_dte) and (dt <= max_back_dte)]
        for b_exp, b_dte in backs:
            pairs.append(Pair(front_exp=f_exp, back_exp=b_exp, front_dte=f_dte, back_dte=b_dte))
            if len(pairs) >= max_pairs_per_ticker:
                return pairs
    return pairs


# -----------------------------
# Screener core
# -----------------------------
def screen_one_symbol(symbol: str,
                      min_front_iv: float,
                      forward_factor_min: float,
                      min_oi: int,
                      min_vol: int,
                      max_spread: float,
                      min_front_dte: int,
                      max_front_dte: int,
                      min_gap_dte: int,
                      max_back_dte: int,
                      max_pairs_per_ticker: int,
                      sleep_s: float = 0.05) -> List[Dict]:
    """
    For a symbol, scan many expiry pairs for LONG ATM CALL CALENDAR.
    Keep rows where forward_factor >= threshold.
    """
    rows: List[Dict] = []
    try:
        t = yf.Ticker(symbol)

        # Spot
        spot = safe_float(t.fast_info.get("last_price", np.nan) if hasattr(t, "fast_info") else np.nan)
        if not np.isfinite(spot) or spot <= 0:
            h = t.history(period="5d", auto_adjust=True)
            if h is None or h.empty:
                return rows
            spot = float(h["Close"].iloc[-1])

        expiries = list(getattr(t, "options", []) or [])
        if not expiries:
            return rows

        pairs = generate_pairs(
            expiries,
            min_front_dte=min_front_dte,
            max_front_dte=max_front_dte,
            min_gap_dte=min_gap_dte,
            max_back_dte=max_back_dte,
            max_pairs_per_ticker=max_pairs_per_ticker
        )
        if not pairs:
            return rows

        # Cache chains per expiry
        chain_cache: Dict[str, any] = {}

        def get_chain(exp: str):
            if exp not in chain_cache:
                chain_cache[exp] = t.option_chain(exp)
            return chain_cache[exp]

        time.sleep(sleep_s)

        for p in pairs:
            ch_f = get_chain(p.front_exp)
            ch_b = get_chain(p.back_exp)

            atm = pick_atm_strike(ch_f.calls, spot)
            if atm is None:
                continue

            rf = row_at_strike(ch_f.calls, atm)
            rb = row_at_strike(ch_b.calls, atm)
            if rf is None or rb is None:
                continue

            # Liquidity filters
            oi_f = int(rf.get("openInterest", 0) or 0)
            oi_b = int(rb.get("openInterest", 0) or 0)
            vol_f = int(rf.get("volume", 0) or 0)
            vol_b = int(rb.get("volume", 0) or 0)

            if oi_f < min_oi or oi_b < min_oi:
                continue
            if vol_f < min_vol or vol_b < min_vol:
                continue

            bid_f = safe_float(rf.get("bid", np.nan))
            ask_f = safe_float(rf.get("ask", np.nan))
            last_f = safe_float(rf.get("lastPrice", np.nan))

            bid_b = safe_float(rb.get("bid", np.nan))
            ask_b = safe_float(rb.get("ask", np.nan))
            last_b = safe_float(rb.get("lastPrice", np.nan))

            mid_f = mid_price(bid_f, ask_f, last_f)
            mid_b = mid_price(bid_b, ask_b, last_b)

            sp_f = spread_pct(bid_f, ask_f, mid_f)
            sp_b = spread_pct(bid_b, ask_b, mid_b)

            if np.isfinite(sp_f) and sp_f > max_spread:
                continue
            if np.isfinite(sp_b) and sp_b > max_spread:
                continue

            iv_f = safe_float(rf.get("impliedVolatility", np.nan))
            iv_b = safe_float(rb.get("impliedVolatility", np.nan))
            if not (np.isfinite(iv_f) and np.isfinite(iv_b)) or iv_f <= 0 or iv_b <= 0:
                continue

            if iv_f < min_front_iv:
                continue

            T1 = years_from_dte(p.front_dte)
            T2 = years_from_dte(p.back_dte)
            fwd = forward_vol(iv_f, iv_b, T1, T2)
            if not np.isfinite(fwd) or fwd <= 0:
                continue

            forward_factor = (iv_f - fwd) / iv_f if iv_f > 0 else float("nan")
            if not np.isfinite(forward_factor) or forward_factor < forward_factor_min:
                continue

            # Strategy pricing (explicit)
            strategy_mid_debit = (mid_b - mid_f) if (np.isfinite(mid_f) and np.isfinite(mid_b)) else float("nan")
            strategy_natural_debit = (ask_b - bid_f) if (np.isfinite(ask_b) and np.isfinite(bid_f)) else float("nan")

            rows.append({
                "symbol": symbol,
                "spot": spot,
                "atm_strike": atm,
                "front_expiry": p.front_exp,
                "back_expiry": p.back_exp,
                "front_dte": p.front_dte,
                "back_dte": p.back_dte,
                "gap_dte": p.back_dte - p.front_dte,

                "front_iv": iv_f,
                "back_iv": iv_b,
                "forward_vol": fwd,
                "forward_factor": forward_factor,
                "front_over_back_iv": (iv_f / iv_b) if iv_b > 0 else float("nan"),

                "front_bid": bid_f,
                "front_ask": ask_f,
                "back_bid": bid_b,
                "back_ask": ask_b,
                "front_mid": mid_f,
                "back_mid": mid_b,

                "strategy_mid_debit": strategy_mid_debit,
                "strategy_natural_debit": strategy_natural_debit,

                "front_spread_pct": sp_f,
                "back_spread_pct": sp_b,
                "front_oi": oi_f,
                "back_oi": oi_b,
                "front_vol": vol_f,
                "back_vol": vol_b,
            })

    except Exception:
        return rows

    return rows


def run_screen(tickers: List[str],
               min_front_iv: float,
               forward_factor_min: float,
               min_oi: int,
               min_vol: int,
               max_spread: float,
               min_front_dte: int,
               max_front_dte: int,
               min_gap_dte: int,
               max_back_dte: int,
               max_pairs_per_ticker: int,
               top_k: int) -> pd.DataFrame:
    all_rows: List[Dict] = []
    for sym in tickers:
        all_rows.extend(
            screen_one_symbol(
                sym,
                min_front_iv=min_front_iv,
                forward_factor_min=forward_factor_min,
                min_oi=min_oi,
                min_vol=min_vol,
                max_spread=max_spread,
                min_front_dte=min_front_dte,
                max_front_dte=max_front_dte,
                min_gap_dte=min_gap_dte,
                max_back_dte=max_back_dte,
                max_pairs_per_ticker=max_pairs_per_ticker,
            )
        )

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    df = df.sort_values(["forward_factor", "front_iv"], ascending=[False, False]).head(top_k).reset_index(drop=True)
    return df
