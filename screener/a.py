from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Optional, Any

import numpy as np
import pandas as pd
import yfinance as yf
import requests


# -----------------------------
# Universe: Yahoo Finance Most Active (robust + cached)
# -----------------------------
YF_MOST_ACTIVE_URL = "https://finance.yahoo.com/markets/stocks/most-active/"
CACHE_FILE = Path("most_active_cache.csv")
CACHE_TTL_SECONDS = 60 * 60  # 1 hour


def fetch_top_most_active_tickers(n: int = 100) -> List[str]:
    """
    Robust 'Most Active' fetch with:
      - cache to disk
      - browser-like User-Agent
      - fallback list if Yahoo rate-limits (HTTP 429)
    """
    # Serve cache if fresh
    try:
        if CACHE_FILE.exists():
            age = time.time() - CACHE_FILE.stat().st_mtime
            if age < CACHE_TTL_SECONDS:
                dfc = pd.read_csv(CACHE_FILE)
                if "Symbol" in dfc.columns:
                    syms = dfc["Symbol"].astype(str).str.upper().tolist()
                    return [s.replace(".", "-") for s in syms[:n]]
    except Exception:
        pass

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
    }

    try:
        resp = requests.get(YF_MOST_ACTIVE_URL, headers=headers, timeout=12)
        resp.raise_for_status()
        tables = pd.read_html(resp.text)
        syms: List[str] = []

        for tbl in tables:
            if "Symbol" in tbl.columns:
                part = tbl["Symbol"].astype(str).str.upper().tolist()
                part = [s.replace(".", "-") for s in part]
                part = [s for s in part if s and s != "NAN"]
                syms.extend(part)

        # de-dupe
        seen, out = set(), []
        for s in syms:
            if s not in seen:
                seen.add(s)
                out.append(s)
            if len(out) >= n:
                break

        if out:
            pd.DataFrame({"Symbol": out}).to_csv(CACHE_FILE, index=False)
            return out[:n]
    except Exception:
        pass

    # Fallback: liquid-ish names
    fallback = [
        "SPY","QQQ","IWM","DIA","AAPL","MSFT","NVDA","TSLA","AMZN","META","GOOGL","AMD","NFLX","AVGO",
        "JPM","BAC","WFC","C","GS","XOM","CVX","KO","PEP","DIS","NKE","WMT","COST",
        "PFE","JNJ","UNH","LLY","ABBV","ADBE","CRM","ORCL","CSCO","QCOM","MU","AMAT",
        "PANW","CRWD","NET","SNOW","PLTR","COIN","PYPL","SQ","SHOP","UBER","DKNG","HOOD","MRVL","TXN"
    ]
    return fallback[:n]


def parse_manual_tickers(text: str) -> List[str]:
    if not text:
        return []
    raw = text.replace(",", " ").split()
    syms = [s.strip().upper() for s in raw if s.strip()]
    syms = [s.replace(".", "-") for s in syms]
    # de-dupe preserving order
    seen, out = set(), []
    for s in syms:
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


# -----------------------------
# Earnings helper
# -----------------------------
def get_next_earnings_date_utc(t: yf.Ticker) -> Optional[datetime]:
    """
    Returns next earnings datetime in UTC if available, else None.
    """
    try:
        cal = getattr(t, "calendar", None)
        if cal is None or cal is False:
            return None
        if hasattr(cal, "empty") and cal.empty:
            return None
        if not isinstance(cal, pd.DataFrame):
            return None
        if "Earnings Date" not in cal.index:
            return None

        v = cal.loc["Earnings Date"]
        if isinstance(v, (pd.Series, pd.DataFrame)):
            v0 = v.iloc[0]
        else:
            v0 = v

        dt = pd.to_datetime(v0, errors="coerce")
        if pd.isna(dt):
            return None

        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        else:
            dt = dt.astimezone(timezone.utc)
        return dt
    except Exception:
        return None


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
# Screener: SHORT calendars
# -----------------------------
def screen_one_symbol_short_calendar(symbol: str,
                                     min_front_iv: float,
                                     forward_factor_threshold_abs: float,
                                     min_oi: int,
                                     min_vol: int,
                                     max_spread: float,
                                     min_front_dte: int,
                                     max_front_dte: int,
                                     min_gap_dte: int,
                                     max_back_dte: int,
                                     max_pairs_per_ticker: int,
                                     exclude_earnings_to_front: bool = True,
                                     sleep_s: float = 0.05) -> List[Dict[str, Any]]:
    """
    SHORT ATM call calendar:
      SELL back call, BUY front call

    Keep only where:
      forward_factor = (front_iv - forward_vol) / front_iv  <=  -threshold_abs
    """
    rows: List[Dict[str, Any]] = []
    try:
        t = yf.Ticker(symbol)

        # Spot
        spot = safe_float(t.fast_info.get("last_price", np.nan) if hasattr(t, "fast_info") else np.nan)
        if not np.isfinite(spot) or spot <= 0:
            h = t.history(period="5d", auto_adjust=True)
            if h is None or h.empty:
                return rows
            spot = float(h["Close"].iloc[-1])

        # Earnings (once per symbol)
        earn_dt = get_next_earnings_date_utc(t)
        now_dt = datetime.now(timezone.utc)

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
        chain_cache: Dict[str, Any] = {}

        def get_chain(exp: str):
            if exp not in chain_cache:
                chain_cache[exp] = t.option_chain(exp)
            return chain_cache[exp]

        time.sleep(sleep_s)

        for p in pairs:
            # Exclude earnings between now and front expiry
            if exclude_earnings_to_front and earn_dt is not None:
                front_dt = datetime.strptime(p.front_exp, "%Y-%m-%d").replace(tzinfo=timezone.utc)
                if now_dt < earn_dt <= front_dt:
                    continue

            ch_f = get_chain(p.front_exp)
            ch_b = get_chain(p.back_exp)

            # ATM based on front calls
            atm = pick_atm_strike(ch_f.calls, spot)
            if atm is None:
                continue

            rf = row_at_strike(ch_f.calls, atm)  # front call (we BUY)
            rb = row_at_strike(ch_b.calls, atm)  # back call  (we SELL)
            if rf is None or rb is None:
                continue

            # Liquidity filters (both legs)
            oi_f = int(rf.get("openInterest", 0) or 0)
            oi_b = int(rb.get("openInterest", 0) or 0)
            vol_f = int(rf.get("volume", 0) or 0)
            vol_b = int(rb.get("volume", 0) or 0)

            if oi_f < min_oi or oi_b < min_oi:
                continue
            if vol_f < min_vol or vol_b < min_vol:
                continue

            bid_f, ask_f, last_f = safe_float(rf.get("bid", np.nan)), safe_float(rf.get("ask", np.nan)), safe_float(rf.get("lastPrice", np.nan))
            bid_b, ask_b, last_b = safe_float(rb.get("bid", np.nan)), safe_float(rb.get("ask", np.nan)), safe_float(rb.get("lastPrice", np.nan))

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

            # Require high near-term IV if you want (kept from your original logic)
            if iv_f < min_front_iv:
                continue

            T1 = years_from_dte(p.front_dte)
            T2 = years_from_dte(p.back_dte)
            fwd = forward_vol(iv_f, iv_b, T1, T2)
            if not np.isfinite(fwd) or fwd <= 0:
                continue

            ff = (iv_f - fwd) / iv_f if iv_f > 0 else float("nan")
            if not np.isfinite(ff):
                continue

            # SHORT condition: forward factor must be <= -threshold_abs
            thr = abs(forward_factor_threshold_abs)
            if ff > -thr:
                continue

            # Strategy credit estimates for SHORT calendar (sell back, buy front)
            strategy_mid_credit = (mid_b - mid_f) if (np.isfinite(mid_f) and np.isfinite(mid_b)) else float("nan")
            strategy_natural_credit = (bid_b - ask_f) if (np.isfinite(bid_b) and np.isfinite(ask_f)) else float("nan")

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
                "forward_factor": ff,                     # negative is "rich forward"
                "front_over_back_iv": (iv_f / iv_b) if iv_b > 0 else float("nan"),

                "front_bid": bid_f,
                "front_ask": ask_f,
                "back_bid": bid_b,
                "back_ask": ask_b,
                "front_mid": mid_f,
                "back_mid": mid_b,

                "strategy_mid_credit": strategy_mid_credit,
                "strategy_natural_credit": strategy_natural_credit,

                "front_spread_pct": sp_f,
                "back_spread_pct": sp_b,
                "front_oi": oi_f,
                "back_oi": oi_b,
                "front_vol": vol_f,
                "back_vol": vol_b,

                "next_earnings_utc": earn_dt.isoformat() if earn_dt is not None else "",
            })

    except Exception:
        return rows

    return rows


def run_short_calendar_screen(tickers: List[str],
                             min_front_iv: float = 0.40,
                             forward_factor_threshold_abs: float = 0.20,
                             min_oi: int = 200,
                             min_vol: int = 0,
                             max_spread: float = 0.30,
                             min_front_dte: int = 7,
                             max_front_dte: int = 45,
                             min_gap_dte: int = 7,
                             max_back_dte: int = 180,
                             max_pairs_per_ticker: int = 60,
                             exclude_earnings_to_front: bool = True,
                             top_k: int = 100) -> pd.DataFrame:
    all_rows: List[Dict[str, Any]] = []
    for sym in tickers:
        all_rows.extend(
            screen_one_symbol_short_calendar(
                sym,
                min_front_iv=min_front_iv,
                forward_factor_threshold_abs=forward_factor_threshold_abs,
                min_oi=min_oi,
                min_vol=min_vol,
                max_spread=max_spread,
                min_front_dte=min_front_dte,
                max_front_dte=max_front_dte,
                min_gap_dte=min_gap_dte,
                max_back_dte=max_back_dte,
                max_pairs_per_ticker=max_pairs_per_ticker,
                exclude_earnings_to_front=exclude_earnings_to_front,
            )
        )

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)

    # For short calendars, "more negative" forward_factor is stronger
    df = df.sort_values(["forward_factor", "front_iv"], ascending=[True, False]).head(top_k).reset_index(drop=True)
    return df


# -----------------------------
# Optional: quick CLI test
# -----------------------------
if __name__ == "__main__":
    tickers = fetch_top_most_active_tickers(25)
    df = run_short_calendar_screen(
        tickers,
        min_front_iv=0.40,
        forward_factor_threshold_abs=0.20,
        min_oi=200,
        max_spread=0.30,
        min_front_dte=7,
        max_front_dte=45,
        min_gap_dte=7,
        max_back_dte=180,
        max_pairs_per_ticker=40,
        exclude_earnings_to_front=True,
        top_k=30
    )
    print(df.head(30).to_string(index=False))
