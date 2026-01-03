from __future__ import annotations

import math
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import yfinance as yf


# =========================================================
# Helpers
# =========================================================

def now_utc():
    return datetime.now(timezone.utc)


def parse_manual_tickers(text: str) -> list[str]:
    if not text:
        return []
    raw = text.replace(",", " ").split()
    return [t.strip().upper() for t in raw if t.strip()]


def fetch_top_most_active_tickers(n: int = 100) -> list[str]:
    """
    Uses Yahoo Finance 'Most Active' table
    """
    url = "https://finance.yahoo.com/most-active"
    tables = pd.read_html(url)
    df = tables[0]
    syms = df["Symbol"].astype(str).tolist()
    return syms[:n]


def get_next_earnings(symbol: str):
    try:
        cal = yf.Ticker(symbol).calendar
        if cal is None or cal.empty:
            return None
        dt = pd.to_datetime(cal.loc["Earnings Date"][0])
        if dt.tzinfo is None:
            dt = dt.tz_localize(timezone.utc)
        return dt
    except Exception:
        return None


def mid(bid, ask):
    if bid <= 0 or ask <= 0:
        return np.nan
    return (bid + ask) / 2.0


def spread_pct(bid, ask):
    m = mid(bid, ask)
    if m <= 0:
        return np.nan
    return (ask - bid) / m


def forward_vol(front_iv, back_iv, front_dte, back_dte):
    """
    Variance interpolation
    """
    T1 = front_dte / 365.0
    T2 = back_dte / 365.0
    if T2 <= T1 or front_iv <= 0 or back_iv <= 0:
        return np.nan

    var_fwd = (back_iv**2 * T2 - front_iv**2 * T1) / (T2 - T1)
    if var_fwd <= 0:
        return np.nan

    return math.sqrt(var_fwd)


def forward_factor(front_iv, fwd_iv):
    if front_iv <= 0 or fwd_iv <= 0:
        return np.nan
    return (fwd_iv / front_iv) - 1.0


# =========================================================
# Core screening logic
# =========================================================

def run_screen(
    *,
    tickers: list[str],
    direction: str = "long",  # "long" or "short"
    min_front_iv: float = 0.0,
    min_forward_factor: float | None = None,
    max_forward_factor: float | None = None,
    min_oi: int = 0,
    min_vol: int = 0,
    max_spread: float = 1.0,
    min_front_dte: int = 7,
    max_front_dte: int = 45,
    min_gap_dte: int = 7,
    max_back_dte: int = 180,
    max_pairs_per_ticker: int = 60,
    exclude_earnings: bool = True,
    top_k: int = 200,
):
    rows = []
    now = now_utc()

    for symbol in tickers:
        try:
            tk = yf.Ticker(symbol)
            spot = tk.fast_info.get("lastPrice")
            if not spot or spot <= 0:
                continue

            earnings_dt = get_next_earnings(symbol)

            expiries = tk.options
            if not expiries:
                continue

            exp_info = []
            for e in expiries:
                dt = pd.to_datetime(e).tz_localize(timezone.utc)
                dte = (dt - now).days
                exp_info.append((e, dt, dte))

            fronts = [x for x in exp_info if min_front_dte <= x[2] <= max_front_dte]

            for f_exp, f_dt, f_dte in fronts:
                if exclude_earnings and earnings_dt:
                    if now < earnings_dt <= f_dt:
                        continue

                backs = [
                    x for x in exp_info
                    if f_dte + min_gap_dte <= x[2] <= max_back_dte
                ]

                pairs_checked = 0

                for b_exp, b_dt, b_dte in backs:
                    if pairs_checked >= max_pairs_per_ticker:
                        break

                    oc_f = tk.option_chain(f_exp).calls
                    oc_b = tk.option_chain(b_exp).calls

                    oc_f = oc_f.iloc[(oc_f["strike"] - spot).abs().argsort()[:1]]
                    oc_b = oc_b.iloc[(oc_b["strike"] - spot).abs().argsort()[:1]]

                    f = oc_f.iloc[0]
                    b = oc_b.iloc[0]

                    if (
                        f["openInterest"] < min_oi
                        or b["openInterest"] < min_oi
                        or f["volume"] < min_vol
                        or b["volume"] < min_vol
                    ):
                        continue

                    f_spread = spread_pct(f["bid"], f["ask"])
                    b_spread = spread_pct(b["bid"], b["ask"])
                    if (
                        np.isnan(f_spread)
                        or np.isnan(b_spread)
                        or f_spread > max_spread
                        or b_spread > max_spread
                    ):
                        continue

                    front_iv = f["impliedVolatility"]
                    back_iv = b["impliedVolatility"]
                    if front_iv < min_front_iv or back_iv <= 0:
                        continue

                    fwd_iv = forward_vol(front_iv, back_iv, f_dte, b_dte)
                    if not fwd_iv:
                        continue

                    ff = forward_factor(front_iv, fwd_iv)
                    if np.isnan(ff):
                        continue

                    if min_forward_factor is not None and ff < min_forward_factor:
                        continue
                    if max_forward_factor is not None and ff > max_forward_factor:
                        continue

                    f_mid = mid(f["bid"], f["ask"])
                    b_mid = mid(b["bid"], b["ask"])

                    strategy_mid = b_mid - f_mid
                    strategy_natural = b["ask"] - f["bid"]

                    rows.append({
                        "symbol": symbol,
                        "spot": spot,
                        "atm_strike": f["strike"],
                        "front_expiry": f_exp,
                        "back_expiry": b_exp,
                        "front_dte": f_dte,
                        "back_dte": b_dte,
                        "gap_dte": b_dte - f_dte,
                        "front_iv": front_iv,
                        "back_iv": back_iv,
                        "forward_vol": fwd_iv,
                        "forward_factor": ff,
                        "front_over_back_iv": front_iv / back_iv,
                        "strategy_mid_debit": strategy_mid,
                        "strategy_natural_debit": strategy_natural,
                        "front_oi": f["openInterest"],
                        "back_oi": b["openInterest"],
                        "front_vol": f["volume"],
                        "back_vol": b["volume"],
                        "front_spread_pct": f_spread,
                        "back_spread_pct": b_spread,
                        "direction": direction,
                    })

                    pairs_checked += 1

        except Exception:
            continue

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    if direction == "long":
        df = df.sort_values("forward_factor", ascending=False)
    else:
        df = df.sort_values("forward_factor", ascending=True)

    return df.head(top_k).reset_index(drop=True)
