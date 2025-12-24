"""
Forward-Vol Mispricing Screener (GUI) — Long ATM CALL Calendar

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
    (If forward vol is much lower than front IV -> forward_factor is big.)

Filters:
  - Front IV >= min_front_iv
  - forward_factor >= forward_factor_min (e.g. 0.20 = 20%)
  - OI/volume/spread filters for BOTH legs at ATM call

Universe:
  - Top N "most active" tickers from Yahoo Finance page (with fallback list)
  - OR paste your own tickers in GUI.

Install:
  pip install yfinance pandas numpy

Run:
  python forward_calendar_gui.py
"""

from __future__ import annotations

import threading
import time
from datetime import datetime, timezone
from dataclasses import dataclass
from typing import List, Dict, Optional

import numpy as np
import pandas as pd
import yfinance as yf

import tkinter as tk
from tkinter import ttk, filedialog, messagebox


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
    # de-dupe and trim
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
    # yfinance style for dots
    syms = [s.replace(".", "-") for s in syms]
    # de-dupe preserving order
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
    Returns NaN if invalid/noisy.
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
                      sleep_s: float = 0.10) -> List[Dict]:
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

        # Cache chains per expiry (big speed-up)
        chain_cache: Dict[str, any] = {}

        def get_chain(exp: str):
            if exp not in chain_cache:
                chain_cache[exp] = t.option_chain(exp)
            return chain_cache[exp]

        time.sleep(sleep_s)

        for p in pairs:
            ch_f = get_chain(p.front_exp)
            ch_b = get_chain(p.back_exp)

            # ATM based on front calls
            atm = pick_atm_strike(ch_f.calls, spot)
            if atm is None:
                continue

            rf = row_at_strike(ch_f.calls, atm)
            rb = row_at_strike(ch_b.calls, atm)
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

            # Require high near-term IV
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

            cal_debit = (mid_b - mid_f) if (np.isfinite(mid_f) and np.isfinite(mid_b)) else float("nan")

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
                "front_mid": mid_f,
                "back_mid": mid_b,
                "calendar_debit_mid": cal_debit,
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
    # Rank: highest forward_factor first, then higher front IV
    df = df.sort_values(["forward_factor", "front_iv"], ascending=[False, False]).head(top_k).reset_index(drop=True)
    return df


# -----------------------------
# GUI
# -----------------------------
COLUMNS = [
    "symbol",
    "front_dte", "back_dte", "gap_dte",
    "spot", "atm_strike",
    "front_expiry", "back_expiry",
    "front_iv", "back_iv",
    "forward_vol", "forward_factor",
    "front_over_back_iv",
    "calendar_debit_mid",
    "front_spread_pct", "back_spread_pct",
    "front_oi", "back_oi",
    "front_vol", "back_vol",
]


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Forward-Vol Mispricing Screener — Long ATM Call Calendar")
        self.geometry("1550x780")

        self.df: Optional[pd.DataFrame] = None

        self._build_controls()
        self._build_table()
        self._build_status()

    def _build_controls(self):
        root = ttk.Frame(self)
        root.pack(fill="x", padx=10, pady=8)

        # Universe
        uni = ttk.LabelFrame(root, text="Universe")
        uni.pack(side="left", padx=6, pady=4, fill="y")

        self.var_use_most_active = tk.BooleanVar(value=True)
        self.var_top_n = tk.IntVar(value=100)

        ttk.Checkbutton(uni, text="Use Yahoo 'Most Active'", variable=self.var_use_most_active).grid(
            row=0, column=0, columnspan=2, sticky="w", padx=6, pady=3
        )
        ttk.Label(uni, text="Top N:").grid(row=1, column=0, sticky="w", padx=6, pady=3)
        ttk.Entry(uni, textvariable=self.var_top_n, width=8).grid(row=1, column=1, sticky="w", padx=6, pady=3)

        ttk.Label(uni, text="Or paste tickers:").grid(row=2, column=0, sticky="w", padx=6, pady=(8, 3))
        self.txt_tickers = tk.Text(uni, width=26, height=4)
        self.txt_tickers.grid(row=3, column=0, columnspan=2, padx=6, pady=3, sticky="w")
        self.txt_tickers.insert("1.0", "SPY, QQQ, AAPL, MSFT, NVDA")

        # Filters
        filt = ttk.LabelFrame(root, text="Filters")
        filt.pack(side="left", padx=6, pady=4, fill="y")

        self.var_min_front_iv = tk.DoubleVar(value=0.40)
        self.var_forward_factor_min = tk.DoubleVar(value=0.20)

        self.var_min_oi = tk.IntVar(value=200)
        self.var_min_vol = tk.IntVar(value=0)
        self.var_max_spread = tk.DoubleVar(value=0.30)

        r = 0
        ttk.Label(filt, text="Min front IV:").grid(row=r, column=0, sticky="w", padx=6, pady=3)
        ttk.Entry(filt, textvariable=self.var_min_front_iv, width=10).grid(row=r, column=1, sticky="w", padx=6, pady=3)
        r += 1
        ttk.Label(filt, text="Min forward factor:").grid(row=r, column=0, sticky="w", padx=6, pady=3)
        ttk.Entry(filt, textvariable=self.var_forward_factor_min, width=10).grid(row=r, column=1, sticky="w", padx=6, pady=3)
        r += 1

        ttk.Separator(filt, orient="horizontal").grid(row=r, column=0, columnspan=2, sticky="ew", padx=6, pady=8)
        r += 1

        ttk.Label(filt, text="Min OI (both legs):").grid(row=r, column=0, sticky="w", padx=6, pady=3)
        ttk.Entry(filt, textvariable=self.var_min_oi, width=10).grid(row=r, column=1, sticky="w", padx=6, pady=3)
        r += 1
        ttk.Label(filt, text="Min volume (both):").grid(row=r, column=0, sticky="w", padx=6, pady=3)
        ttk.Entry(filt, textvariable=self.var_min_vol, width=10).grid(row=r, column=1, sticky="w", padx=6, pady=3)
        r += 1
        ttk.Label(filt, text="Max (ask-bid)/mid:").grid(row=r, column=0, sticky="w", padx=6, pady=3)
        ttk.Entry(filt, textvariable=self.var_max_spread, width=10).grid(row=r, column=1, sticky="w", padx=6, pady=3)

        # DTE scan
        dtef = ttk.LabelFrame(root, text="DTE Scan")
        dtef.pack(side="left", padx=6, pady=4, fill="y")

        self.var_min_front_dte = tk.IntVar(value=7)
        self.var_max_front_dte = tk.IntVar(value=45)
        self.var_min_gap_dte = tk.IntVar(value=7)
        self.var_max_back_dte = tk.IntVar(value=180)
        self.var_max_pairs = tk.IntVar(value=120)
        self.var_top_k = tk.IntVar(value=50)

        rr = 0
        ttk.Label(dtef, text="Min front DTE:").grid(row=rr, column=0, sticky="w", padx=6, pady=3)
        ttk.Entry(dtef, textvariable=self.var_min_front_dte, width=8).grid(row=rr, column=1, sticky="w", padx=6, pady=3)
        rr += 1
        ttk.Label(dtef, text="Max front DTE:").grid(row=rr, column=0, sticky="w", padx=6, pady=3)
        ttk.Entry(dtef, textvariable=self.var_max_front_dte, width=8).grid(row=rr, column=1, sticky="w", padx=6, pady=3)
        rr += 1
        ttk.Label(dtef, text="Min gap DTE:").grid(row=rr, column=0, sticky="w", padx=6, pady=3)
        ttk.Entry(dtef, textvariable=self.var_min_gap_dte, width=8).grid(row=rr, column=1, sticky="w", padx=6, pady=3)
        rr += 1
        ttk.Label(dtef, text="Max back DTE:").grid(row=rr, column=0, sticky="w", padx=6, pady=3)
        ttk.Entry(dtef, textvariable=self.var_max_back_dte, width=8).grid(row=rr, column=1, sticky="w", padx=6, pady=3)
        rr += 1
        ttk.Label(dtef, text="Max pairs per ticker:").grid(row=rr, column=0, sticky="w", padx=6, pady=3)
        ttk.Entry(dtef, textvariable=self.var_max_pairs, width=8).grid(row=rr, column=1, sticky="w", padx=6, pady=3)
        rr += 1
        ttk.Label(dtef, text="Keep top K rows:").grid(row=rr, column=0, sticky="w", padx=6, pady=3)
        ttk.Entry(dtef, textvariable=self.var_top_k, width=8).grid(row=rr, column=1, sticky="w", padx=6, pady=3)

        # Buttons
        btns = ttk.Frame(root)
        btns.pack(side="right", padx=6, pady=4, fill="y")

        self.btn_run = ttk.Button(btns, text="Run Screener", command=self._on_run)
        self.btn_run.pack(fill="x", pady=4)

        self.btn_export = ttk.Button(btns, text="Export CSV", command=self._on_export, state="disabled")
        self.btn_export.pack(fill="x", pady=4)

        self.btn_clear = ttk.Button(btns, text="Clear", command=self._clear)
        self.btn_clear.pack(fill="x", pady=4)

    def _build_table(self):
        frame = ttk.Frame(self)
        frame.pack(fill="both", expand=True, padx=10, pady=6)

        self.tree = ttk.Treeview(frame, columns=COLUMNS, show="headings")
        vsb = ttk.Scrollbar(frame, orient="vertical", command=self.tree.yview)
        hsb = ttk.Scrollbar(frame, orient="horizontal", command=self.tree.xview)
        self.tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)

        self.tree.pack(side="left", fill="both", expand=True)
        vsb.pack(side="right", fill="y")
        hsb.pack(side="bottom", fill="x")

        for col in COLUMNS:
            self.tree.heading(col, text=col, command=lambda c=col: self._sort_by(c))
            self.tree.column(col, width=125, anchor="center")

        # Wider useful columns
        self.tree.column("front_expiry", width=110)
        self.tree.column("back_expiry", width=110)
        self.tree.column("forward_factor", width=140)
        self.tree.column("calendar_debit_mid", width=150)

    def _build_status(self):
        self.status = tk.StringVar(value="Ready.")
        self.pb = ttk.Progressbar(self, mode="indeterminate")
        self.pb.pack(fill="x", padx=10, pady=(0, 4))
        ttk.Label(self, textvariable=self.status).pack(fill="x", padx=10, pady=(0, 8))

    def _clear(self):
        for i in self.tree.get_children():
            self.tree.delete(i)
        self.df = None
        self.btn_export.config(state="disabled")
        self.status.set("Cleared. Ready.")

    def _set_running(self, running: bool):
        if running:
            self.btn_run.config(state="disabled")
            self.btn_export.config(state="disabled")
            self.pb.start(10)
        else:
            self.btn_run.config(state="normal")
            self.pb.stop()

    def _get_tickers(self) -> List[str]:
        if self.var_use_most_active.get():
            return fetch_top_most_active_tickers(int(self.var_top_n.get()))
        return parse_manual_tickers(self.txt_tickers.get("1.0", "end").strip())

    def _on_run(self):
        self._clear()
        self._set_running(True)

        tickers = self._get_tickers()
        if not tickers:
            self._set_running(False)
            messagebox.showerror("No tickers", "Provide tickers or enable 'Most Active'.")
            return

        self.status.set(f"Running… scanning {len(tickers)} tickers across expiry pairs…")

        def worker():
            try:
                df = run_screen(
                    tickers=tickers,
                    min_front_iv=float(self.var_min_front_iv.get()),
                    forward_factor_min=float(self.var_forward_factor_min.get()),
                    min_oi=int(self.var_min_oi.get()),
                    min_vol=int(self.var_min_vol.get()),
                    max_spread=float(self.var_max_spread.get()),
                    min_front_dte=int(self.var_min_front_dte.get()),
                    max_front_dte=int(self.var_max_front_dte.get()),
                    min_gap_dte=int(self.var_min_gap_dte.get()),
                    max_back_dte=int(self.var_max_back_dte.get()),
                    max_pairs_per_ticker=int(self.var_max_pairs.get()),
                    top_k=int(self.var_top_k.get()),
                )
                self.df = df
                self.after(0, self._render_results)
            except Exception as e:
                self.after(0, lambda: messagebox.showerror("Error", str(e)))
            finally:
                self.after(0, lambda: self._set_running(False))

        threading.Thread(target=worker, daemon=True).start()

    def _render_results(self):
        if self.df is None or self.df.empty:
            self.status.set("No matches. Try relaxing filters (lower min IV, lower forward factor, lower OI, higher spread).")
            return

        for _, row in self.df.iterrows():
            vals = []
            for c in COLUMNS:
                v = row.get(c, "")
                if isinstance(v, float):
                    if c in ("spot", "atm_strike", "front_mid", "back_mid", "calendar_debit_mid"):
                        vals.append(f"{v:.2f}")
                    elif "iv" in c or "spread_pct" in c or c in ("forward_vol", "forward_factor", "front_over_back_iv"):
                        vals.append(f"{v:.4f}")
                    else:
                        vals.append(f"{v:.4f}")
                else:
                    vals.append(str(v))
            self.tree.insert("", "end", values=vals)

        self.btn_export.config(state="normal")
        top = self.df.iloc[0]
        self.status.set(
            f"Done. {len(self.df)} rows. Top: {top['symbol']} "
            f"FF={top['forward_factor']:.3f}  frontIV={top['front_iv']:.3f}  "
            f"{int(top['front_dte'])}->{int(top['back_dte'])}D"
        )

    def _on_export(self):
        if self.df is None or self.df.empty:
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv")],
            initialfile="forward_calendar_gui_results.csv",
        )
        if not path:
            return
        self.df.to_csv(path, index=False)
        messagebox.showinfo("Exported", f"Saved CSV to:\n{path}")

    def _sort_by(self, col: str):
        if self.df is None or self.df.empty:
            return
        # numeric sort desc if possible
        try:
            s = pd.to_numeric(self.df[col], errors="coerce")
            if s.notna().any():
                self.df = self.df.assign(_sortkey=s).sort_values("_sortkey", ascending=False).drop(columns="_sortkey")
            else:
                self.df = self.df.sort_values(col, ascending=True)
        except Exception:
            self.df = self.df.sort_values(col, ascending=True)

        for i in self.tree.get_children():
            self.tree.delete(i)
        self._render_results()


if __name__ == "__main__":
    App().mainloop()
