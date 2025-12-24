# app.py
# Streamlit web UI for: Forward-Vol Mispricing Screener — Long ATM Call Calendar
#
# Assumes you have a screener.py in the same folder that exposes:
#   - run_screen(tickers, min_front_iv, forward_factor_min, min_oi, min_vol, max_spread,
#                min_front_dte, max_front_dte, min_gap_dte, max_back_dte, max_pairs_per_ticker, top_k)
#   - fetch_top_most_active_tickers(n)
#   - parse_manual_tickers(text)   (optional but recommended)
#
# Install requirements:
#   pip install streamlit yfinance pandas numpy lxml html5lib
#
# Run locally:
#   streamlit run app.py

from __future__ import annotations

import time
import pandas as pd
import streamlit as st

# ---- Import your screener logic ----
# If you don't have parse_manual_tickers in screener.py, you can delete that import and use the fallback below.
from screener import run_screen, fetch_top_most_active_tickers

try:
    from screener import parse_manual_tickers  # optional helper
except Exception:
    def parse_manual_tickers(text: str):
        raw = text.replace(",", " ").split()
        syms = [s.strip().upper() for s in raw if s.strip()]
        syms = [s.replace(".", "-") for s in syms]
        seen, out = set(), []
        for s in syms:
            if s not in seen:
                seen.add(s)
                out.append(s)
        return out


st.set_page_config(page_title="Forward Calendar Screener", layout="wide")
st.title("Forward-Vol Mispricing Screener — Long ATM Call Calendar")
st.caption(
    "Filters for high near-term (front) IV and forward-factor mispricing across expiry pairs. "
    "Designed to match your Python logic, now usable in a browser."
)

# -----------------------------
# Sidebar controls
# -----------------------------
with st.sidebar:
    st.header("Universe")

    use_most_active = st.checkbox("Use Yahoo 'Most Active'", value=True)
    top_n = st.number_input("Top N (Most Active)", min_value=1, max_value=500, value=100, step=5)

    manual = st.text_area(
        "Or paste tickers (comma/space/newline separated)",
        value="SPY, QQQ, AAPL, MSFT, NVDA",
        height=110,
        disabled=use_most_active,
    )

    st.divider()
    st.header("Core filters")

    min_front_iv = st.number_input("Min front IV", min_value=0.0, max_value=5.0, value=0.40, step=0.05, format="%.2f")
    forward_factor_min = st.number_input("Min forward factor (>= 0.20 = 20%)",
                                         min_value=0.0, max_value=1.0, value=0.20, step=0.01, format="%.2f")

    st.divider()
    st.header("Liquidity / quotes")

    min_oi = st.number_input("Min OI (both legs)", min_value=0, max_value=1_000_000, value=200, step=50)
    min_vol = st.number_input("Min option volume (both legs)", min_value=0, max_value=1_000_000, value=0, step=50)
    max_spread = st.number_input("Max (ask-bid)/mid (both legs)", min_value=0.0, max_value=5.0, value=0.30, step=0.05, format="%.2f")

    st.divider()
    st.header("Expiry scan")

    min_front_dte = st.number_input("Min front DTE", min_value=1, max_value=365, value=7, step=1)
    max_front_dte = st.number_input("Max front DTE", min_value=1, max_value=365, value=45, step=1)
    min_gap_dte = st.number_input("Min gap DTE", min_value=1, max_value=365, value=7, step=1)
    max_back_dte = st.number_input("Max back DTE", min_value=1, max_value=3650, value=180, step=5)

    st.divider()
    st.header("Runtime / output")

    max_pairs_per_ticker = st.number_input(
        "Max expiry pairs per ticker", min_value=1, max_value=500, value=60, step=10,
        help="Higher = more coverage, slower. For Top 100 online, 40–80 is a good range."
    )
    top_k = st.number_input("Keep top K rows", min_value=1, max_value=5000, value=100, step=10)

    cache_minutes = st.number_input("Cache results (minutes)", min_value=0, max_value=120, value=10, step=5)
    run_btn = st.button("Run Screener", type="primary", use_container_width=True)


# -----------------------------
# Caching wrapper
# -----------------------------
def _cache_ttl_seconds(minutes: int) -> int:
    return max(0, int(minutes)) * 60


@st.cache_data(ttl=_cache_ttl_seconds(cache_minutes))
def cached_run_screen(
    tickers,
    min_front_iv,
    forward_factor_min,
    min_oi,
    min_vol,
    max_spread,
    min_front_dte,
    max_front_dte,
    min_gap_dte,
    max_back_dte,
    max_pairs_per_ticker,
    top_k,
):
    # Ensure cache key is stable
    tickers = list(tickers)
    return run_screen(
        tickers=tickers,
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
        top_k=top_k,
    )


# -----------------------------
# Main action
# -----------------------------
def build_universe():
    if use_most_active:
        tks = fetch_top_most_active_tickers(int(top_n))
    else:
        tks = parse_manual_tickers(manual)
    # dedupe + sanitize
    seen, out = set(), []
    for t in tks:
        t = str(t).strip().upper().replace(".", "-")
        if t and t not in seen:
            seen.add(t)
            out.append(t)
    return out


if run_btn:
    tickers = build_universe()

    if not tickers:
        st.error("No tickers selected. Enable 'Most Active' or paste tickers.")
        st.stop()

    st.info(f"Scanning **{len(tickers)}** tickers across expiry pairs…")

    with st.spinner("Running scan (this can take a bit for Top 100)…"):
        start = time.time()
        df = cached_run_screen(
            tickers=tickers,
            min_front_iv=float(min_front_iv),
            forward_factor_min=float(forward_factor_min),
            min_oi=int(min_oi),
            min_vol=int(min_vol),
            max_spread=float(max_spread),
            min_front_dte=int(min_front_dte),
            max_front_dte=int(max_front_dte),
            min_gap_dte=int(min_gap_dte),
            max_back_dte=int(max_back_dte),
            max_pairs_per_ticker=int(max_pairs_per_ticker),
            top_k=int(top_k),
        )
        elapsed = time.time() - start

    if df is None or df.empty:
        st.warning(
            f"No matches (ran in {elapsed:.1f}s). Try relaxing filters:\n"
            "- lower Min front IV\n"
            "- lower Min forward factor\n"
            "- lower Min OI / Min option volume\n"
            "- increase Max spread\n"
            "- reduce Top N or Max pairs per ticker (if rate-limited)"
        )
    else:
        st.success(f"Found {len(df)} rows (ran in {elapsed:.1f}s).")

        # Nice ordering if columns exist
        preferred = [
            "symbol", "front_dte", "back_dte", "gap_dte",
            "spot", "atm_strike",
            "front_expiry", "back_expiry",
            "front_iv", "back_iv",
            "forward_vol", "forward_factor",
            "front_over_back_iv",
            "strategy_mid_debit", "strategy_natural_debit",
            "calendar_debit_mid",
            "front_spread_pct", "back_spread_pct",
            "front_oi", "back_oi",
            "front_vol", "back_vol",
        ]
        cols = [c for c in preferred if c in df.columns] + [c for c in df.columns if c not in preferred]
        df = df[cols].copy()

        # Display + quick sort control
        st.subheader("Results")
        sort_col = st.selectbox(
            "Sort by",
            options=[c for c in df.columns if c not in ("front_expiry", "back_expiry")] or list(df.columns),
            index=(df.columns.get_loc("forward_factor") if "forward_factor" in df.columns else 0),
        )
        df_sorted = df.sort_values(sort_col, ascending=False, na_position="last")

        st.dataframe(df_sorted, use_container_width=True, height=520)

        # Download
        st.download_button(
            "Download CSV",
            df_sorted.to_csv(index=False).encode("utf-8"),
            file_name="forward_calendar_results.csv",
            mime="text/csv",
            use_container_width=True,
        )

        with st.expander("Universe used"):
            st.write(tickers)
else:
    st.write("Set your filters in the sidebar, then click **Run Screener**.")
    st.write("Tip: For Top 100 online, keep **Max expiry pairs per ticker** around **40–80** for reliability.")
