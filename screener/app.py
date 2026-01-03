from __future__ import annotations

import time
import streamlit as st
import pandas as pd

from screener import (
    run_screen,
    fetch_top_most_active_tickers,
    parse_manual_tickers,
)

st.set_page_config(page_title="Calendar Mispricing Screener", layout="wide")

st.title("📆 Calendar Forward-Vol Screener")

# -------------------------------------------------
# Sidebar
# -------------------------------------------------
with st.sidebar:
    st.header("Universe")

    use_most_active = st.checkbox("Use Yahoo Most Active", value=True)
    top_n = st.number_input("Top N", 1, 500, 100, step=5)

    manual_tickers = st.text_area(
        "Manual tickers",
        value="SPY, QQQ, AAPL, MSFT",
        height=90,
        disabled=use_most_active,
    )

    st.divider()
    st.header("Trade Type")

    direction = st.radio(
        "Calendar Direction",
        ["Long Call Calendar", "Short Call Calendar"],
    )

    if direction == "Long Call Calendar":
        min_forward_factor = st.number_input(
            "Min forward factor (+)",
            0.0, 2.0, 0.20, step=0.05
        )
        max_forward_factor = None
    else:
        max_forward_factor = st.number_input(
            "Max forward factor (−)",
            -2.0, 0.0, -0.20, step=0.05
        )
        min_forward_factor = None

    st.divider()
    st.header("IV + Liquidity")

    min_front_iv = st.number_input("Min front IV", 0.0, 5.0, 0.40, step=0.05)
    min_oi = st.number_input("Min OI (both)", 0, 1_000_000, 200)
    max_spread = st.number_input("Max bid/ask %", 0.0, 1.0, 0.30, step=0.05)

    st.divider()
    st.header("Expiries")

    min_front_dte = st.number_input("Min front DTE", 1, 365, 7)
    max_front_dte = st.number_input("Max front DTE", 1, 365, 45)
    min_gap_dte = st.number_input("Min gap DTE", 1, 365, 7)
    max_back_dte = st.number_input("Max back DTE", 1, 3650, 180)

    st.divider()
    exclude_earnings = st.checkbox(
        "Exclude earnings before front expiry",
        value=True
    )

    max_pairs = st.number_input("Max expiry pairs / ticker", 10, 200, 60)
    top_k = st.number_input("Max rows", 10, 2000, 200)

    run_btn = st.button("🚀 Run Screener", type="primary")

# -------------------------------------------------
# Helpers
# -------------------------------------------------
def build_universe():
    if use_most_active:
        return fetch_top_most_active_tickers(int(top_n))
    return parse_manual_tickers(manual_tickers)

@st.cache_data(ttl=600)
def cached_run(**kwargs):
    return run_screen(**kwargs)

# -------------------------------------------------
# Run
# -------------------------------------------------
if run_btn:
    tickers = build_universe()

    st.info(f"Scanning {len(tickers)} tickers…")

    with st.spinner("Crunching option chains…"):
        t0 = time.time()
        df = cached_run(
            tickers=tickers,
            direction="long" if direction.startswith("Long") else "short",
            min_front_iv=min_front_iv,
            min_forward_factor=min_forward_factor,
            max_forward_factor=max_forward_factor,
            min_oi=min_oi,
            max_spread=max_spread,
            min_front_dte=min_front_dte,
            max_front_dte=max_front_dte,
            min_gap_dte=min_gap_dte,
            max_back_dte=max_back_dte,
            max_pairs_per_ticker=max_pairs,
            exclude_earnings=exclude_earnings,
            top_k=top_k,
        )

    if df.empty:
        st.warning("No matches found.")
        st.stop()

    st.success(f"Found {len(df)} trades in {time.time()-t0:.1f}s")

    df = df.sort_values("forward_factor", ascending=False)

    st.dataframe(df, use_container_width=True, height=550)
    st.download_button(
        "Download CSV",
        df.to_csv(index=False),
        "calendar_results.csv",
        "text/csv"
    )
