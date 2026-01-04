# app.py
# Streamlit UI for Forward-Vol Mispricing Screener — Long ATM Call Calendar
#
# Folder structure:
# underpriced-calendar-screener/
# ├─ requirements.txt
# └─ screener/
#    ├─ app.py        ← THIS FILE
#    ├─ screener.py   ← your tkinter-free logic
#    └─ __init__.py

from __future__ import annotations

import time
import streamlit as st
import pandas as pd

from long_calendar import (
    run_screen,
    fetch_top_most_active_tickers,
    parse_manual_tickers,
)

# -------------------------------------------------
# Page setup
# -------------------------------------------------
st.set_page_config(
    page_title="Forward Calendar Screener",
    layout="wide",
)

st.title("Forward-Vol Mispricing for Long Call Calendars Screener")
st.caption(
    "Long ATM **call calendars** where near-term IV is high "
    "and forward volatility is materially cheaper (mispricing)."
)

# -------------------------------------------------
# Sidebar controls
# -------------------------------------------------
with st.sidebar:
    st.header("Universe")

    use_most_active = st.checkbox("Use Yahoo Most Active", value=True)
    top_n = st.number_input("Top N (most active)", 1, 500, 100, step=5)

    manual_tickers = st.text_area(
        "Or paste tickers (comma / space / newline)",
        value="SPY, QQQ, AAPL, MSFT, NVDA",
        height=110,
        disabled=use_most_active,
    )

    st.divider()
    st.header("Core filters")

    min_front_iv = st.number_input(
        "Min front IV", 0.0, 5.0, 0.40, step=0.05, format="%.2f"
    )
    min_forward_factor = st.number_input(
        "Min forward factor (20% = 0.20)",
        0.0, 1.0, 0.20, step=0.01, format="%.2f"
    )

    st.divider()
    st.header("Liquidity")

    min_oi = st.number_input("Min OI (both legs)", 0, 1_000_000, 200, step=50)
    min_vol = st.number_input("Min option volume (both)", 0, 1_000_000, 0, step=50)
    max_spread = st.number_input(
        "Max (ask-bid)/mid", 0.0, 5.0, 0.30, step=0.05, format="%.2f"
    )

    st.divider()
    st.header("Expiry scan")

    min_front_dte = st.number_input("Min front DTE", 1, 365, 7)
    max_front_dte = st.number_input("Max front DTE", 1, 365, 45)
    min_gap_dte = st.number_input("Min gap DTE", 1, 365, 7)
    max_back_dte = st.number_input("Max back DTE", 1, 3650, 180)

    st.divider()
    st.header("Runtime")

    max_pairs = st.number_input(
        "Max expiry pairs per ticker",
        10, 300, 60, step=10,
        help="Lower this if Streamlit is slow or rate-limited"
    )
    top_k = st.number_input("Keep top K rows", 1, 2000, 100, step=10)

    cache_minutes = st.number_input(
        "Cache results (minutes)", 0, 120, 10, step=5
    )

    run_btn = st.button("🚀 Run Screener", type="primary", use_container_width=True)

# -------------------------------------------------
# Helpers
# -------------------------------------------------
def build_universe():
    if use_most_active:
        tickers = fetch_top_most_active_tickers(int(top_n))
    else:
        tickers = parse_manual_tickers(manual_tickers)

    # sanitize + dedupe
    out, seen = [], set()
    for t in tickers:
        t = t.strip().upper().replace(".", "-")
        if t and t not in seen:
            seen.add(t)
            out.append(t)
    return out


@st.cache_data(ttl=int(cache_minutes * 60))
def cached_run_screen(**kwargs):
    return run_screen(**kwargs)

# -------------------------------------------------
# Run screener
# -------------------------------------------------
if run_btn:
    tickers = build_universe()

    if not tickers:
        st.error("No tickers selected.")
        st.stop()

    st.info(f"Scanning **{len(tickers)}** tickers across expiry pairs…")

    with st.spinner("Fetching option chains and computing forward volatility…"):
        start = time.time()
        df = cached_run_screen(
            tickers=tickers,
            min_front_iv=min_front_iv,
            forward_factor_min=min_forward_factor,
            min_oi=min_oi,
            min_vol=min_vol,
            max_spread=max_spread,
            min_front_dte=min_front_dte,
            max_front_dte=max_front_dte,
            min_gap_dte=min_gap_dte,
            max_back_dte=max_back_dte,
            max_pairs_per_ticker=max_pairs,
            top_k=top_k,
        )
        elapsed = time.time() - start

    if df.empty:
        st.warning(
            f"No matches found ({elapsed:.1f}s). "
            "Try relaxing filters or reducing Top N / pairs."
        )
        st.stop()

    st.success(f"Found **{len(df)}** rows in {elapsed:.1f}s")

    # -------------------------------------------------
    # Display
    # -------------------------------------------------
    preferred_cols = [
        "symbol",
        "front_dte", "back_dte", "gap_dte",
        "spot", "atm_strike",
        "front_expiry", "back_expiry",
        "front_iv", "back_iv",
        "forward_vol", "forward_factor",
        "front_over_back_iv",
        "strategy_mid_debit", "strategy_natural_debit",
        "front_spread_pct", "back_spread_pct",
        "front_oi", "back_oi",
        "front_vol", "back_vol",
    ]

    cols = [c for c in preferred_cols if c in df.columns] + \
           [c for c in df.columns if c not in preferred_cols]

    df = df[cols]

    sort_col = st.selectbox(
        "Sort by",
        options=[c for c in df.columns if c not in ("front_expiry", "back_expiry")],
        index=df.columns.get_loc("forward_factor")
        if "forward_factor" in df.columns else 0,
    )

    df = df.sort_values(sort_col, ascending=False, na_position="last")

    st.dataframe(df, use_container_width=True, height=520)

    st.download_button(
        "⬇️ Download CSV",
        df.to_csv(index=False),
        file_name="forward_calendar_results.csv",
        mime="text/csv",
        use_container_width=True,
    )

    with st.expander("Universe used"):
        st.write(tickers)

else:
    st.write("⬅️ Configure filters in the sidebar, then click **Run Screener**.")
    st.write(
        "Tip: For online use, **Top 50–100** tickers and **40–60 expiry pairs** "
        "is the sweet spot."
    )
