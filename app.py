"""
Aerithius Apollo — Streamlit app
================================
Open a live AU/NZ meeting, pick a race, get calibrated win tips. Designed to run
on Streamlit Community Cloud (free) and be used from a phone browser.

Data:
  • Live  — Betfair Exchange API (needs BETFAIR_* secrets; AU/NZ markets).
  • Offline/demo — committed fixtures, used automatically when no secrets are set
    or when live login is geo-blocked from the host.
"""
from __future__ import annotations

import os

import pandas as pd
import streamlit as st

# Promote Streamlit secrets to env vars so the data layer can read them.
# Guarded: st.secrets raises if no secrets.toml exists (e.g. local/demo runs).
try:
    for _k in ("BETFAIR_APP_KEY", "BETFAIR_USERNAME", "BETFAIR_PASSWORD"):
        if _k in st.secrets:
            os.environ[_k] = str(st.secrets[_k])
except Exception:
    pass

import db                              # noqa: E402
from data_sources import BetfairSource  # noqa: E402
from predict import load_brain, predict_racecard  # noqa: E402

st.set_page_config(page_title="Apollo Racing Tips", page_icon="🐎", layout="wide")

HAS_LIVE = all(os.environ.get(k) for k in ("BETFAIR_APP_KEY", "BETFAIR_USERNAME", "BETFAIR_PASSWORD"))


@st.cache_resource
def get_brain():
    try:
        return load_brain()
    except Exception as e:
        st.error(f"Model not loaded: {e}")
        st.stop()


@st.cache_data(ttl=300, show_spinner="Fetching racecards …")
def get_racecards(day: str):
    """Return (cards, source_label). Falls back to fixtures on any live failure."""
    if HAS_LIVE:
        try:
            cards = BetfairSource().racecards(day=day)
            if cards:
                return cards, "live (Betfair AU/NZ)"
        except Exception as e:  # geo-block, login, etc. — degrade gracefully
            st.warning(f"Live Betfair unavailable ({e}). Showing demo data.")
    cards = BetfairSource(fixtures="tests/fixtures").racecards()
    return cards, "demo (fixtures)"


def race_label(c: dict) -> str:
    t = c.get("market_start_time", "") or ""
    hhmm = t[11:16] if len(t) >= 16 else ""
    return f"{c.get('track','?')} R{c.get('race_no','?')} · {c.get('distance_m','?')}m {hhmm}".strip()


def main() -> None:
    brain = get_brain()
    m = brain["metrics"]

    st.title("🐎 Apollo Racing Tips")
    st.caption(
        f"Model v{brain['version']} · win AUC **{m['win_auc']:.3f}** · "
        f"top-1 hit **{m['win_top1_hit']*100:.0f}%** (naive {m['naive_hit']*100:.0f}%) · "
        f"trained on {m['n_train']:,} runs / {m['n_races']:,} races"
    )

    with st.sidebar:
        st.header("Controls")
        day = st.radio("Day", ["today", "tomorrow"], horizontal=True)
        if st.button("🔄 Refresh racecards"):
            get_racecards.clear()
        counts = db.row_counts()
        st.metric("Races in DB", f"{counts['races']:,}")
        st.metric("Horses tracked", f"{counts['horses']:,}")
        st.caption("Status: " + ("🟢 live secrets set" if HAS_LIVE else "🟡 demo mode (no secrets)"))

    cards, source_label = get_racecards(day)
    st.caption(f"Source: {source_label} · {len(cards)} race(s)")
    if not cards:
        st.info("No racecards available right now.")
        return

    labels = [race_label(c) for c in cards]
    idx = st.selectbox("Pick a race", range(len(cards)), format_func=lambda i: labels[i])
    card = cards[idx]

    results = predict_racecard(card, brain)
    df = pd.DataFrame(results)
    df.insert(0, "Rank", range(1, len(df) + 1))
    df = df.rename(columns={
        "horse": "Horse", "win_pct": "Win %", "place_pct": "Place %",
        "fair_odds": "Fair $", "data_coverage": "Data %", "history_n": "Runs",
    })
    show = df[["Rank", "Horse", "Win %", "Place %", "Fair $", "Data %", "Runs"]]
    st.subheader(race_label(card))
    st.dataframe(show, hide_index=True, width="stretch",
                 column_config={"Win %": st.column_config.ProgressColumn(
                     "Win %", min_value=0, max_value=float(df["Win %"].max() or 1), format="%.1f")})
    st.caption("Win % is field-normalized & calibrated. Fair $ = 1 / win probability. "
               "Place % shows '—' on the free Betfair feed (no finishing positions).")


if __name__ == "__main__":
    main()
