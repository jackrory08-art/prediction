"""
Aerithius Apollo — Prediction engine v4.0
=========================================
Pure library + small CLI. Shares the `features` module with training so the two
can't drift. Takes a racecard (from the Betfair Exchange API or a fixture) and
returns calibrated, field-normalized win tips.
"""
from __future__ import annotations

import os
import pickle

import numpy as np
import pandas as pd

import db
from features import (
    FEATURE_NAMES,
    build_features_from_history,
    build_inference_history,
    softmax_normalize,
)

MODEL_PKL = "apollo_brain.pkl"


def load_brain(path: str = MODEL_PKL) -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} missing — run `python train_model.py` first.")
    with open(path, "rb") as f:
        brain = pickle.load(f)
    if brain.get("feature_names") != FEATURE_NAMES:
        raise ValueError("Model feature names don't match the current features module. Retrain.")
    return brain


def predict_racecard(racecard: dict, brain: dict | None = None,
                     as_of: pd.Timestamp | None = None) -> list[dict]:
    """Rank the runners in one race. Returns dicts sorted by win probability."""
    brain = brain or load_brain()
    cutoff = as_of or pd.Timestamp.now()
    runners = racecard.get("runners") or []
    field_size = len(runners)

    rows = []
    for r in runners:
        hid = db.normalize_horse_id(r["horse"])
        hist = build_inference_history(hid, as_of=cutoff)
        target = {
            "race_date": cutoff,
            "distance_m": racecard.get("distance_m"),
            "track": racecard.get("track"),
            "going": racecard.get("going"),
            "draw": r.get("draw"),
            "weight": r.get("weight"),
            "field_size": field_size,
            "region": racecard.get("region"),
        }
        feats = build_features_from_history(hist, target)
        rows.append({"horse": r["horse"], "history_n": len(hist), "features": feats})

    X = pd.DataFrame([r["features"] for r in rows])[FEATURE_NAMES].values
    win_raw = brain["win_model"].predict_proba(X)[:, 1]
    win_norm = softmax_normalize(win_raw)

    place_norm = [None] * field_size
    if brain.get("place_model") is not None:
        place_raw = brain["place_model"].predict_proba(X)[:, 1]
        expected = min(3.0, float(field_size))
        if place_raw.sum() > 0:
            place_norm = np.clip(place_raw * (expected / place_raw.sum()), 0.01, 0.99)

    results = []
    for r, wp, pp in zip(rows, win_norm, place_norm):
        results.append({
            "horse": r["horse"],
            "history_n": r["history_n"],
            "win_pct": round(float(wp) * 100, 1),
            "place_pct": None if pp is None else round(float(pp) * 100, 1),
            "fair_odds": round(1.0 / wp, 2) if wp > 1e-3 else 99.0,
            "data_coverage": round(r["features"]["data_confidence"] * 100),
            "form_trend": r["features"]["form_trend_win"],
        })
    results.sort(key=lambda x: x["win_pct"], reverse=True)
    return results


def print_card(racecard: dict, results: list[dict], brain: dict) -> None:
    m = brain["metrics"]
    title = f"{racecard.get('track','?')} R{racecard.get('race_no','?')} · {racecard.get('distance_m','?')}m"
    print("=" * 86)
    print(f"  APOLLO v{brain['version']}  ·  {title}  ·  going={racecard.get('going') or 'n/a'}")
    print(f"  win AUC={m['win_auc']:.3f}  top-1 hit={m['win_top1_hit']*100:.0f}% (naive {m['naive_hit']*100:.0f}%)")
    print("=" * 86)
    print(f"  {'#':>2} {'Horse':<22} {'Win%':>6} {'Place%':>7} {'Fair$':>7} {'Data':>5} {'Trend':>6}")
    print("-" * 86)
    for i, r in enumerate(results, 1):
        trend = "↑" if r["form_trend"] > 0.1 else ("↓" if r["form_trend"] < -0.1 else "·")
        place = "  —  " if r["place_pct"] is None else f"{r['place_pct']:>5.1f}%"
        flag = "  [new]" if r["history_n"] == 0 else ""
        print(f"  {i:>2} {r['horse']:<22} {r['win_pct']:>5.1f}% {place} "
              f"{r['fair_odds']:>7.2f} {r['data_coverage']:>4}% {trend:>5s}{flag}")
    print("=" * 86)


if __name__ == "__main__":
    import json
    import glob

    brain = load_brain()
    cards = sorted(glob.glob("tests/fixtures/racecard_*.json"))
    if not cards:
        raise SystemExit("No fixture racecard found. Run `python tests/make_fixtures.py`.")
    racecard = json.loads(open(cards[0]).read())
    results = predict_racecard(racecard, brain)
    print_card(racecard, results, brain)
