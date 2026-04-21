"""
Aerithius Apollo — Prediction CLI v3.0
======================================
Uses the shared `features` module so it cannot drift from training.
Normalizes win probabilities across the field (racing is a ranking problem —
only one horse can win).
"""
from __future__ import annotations
import os
import sys
import pickle
import numpy as np
import pandas as pd

from features import (
    FEATURE_NAMES,
    build_features_from_history,
    build_inference_history,
    softmax_normalize,
    parse_date,
)

FORM_CSV  = "master_horse_form.csv"
MODEL_PKL = "apollo_brain.pkl"


def load_brain():
    if not os.path.exists(MODEL_PKL):
        sys.exit(f"ERROR: {MODEL_PKL} missing — run `python train_model.py` first.")
    with open(MODEL_PKL, "rb") as f:
        brain = pickle.load(f)
    if brain.get("feature_names") != FEATURE_NAMES:
        sys.exit("ERROR: model feature names do not match the current `features` module. Retrain.")
    return brain


def predict_race(
    condition: str | None,
    distance_m: int,
    runners: list[str] | dict,
    race_name: str = "Live Race",
    as_of: str | None = None,
    track: str | None = None,
) -> list[dict]:
    """
    Parameters
    ----------
    runners : list[str]  OR  dict {horse: {"barrier": int, "weight": float}}
    as_of   : optional 'DD-Mon-YY' cutoff — use only history before this date
              (defaults to "now", i.e. all scraped history).
    track   : track name for the race (e.g. 'Ayr'). Matches the training-time
              `track` feature so the track_* history features actually fire at
              inference. Leave as None if genuinely unknown.
    """
    brain = load_brain()
    form_df = pd.read_csv(FORM_CSV) if os.path.exists(FORM_CSV) else pd.DataFrame()

    cutoff = parse_date(as_of) if as_of else pd.Timestamp.now()

    # Normalize runners to a uniform dict
    if isinstance(runners, list):
        runners = {h: {} for h in runners}

    field_size = len(runners)

    rows = []
    for horse, ctx in runners.items():
        history = build_inference_history(form_df, horse, as_of=cutoff)
        in_db = len(history) > 0

        target = {
            "race_date":  cutoff,
            "distance_m": int(distance_m),
            "track":      track,
            "condition":  condition,
            "barrier":    ctx.get("barrier"),
            "weight":     ctx.get("weight"),
            "field_size": field_size,
        }
        feats = build_features_from_history(history, target)
        rows.append({"horse": horse, "in_db": in_db, "history_n": len(history), "features": feats})

    # Batch inference
    X = pd.DataFrame([r["features"] for r in rows])[FEATURE_NAMES].values
    X_sc = brain["scaler"].transform(X)

    win_raw   = brain["win_model"].predict_proba(X_sc)[:, 1]
    place_raw = brain["place_model"].predict_proba(X_sc)[:, 1]
    score_raw = brain["score_model"].predict(X_sc)

    # Field-level normalization: one horse wins, so win probs sum to 1.
    win_norm = softmax_normalize(win_raw)
    # Place probs scaled so expected place count ≈ 3 (or field_size if <3).
    expected_places = min(3.0, float(field_size))
    if place_raw.sum() > 0:
        place_norm = place_raw * (expected_places / place_raw.sum())
        place_norm = np.clip(place_norm, 0.01, 0.99)
    else:
        place_norm = place_raw

    results = []
    for r, wp, pp, sc in zip(rows, win_norm, place_norm, score_raw):
        fair_odds = 1.0 / wp if wp > 1e-3 else 99.0
        # data_confidence is already a feature the model sees; don't double-damp.
        data_conf = r["features"]["data_confidence"]
        composite = (0.5 * wp + 0.3 * pp + 0.2 * max(0, min(1, sc))) * 100
        results.append({
            "horse":          r["horse"],
            "in_database":    r["in_db"],
            "history_n":      r["history_n"],
            "confidence":     round(composite, 1),
            "win_pct":        round(wp * 100, 1),
            "place_pct":      round(pp * 100, 1),
            "fair_odds":      round(fair_odds, 2),
            "data_coverage":  round(data_conf * 100),
            "form_trend":     r["features"]["form_trend"],
        })

    results.sort(key=lambda r: r["confidence"], reverse=True)

    # ── Pretty print ──
    print("=" * 92)
    print(f"  APOLLO v{brain['version']}  ·  {race_name}  ·  {distance_m}m  ·  cond={condition or 'n/a'}")
    print(f"  metrics: win AUC={brain['metrics']['win_auc_mean']:.3f}  place AUC={brain['metrics']['place_auc_mean']:.3f}")
    print("=" * 92)
    print(f"  {'#':>2} {'Horse':<22} {'Conf':>6} {'Win%':>6} {'Place%':>7} {'Fair$':>7} {'Data':>5} {'Trend':>6}")
    print("-" * 92)
    for i, r in enumerate(results, 1):
        trend = r["form_trend"]
        trend_s = "↑" if trend > 0.1 else ("↓" if trend < -0.1 else "·")
        flag = "  [new]" if not r["in_database"] else ("  [?]" if r["data_coverage"] < 30 else "")
        print(f"  {i:>2} {r['horse']:<22} "
              f"{r['confidence']:>5.0f}% "
              f"{r['win_pct']:>5.1f}% "
              f"{r['place_pct']:>6.1f}% "
              f"{r['fair_odds']:>7.2f} "
              f"{r['data_coverage']:>4}% "
              f"{trend_s:>5s}{flag}")
    print("=" * 92)
    return results


if __name__ == "__main__":
    TRACK     = "Ayr"        # course name — matches training-time track feature
    CONDITION = "Good"       # track condition
    DIST      = 1200
    NAME      = "Ayr Race 2"
    RUNNERS = [

    "Worth",
    "Venator",
    "Star Wave",
    "Counter Seven",
    "Gatto Nero",
    "Love Diva",
    "Kyoei Bonita",
    "Meisho Yozora",
    "Smooth Velvet",
    "Viva Crown",
    "Jacquard",
    "Aoi Regina"
]
    predict_race(CONDITION, DIST, RUNNERS, NAME, track=TRACK)
