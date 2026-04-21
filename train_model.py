"""
Aerithius Apollo — Trainer v3.0
===============================
Point-in-time features. No data leakage. Race-aware cross-validation so we
never score a horse using a model that was trained on its own races.
"""
from __future__ import annotations
import os
import sys
import pickle
import warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.preprocessing import StandardScaler

from features import FEATURE_NAMES, build_training_dataset

warnings.filterwarnings("ignore")

FORM_CSV  = "master_horse_form.csv"
STATS_CSV = "master_horse_stats.csv"   # kept alongside for inference convenience only
MODEL_OUT = "apollo_brain.pkl"
CACHE_PKL = ".apollo_training_cache.pkl"  # auto-invalidated when form CSV changes

MIN_PRIOR_RACES = 1  # rows with 0 prior history are dropped (no signal)


def _load_or_build_dataset(form_df: pd.DataFrame):
    """Feature-build is the slow step (~90s on 25k rows). Cache it keyed on CSV mtime+size."""
    import os as _os
    sig = (_os.path.getmtime(FORM_CSV), _os.path.getsize(FORM_CSV), MIN_PRIOR_RACES)
    if _os.path.exists(CACHE_PKL):
        try:
            with open(CACHE_PKL, "rb") as f:
                cached = pickle.load(f)
            if cached.get("sig") == sig:
                print("  (loaded from cache)")
                return cached["X"], cached["y_win"], cached["y_place"], cached["y_score"], cached["groups"]
        except Exception:
            pass
    X, yw, yp, ys, g = build_training_dataset(form_df, min_prior=MIN_PRIOR_RACES)
    with open(CACHE_PKL, "wb") as f:
        pickle.dump({"sig": sig, "X": X, "y_win": yw, "y_place": yp, "y_score": ys, "groups": g}, f)
    return X, yw, yp, ys, g


def main() -> None:
    if not os.path.exists(FORM_CSV):
        sys.exit(f"ERROR: {FORM_CSV} not found.")

    print("=" * 72)
    print("  Aerithius Apollo — Training v3.0")
    print("=" * 72)
    print(f"\nLoading {FORM_CSV} ...")
    form_df = pd.read_csv(FORM_CSV)
    print(f"  Raw rows: {len(form_df):,}")

    print("\nBuilding point-in-time training dataset ...")
    X, y_win, y_place, y_score, groups = _load_or_build_dataset(form_df)

    if len(X) < 500:
        sys.exit(f"ERROR: only {len(X)} usable samples — need more scraped form history.")

    print(f"\n  Features : {X.shape[1]}")
    print(f"  Samples  : {X.shape[0]:,}")
    print(f"  Races    : {len(set(groups)):,}")
    print(f"  Win rate : {y_win.mean()*100:.1f}%")
    print(f"  Place rt : {y_place.mean()*100:.1f}%")

    # ── Scaling ──
    # GBMs don't need it, but the isotonic calibrator's internal splits behave
    # more predictably on standardized inputs, and the API preserves the
    # scaler for consistency with any future linear/NN head.
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X.values)

    # ── Race-aware CV ──
    # CRITICAL: horses within the same race share context. GroupKFold keeps
    # whole races together so we never train on a race and test on it.
    n_races = len(set(groups))
    n_splits = min(5, max(2, n_races // 10))
    gkf = GroupKFold(n_splits=n_splits)
    print(f"\n  CV: GroupKFold({n_splits}) on race-id groups\n")

    # ── Win model ──
    print("Training WIN model ...")
    win_base = GradientBoostingClassifier(
        n_estimators=100, max_depth=3, learning_rate=0.1,
        min_samples_leaf=20, subsample=0.8, random_state=42,
    )
    win_auc = cross_val_score(win_base, X_scaled, y_win, cv=gkf.split(X_scaled, y_win, groups),
                              scoring="roc_auc", n_jobs=-1)
    print(f"  AUC (race-grouped CV): {win_auc.mean():.3f}  ± {win_auc.std():.3f}")
    win_base.fit(X_scaled, y_win)
    # Calibrate with race-grouped folds so horses from the same race never
    # leak across calibration splits (same reason we use GroupKFold above).
    cal_splits = min(3, n_splits)
    win_cal = CalibratedClassifierCV(
        win_base,
        cv=list(GroupKFold(cal_splits).split(X_scaled, y_win, groups)),
        method="isotonic",
    )
    win_cal.fit(X_scaled, y_win)

    # ── Place model ──
    print("\nTraining PLACE model ...")
    place_base = GradientBoostingClassifier(
        n_estimators=100, max_depth=3, learning_rate=0.1,
        min_samples_leaf=20, subsample=0.8, random_state=42,
    )
    place_auc = cross_val_score(place_base, X_scaled, y_place, cv=gkf.split(X_scaled, y_place, groups),
                                scoring="roc_auc", n_jobs=-1)
    print(f"  AUC (race-grouped CV): {place_auc.mean():.3f}  ± {place_auc.std():.3f}")
    place_base.fit(X_scaled, y_place)
    place_cal = CalibratedClassifierCV(
        place_base,
        cv=list(GroupKFold(cal_splits).split(X_scaled, y_place, groups)),
        method="isotonic",
    )
    place_cal.fit(X_scaled, y_place)

    # ── Score regressor ──
    print("\nTraining SCORE regressor ...")
    score_model = GradientBoostingRegressor(
        n_estimators=100, max_depth=3, learning_rate=0.1,
        min_samples_leaf=20, subsample=0.8, random_state=42,
    )
    score_r2 = cross_val_score(score_model, X_scaled, y_score, cv=gkf.split(X_scaled, y_score, groups),
                               scoring="r2", n_jobs=-1)
    print(f"  R²  (race-grouped CV): {score_r2.mean():.3f}  ± {score_r2.std():.3f}")
    score_model.fit(X_scaled, y_score)

    # ── Feature importance ──
    print("\nTop 10 features (WIN model):")
    importances = pd.Series(win_base.feature_importances_, index=FEATURE_NAMES).sort_values(ascending=False)
    for feat, imp in importances.head(10).items():
        bar = "█" * int(imp * 100)
        print(f"  {feat:<28s} {imp:.3f}  {bar}")

    # ── Persist ──
    bundle = {
        "version":       "3.0",
        "win_model":     win_cal,
        "place_model":   place_cal,
        "score_model":   score_model,
        "scaler":        scaler,
        "feature_names": FEATURE_NAMES,
        "metrics": {
            "win_auc_mean":   float(win_auc.mean()),
            "win_auc_std":    float(win_auc.std()),
            "place_auc_mean": float(place_auc.mean()),
            "place_auc_std":  float(place_auc.std()),
            "score_r2_mean":  float(score_r2.mean()),
            "n_train":        int(X.shape[0]),
            "n_races":        int(n_races),
        },
    }
    with open(MODEL_OUT, "wb") as f:
        pickle.dump(bundle, f)

    print(f"\n✓ Apollo v3.0 saved to {MODEL_OUT}")
    print(f"  {X.shape[1]} features · {X.shape[0]:,} samples · {n_races:,} races")


if __name__ == "__main__":
    main()
