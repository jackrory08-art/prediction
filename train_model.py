"""
Aerithius Apollo — Trainer v4.0
===============================
Point-in-time features, race-aware cross-validation, calibrated probabilities.

  • WIN model   — always trained (needs only win/lose + Betfair SP + form),
                  so it works on the free Betfair feed.
  • PLACE/SCORE — trained only when finishing positions are available (a richer
                  paid source); otherwise skipped cleanly.

Evaluation goes beyond AUC: top-1 hit rate (did our #1 pick win?) is what a
tipster actually cares about, computed from out-of-fold, race-grouped preds.
"""
from __future__ import annotations

import pickle
import warnings
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.metrics import roc_auc_score, log_loss

import db
from features import FEATURE_NAMES, build_training_dataset

warnings.filterwarnings("ignore")

MODEL_OUT = "apollo_brain.pkl"
MIN_PRIOR_RACES = 1
MIN_LABELS_FOR_OPTIONAL = 300   # need this many positions to train place/score


def _splits(n_races: int) -> int:
    return min(5, max(2, n_races // 10))


def _top1_hit_rate(groups, y_win, win_prob) -> float:
    """Per race, take the horse with the highest predicted win prob; what
    fraction actually won? The headline tipping metric."""
    d = pd.DataFrame({"g": groups, "y": y_win, "p": win_prob})
    hits = d.loc[d.groupby("g")["p"].idxmax()]["y"]
    return float(hits.mean()) if len(hits) else float("nan")


def _new_win_model() -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        max_depth=3, learning_rate=0.05, max_iter=300,
        min_samples_leaf=40, l2_regularization=1.0, random_state=42,
    )


def main() -> None:
    print("=" * 72)
    print("  Aerithius Apollo — Training v4.0")
    print("=" * 72)

    counts = db.row_counts()
    print(f"\nDB: {counts['races']:,} races · {counts['runners']:,} runners · "
          f"{counts['horses']:,} horses")

    print("\nBuilding point-in-time training dataset ...")
    X, y_win, y_place, y_score, groups = build_training_dataset(min_prior=MIN_PRIOR_RACES)

    win_mask = np.isfinite(y_win)
    Xw, yw, gw = X[win_mask], y_win[win_mask].astype(int), groups[win_mask]
    if len(Xw) < 200:
        raise SystemExit(f"ERROR: only {len(Xw)} usable samples — ingest more history.")

    n_races = len(set(gw))
    n_splits = _splits(n_races)
    print(f"\n  Features : {X.shape[1]}")
    print(f"  Samples  : {len(Xw):,}")
    print(f"  Races    : {n_races:,}")
    print(f"  Win rate : {yw.mean()*100:.1f}%")
    print(f"  CV       : GroupKFold({n_splits}) on race ids")

    # ── WIN model ──
    print("\nTraining WIN model ...")
    gkf = GroupKFold(n_splits=n_splits)
    base = _new_win_model()
    oof = cross_val_predict(base, Xw.values, yw, groups=gw, cv=gkf,
                            method="predict_proba", n_jobs=-1)[:, 1]
    auc = roc_auc_score(yw, oof)
    ll = log_loss(yw, np.clip(oof, 1e-6, 1 - 1e-6))
    hit1 = _top1_hit_rate(gw, yw, oof)
    base_hit = yw.mean()  # naive baseline = overall win rate
    print(f"  AUC (oof)        : {auc:.3f}")
    print(f"  LogLoss (oof)    : {ll:.3f}")
    print(f"  Top-1 hit rate   : {hit1*100:.1f}%   (naive {base_hit*100:.1f}%)")

    cal_method = "isotonic" if len(Xw) >= 1000 else "sigmoid"
    win_cal = CalibratedClassifierCV(
        _new_win_model(),
        cv=list(GroupKFold(min(3, n_splits)).split(Xw.values, yw, gw)),
        method=cal_method,
    )
    win_cal.fit(Xw.values, yw)

    # ── optional PLACE / SCORE ──
    place_cal = score_model = None
    place_auc = score_r2 = None
    place_mask = np.isfinite(y_place)
    if int(place_mask.sum()) >= MIN_LABELS_FOR_OPTIONAL and len(set(y_place[place_mask])) > 1:
        print("\nTraining PLACE model (positions available) ...")
        Xp, yp, gp = X[place_mask], y_place[place_mask].astype(int), groups[place_mask]
        p_oof = cross_val_predict(_new_win_model(), Xp.values, yp, groups=gp,
                                  cv=GroupKFold(_splits(len(set(gp)))),
                                  method="predict_proba", n_jobs=-1)[:, 1]
        place_auc = roc_auc_score(yp, p_oof)
        print(f"  PLACE AUC (oof)  : {place_auc:.3f}")
        place_cal = CalibratedClassifierCV(
            _new_win_model(),
            cv=list(GroupKFold(min(3, _splits(len(set(gp))))).split(Xp.values, yp, gp)),
            method=("isotonic" if len(Xp) >= 1000 else "sigmoid"),
        )
        place_cal.fit(Xp.values, yp)

        score_mask = np.isfinite(y_score)
        Xs, ys = X[score_mask], y_score[score_mask]
        score_model = HistGradientBoostingRegressor(
            max_depth=3, learning_rate=0.05, max_iter=300,
            min_samples_leaf=40, l2_regularization=1.0, random_state=42)
        score_model.fit(Xs.values, ys)
        print("  SCORE regressor  : trained")
    else:
        print("\nPLACE/SCORE models skipped — no finishing positions in this data "
              "(free Betfair feed). Win tips are fully functional; add a positions "
              "source to enable them.")

    # ── feature importance (permutation-free quick proxy via the win model) ──
    fitted = _new_win_model().fit(Xw.values, yw)
    try:
        from sklearn.inspection import permutation_importance
        imp = permutation_importance(fitted, Xw.values, yw, n_repeats=3,
                                     random_state=42, n_jobs=-1)
        order = pd.Series(imp.importances_mean, index=FEATURE_NAMES).sort_values(ascending=False)
        print("\nTop 10 features (WIN):")
        for feat, val in order.head(10).items():
            print(f"  {feat:<26s} {val:.4f}  {'█' * int(max(0, val) * 300)}")
    except Exception:
        pass

    bundle = {
        "version": "4.0",
        "win_model": win_cal,
        "place_model": place_cal,
        "score_model": score_model,
        "feature_names": FEATURE_NAMES,
        "metrics": {
            "win_auc": float(auc),
            "win_logloss": float(ll),
            "win_top1_hit": float(hit1),
            "naive_hit": float(base_hit),
            "place_auc": None if place_auc is None else float(place_auc),
            "n_train": int(len(Xw)),
            "n_races": int(n_races),
        },
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(MODEL_OUT, "wb") as f:
        pickle.dump(bundle, f)
    print(f"\n✓ Apollo v4.0 saved to {MODEL_OUT}")
    print(f"  {X.shape[1]} features · {len(Xw):,} samples · {n_races:,} races")


if __name__ == "__main__":
    main()
