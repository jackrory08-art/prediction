"""
Aerithius Apollo — Feature Module v4.0
======================================
Single source of truth for feature extraction. Imported by BOTH training and
inference so the two cannot drift.

Design principles (unchanged from v3, extended for the new data layer)
---------------------------------------------------------------------
1. POINT-IN-TIME: every feature for race R uses only that horse's races BEFORE
   R's date. No leakage.
2. History comes from the SQLite DB (system of record), loaded once into a
   DataFrame for the heavy training loop, or per-horse for live inference.
3. Bayesian smoothing on all rate features so tiny samples don't blow up.
4. Win-centric: the primary signal is win/lose + Betfair SP, which the free
   Betfair feed always provides. Position-based features (place rate, finish
   score) are computed when finishing positions exist and fall back to neutral
   otherwise — so a richer paid source upgrades the model with no code change.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import db

# ─────────────────────────── Feature schema (the contract) ───────────────────────────
FEATURE_NAMES = [
    # Volume / recency
    "prior_starts",
    "days_since_last",
    "days_since_last_known",
    # Win performance
    "prior_win_rate",
    "form_last3_win",
    "form_trend_win",
    # Market / class (Betfair SP)
    "prior_avg_bsp",
    "prior_best_bsp",
    "prior_log_best_bsp",
    "has_bsp_data",
    # Track-specific
    "track_starts",
    "track_win_rate",
    "has_track_data",
    # Distance bucket
    "dist_starts",
    "dist_win_rate",
    "has_dist_data",
    # Going / condition
    "going_win_rate",
    "has_going_data",
    # Position-based (optional — neutral when finishing order is unknown)
    "prior_place_rate",
    "prior_avg_finish_score",
    "has_pos_data",
    # Current race context
    "race_distance_m",
    "race_field_size",
    "race_draw",
    "race_draw_known",
    "race_weight",
    "race_weight_known",
    "region_code",
    # Meta
    "data_confidence",
]

REGION_CODE = {"AUS": 1, "NZ": 2, "GB": 3, "UK": 3, "IRE": 4, "USA": 5, "FR": 6}


# ─────────────────────────── small parsers / math ───────────────────────────

def parse_date(raw) -> pd.Timestamp:
    """ISO `'2026-04-09'` (or `'09-Apr-26'`) -> Timestamp. NaT on failure."""
    ts = pd.to_datetime(raw, format="%Y-%m-%d", errors="coerce")
    if pd.isna(ts):
        ts = pd.to_datetime(raw, errors="coerce")
    return ts


def smooth(count: float, total: float, prior_rate: float = 0.1, strength: float = 5.0) -> float:
    """Bayesian-smoothed rate. Small samples pull toward `prior_rate`."""
    return (count + prior_rate * strength) / (total + strength)


def distance_bucket(meters) -> str | None:
    if meters is None or (isinstance(meters, float) and np.isnan(meters)):
        return None
    m = int(meters)
    if m <= 1000:   return "0-1000"
    if m <= 1300:   return "1001-1300"
    if m <= 1600:   return "1301-1600"
    if m <= 2000:   return "1601-2000"
    if m <= 2400:   return "2001-2400"
    return "2401+"


def finish_score(pos, field_size) -> float | None:
    """1.0 = win, 0.0 = last. None if position unknown."""
    if pos is None or field_size is None:
        return None
    try:
        pos, field_size = int(pos), int(field_size)
    except (TypeError, ValueError):
        return None
    if field_size < 2:
        return 0.5
    return 1.0 - (pos - 1) / (field_size - 1)


def softmax_normalize(win_probs: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Renormalize per-horse win probabilities across a field so they sum to 1.
       Racing is a ranking problem — only one horse can win."""
    logits = np.log(np.clip(win_probs, 1e-6, 1.0)) / temperature
    logits -= logits.max()
    e = np.exp(logits)
    return e / e.sum()


# ─────────────────────────── data prep ───────────────────────────

def prepare_form_df(df: pd.DataFrame) -> pd.DataFrame:
    """Clean + parse the raw DB DataFrame (columns per db._FORM_SQL)."""
    if df.empty:
        return df.assign(_date=pd.Series(dtype="datetime64[ns]"))
    df = df.copy()
    df = df[~df["Track"].astype(str).str.contains("TRIAL", na=False, case=False)]
    df["_date"] = df["Date"].apply(parse_date)
    df = df.dropna(subset=["_date"]).reset_index(drop=True)
    df["_won"] = pd.to_numeric(df.get("Won"), errors="coerce")
    df["_bsp"] = pd.to_numeric(df.get("BSP"), errors="coerce")
    df["_pos"] = pd.to_numeric(df.get("Pos"), errors="coerce")
    df["_field"] = pd.to_numeric(df.get("FieldSize"), errors="coerce")
    df["_dist"] = pd.to_numeric(df.get("DistanceM"), errors="coerce")
    return df


# ─────────────────────────── core feature builder ───────────────────────────

def _defaults() -> dict:
    feats = {name: 0.0 for name in FEATURE_NAMES}
    feats.update({
        "prior_win_rate":         smooth(0, 0),
        "form_last3_win":         smooth(0, 0),
        "prior_avg_bsp":          15.0,
        "prior_best_bsp":         15.0,
        "prior_log_best_bsp":     float(np.log(15.0)),
        "track_win_rate":         smooth(0, 0),
        "dist_win_rate":          smooth(0, 0),
        "going_win_rate":         smooth(0, 0),
        "prior_place_rate":       smooth(0, 0, prior_rate=0.3),
        "prior_avg_finish_score": 0.5,
        "race_weight":            57.0,
    })
    return feats


def build_features_from_history(history: pd.DataFrame, target_race: dict) -> dict:
    """Feature dict for one horse/race. `history` = that horse's prepared runs
    STRICTLY BEFORE the target race, newest-first. May be empty."""
    feats = _defaults()

    # Current-race context (always knowable)
    feats["race_distance_m"] = float(target_race.get("distance_m") or 1200)
    feats["race_field_size"] = float(target_race.get("field_size") or 10)
    draw = target_race.get("draw")
    feats["race_draw"] = float(draw) if draw not in (None, "") else 0.0
    feats["race_draw_known"] = 1.0 if draw not in (None, "") else 0.0
    wt = target_race.get("weight")
    feats["race_weight"] = float(wt) if wt not in (None, "") else 57.0
    feats["race_weight_known"] = 1.0 if wt not in (None, "") else 0.0
    feats["region_code"] = float(REGION_CODE.get(str(target_race.get("region", "")).upper(), 0))

    n = len(history)
    if n == 0:
        feats["data_confidence"] = 0.0
        return feats

    won = history["_won"].fillna(0).astype(int).tolist()
    bsp = [b for b in history["_bsp"].tolist() if pd.notna(b) and b > 0]
    pos = history["_pos"].tolist()
    field = history["_field"].tolist()

    wins = sum(won)
    feats["prior_starts"] = float(n)
    feats["prior_win_rate"] = smooth(wins, n)

    last3 = won[:3]
    older = won[3:]
    feats["form_last3_win"] = float(np.mean(last3)) if last3 else smooth(0, 0)
    feats["form_trend_win"] = float(np.mean(last3) - np.mean(older)) if (last3 and older) else 0.0

    # Market / class
    if bsp:
        feats["prior_avg_bsp"] = float(np.mean(bsp))
        feats["prior_best_bsp"] = float(np.min(bsp))
        feats["prior_log_best_bsp"] = float(np.log(max(1.01, np.min(bsp))))
        feats["has_bsp_data"] = 1.0

    # Days since last
    tdate = target_race.get("race_date")
    if tdate is not None and not pd.isna(tdate):
        last_date = history.iloc[0]["_date"]
        if pd.notna(last_date):
            feats["days_since_last"] = float(max(0, (tdate - last_date).days))
            feats["days_since_last_known"] = 1.0

    # Track-specific
    track = target_race.get("track")
    if track:
        tmask = history["Track"].astype(str).str.lower() == str(track).lower()
        _agg_win(history[tmask], feats, "track")

    # Distance bucket
    tgt_bucket = distance_bucket(target_race.get("distance_m"))
    if tgt_bucket:
        hb = history["_dist"].apply(distance_bucket)
        _agg_win(history[hb == tgt_bucket], feats, "dist")

    # Going
    going = target_race.get("going")
    if going and "Going" in history.columns:
        gmask = history["Going"].astype(str).str.lower() == str(going).lower()
        sub = history[gmask]
        if len(sub):
            gw = int(sub["_won"].fillna(0).sum())
            feats["going_win_rate"] = smooth(gw, len(sub))
            feats["has_going_data"] = 1.0

    # Position-based (optional)
    known_pos = [(p, f) for p, f in zip(pos, field) if pd.notna(p) and pd.notna(f)]
    if known_pos:
        places = sum(1 for p, _ in known_pos if int(p) <= 3)
        scores = [finish_score(p, f) for p, f in known_pos]
        scores = [s for s in scores if s is not None]
        feats["prior_place_rate"] = smooth(places, len(known_pos), prior_rate=0.3)
        feats["prior_avg_finish_score"] = float(np.mean(scores)) if scores else 0.5
        feats["has_pos_data"] = 1.0

    feats["data_confidence"] = min(n / 10.0, 1.0)
    return feats


def _agg_win(subset: pd.DataFrame, feats: dict, prefix: str) -> None:
    n = len(subset)
    if n == 0:
        return
    wins = int(subset["_won"].fillna(0).sum())
    feats[f"{prefix}_starts"] = float(n)
    feats[f"{prefix}_win_rate"] = smooth(wins, n)
    feats[f"has_{prefix}_data"] = 1.0


# ─────────────────────────── training dataset ───────────────────────────

def build_training_dataset(form_df: pd.DataFrame | None = None, min_prior: int = 1,
                           verbose: bool = True):
    """For every runner, compute features from that horse's PRIOR races only.

    Returns X (DataFrame), y_win, y_place (may contain NaN), y_score (may contain
    NaN), groups (race_id). Place/score targets are NaN where finishing order is
    unknown — the trainer simply skips those models if too few are populated.
    """
    if form_df is None:
        form_df = db.load_form_df()
    df = prepare_form_df(form_df)
    if verbose:
        print(f"  Clean rows: {len(df):,} across {df['HorseId'].nunique():,} horses")

    df = df.sort_values(["HorseId", "_date"], ascending=[True, False]).reset_index(drop=True)
    horse_groups = dict(tuple(df.groupby("HorseId", sort=False)))

    X_rows, y_win, y_place, y_score, groups = [], [], [], [], []
    for _, row in df.iterrows():
        hist = horse_groups[row["HorseId"]]
        hist = hist[hist["_date"] < row["_date"]]
        if len(hist) < min_prior:
            continue
        target = {
            "race_date": row["_date"],
            "distance_m": None if pd.isna(row["_dist"]) else int(row["_dist"]),
            "track": row["Track"],
            "going": row.get("Going"),
            "draw": None if pd.isna(row.get("Draw")) else row.get("Draw"),
            "weight": None if pd.isna(row.get("Weight")) else row.get("Weight"),
            "field_size": None if pd.isna(row["_field"]) else int(row["_field"]),
            "region": row.get("Region"),
        }
        X_rows.append(build_features_from_history(hist, target))
        y_win.append(int(row["_won"]) if pd.notna(row["_won"]) else np.nan)
        y_place.append(int(row["_pos"] <= 3) if pd.notna(row["_pos"]) else np.nan)
        y_score.append(finish_score(row["_pos"], row["_field"])
                       if pd.notna(row["_pos"]) else np.nan)
        groups.append(row["RaceId"])

        if verbose and len(X_rows) % 5000 == 0:
            print(f"    ... built {len(X_rows):,} rows")

    if verbose:
        print(f"  Built {len(X_rows):,} training samples (need {min_prior}+ prior races)")
    X = pd.DataFrame(X_rows)[FEATURE_NAMES]
    return (X, np.array(y_win, dtype=float), np.array(y_place, dtype=float),
            np.array(y_score, dtype=float), np.array(groups))


# ─────────────────────────── inference ───────────────────────────

def build_inference_history(horse_id: str, as_of: pd.Timestamp | None = None) -> pd.DataFrame:
    """The horse's prepared form before `as_of` (default now), newest first."""
    df = prepare_form_df(db.load_form_df(horse_id=horse_id))
    if df.empty:
        return df
    if as_of is not None:
        df = df[df["_date"] < as_of]
    return df.sort_values("_date", ascending=False).reset_index(drop=True)
