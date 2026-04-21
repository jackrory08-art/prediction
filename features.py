"""
Aerithius Apollo — Feature Module v3.0
======================================
Single source of truth for feature extraction. Imported by BOTH training
and inference so the two cannot drift.

Design principles
-----------------
1. POINT-IN-TIME: every feature for race R uses only races that occurred
   BEFORE R's date. No leakage.
2. Features come from `form_df` (per-race history) — NOT from the aggregate
   stats CSV, which is a "snapshot" that includes the outcomes we're trying
   to predict.
3. Bayesian smoothing on all rate features so tiny samples don't blow up.
4. "Known" flags for optional fields (barrier, weight, condition) so the
   model learns to weight them only when present.
"""

from __future__ import annotations
import numpy as np
import pandas as pd


# ─────────────────────────── Feature schema ───────────────────────────
# This list is the contract. Both training and inference produce DataFrames
# with exactly these columns in this order.

FEATURE_NAMES = [
    # Volume / recency
    "prior_starts",
    "days_since_last",
    "days_since_last_known",

    # Overall prior performance
    "prior_win_rate",
    "prior_place_rate",
    "prior_avg_finish_score",

    # Form trend (last 3 vs older)
    "form_last3_score",
    "form_last3_n",
    "form_trend",

    # Track-specific history
    "track_starts",
    "track_win_rate",
    "track_place_rate",
    "has_track_data",

    # Distance-bucket history
    "dist_starts",
    "dist_win_rate",
    "dist_place_rate",
    "has_dist_data",

    # Track-condition history (populated only when scraper provides it)
    "cond_starts",
    "cond_win_rate",
    "cond_place_rate",
    "has_cond_data",

    # Class / quality proxies (from prior races)
    "prior_avg_sp",
    "prior_best_sp",
    "prior_avg_margin",
    "prior_avg_field_size",

    # Current race context
    "race_distance_m",
    "race_field_size",
    "race_barrier",
    "race_barrier_known",
    "race_weight",
    "race_weight_known",

    # Meta
    "data_confidence",
]


# ─────────────────────────── Parsers ───────────────────────────

def parse_finish_pos(raw) -> tuple[int | None, int | None]:
    """`'3/12'` → `(3, 12)`. Returns `(None, None)` on failure."""
    try:
        parts = str(raw).split("/")
        return int(parts[0]), int(parts[1])
    except Exception:
        return None, None


def parse_distance(raw) -> int | None:
    """`'1200m'` → `1200`. Returns `None` on failure."""
    try:
        s = str(raw).lower().replace("m", "").strip()
        return int(s) if s.isdigit() else None
    except Exception:
        return None


def parse_date(raw) -> pd.Timestamp:
    """`'09-Apr-26'` → `Timestamp('2026-04-09')`. Returns `NaT` on failure."""
    return pd.to_datetime(raw, format="%d-%b-%y", errors="coerce")


def parse_float(raw) -> float | None:
    try:
        s = str(raw).replace("$", "").replace(",", "").replace("%", "").strip()
        if s in ("", "-", "nan", "None", "NaN"):
            return None
        return float(s)
    except Exception:
        return None


def parse_int(raw) -> int | None:
    v = parse_float(raw)
    return int(v) if v is not None else None


def distance_bucket(meters: int | None) -> str | None:
    if meters is None:
        return None
    if meters <= 1000:   return "0-1000"
    if meters <= 1300:   return "1001-1300"
    if meters <= 1600:   return "1301-1600"
    if meters <= 2000:   return "1601-2000"
    if meters <= 2400:   return "2001-2400"
    return "2401+"


def smooth(count: float, total: float, prior_rate: float = 0.1, strength: float = 5.0) -> float:
    """Bayesian-smoothed rate. Small samples pull toward `prior_rate`."""
    return (count + prior_rate * strength) / (total + strength)


def finish_score(pos: int | None, field_size: int | None) -> float:
    """1.0 = win, 0.0 = last. Neutral 0.5 if unknown."""
    if pos is None or field_size is None or field_size < 2:
        return 0.5
    return 1.0 - (pos - 1) / (field_size - 1)


# ─────────────────────────── Core feature builder ───────────────────────────

def _defaults() -> dict:
    """Neutral defaults for a horse with no prior history."""
    return {name: 0.0 for name in FEATURE_NAMES} | {
        "prior_win_rate":       smooth(0, 0),
        "prior_place_rate":     smooth(0, 0, prior_rate=0.3),
        "prior_avg_finish_score": 0.5,
        "form_last3_score":     0.5,
        "track_win_rate":       smooth(0, 0),
        "track_place_rate":     smooth(0, 0, prior_rate=0.3),
        "dist_win_rate":        smooth(0, 0),
        "dist_place_rate":      smooth(0, 0, prior_rate=0.3),
        "cond_win_rate":        smooth(0, 0),
        "cond_place_rate":      smooth(0, 0, prior_rate=0.3),
        "prior_avg_sp":         15.0,     # neutral-ish
        "prior_best_sp":        15.0,
        "prior_avg_margin":     5.0,
        "prior_avg_field_size": 10.0,
    }


def build_features_from_history(history_df: pd.DataFrame, target_race: dict) -> dict:
    """
    Build a feature dict for a single horse/race.

    Parameters
    ----------
    history_df : DataFrame
        The horse's form entries STRICTLY BEFORE `target_race`. Must be sorted
        newest-first. Trials already removed. May be empty.
    target_race : dict with keys
        race_date (Timestamp), distance_m (int), track (str),
        condition (str|None), barrier (int|None), weight (float|None),
        field_size (int|None)
    """
    feats = _defaults()

    # ── Current-race context (always knowable) ──
    feats["race_distance_m"]     = float(target_race.get("distance_m") or 1200)
    feats["race_field_size"]     = float(target_race.get("field_size") or 10)
    bar = target_race.get("barrier")
    feats["race_barrier"]        = float(bar) if bar is not None else 0.0
    feats["race_barrier_known"]  = 1.0 if bar is not None else 0.0
    wt = target_race.get("weight")
    feats["race_weight"]         = float(wt) if wt is not None else 57.0
    feats["race_weight_known"]   = 1.0 if wt is not None else 0.0

    n = len(history_df)
    if n == 0:
        feats["data_confidence"] = 0.0
        return feats

    # ── Parse all prior races once ──
    positions, fields, scores = [], [], []
    for _, r in history_df.iterrows():
        p, f = parse_finish_pos(r.get("Finish_Pos"))
        positions.append(p)
        fields.append(f)
        scores.append(finish_score(p, f))

    wins   = sum(1 for p in positions if p == 1)
    places = sum(1 for p in positions if p is not None and p <= 3)

    feats["prior_starts"]            = float(n)
    feats["prior_win_rate"]          = smooth(wins, n)
    feats["prior_place_rate"]        = smooth(places, n, prior_rate=0.3)
    feats["prior_avg_finish_score"]  = float(np.mean(scores))

    # ── Form trend (last 3 vs older) ──
    last3 = scores[:3]
    older = scores[3:]
    feats["form_last3_score"] = float(np.mean(last3)) if last3 else 0.5
    feats["form_last3_n"]     = float(len(last3))
    feats["form_trend"]       = float(np.mean(last3) - np.mean(older)) if last3 and older else 0.0

    # ── Days since last run ──
    target_date = target_race.get("race_date")
    if target_date is not None and not pd.isna(target_date) and "_date_parsed" in history_df.columns:
        last_date = history_df.iloc[0]["_date_parsed"]
        if pd.notna(last_date):
            feats["days_since_last"]        = float(max(0, (target_date - last_date).days))
            feats["days_since_last_known"]  = 1.0

    # ── Track-specific ──
    track = target_race.get("track")
    if track and "Track" in history_df.columns:
        tmask = history_df["Track"].astype(str).str.lower() == str(track).lower()
        _agg_subset(history_df[tmask], feats, prefix="track")

    # ── Distance bucket ──
    tgt_bucket = distance_bucket(target_race.get("distance_m"))
    if tgt_bucket and "Distance" in history_df.columns:
        hist_buckets = history_df["Distance"].apply(lambda x: distance_bucket(parse_distance(x)))
        _agg_subset(history_df[hist_buckets == tgt_bucket], feats, prefix="dist")

    # ── Track condition ──
    cond = target_race.get("condition")
    if cond and "Condition" in history_df.columns:
        cmask = history_df["Condition"].astype(str).str.lower() == str(cond).lower()
        _agg_subset(history_df[cmask], feats, prefix="cond")

    # ── Class / quality proxies ──
    sps = [parse_float(v) for v in history_df.get("SP", pd.Series([], dtype=float))]
    sps = [x for x in sps if x is not None and x > 0]
    if sps:
        feats["prior_avg_sp"]  = float(np.mean(sps))
        feats["prior_best_sp"] = float(np.min(sps))

    margins = [parse_float(v) for v in history_df.get("Margin", pd.Series([], dtype=float))]
    margins = [x for x in margins if x is not None and x >= 0]
    if margins:
        feats["prior_avg_margin"] = float(np.mean(margins))

    valid_fields = [f for f in fields if f is not None]
    if valid_fields:
        feats["prior_avg_field_size"] = float(np.mean(valid_fields))

    feats["data_confidence"] = min(n / 10.0, 1.0)
    return feats


def _agg_subset(subset: pd.DataFrame, feats: dict, *, prefix: str) -> None:
    """Fill `{prefix}_starts`, `{prefix}_win_rate`, `{prefix}_place_rate`,
       `has_{prefix}_data` from a slice of history."""
    n = len(subset)
    if n == 0:
        return
    wins = places = 0
    for _, r in subset.iterrows():
        p, _ = parse_finish_pos(r.get("Finish_Pos"))
        if p == 1:
            wins += 1
        if p is not None and p <= 3:
            places += 1
    feats[f"{prefix}_starts"]      = float(n)
    feats[f"{prefix}_win_rate"]    = smooth(wins, n)
    feats[f"{prefix}_place_rate"]  = smooth(places, n, prior_rate=0.3)
    feats[f"has_{prefix}_data"]    = 1.0


# ─────────────────────────── Training dataset builder ───────────────────────────

def prepare_form_df(form_df: pd.DataFrame) -> pd.DataFrame:
    """Clean + parse the raw form CSV. Trials removed, dates parsed, etc."""
    df = form_df.copy()
    df = df[~df["Track"].astype(str).str.contains("TRIAL", na=False, case=False)]
    df["_date_parsed"] = df["Date"].apply(parse_date)
    df = df.dropna(subset=["_date_parsed"]).reset_index(drop=True)

    pos_field = df["Finish_Pos"].apply(lambda x: pd.Series(parse_finish_pos(x), index=["_pos", "_field"]))
    df[["_pos", "_field"]] = pos_field
    df["_dist_m"] = df["Distance"].apply(parse_distance)

    df = df.dropna(subset=["_pos", "_field", "_dist_m"])
    df["_pos"]    = df["_pos"].astype(int)
    df["_field"]  = df["_field"].astype(int)
    df["_dist_m"] = df["_dist_m"].astype(int)
    return df.reset_index(drop=True)


def build_training_dataset(form_df: pd.DataFrame, min_prior: int = 1, verbose: bool = True):
    """
    For every form row, compute features from that horse's PRIOR races only.

    Returns
    -------
    X        : DataFrame with columns == FEATURE_NAMES
    y_win    : 1 if this race was won, else 0
    y_place  : 1 if placed top-3, else 0
    y_score  : continuous 0-1 finish score
    groups   : race-group keys (same date+track+distance = same race)
    """
    df = prepare_form_df(form_df)
    if verbose:
        print(f"  Clean form rows: {len(df):,} across {df['Horse'].nunique():,} horses")

    # Pre-sort once (speeds up per-horse slicing massively)
    df = df.sort_values(["Horse", "_date_parsed"], ascending=[True, False]).reset_index(drop=True)
    horse_groups = dict(tuple(df.groupby("Horse", sort=False)))

    has_condition = "Condition" in df.columns

    X_rows, y_win, y_place, y_score, groups = [], [], [], [], []

    for idx, row in df.iterrows():
        horse = row["Horse"]
        race_date = row["_date_parsed"]

        horse_hist = horse_groups[horse]
        hist = horse_hist[horse_hist["_date_parsed"] < race_date]
        if len(hist) < min_prior:
            continue

        target = {
            "race_date":   race_date,
            "distance_m":  int(row["_dist_m"]),
            "track":       row["Track"],
            "condition":   row["Condition"] if has_condition and pd.notna(row.get("Condition")) else None,
            "barrier":     parse_int(row.get("Barrier")),
            "weight":      parse_float(row.get("Weight")),
            "field_size":  int(row["_field"]),
        }

        feats = build_features_from_history(hist, target)
        X_rows.append(feats)

        pos, field = int(row["_pos"]), int(row["_field"])
        y_win.append(1 if pos == 1 else 0)
        y_place.append(1 if pos <= 3 else 0)
        y_score.append(finish_score(pos, field))
        groups.append(f"{row['Date']}|{row['Track']}|{row['Distance']}")

        if verbose and (len(X_rows) % 5000 == 0):
            print(f"    ... built {len(X_rows):,} training rows")

    if verbose:
        print(f"  Built {len(X_rows):,} training samples (horses need {min_prior}+ prior races)")

    X = pd.DataFrame(X_rows)[FEATURE_NAMES]
    return X, np.array(y_win), np.array(y_place), np.array(y_score), np.array(groups)


# ─────────────────────────── Inference helpers ───────────────────────────

def build_inference_history(form_df: pd.DataFrame, horse: str, as_of: pd.Timestamp | None = None) -> pd.DataFrame:
    """Return the horse's form history before `as_of` (default: now), newest first.
       Accepts the RAW form_df — will clean/parse internally."""
    clean = prepare_form_df(form_df)
    clean = clean[clean["Horse"] == horse]
    if as_of is not None:
        clean = clean[clean["_date_parsed"] < as_of]
    return clean.sort_values("_date_parsed", ascending=False).reset_index(drop=True)


def softmax_normalize(win_probs: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Renormalize per-horse win probabilities across a field so they sum to 1.
       Racing is a ranking problem — only one horse can win."""
    logits = np.log(np.clip(win_probs, 1e-6, 1.0)) / temperature
    logits -= logits.max()
    e = np.exp(logits)
    return e / e.sum()
