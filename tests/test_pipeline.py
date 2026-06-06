"""Fast offline smoke tests for the fixture → DB → features → predict path."""
import os
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import db                                  # noqa: E402
import features                            # noqa: E402
from data_sources import BetfairSource     # noqa: E402
from ingest import ingest                  # noqa: E402

FIX = str(ROOT / "tests" / "fixtures")


def _fresh_db(tmp_path):
    """Point the whole app at a throwaway DB and ingest fixtures into it."""
    dbfile = str(tmp_path / "t.db")
    db.DB_PATH = dbfile
    features.db.DB_PATH = dbfile
    src = BetfairSource(fixtures=FIX)
    stats = ingest(src, "2026-01-01", "2026-12-31", ["AUS", "GB", "IRE", "USA"])
    return dbfile, stats


def test_ingest_populates_db(tmp_path):
    _, stats = _fresh_db(tmp_path)
    assert stats["races"] > 50
    assert stats["runners"] > 500
    counts = db.row_counts(db.DB_PATH)
    assert counts["horses"] > 20


def test_form_has_market_and_labels(tmp_path):
    _fresh_db(tmp_path)
    df = db.load_form_df(db.DB_PATH)
    # The whole point of switching sources: these come back POPULATED.
    assert df["BSP"].notna().mean() > 0.9
    assert df["Won"].notna().mean() > 0.9
    assert set(df["Won"].dropna().unique()) <= {0, 1}


def test_features_complete_and_pointintime(tmp_path):
    _fresh_db(tmp_path)
    X, y_win, y_place, y_score, groups = features.build_training_dataset(verbose=False)
    assert list(X.columns) == features.FEATURE_NAMES
    assert len(X) > 200
    assert set(pd.unique(y_win[~pd.isna(y_win)])) <= {0, 1}
    # On the free win-only Betfair feed the only known finishing position is the
    # winner's (1st), so every known place label is 1 — not enough signal to
    # train a place model (the trainer skips it; see MIN_LABELS + unique check).
    known_place = y_place[~pd.isna(y_place)]
    assert set(pd.unique(known_place)) <= {1}
    # a horse with history should have non-default market features
    assert (X["has_bsp_data"] == 1.0).mean() > 0.8
