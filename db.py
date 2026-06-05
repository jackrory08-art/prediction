"""
Apollo storage layer — SQLite
=============================
Replaces the old flat CSVs with a real (embedded, zero-ops) database.

System of record for all horse-racing history. Two tables:

  races   — one row per race (date, track, distance, going, region, …)
  runners — one row per horse-in-a-race (finish, draw, weight, BSP, …)

The feature builder loads this into a DataFrame for the heavy point-in-time
training loop; the live app queries it for a single horse's prior form.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

DB_PATH = str(Path(__file__).with_name("apollo.db"))

SCHEMA = """
CREATE TABLE IF NOT EXISTS races (
    race_id      TEXT PRIMARY KEY,   -- stable hash of date|track|race_no
    date         TEXT NOT NULL,      -- ISO YYYY-MM-DD
    track        TEXT NOT NULL,
    race_no      INTEGER,
    distance_m   INTEGER,
    going        TEXT,
    field_size   INTEGER,
    region       TEXT                -- AUS, GB, IRE, USA, ...
);

CREATE TABLE IF NOT EXISTS runners (
    runner_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    race_id      TEXT NOT NULL REFERENCES races(race_id),
    horse        TEXT NOT NULL,
    horse_id     TEXT NOT NULL,      -- normalized horse key (lower, no spaces)
    finish_pos   INTEGER,            -- 1 = winner; NULL if scratched/unknown
    draw         INTEGER,
    weight       REAL,               -- kg
    jockey       TEXT,
    trainer      TEXT,
    bsp          REAL,               -- Betfair Starting Price (decimal odds)
    won          INTEGER,            -- 1/0
    placed       INTEGER,            -- 1/0 (top 3)
    UNIQUE(race_id, horse_id)
);

CREATE INDEX IF NOT EXISTS idx_runners_horse ON runners(horse_id, race_id);
CREATE INDEX IF NOT EXISTS idx_races_date    ON races(date);
"""


def connect(db_path: str | None = None) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path or DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    return conn


def init_db(db_path: str | None = None) -> None:
    """Create tables/indexes if they don't exist. Idempotent."""
    conn = connect(db_path)
    try:
        conn.executescript(SCHEMA)
        conn.commit()
    finally:
        conn.close()


def normalize_horse_id(name: str) -> str:
    """Stable key for a horse across races. `'Star Wave (GB)'` -> `'starwave'`."""
    import re

    s = str(name).lower()
    s = re.sub(r"\(.*?\)", "", s)          # drop country suffix e.g. (GB)
    s = re.sub(r"[^a-z0-9]", "", s)        # keep alnum only
    return s


def upsert_race(conn: sqlite3.Connection, race: dict) -> None:
    conn.execute(
        """INSERT INTO races (race_id, date, track, race_no, distance_m, going, field_size, region)
           VALUES (:race_id, :date, :track, :race_no, :distance_m, :going, :field_size, :region)
           ON CONFLICT(race_id) DO UPDATE SET
             date=excluded.date, track=excluded.track, race_no=excluded.race_no,
             distance_m=excluded.distance_m, going=excluded.going,
             field_size=excluded.field_size, region=excluded.region""",
        race,
    )


def upsert_runner(conn: sqlite3.Connection, runner: dict) -> None:
    conn.execute(
        """INSERT INTO runners
             (race_id, horse, horse_id, finish_pos, draw, weight, jockey, trainer, bsp, won, placed)
           VALUES
             (:race_id, :horse, :horse_id, :finish_pos, :draw, :weight, :jockey, :trainer, :bsp, :won, :placed)
           ON CONFLICT(race_id, horse_id) DO UPDATE SET
             horse=excluded.horse, finish_pos=excluded.finish_pos, draw=excluded.draw,
             weight=excluded.weight, jockey=excluded.jockey, trainer=excluded.trainer,
             bsp=excluded.bsp, won=excluded.won, placed=excluded.placed""",
        runner,
    )


# ─────────────────────────── Read helpers ───────────────────────────

# Single JOIN that both training (whole table) and inference (one horse) reuse.
_FORM_SQL = """
SELECT  ru.horse        AS Horse,
        ru.horse_id     AS HorseId,
        ra.date         AS Date,
        ra.track        AS Track,
        ra.distance_m   AS DistanceM,
        ru.finish_pos   AS Pos,
        ru.won          AS Won,
        ru.placed       AS Placed,
        ra.field_size   AS FieldSize,
        ru.draw         AS Draw,
        ru.weight       AS Weight,
        ru.jockey       AS Jockey,
        ru.trainer      AS Trainer,
        ru.bsp          AS BSP,
        ra.going        AS Going,
        ra.region       AS Region,
        ra.race_id      AS RaceId
FROM runners ru
JOIN races   ra ON ra.race_id = ru.race_id
"""


def load_form_df(db_path: str | None = None, horse_id: str | None = None) -> pd.DataFrame:
    """Load history as a DataFrame. Whole table for training, or one horse for
    inference. Columns are the canonical names the feature module expects."""
    conn = connect(db_path)
    try:
        if horse_id is not None:
            df = pd.read_sql_query(_FORM_SQL + " WHERE ru.horse_id = ?", conn, params=[horse_id])
        else:
            df = pd.read_sql_query(_FORM_SQL, conn)
    finally:
        conn.close()
    return df


def row_counts(db_path: str | None = None) -> dict:
    conn = connect(db_path)
    try:
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM races")
        races = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM runners")
        runners = cur.fetchone()[0]
        cur.execute("SELECT COUNT(DISTINCT horse_id) FROM runners")
        horses = cur.fetchone()[0]
        return {"races": races, "runners": runners, "horses": horses}
    finally:
        conn.close()
