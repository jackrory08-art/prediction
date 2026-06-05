"""
Apollo ingest — pull racing results into SQLite
===============================================
Reads normalized race records from a DataSource (default: Betfair free) and
upserts them into `apollo.db`. Idempotent: safe to re-run daily; existing races
are updated in place.

Usage:
  python ingest.py --from-fixtures        # offline sample data (no key/egress)
  python ingest.py --days 365             # last 365 days of live Betfair BSP
  python ingest.py --start 2025-01-01 --end 2025-12-31 --regions AUS,GB,IRE,USA
"""
from __future__ import annotations

import argparse
import hashlib
from datetime import date, timedelta

import db
from data_sources import BetfairSource


def _race_id(rec: dict) -> str:
    key = f"{rec.get('date')}|{rec.get('track')}|{rec.get('race_no')}|{rec.get('distance_m')}"
    return hashlib.sha1(key.encode()).hexdigest()[:16]


def ingest(source, start: str, end: str, regions: list[str]) -> dict:
    db.init_db()
    conn = db.connect()
    n_races = n_runners = 0
    try:
        for rec in source.results(start, end, regions):
            runners = rec.get("runners") or []
            if not rec.get("date") or not runners:
                continue
            rid = _race_id(rec)
            db.upsert_race(conn, {
                "race_id": rid,
                "date": rec["date"],
                "track": rec.get("track") or "Unknown",
                "race_no": rec.get("race_no"),
                "distance_m": rec.get("distance_m"),
                "going": rec.get("going"),
                "field_size": len(runners),
                "region": rec.get("region") or "UNK",
            })
            n_races += 1
            for r in runners:
                pos = r.get("finish_pos")
                won = r.get("won")
                if won is None and pos is not None:
                    won = 1 if pos == 1 else 0
                placed = r.get("placed")
                if placed is None and pos is not None:
                    placed = 1 if pos <= 3 else 0
                db.upsert_runner(conn, {
                    "race_id": rid,
                    "horse": r["horse"],
                    "horse_id": db.normalize_horse_id(r["horse"]),
                    "finish_pos": pos,
                    "draw": r.get("draw"),
                    "weight": r.get("weight"),
                    "jockey": r.get("jockey"),
                    "trainer": r.get("trainer"),
                    "bsp": r.get("bsp"),
                    "won": won,
                    "placed": placed,
                })
                n_runners += 1
            if n_races % 500 == 0:
                conn.commit()
        conn.commit()
    finally:
        conn.close()
    return {"races": n_races, "runners": n_runners}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-fixtures", action="store_true",
                    help="use committed sample data in tests/fixtures (offline)")
    ap.add_argument("--days", type=int, default=None, help="ingest the last N days")
    ap.add_argument("--start", type=str, default=None, help="ISO start date")
    ap.add_argument("--end", type=str, default=None, help="ISO end date")
    ap.add_argument("--regions", type=str, default="AUS,GB,IRE,USA")
    args = ap.parse_args()

    source = BetfairSource(fixtures="tests/fixtures" if args.from_fixtures else None)

    if args.days:
        end = date.today()
        start = end - timedelta(days=args.days)
        start_s, end_s = start.isoformat(), end.isoformat()
    else:
        start_s = args.start or "2026-01-01"
        end_s = args.end or "2026-12-31"

    regions = [r.strip().upper() for r in args.regions.split(",") if r.strip()]

    print(f"Ingesting {source.name} {start_s}..{end_s} regions={regions}"
          f"{' (fixtures)' if args.from_fixtures else ''}")
    stats = ingest(source, start_s, end_s, regions)
    counts = db.row_counts()
    print(f"  added/updated: {stats['races']:,} races, {stats['runners']:,} runners")
    print(f"  DB now holds : {counts['races']:,} races, {counts['runners']:,} runners, "
          f"{counts['horses']:,} horses")


if __name__ == "__main__":
    main()
