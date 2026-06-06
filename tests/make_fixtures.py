"""
Generate synthetic Betfair BSP fixture files for offline testing.
=================================================================
Writes `tests/fixtures/bsp_*.csv` (Betfair SP format) and a sample racecard.
Data is deterministic (seeded) and has real structure so the model can learn:

  • a pool of horses, each with a latent "ability"
  • BSP derived from ability (+ noise) — so the favourite is usually best
  • the winner sampled by ability within each race — so past wins predict future

This stands in for the real promo.betfair.com BSP downloads, which we can't
reach from the sandbox. Run: `python tests/make_fixtures.py`.
"""
from __future__ import annotations

import csv
import json
import math
import random
from datetime import date, timedelta
from pathlib import Path

FIX = Path(__file__).with_name("fixtures")
FIX.mkdir(exist_ok=True)

rng = random.Random(42)

REGIONS = [("AUS", ["Flemington", "Randwick", "Caulfield", "Eagle Farm"]),
           ("GB", ["Ascot", "York", "Newmarket"]),
           ("IRE", ["Leopardstown", "Curragh"]),
           ("USA", ["Belmont", "Santa Anita"])]
DISTANCES = [1000, 1100, 1200, 1400, 1600, 2000, 2400]

# Horse pool with latent ability ~ N(0,1)
HORSES = [f"Horse{n:03d}" for n in range(180)]
ABILITY = {h: rng.gauss(0, 1.0) for h in HORSES}


def bsp_from_ability(a: float) -> float:
    """Higher ability -> shorter price. Add noise; clamp to plausible range."""
    implied = 1 / (1 + math.exp(-(a + rng.gauss(0, 0.4))))      # 0..1 strength
    odds = 1.0 / max(0.02, min(0.9, implied * 0.5 + 0.05))
    return round(min(101.0, max(1.2, odds)), 2)


def pick_winner(field: list[str]) -> str:
    weights = [math.exp(ABILITY[h]) for h in field]
    total = sum(weights)
    r = rng.random() * total
    upto = 0.0
    for h, w in zip(field, weights):
        upto += w
        if upto >= r:
            return h
    return field[-1]


def main() -> None:
    start = date(2026, 1, 1)
    n_days = 90
    eid = 1000

    for d in range(n_days):
        day = start + timedelta(days=d)
        region, tracks = REGIONS[d % len(REGIONS)]
        code = {"AUS": "aus", "GB": "uk", "IRE": "uk", "USA": "usa"}[region]
        rows = []
        n_races = rng.randint(2, 4)
        for race_no in range(1, n_races + 1):
            eid += 1
            track = rng.choice(tracks)
            dist = rng.choice(DISTANCES)
            field = rng.sample(HORSES, rng.randint(7, 12))
            winner = pick_winner(field)
            menu = f"{region} / {track} ({region}) {day.strftime('%dth %b')}"
            ename = f"{race_no*100:04d} R{race_no} {dist}m"
            edt = day.strftime("%d-%m-%Y") + f" 0{race_no}:30"
            for sid, h in enumerate(field, 1):
                rows.append({
                    "EVENT_ID": eid, "MENU_HINT": menu, "EVENT_NAME": ename,
                    "EVENT_DT": edt, "SELECTION_ID": eid * 100 + sid,
                    "SELECTION_NAME": h,
                    "WIN_LOSE": 1 if h == winner else 0,
                    "BSP": bsp_from_ability(ABILITY[h]),
                })
        # one file per region-day, matching the real per-day BSP files
        path = FIX / f"bsp_{code}_{day.strftime('%Y%m%d')}.csv"
        with path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    # A sample live racecard (AU/NZ) for offline app/predict testing.
    field = rng.sample(HORSES, 9)
    racecard = {
        "date": (start + timedelta(days=n_days)).strftime("%Y-%m-%d"),
        "track": "Flemington", "race_no": 5, "distance_m": 1200,
        "going": "Good", "region": "AUS", "market_id": "1.999",
        "runners": [{"horse": h, "draw": i + 1, "weight": 57.0,
                     "jockey": None, "trainer": None, "bsp": None,
                     "finish_pos": None, "won": None, "placed": None}
                    for i, h in enumerate(field)],
    }
    (FIX / "racecard_flemington_r5.json").write_text(json.dumps(racecard, indent=2))

    files = list(FIX.glob("bsp_*.csv"))
    print(f"Wrote {len(files)} BSP fixture files + 1 racecard to {FIX}")


if __name__ == "__main__":
    main()
