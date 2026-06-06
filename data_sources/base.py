"""
DataSource interface
====================
The model never talks to a vendor directly — it talks to this interface. That
means swapping Betfair (free) for Punting Form / The Racing API (paid, richer)
later is a drop-in: implement these three methods, change one line in config.

Normalized shapes (so every adapter is interchangeable):

  results() -> Iterable[dict] race records, each:
    {
      "date": "YYYY-MM-DD", "track": str, "race_no": int|None,
      "distance_m": int|None, "going": str|None, "region": str,
      "runners": [
        {"horse": str, "finish_pos": int|None, "draw": int|None,
         "weight": float|None, "jockey": str|None, "trainer": str|None,
         "bsp": float|None, "won": int|None, "placed": int|None},
        ...
      ]
    }

  racecards() -> list[dict] upcoming races, same race shape but runners carry
    pre-race fields only (no finish_pos/won).
"""
from __future__ import annotations

from typing import Iterable, Protocol


class DataSource(Protocol):
    name: str

    def results(self, start: str, end: str, regions: list[str]) -> Iterable[dict]:
        """Historical race records for training/ingest."""
        ...

    def racecards(self, day: str = "today") -> list[dict]:
        """Upcoming races with runners, for live prediction."""

    def horse_form(self, horse_id: str) -> list[dict]:
        """A horse's prior runs (usually served from the local DB)."""
