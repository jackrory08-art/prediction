"""
Betfair data source (free)
==========================
Two free Betfair feeds, one adapter:

  • TRAINING  — historical "Betfair Starting Price" (BSP) CSV files, published
    daily per region at promo.betfair.com/betfairsp/prices/. These give, per
    horse: the BSP (decimal odds) and WIN_LOSE. That's enough for a *win*
    tipping model (market price is the strongest signal; win/lose is the label).
    They do NOT include full finishing order, so place/score models are trained
    only if a richer source later provides finish_pos.

  • LIVE      — the Exchange API (betfairlightweight) for today's AU/NZ win
    markets + runners.

`fixtures=<dir>` makes both paths read committed sample CSVs instead of the
network, so the whole pipeline runs with no key and no egress.
"""
from __future__ import annotations

import os
import re
import glob
from datetime import datetime, timedelta
from pathlib import Path

import requests

# Region -> Betfair SP file code. The SP files bundle GB+IRE under "uk".
_REGION_FILECODE = {
    "GB": "uk", "IRE": "uk", "UK": "uk",
    "AUS": "aus", "NZ": "aus",
    "USA": "usa",
}
_BSP_BASE = "https://promo.betfair.com/betfairsp/prices"

# MENU_HINT looks like "AUS / Flemington (AUS) 9th Mar" ; EVENT_NAME like
# "R7 1600m Hcap" or "1845 R7 1600m". We pull region/track/race_no/distance.
_RACE_NO_RE = re.compile(r"\bR(\d{1,2})\b", re.IGNORECASE)
_DIST_RE = re.compile(r"(\d{3,4})\s*m\b", re.IGNORECASE)


class BetfairSource:
    name = "betfair"

    def __init__(self, fixtures: str | None = None):
        # Explicit dir, or env toggle, or None (live).
        self.fixtures = fixtures or os.environ.get("APOLLO_FIXTURES") or None
        self._session = requests.Session()
        self._session.headers["User-Agent"] = "apollo-tipster/1.0"

    # ─────────────────────────── TRAINING ───────────────────────────
    def results(self, start: str, end: str, regions: list[str] | None = None):
        """Yield normalized race dicts between `start` and `end` (ISO dates)."""
        regions = regions or ["AUS", "GB", "IRE", "USA"]
        if self.fixtures:
            yield from self._results_from_fixtures()
            return

        codes = sorted({_REGION_FILECODE.get(r.upper(), "") for r in regions} - {""})
        d0 = datetime.strptime(start, "%Y-%m-%d").date()
        d1 = datetime.strptime(end, "%Y-%m-%d").date()
        day = d0
        while day <= d1:
            for code in codes:
                csv_text = self._download_bsp(code, day)
                if csv_text:
                    yield from self._parse_bsp_csv(csv_text)
            day += timedelta(days=1)

    def _download_bsp(self, code: str, day) -> str | None:
        fname = f"dwbfprices{code}win{day.strftime('%d%m%Y')}.csv"
        url = f"{_BSP_BASE}/{fname}"
        try:
            r = self._session.get(url, timeout=30)
            if r.status_code == 200 and r.text.strip():
                return r.text
        except requests.RequestException:
            pass
        return None  # missing day/region is normal (no racing) — skip quietly

    def _results_from_fixtures(self):
        for path in sorted(glob.glob(str(Path(self.fixtures) / "bsp_*.csv"))):
            yield from self._parse_bsp_csv(Path(path).read_text())

    @staticmethod
    def _parse_bsp_csv(csv_text: str):
        """Group BSP rows by event into normalized race dicts."""
        import csv
        import io

        events: dict[str, dict] = {}
        for row in csv.DictReader(io.StringIO(csv_text)):
            eid = (row.get("EVENT_ID") or "").strip()
            if not eid:
                continue
            ev = events.setdefault(eid, {"menu": row.get("MENU_HINT", ""),
                                         "name": row.get("EVENT_NAME", ""),
                                         "dt": row.get("EVENT_DT", ""),
                                         "runners": []})
            bsp = _to_float(row.get("BSP"))
            won = _to_int(row.get("WIN_LOSE"))
            ev["runners"].append({
                "horse": (row.get("SELECTION_NAME") or "").strip(),
                "bsp": bsp,
                "won": won,
                "finish_pos": 1 if won == 1 else None,
                "placed": None,        # not available in win-only BSP files
                "draw": None, "weight": None, "jockey": None, "trainer": None,
            })

        for eid, ev in events.items():
            region, track = _parse_menu_hint(ev["menu"])
            race_no, distance = _parse_event_name(ev["name"])
            runners = [r for r in ev["runners"] if r["horse"]]
            if not runners:
                continue
            yield {
                "date": _parse_event_dt(ev["dt"]),
                "track": track,
                "race_no": race_no,
                "distance_m": distance,
                "going": None,
                "region": region,
                "runners": runners,
            }

    # ─────────────────────────── LIVE ───────────────────────────
    def racecards(self, day: str = "today") -> list[dict]:
        if self.fixtures:
            return self._racecards_from_fixtures()
        return self._racecards_from_exchange(day)

    def _racecards_from_fixtures(self) -> list[dict]:
        import json

        cards = []
        for path in sorted(glob.glob(str(Path(self.fixtures) / "racecard_*.json"))):
            cards.append(json.loads(Path(path).read_text()))
        return cards

    def _racecards_from_exchange(self, day: str) -> list[dict]:
        """Live AU/NZ win markets via the Exchange API. Raises on geo/login
        failure so the caller can fall back to the DB."""
        try:
            import betfairlightweight
            from betfairlightweight import filters
        except ImportError as e:
            raise RuntimeError("betfairlightweight not installed — `pip install betfairlightweight`") from e

        app_key = os.environ.get("BETFAIR_APP_KEY")
        username = os.environ.get("BETFAIR_USERNAME")
        password = os.environ.get("BETFAIR_PASSWORD")
        if not (app_key and username and password):
            raise RuntimeError("Missing BETFAIR_APP_KEY / BETFAIR_USERNAME / BETFAIR_PASSWORD")

        trading = betfairlightweight.APIClient(
            username, password, app_key=app_key, locale="australia"
        )
        trading.login_interactive()  # no SSL cert needed
        try:
            start = datetime.utcnow()
            end = start + timedelta(days=1 if day == "tomorrow" else 0, hours=24)
            mf = filters.market_filter(
                event_type_ids=["7"],  # 7 = Horse Racing
                market_countries=["AU", "NZ"],
                market_type_codes=["WIN"],
                market_start_time={"from": start.isoformat(), "to": end.isoformat()},
            )
            catalogues = trading.betting.list_market_catalogue(
                filter=mf,
                market_projection=["EVENT", "MARKET_START_TIME", "RUNNER_DESCRIPTION"],
                max_results=200,
            )
            return [_catalogue_to_racecard(c) for c in catalogues]
        finally:
            trading.logout()

    # ─────────────────────────── DB-backed form ───────────────────────────
    def horse_form(self, horse_id: str) -> list[dict]:
        import db

        return db.load_form_df(horse_id=horse_id).to_dict("records")


# ─────────────────────────── parsing helpers ───────────────────────────

def _to_float(v) -> float | None:
    try:
        s = str(v).strip()
        return float(s) if s and s.upper() != "NULL" else None
    except (TypeError, ValueError):
        return None


def _to_int(v) -> int | None:
    f = _to_float(v)
    return int(round(f)) if f is not None else None


def _parse_menu_hint(menu: str) -> tuple[str, str]:
    """`'AUS / Flemington (AUS) 9th Mar'` -> `('AUS', 'Flemington')`."""
    menu = (menu or "").strip()
    region = "UNK"
    track = menu
    if "/" in menu:
        left, right = menu.split("/", 1)
        region = left.strip().upper()[:3] or "UNK"
        track = right.strip()
    track = re.sub(r"\(.*?\)", "", track)                       # drop "(AUS)"
    track = re.sub(r"\d{1,2}(st|nd|rd|th)?\s*\w*$", "", track)  # drop trailing date
    return region, track.strip() or "Unknown"


def _parse_event_name(name: str) -> tuple[int | None, int | None]:
    race_no = None
    m = _RACE_NO_RE.search(name or "")
    if m:
        race_no = int(m.group(1))
    distance = None
    m = _DIST_RE.search(name or "")
    if m:
        distance = int(m.group(1))
    return race_no, distance


def _parse_event_dt(dt: str) -> str:
    """Betfair `EVENT_DT` like `'09-03-2026 05:45'` -> ISO `'2026-03-09'`."""
    dt = (dt or "").strip()
    for fmt in ("%d-%m-%Y %H:%M", "%d-%m-%Y %H:%M:%S", "%Y-%m-%d %H:%M:%S", "%d/%m/%Y %H:%M"):
        try:
            return datetime.strptime(dt, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    # last resort: leading date token
    return (dt.split(" ")[0] if dt else "")


def _catalogue_to_racecard(cat) -> dict:
    """betfairlightweight MarketCatalogue -> normalized racecard dict."""
    event = getattr(cat, "event", None)
    name = getattr(cat, "market_name", "") or ""
    race_no, distance = _parse_event_name(name)
    venue = getattr(event, "venue", None) or getattr(event, "name", "") or "Unknown"
    runners = []
    for r in (getattr(cat, "runners", None) or []):
        hn = getattr(r, "runner_name", None)
        if hn:
            runners.append({
                "horse": hn, "draw": getattr(r, "sort_priority", None),
                "weight": None, "jockey": None, "trainer": None,
                "bsp": None, "finish_pos": None, "won": None, "placed": None,
            })
    start = getattr(cat, "market_start_time", None)
    return {
        "date": start.strftime("%Y-%m-%d") if start else "",
        "track": venue, "race_no": race_no, "distance_m": distance,
        "going": None, "region": "AUS",
        "market_id": getattr(cat, "market_id", None),
        "market_start_time": start.isoformat() if start else None,
        "runners": runners,
    }
