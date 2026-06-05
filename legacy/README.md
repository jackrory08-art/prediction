# Legacy (archived — not used by the app)

Retired when Apollo moved from scraping to a structured data source.

- `scraper_engine.py` — Playwright scraper of punters.com.au. Abandoned because
  the site is bot-protected (HTTP 403) and its per-race fields (barrier, weight,
  SP, condition) came back mostly empty via regex extraction.
- `master_horse_form.csv`, `master_horse_stats.csv` — the old AU/Japan datasets
  produced by that scraper. Kept for reference only.

The current pipeline ingests from Betfair into `apollo.db` instead. See the
top-level `README.md`.
