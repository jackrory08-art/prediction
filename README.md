# 🐎 Aerithius Apollo — Racing Tipster

A horse-racing **win-tipping** model with a phone-friendly web app. Trained on an
international history of results + Betfair Starting Prices, served as calibrated,
field-normalized win probabilities and fair odds.

Everything runs **in the cloud** — GitHub Actions refreshes the data and retrains
the model; Streamlit Community Cloud hosts the app. You only ever open a URL on
your phone.

```
GitHub Actions (cron, free)              Streamlit Community Cloud (free)
  ├─ ingest.py  (Betfair history) ─┐       ├─ app.py reads apollo.db + apollo_brain.pkl
  ├─ train_model.py                │commits├─ fetches live AU/NZ racecards (Betfair)
  └─ commit apollo.db + .pkl ──────┘to repo└─ auto-redeploys on each push
```

## What the model considers

Per horse, computed **point-in-time** (only races *before* the one being predicted,
so there's no leakage):

- **Market price** — the Betfair Starting Price, and the horse's prior avg/best
  price as a class indicator (the single strongest signal in racing).
- **Recent form & trend** — last-3 results vs older, days since last run, starts.
- **Win rate** — career strike rate, Bayesian-smoothed for small samples.
- **Track / distance / going** records.
- **Draw, weight, field size, region.**
- **Data confidence** — so thin records are trusted less.

Win probabilities are **field-normalized** (only one horse wins) and
**isotonic-calibrated** (a "20%" really means 20%). Place/score models train
automatically only if a data source provides full finishing positions (the free
Betfair feed gives win/lose only, so place shows "—").

## Quick start (offline demo — no account needed)

```bash
pip install -r requirements.txt
python tests/make_fixtures.py     # synthetic sample data
python ingest.py --from-fixtures  # build apollo.db
python train_model.py             # build apollo_brain.pkl
python predict.py                 # CLI tips for the sample racecard
streamlit run app.py              # the app, in demo mode
pytest -q                         # smoke tests
```

## Go live, all from your phone (free)

**1 · Get a free Betfair API key.** At **betfair.com.au**, create an account,
then in the Betfair **Accounts API visualiser** run **`createDeveloperAppKeys`**
with any unique app name. Copy the **Delayed** key — it's free and active
instantly. (The *Live* key costs money; you don't need it.)

**2 · Add GitHub Secrets.** Repo → Settings → Secrets and variables → **Actions**
→ add `BETFAIR_APP_KEY`, `BETFAIR_USERNAME`, `BETFAIR_PASSWORD`.

**3 · Run the refresh.** Actions tab → **Refresh data & retrain** → *Run
workflow*. It pulls real history, retrains, and commits `apollo.db` +
`apollo_brain.pkl`. (It also runs daily on a schedule.)

**4 · Deploy the app.** At **share.streamlit.io**, connect this repo, set the main
file to `app.py`, and paste the same three values into the app's **Secrets**.
You get a public HTTPS URL — open it on your phone. Done.

> **Geo note:** the free historical data downloads from anywhere (so training
> always works in CI), but Betfair's *live* exchange login may be limited to AU
> IPs. If the cloud host can't log in, the app automatically falls back to the
> last refreshed data instead of breaking.

## Upgrading the data (optional, paid)

The model talks to a pluggable `DataSource` (`data_sources/`). To get richer
features (jockey/trainer/going/finishing positions → enables the place model),
drop in an adapter for **Punting Form** ($59 AUD/mo) or **The Racing API** and
point `ingest.py` at it — no model code changes.

## Files

| File | Role |
|---|---|
| `db.py` | SQLite store (`races`, `runners`) |
| `data_sources/` | Pluggable feeds; `betfair.py` = free default |
| `ingest.py` | Pull results → SQLite |
| `features.py` | Point-in-time feature engine (shared by train + predict) |
| `train_model.py` | Calibrated HistGradientBoosting + race-grouped CV |
| `predict.py` | Racecard → ranked tips (library + CLI) |
| `app.py` | Streamlit UI |
| `.github/workflows/refresh.yml` | Cloud cron: ingest + train + commit |
| `legacy/` | Retired punters.com.au scraper + old CSVs (reference only) |
