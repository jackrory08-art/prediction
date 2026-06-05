"""
Aerithius Data Engine v3.1
--------------------------
Keeps v2.0's proven search-and-navigate flow (which works on punters.com.au)
and layers v3's improvements on top:

  • Captures TRACK CONDITION per race (new — was missing entirely)
  • Emits a `Condition` column in master_horse_form.csv
  • Backward-compatible: old CSVs without Condition are auto-upgraded
"""

import asyncio
import os
import re
import pandas as pd
from playwright.async_api import async_playwright


# ═══════════════════════════════════════════════════════════════
# UPDATE THIS FOR EACH RACE
# ═══════════════════════════════════════════════════════════════
HORSE_NAMES = [
    "Worth",
    "Venator",
    "Star Wave",
    "Counter Seven",
    "Gatto Nero",
    "Love Diva",
    "Kyoei Bonita",
    "Meisho Yozora",
    "Smooth Velvet",
    "Viva Crown",
    "Jacquard",
    "Aoi Regina"
]

# ─── Track-condition regex ───
# Matches "Good4", "Soft 6", "Heavy10", "Synthetic", etc. We keep only the word.
COND_RE = re.compile(
    r"\b(Firm|Good|Soft|Heavy|Synthetic|Slow|Dead|Fast|Cushion|Poly)\s*\d{0,2}\b",
    re.IGNORECASE,
)


def _extract_condition(text: str) -> str:
    """Pull 'Good' / 'Soft' / 'Heavy' / etc from a text blob."""
    m = COND_RE.search(text or "")
    return m.group(1).capitalize() if m else ""


# ═══════════════════════════════════════════════════════════════
# STATS TAB — unchanged from v2.0 (proven working)
# ═══════════════════════════════════════════════════════════════
async def scrape_stats(page, horse_name, base_url):
    stats_url = f"{base_url.rstrip('/')}/Stats/"
    try:
        await page.goto(stats_url, wait_until="domcontentloaded", timeout=60000)
        await asyncio.sleep(2)
        data = await page.evaluate("""() => {
            const results = [];
            document.querySelectorAll('table tr').forEach(row => {
                const cells = Array.from(row.querySelectorAll('td')).map(c => c.innerText.trim());
                if (cells.length >= 8) results.push(cells);
            });
            return results;
        }""")
        return [[horse_name] + r for r in data]
    except Exception as e:
        print(f"  Stats error: {e}")
        return []


# ═══════════════════════════════════════════════════════════════
# FORM TAB — v2.0's structure + condition capture
# ═══════════════════════════════════════════════════════════════
async def scrape_form_enhanced(page, horse_name, base_url):
    try:
        await page.goto(base_url, wait_until="domcontentloaded", timeout=60000)
        await asyncio.sleep(3)

        enhanced_data = await page.evaluate("""() => {
            const results = [];
            const rows = document.querySelectorAll(
                '.form-guide-table tr, .horse-form-table tr, table.table tr, .results-table tr'
            );
            rows.forEach(row => {
                const cells = Array.from(row.querySelectorAll('td'));
                if (cells.length < 3) return;
                results.push({
                    text: row.innerText.trim(),
                    cells: cells.map(c => c.innerText.trim()),
                });
            });
            return { rows: results, fullText: document.body.innerText };
        }""")

        entries = []

        # Method 1 — structured rows
        for row_data in enhanced_data.get("rows", []):
            cells = row_data.get("cells", [])
            if len(cells) >= 5:
                entry = parse_form_row(horse_name, cells, row_data.get("text", ""))
                if entry:
                    entries.append(entry)

        # Method 2 — regex fallback
        if not entries:
            entries = parse_form_regex(horse_name, enhanced_data.get("fullText", ""))

        return entries

    except Exception as e:
        print(f"  Form error: {e}")
        return []


def parse_form_row(horse_name, cells, full_text):
    """Parse a single structured table row. Returns a 10-field list or None."""
    try:
        # Finish position
        pos_cell = None
        for cell in cells:
            if re.match(r"^\d+/\d+$", cell.strip()):
                pos_cell = cell.strip()
                break
            m = re.search(r"(\d+)\s*(?:of|/)\s*(\d+)", cell)
            if m:
                pos_cell = f"{m.group(1)}/{m.group(2)}"
                break
        if not pos_cell:
            return None

        # Date
        date = None
        for cell in cells:
            m = re.search(r"(\d{2}-[A-Za-z]{3}-\d{2})", cell)
            if m:
                date = m.group(1)
                break

        # Distance
        dist = None
        for cell in cells:
            m = re.search(r"(\d{3,4})m", cell)
            if m:
                dist = m.group(0)
                break

        # Track
        track = None
        for cell in cells:
            if (
                cell
                and not re.match(r"^[\d.$%\-/]+$", cell.strip())
                and 2 < len(cell) < 30
                and not re.search(r"\d{2}-[A-Za-z]{3}", cell)
            ):
                track = cell.strip()
                break

        # Barrier / Weight / Margin / SP
        barrier = ""
        m = re.search(r"(?:Bar|Barrier|Gate)\s*(\d{1,2})", full_text, re.IGNORECASE)
        if m: barrier = m.group(1)

        weight = ""
        m = re.search(r"(\d{2}(?:\.\d)?)\s*kg", full_text, re.IGNORECASE)
        if m: weight = m.group(1)

        margin = ""
        m = re.search(r"(\d+\.?\d*)\s*(?:len|lengths?|L)", full_text, re.IGNORECASE)
        if m: margin = m.group(1)

        sp = ""
        m = re.search(r"\$(\d+\.?\d*)", full_text)
        if m: sp = m.group(1)

        condition = _extract_condition(full_text)   # NEW

        if date and dist:
            return [
                horse_name, date, track or "Unknown", dist, pos_cell,
                barrier, weight, margin, sp, condition,
            ]
        return None
    except Exception:
        return None


def parse_form_regex(horse_name, page_text):
    """Fallback regex over the full page text. Returns 10-field lists."""
    entries = []
    pattern = r"(\d+)\s+of\s+(\d+)\s*(.*?)\s+(\d{2}-[a-zA-Z]{3}-\d{2})\s+(\d+m)"
    matches = re.findall(pattern, page_text)

    for m in matches:
        finish_pos = f"{m[0]}/{m[1]}"
        track = m[2].strip()
        date = m[3]
        distance = m[4]

        idx = page_text.find(m[3])
        # Narrow context for per-race fields. The old [-100, +300] window
        # could spill into the next race's block and mislabel condition.
        # Keep a small forward window — enough for barrier/weight/margin/SP
        # which sit right next to the result — and no pre-match history.
        ctx = page_text[idx: idx + 150] if idx > 0 else ""
        # Even narrower window for condition since it tends to sit tight to
        # the race header and is the most leak-prone field.
        cond_ctx = page_text[idx: idx + 80] if idx > 0 else ""

        barrier = weight = margin = sp = condition = ""
        if ctx:
            bm = re.search(r"(?:Bar|B)\.?\s*(\d{1,2})", ctx)
            if bm: barrier = bm.group(1)
            wm = re.search(r"(\d{2}\.?\d?)\s*kg", ctx, re.IGNORECASE)
            if wm: weight = wm.group(1)
            mg = re.search(r"(\d+\.?\d*)\s*(?:len|L)", ctx, re.IGNORECASE)
            if mg: margin = mg.group(1)
            sm = re.search(r"\$(\d+\.?\d*)", ctx)
            if sm: sp = sm.group(1)
            condition = _extract_condition(cond_ctx)

        entries.append([
            horse_name, date, track, distance, finish_pos,
            barrier, weight, margin, sp, condition,
        ])
    return entries


# ═══════════════════════════════════════════════════════════════
# NAVIGATION — v2.0's search flow (restored verbatim)
# ═══════════════════════════════════════════════════════════════
async def get_horse_data(browser_context, horse_name):
    """Navigate to horse page and scrape both stats and form."""
    page = await browser_context.new_page()
    try:
        if "_" in horse_name:
            base_url = f"https://www.punters.com.au/horses/{horse_name}/"
        else:
            search_url = f"https://www.punters.com.au/search/?q={horse_name.replace(' ', '+')}"
            await page.goto(search_url, wait_until="domcontentloaded")
            horse_link = page.locator(".search-results a[href*='/horses/']").first
            if await horse_link.count() > 0:
                base_url = await horse_link.get_attribute("href")
                if not base_url.startswith("http"):
                    base_url = f"https://www.punters.com.au{base_url}"
            else:
                print(f"  Could not find link for {horse_name}")
                await page.close()
                return [], []

        print(f"  URL: {base_url}")
        f_data = await scrape_form_enhanced(page, horse_name, base_url)
        s_data = await scrape_stats(page, horse_name, base_url)
        await page.close()
        return s_data, f_data
    except Exception as e:
        print(f"  Navigation error: {e}")
        await page.close()
        return [], []


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════
async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        context = await browser.new_context()
        all_stats, all_form = [], []

        for i, name in enumerate(HORSE_NAMES, 1):
            print(f"\n[{i}/{len(HORSE_NAMES)}] Scraping {name}...")
            s_data, f_data = await get_horse_data(context, name)
            all_stats.extend(s_data)
            all_form.extend(f_data)
            print(f"  Got {len(s_data)} stat rows, {len(f_data)} form entries")

        # ── Stats CSV (schema unchanged) ──
        if all_stats:
            file = "master_horse_stats.csv"
            cols = ["Horse", "Category", "Starts", "1st", "2nd", "3rd",
                    "Win%", "Place%", "Odds", "ROI"]
            new_df = pd.DataFrame(all_stats, columns=cols)
            if os.path.isfile(file):
                old_df = pd.read_csv(file)
                combined = pd.concat([old_df, new_df], ignore_index=True)
            else:
                combined = new_df
            combined.drop_duplicates(subset=["Horse", "Category"], keep="last", inplace=True)
            combined.to_csv(file, index=False)
            print(f"\nStats updated: {len(combined)} total rows in {file}")

        # ── Form CSV (new Condition column) ──
        if all_form:
            file = "master_horse_form.csv"
            cols = ["Horse", "Date", "Track", "Distance", "Finish_Pos",
                    "Barrier", "Weight", "Margin", "SP", "Condition"]
            new_df = pd.DataFrame(all_form, columns=cols)

            if os.path.isfile(file):
                old_df = pd.read_csv(file)
                for col in cols:
                    if col not in old_df.columns:
                        old_df[col] = ""     # upgrade old CSVs that lack Condition
                old_df = old_df[cols]        # reorder/trim to match new schema
                combined = pd.concat([old_df, new_df], ignore_index=True)
            else:
                combined = new_df

            combined.drop_duplicates(subset=["Horse", "Date", "Distance"], keep="last", inplace=True)
            combined.to_csv(file, index=False)
            print(f"Form updated: {len(combined)} total entries in {file}")
        else:
            print("\nNo form data collected.")

        await browser.close()

        print(f"\n{'='*50}")
        print("SCRAPE COMPLETE")
        print(f"{'='*50}")
        print(f"Horses processed: {len(HORSE_NAMES)}")
        print(f"Stats rows:       {len(all_stats)}")
        print(f"Form entries:     {len(all_form)}")
        print("\nNext steps:")
        print("  1. python train_model.py   (retrain Apollo)")
        print("  2. python predict.py       (run predictions)")


if __name__ == "__main__":
    asyncio.run(main())
