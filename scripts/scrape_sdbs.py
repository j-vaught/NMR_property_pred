"""
Scrape SDBS (Spectral Database for Organic Compounds) using Playwright.

After accepting disclaimer, lands on SearchInformation.aspx with a search form.
Searches by SDBS number, extracts compound info from results page.

Usage: python scrape_sdbs.py [start] [end]
"""

import asyncio
import csv
import json
import re
import sys
import time
from pathlib import Path

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "data" / "nmr" / "sdbs"


async def scrape_sdbs(start: int = 1, end: int = 35000):
    from playwright.async_api import async_playwright

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"sdbs_compounds_{start}_{end}.csv"

    existing = set()
    if out_path.exists():
        with open(out_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                existing.add(int(row["sdbs_no"]))
        print(f"Resuming: {len(existing)} already scraped")

    fieldnames = ["sdbs_no", "name", "cas", "formula", "mw",
                  "has_h_nmr", "has_c_nmr", "has_ir", "has_ms"]
    write_header = not out_path.exists() or len(existing) == 0

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()

        # Accept disclaimer
        print("Accepting SDBS disclaimer...")
        await page.goto("https://sdbs.db.aist.go.jp/sdbs/cgi-bin/landingpage?sdbsno=1",
                        timeout=30000)
        try:
            await page.click("input[value='I agree the disclaimer and use SDBS.']",
                             timeout=10000)
            await page.wait_for_load_state("networkidle", timeout=10000)
            print("Disclaimer accepted, on search page")
        except Exception:
            print("Could not find disclaimer button")

        f = open(out_path, "a", newline="")
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if write_header:
            writer.writeheader()

        n_found = 0
        n_tried = 0
        t0 = time.time()

        for sdbsno in range(start, end + 1):
            if sdbsno in existing:
                continue

            try:
                # Fill SDBS number and search
                await page.fill("#BodyContentPlaceHolder_INP_sdbsno", str(sdbsno))
                await page.click("input[value='Search']", timeout=5000)
                await page.wait_for_load_state("domcontentloaded", timeout=10000)
                await asyncio.sleep(0.5)

                text = await page.inner_text("body")

                # Check if we got a result
                if "No data" in text or "no data" in text.lower() or len(text.strip()) < 100:
                    # Go back to search
                    await page.goto("https://sdbs.db.aist.go.jp/SearchInformation.aspx",
                                    timeout=10000)
                    await asyncio.sleep(0.3)
                    n_tried += 1
                    continue

                result = {"sdbs_no": sdbsno}

                # Extract compound name
                name_match = re.search(r'(?:Compound Name|Name)[:\s]+([^\n]+)', text)
                if name_match:
                    result["name"] = name_match.group(1).strip()

                # CAS number
                cas_match = re.search(r'\b(\d{2,7}-\d{2}-\d)\b', text)
                if cas_match:
                    result["cas"] = cas_match.group(1)

                # Molecular formula
                formula_match = re.search(r'(C\d+H\d+[A-Za-z0-9]*)', text)
                if formula_match:
                    result["formula"] = formula_match.group(1)

                # Molecular weight
                mw_match = re.search(r'(?:MW|Mol\.?\s*Wt\.?|Molecular Weight)[:\s]*(\d+\.?\d*)', text, re.I)
                if mw_match:
                    result["mw"] = mw_match.group(1)

                # Check which spectra are available
                result["has_h_nmr"] = "1H NMR" in text or "HNMR" in text
                result["has_c_nmr"] = "13C NMR" in text or "CNMR" in text
                result["has_ir"] = "IR" in text and "infrared" not in text.lower()[:50]
                result["has_ms"] = "MS" in text or "mass" in text.lower()

                if any(k in result for k in ["name", "cas", "formula"]):
                    writer.writerow(result)
                    n_found += 1

                # Go back to search page for next query
                await page.goto("https://sdbs.db.aist.go.jp/SearchInformation.aspx",
                                timeout=10000)
                await asyncio.sleep(0.3)

            except Exception as e:
                # Try to recover to search page
                try:
                    await page.goto("https://sdbs.db.aist.go.jp/SearchInformation.aspx",
                                    timeout=10000)
                except Exception:
                    pass

            n_tried += 1

            if n_tried % 50 == 0:
                f.flush()
                elapsed = time.time() - t0
                rate = n_tried / elapsed if elapsed > 0 else 0
                eta = (end - sdbsno) / rate / 60 if rate > 0 else 0
                print(f"  #{sdbsno}: {n_found}/{n_tried} found "
                      f"({rate:.1f}/sec, ETA {eta:.0f} min)", flush=True)

            # Refresh browser every 500 compounds to avoid memory/session issues
            if n_tried % 500 == 0:
                await page.close()
                await browser.close()
                browser = await p.chromium.launch(headless=True)
                page = await browser.new_page()
                await page.goto("https://sdbs.db.aist.go.jp/sdbs/cgi-bin/landingpage?sdbsno=1",
                                timeout=30000)
                try:
                    await page.click("input[value='I agree the disclaimer and use SDBS.']",
                                     timeout=10000)
                    await page.wait_for_load_state("networkidle", timeout=10000)
                except Exception:
                    pass

        f.close()
        await browser.close()

    elapsed = time.time() - t0
    print(f"\nDone: {n_found}/{n_tried} found in {elapsed/60:.1f} min")


if __name__ == "__main__":
    start = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    end = int(sys.argv[2]) if len(sys.argv) > 2 else 35000
    asyncio.run(scrape_sdbs(start, end))
