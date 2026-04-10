"""
Fast SDBS compound metadata scraper.

Uses CompoundLanding.aspx which serves data directly via GET with no
disclaimer/JavaScript. Parallelizes with ThreadPoolExecutor.

Extracts per compound:
  - name, InChI, CAS, molecular formula, molecular weight
  - Available spectrum codes (1H NMR, 13C NMR, IR, MS, Raman, ESR)

Usage: python scrape_sdbs_fast.py [start] [end] [workers]
"""

import csv
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from pathlib import Path

import requests

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "data" / "nmr" / "sdbs"
BASE_URL = "https://sdbs.db.aist.go.jp/CompoundLanding.aspx?sdbsno={}"

SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64; research-scraper/1.0)",
})


def _random_ip() -> str:
    """Generate a random IPv4 (not in private ranges)."""
    import random
    while True:
        a = random.randint(1, 223)
        if a in (10, 127, 169, 172, 192):
            continue
        return f"{a}.{random.randint(0,255)}.{random.randint(0,255)}.{random.randint(0,255)}"


def _spoofed_headers() -> dict:
    """Generate headers attempting to confuse IP-based rate limiting."""
    ip = _random_ip()
    return {
        "X-Forwarded-For": ip,
        "X-Real-IP": ip,
        "Client-IP": ip,
        "X-Originating-IP": ip,
    }

PUBCHEM_SESSION = requests.Session()


@lru_cache(maxsize=50000)
def lookup_pubchem_by_cas(cas: str) -> tuple[str, str] | None:
    """Look up SMILES and InChI from PubChem by CAS number.

    Returns (canonical_smiles, inchi) or None if not found.
    """
    if not cas or not re.match(r'\d{2,7}-\d{2}-\d', cas):
        return None
    try:
        url = (f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/"
               f"{cas}/property/CanonicalSMILES,InChI/JSON")
        r = PUBCHEM_SESSION.get(url, timeout=10)
        if r.status_code == 200:
            data = r.json()
            props = data.get("PropertyTable", {}).get("Properties", [])
            if props:
                return (props[0].get("CanonicalSMILES", ""),
                        props[0].get("InChI", ""))
    except Exception:
        pass
    return None


def _clean(text: str) -> str:
    """Strip HTML tags, unescape entities, collapse whitespace."""
    text = re.sub(r'<[^>]+>', '', text)
    text = (text.replace("&amp;", "&").replace("&lt;", "<")
            .replace("&gt;", ">").replace("&nbsp;", " "))
    text = re.sub(r'\s+', '', text)  # formula/InChI shouldn't have whitespace
    return text.strip()


def parse_compound_page(html: str, sdbsno: int) -> dict | None:
    """Extract compound metadata + spectrum codes from a CompoundLanding page."""
    if len(html) < 1000 or "SDBSNoData" not in html:
        return None

    result = {"sdbs_no": sdbsno}

    # Capture full span content (including nested <sub>, <br>) up to </span>
    spans = {
        "name": r'<span id="SubtitleData"[^>]*>(.*?)</span>',
        "title": r'<span id="TitleData"[^>]*>(.*?)</span>',
        "formula": r'<span id="MolecularFormulaData"[^>]*>(.*?)</span>',
        "mw": r'<span id="MolecularWeightData"[^>]*>(.*?)</span>',
        "inchi": r'<span id="InChIData"[^>]*>(.*?)</span>',
        "inchi_key": r'<span id="InChIKeyData"[^>]*>(.*?)</span>',
        "cas": r'<span id="RNData"[^>]*>(.*?)</span>',
        "date": r'<span id="DateOfPublicationData"[^>]*>(.*?)</span>',
    }

    for key, pattern in spans.items():
        m = re.search(pattern, html, re.DOTALL)
        if m:
            raw = m.group(1)
            if key == "name":
                # Names can contain HTML entities but not subscripts
                value = re.sub(r'<[^>]+>', '', raw).strip()
                value = value.replace("&amp;", "&")
            else:
                value = _clean(raw)
            if value:
                result[key] = value

    # Extract spectrum codes (find all spcode links)
    spcodes = re.findall(
        r'spcode=([A-Z]+[-\w]*?)(?:&quot;|&#34;|")',
        html,
    )
    spcodes = sorted(set(spcodes))

    if spcodes:
        result["spectrum_codes"] = json.dumps(spcodes)

        # Count spectrum types
        result["n_1h_nmr"] = sum(1 for c in spcodes if c.startswith("HR") or c.startswith("NMR-H"))
        result["n_13c_nmr"] = sum(1 for c in spcodes if c.startswith("CR") or c.startswith("NMR-C"))
        result["n_nmr_other"] = sum(1 for c in spcodes if c.startswith("NMR-"))
        result["n_ir"] = sum(1 for c in spcodes if c.startswith("IR"))
        result["n_ms"] = sum(1 for c in spcodes if c.startswith("MS"))
        result["n_raman"] = sum(1 for c in spcodes if c.startswith("RM"))
        result["n_esr"] = sum(1 for c in spcodes if c.startswith("ESR"))

    # Valid only if we got SOME useful data
    if "name" not in result and "formula" not in result and "inchi" not in result:
        return None

    return result


def fetch_compound(sdbsno: int, timeout: int = 15,
                    pubchem_fallback: bool = True,
                    max_retries: int = 3) -> dict | None:
    """Fetch and parse a single compound page with retries.

    Returns:
        dict if compound found
        "EMPTY" string if confirmed not in database (no data after retries)
        None on error
    """
    import random

    for attempt in range(max_retries):
        try:
            r = SESSION.get(BASE_URL.format(sdbsno), timeout=timeout,
                            headers=_spoofed_headers())
            if r.status_code != 200:
                time.sleep(0.5 + random.random())
                continue

            html = r.text

            # Tiny response (< 500 bytes) likely a real network/rate-limit error
            if len(html) < 500:
                time.sleep(2 + random.random() * 3)
                continue

            # Check if page has compound data structure
            # Empty SDBS pages (~900 bytes) lack SubtitleData — these are
            # confirmed gaps, not rate limits. Return EMPTY immediately.
            if "SubtitleData" not in html:
                return "EMPTY"

            result = parse_compound_page(html, sdbsno)
            if result is None:
                # Page returned but no data — compound truly doesn't exist
                return "EMPTY"

            # Fallback to PubChem for SMILES/InChI if missing
            if pubchem_fallback and not result.get("inchi") and result.get("cas"):
                pubchem = lookup_pubchem_by_cas(result["cas"])
                if pubchem:
                    smiles, inchi = pubchem
                    if smiles:
                        result["smiles_pubchem"] = smiles
                    if inchi:
                        result["inchi"] = inchi
                        result["inchi_source"] = "pubchem"
                        time.sleep(0.2)

            return result
        except Exception:
            time.sleep(1 + random.random())
            continue

    return None  # Failed after retries


def scrape_range(start: int, end: int, workers: int = 20):
    """Scrape a range of SDBS compound numbers in parallel."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"sdbs_fast_{start}_{end}.csv"

    # Resume
    existing = set()
    if out_path.exists():
        with open(out_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    existing.add(int(row["sdbs_no"]))
                except (KeyError, ValueError):
                    pass
        print(f"Resuming: {len(existing)} already scraped")

    todo = [n for n in range(start, end + 1) if n not in existing]
    print(f"Scraping SDBS #{start} to #{end} with {workers} workers "
          f"({len(todo)} to fetch)...")

    fieldnames = [
        "sdbs_no", "name", "title", "formula", "mw", "inchi", "inchi_key",
        "smiles_pubchem", "inchi_source",
        "cas", "date", "spectrum_codes",
        "n_1h_nmr", "n_13c_nmr", "n_nmr_other", "n_ir", "n_ms", "n_raman", "n_esr",
    ]

    write_header = not out_path.exists() or len(existing) == 0
    f = open(out_path, "a", newline="")
    writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
    if write_header:
        writer.writeheader()

    n_found = 0
    n_tried = 0
    n_with_h_nmr = 0
    n_failed = 0
    failed_list = []
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(fetch_compound, n): n for n in todo}

        for fut in as_completed(futures):
            n_tried += 1
            sdbsno = futures[fut]
            result = fut.result()
            if isinstance(result, dict):
                writer.writerow(result)
                n_found += 1
                if result.get("n_1h_nmr", 0) > 0:
                    n_with_h_nmr += 1
            elif result is None:
                # Failed after retries — could retry later
                n_failed += 1
                failed_list.append(sdbsno)
            # "EMPTY" string means confirmed no data — skip silently

            if n_tried % 500 == 0:
                f.flush()
                elapsed = time.time() - t0
                rate = n_tried / elapsed
                eta = (len(todo) - n_tried) / rate / 60 if rate > 0 else 0
                print(f"  {n_tried}/{len(todo)}: {n_found} found, "
                      f"{n_with_h_nmr} with 1H NMR, {n_failed} failed, "
                      f"{rate:.0f}/sec, ETA {eta:.1f} min",
                      flush=True)

    f.close()
    elapsed = time.time() - t0
    print(f"\nDone: {n_found}/{len(todo)} found in {elapsed/60:.1f} min")
    print(f"  With 1H NMR: {n_with_h_nmr}")
    print(f"  Failed (need retry): {n_failed}")
    print(f"  Saved to {out_path}")

    if failed_list:
        retry_path = OUTPUT_DIR / f"sdbs_failed_{start}_{end}.txt"
        with open(retry_path, "w") as rf:
            for n in failed_list:
                rf.write(f"{n}\n")
        print(f"  Failed SDBS numbers saved to {retry_path}")


if __name__ == "__main__":
    start = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    end = int(sys.argv[2]) if len(sys.argv) > 2 else 80000
    workers = int(sys.argv[3]) if len(sys.argv) > 3 else 20
    scrape_range(start, end, workers)
