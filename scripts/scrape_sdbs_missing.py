"""
Identify SDBS numbers we missed in the first pass and rescrape them.

Reads existing CSV, finds gaps, then scrapes only the missing numbers
with lower concurrency and retries to avoid rate limits.
"""

import csv
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from scrape_sdbs_fast import fetch_compound, OUTPUT_DIR


def find_missing(max_sdbs: int = 40000) -> list[int]:
    """Find SDBS numbers we don't have yet."""
    have = set()
    for csv_file in OUTPUT_DIR.glob("sdbs_fast_*.csv"):
        with open(csv_file) as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    have.add(int(row["sdbs_no"]))
                except (KeyError, ValueError):
                    pass
    print(f"Already have: {len(have)} compounds")
    missing = [n for n in range(1, max_sdbs + 1) if n not in have]
    print(f"Missing: {len(missing)} numbers to retry")
    return missing


def rescrape_missing(workers: int = 8, max_sdbs: int = 40000):
    """Rescrape missing SDBS numbers with low concurrency + retries."""
    missing = find_missing(max_sdbs)
    if not missing:
        print("Nothing missing!")
        return

    out_path = OUTPUT_DIR / "sdbs_fast_retry.csv"

    fieldnames = [
        "sdbs_no", "name", "title", "formula", "mw", "inchi", "inchi_key",
        "smiles_pubchem", "inchi_source",
        "cas", "date", "spectrum_codes",
        "n_1h_nmr", "n_13c_nmr", "n_nmr_other", "n_ir", "n_ms", "n_raman", "n_esr",
    ]

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
    todo = [n for n in missing if n not in existing]
    print(f"To process: {len(todo)} (already retried: {len(existing)})")

    write_header = not out_path.exists() or len(existing) == 0
    f = open(out_path, "a", newline="")
    writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
    if write_header:
        writer.writeheader()

    n_found = 0
    n_tried = 0
    n_with_h_nmr = 0
    n_failed = 0
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(fetch_compound, n, max_retries=5): n for n in todo}

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
                n_failed += 1

            if n_tried % 200 == 0:
                f.flush()
                elapsed = time.time() - t0
                rate = n_tried / elapsed
                eta = (len(todo) - n_tried) / rate / 60 if rate > 0 else 0
                print(f"  {n_tried}/{len(todo)}: {n_found} found, "
                      f"{n_with_h_nmr} with 1H NMR, {n_failed} failed, "
                      f"{rate:.1f}/sec, ETA {eta:.1f} min",
                      flush=True)

    f.close()
    elapsed = time.time() - t0
    print(f"\nDone: {n_found}/{len(todo)} new compounds in {elapsed/60:.1f} min")
    print(f"  With 1H NMR: {n_with_h_nmr}")
    print(f"  Still failed: {n_failed}")


if __name__ == "__main__":
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    max_sdbs = int(sys.argv[2]) if len(sys.argv) > 2 else 40000
    rescrape_missing(workers, max_sdbs)
