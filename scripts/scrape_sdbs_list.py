"""
Scrape SDBS compound numbers from a list file.

Usage: python scrape_sdbs_list.py list.txt output.csv [workers]
"""

import csv
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from scrape_sdbs_fast import fetch_compound


def scrape_list(list_path: str, out_path: str, workers: int = 6):
    with open(list_path) as f:
        all_nums = [int(line.strip()) for line in f if line.strip()]

    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    # Resume
    existing = set()
    if out_file.exists():
        with open(out_file) as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    existing.add(int(row["sdbs_no"]))
                except (KeyError, ValueError):
                    pass
    todo = [n for n in all_nums if n not in existing]
    print(f"Total: {len(all_nums)}, already done: {len(existing)}, todo: {len(todo)}")

    fieldnames = [
        "sdbs_no", "name", "title", "formula", "mw", "inchi", "inchi_key",
        "smiles_pubchem", "inchi_source",
        "cas", "date", "spectrum_codes",
        "n_1h_nmr", "n_13c_nmr", "n_nmr_other", "n_ir", "n_ms", "n_raman", "n_esr",
    ]

    write_header = not out_file.exists() or len(existing) == 0
    f = open(out_file, "a", newline="")
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
                      f"{n_with_h_nmr} 1H NMR, {n_failed} failed, "
                      f"{rate:.1f}/sec, ETA {eta:.1f} min", flush=True)

    f.close()
    elapsed = time.time() - t0
    print(f"\nDone: {n_found}/{len(todo)} in {elapsed/60:.1f} min")
    print(f"  With 1H NMR: {n_with_h_nmr}")
    print(f"  Failed: {n_failed}")


if __name__ == "__main__":
    list_path = sys.argv[1]
    out_path = sys.argv[2]
    workers = int(sys.argv[3]) if len(sys.argv) > 3 else 6
    scrape_list(list_path, out_path, workers)
