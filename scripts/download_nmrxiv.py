"""
Standalone nmrXiv bulk downloader.

Uses the correct API routes (verified via OpenAPI spec + source code):
  - GET /api/v1/list/projects    — paginated, has download_url per project
  - GET /api/v1/list/datasets    — paginated, per-spectrum with molecule SMILES
  - GET /api/v1/list/samples     — paginated, per-molecule metadata

Every project has a pre-generated ZIP bundle on the Uni-Jena S3 bucket
(s3.uni-jena.de/nmrxiv/production/archive/...). No auth required.

Writes to ~/NMR_property_pred/data/nmr/nmrxiv/:
  - projects_page_*.json     — raw API responses
  - datasets_page_*.json     — all spectrum metadata with SMILES
  - samples_page_*.json      — all molecule metadata
  - zips/                    — downloaded raw FID bundles per project
"""

import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

# Path detection
CANDIDATES = [
    Path("/Users/user/Downloads/untitled folder"),
    Path.home() / "NMR_property_pred",
]
for _p in CANDIDATES:
    if (_p / "data").exists():
        ROOT = _p
        break
else:
    ROOT = Path.home() / "NMR_property_pred"

DS_DIR = ROOT / "data" / "nmr" / "nmrxiv"
ZIPS_DIR = DS_DIR / "zips"
DS_DIR.mkdir(parents=True, exist_ok=True)
ZIPS_DIR.mkdir(exist_ok=True)

BASE = "https://nmrxiv.org/api/v1"

SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": "Mozilla/5.0 (research-scraper nmrxiv/1.0)",
    "Accept": "application/json",
})


def log(msg: str):
    ts = time.strftime("%Y-%m-%dT%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    (DS_DIR / "download.log").open("a").write(line + "\n")


def fetch_list(model: str) -> list[dict]:
    """Paginate through /api/v1/list/{model} and save each page."""
    log(f"Fetching list: {model}")
    all_records = []
    page = 1
    while True:
        url = f"{BASE}/list/{model}?page={page}"
        try:
            r = SESSION.get(url, timeout=30)
            if r.status_code != 200:
                log(f"  page {page}: HTTP {r.status_code}, stopping")
                break
            data = r.json()
        except Exception as e:
            log(f"  page {page}: error {e}, stopping")
            break

        page_file = DS_DIR / f"{model}_page_{page:03d}.json"
        page_file.write_text(json.dumps(data, indent=2))

        records = data.get("data", [])
        if not records:
            log(f"  page {page}: empty, stopping")
            break

        all_records.extend(records)

        meta = data.get("meta", {})
        current = meta.get("current_page", page)
        last = meta.get("last_page", page)
        total = meta.get("total", "?")
        log(f"  page {current}/{last}: {len(records)} records (total={total})")

        if current >= last:
            break
        page += 1
        time.sleep(0.3)  # polite delay

    log(f"  Total {model}: {len(all_records)}")
    return all_records


def download_project_zip(project: dict) -> tuple[bool, int, str]:
    """Download one project's ZIP bundle from its download_url."""
    url = project.get("download_url")
    ident = project.get("identifier", "?")
    if not url:
        return False, 0, "no download_url"

    filename = url.rsplit("/", 1)[-1]
    dest = ZIPS_DIR / f"{ident}_{filename}"

    # Resume support
    start_byte = dest.stat().st_size if dest.exists() else 0

    try:
        headers = {}
        if start_byte > 0:
            headers["Range"] = f"bytes={start_byte}-"

        r = requests.get(url, headers=headers, stream=True, timeout=(30, 600))
        if r.status_code == 416:
            return True, start_byte, "already complete"
        if r.status_code not in (200, 206):
            return False, 0, f"HTTP {r.status_code}"

        mode = "ab" if r.status_code == 206 else "wb"
        bytes_written = 0
        with open(dest, mode) as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
                    bytes_written += len(chunk)
        return True, dest.stat().st_size, "ok"
    except Exception as e:
        return False, 0, str(e)


def main():
    log("=" * 60)
    log("nmrXiv bulk download")
    log("=" * 60)

    # 1. Fetch all metadata lists
    projects = fetch_list("projects")
    datasets = fetch_list("datasets")
    samples = fetch_list("samples")

    # Save combined JSONs for easy loading later
    (DS_DIR / "all_projects.json").write_text(json.dumps(projects, indent=2))
    (DS_DIR / "all_datasets.json").write_text(json.dumps(datasets, indent=2))
    (DS_DIR / "all_samples.json").write_text(json.dumps(samples, indent=2))

    log(f"\nSaved all metadata:")
    log(f"  projects: {len(projects)}")
    log(f"  datasets: {len(datasets)}")
    log(f"  samples: {len(samples)}")

    # 2. Download all project ZIPs in parallel (but only 4 at a time)
    to_download = [p for p in projects if p.get("download_url")]
    log(f"\nDownloading {len(to_download)} project ZIPs (4 parallel)...")

    n_ok = 0
    n_fail = 0
    total_bytes = 0
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(download_project_zip, p): p for p in to_download}
        for i, fut in enumerate(as_completed(futures)):
            p = futures[fut]
            ident = p.get("identifier", "?")
            ok, size, msg = fut.result()
            if ok:
                n_ok += 1
                total_bytes += size
            else:
                n_fail += 1
            if (i + 1) % 5 == 0 or n_fail > 0:
                elapsed = time.time() - t0
                log(f"  {i+1}/{len(to_download)}: {ident} {'✓' if ok else '✗'} "
                    f"{size/1e6:.1f}MB — {msg[:60]} "
                    f"| total {total_bytes/1e9:.2f}GB, {elapsed/60:.1f}min")

    log(f"\nDone: {n_ok} ok, {n_fail} failed")
    log(f"Total downloaded: {total_bytes/1e9:.2f} GB")

    # 3. Write manifest
    manifest = {
        "dataset": "nmrxiv",
        "projects_count": len(projects),
        "datasets_count": len(datasets),
        "samples_count": len(samples),
        "zips_downloaded": n_ok,
        "zips_failed": n_fail,
        "total_bytes": total_bytes,
        "status": "complete" if n_ok > 0 else "failed",
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    (DS_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))
    log("Manifest saved")


if __name__ == "__main__":
    main()
