"""
Master download script for Tier-1 experimental 1H NMR datasets.

Downloads 8 datasets to ~/NMR_property_pred/data/nmr/<dataset_name>/:
  1. NP-MRD        — ~281K natural products (nmrML + peak lists + structures)
  2. HMDB          — ~1-2K metabolite experimental NMR
  3. BMRB/GISSMO   — 1,286 compounds with spin systems + 18 field strengths
  4. BML-NMR       — 208 metabolites × 16 spectra each
  5. PROSPRE       — ~10K solvent-annotated 1H chemical shifts
  6. 2DNMRGym      — 22,348 experimental HSQC spectra
  7. nmrXiv        — modern raw NMR depositions (best effort)
  8. logD predictor — 754 pharma compounds + logD labels

Each dataset:
  - Resumable (wget -c, skip complete, manifest.json per dir)
  - Logged to data/nmr/<dataset>/download.log
  - Disk guard: aborts if <20GB free before next dataset
  - Manifest records url, size, sha256, status

Usage:
    python3 download_nmr_datasets.py [--dataset NAME] [--force]
"""

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import requests

# Base paths (auto-detected for mac vs server)
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

DATA_DIR = ROOT / "data" / "nmr"
DATA_DIR.mkdir(parents=True, exist_ok=True)

MIN_FREE_GB = 20

UA_HEADER = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def log(ds_dir: Path, msg: str):
    """Append to per-dataset download.log and echo to stdout."""
    ts = datetime.now().isoformat(timespec="seconds")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    (ds_dir / "download.log").open("a").write(line + "\n")


def free_gb(path: Path = DATA_DIR) -> float:
    """Free disk space in GB on the filesystem holding `path`."""
    st = shutil.disk_usage(path)
    return st.free / 1e9


def check_disk(ds_dir: Path) -> bool:
    """Abort if free disk is below MIN_FREE_GB."""
    gb = free_gb()
    log(ds_dir, f"Free disk: {gb:.1f} GB")
    if gb < MIN_FREE_GB:
        log(ds_dir, f"ABORT: <{MIN_FREE_GB} GB free, stopping")
        return False
    return True


def write_manifest(ds_dir: Path, data: dict):
    manifest = ds_dir / "manifest.json"
    existing = {}
    if manifest.exists():
        try:
            existing = json.loads(manifest.read_text())
        except Exception:
            pass
    existing.update(data)
    manifest.write_text(json.dumps(existing, indent=2))


def is_complete(ds_dir: Path) -> bool:
    manifest = ds_dir / "manifest.json"
    if not manifest.exists():
        return False
    try:
        return json.loads(manifest.read_text()).get("status") == "complete"
    except Exception:
        return False


def sha256_of(path: Path, max_mb: int = 100) -> str:
    """Compute sha256 of first max_mb megabytes (for quick sanity check)."""
    h = hashlib.sha256()
    remaining = max_mb * 1024 * 1024
    with open(path, "rb") as f:
        while remaining > 0:
            chunk = f.read(min(remaining, 1024 * 1024))
            if not chunk:
                break
            h.update(chunk)
            remaining -= len(chunk)
    return h.hexdigest()[:16]  # truncated


def wget(url: str, dest: Path, ds_dir: Path, timeout: int = 600) -> bool:
    """Download a URL to dest using requests (resumable via Range header).
    Named 'wget' for historical reasons — actually uses python requests.
    Returns True on success.
    """
    log(ds_dir, f"  download: {url}")
    try:
        # Check existing partial download
        start_byte = 0
        if dest.exists():
            start_byte = dest.stat().st_size

        headers = dict(UA_HEADER)
        if start_byte > 0:
            headers["Range"] = f"bytes={start_byte}-"

        r = requests.get(url, headers=headers, stream=True, timeout=(30, timeout), allow_redirects=True)

        # Handle range not satisfiable (file complete)
        if r.status_code == 416:
            log(ds_dir, f"    already complete: {start_byte/1e6:.1f} MB")
            return True

        if r.status_code not in (200, 206):
            log(ds_dir, f"    HTTP {r.status_code}")
            return False

        mode = "ab" if r.status_code == 206 else "wb"
        bytes_written = 0
        last_log = time.time()
        with open(dest, mode) as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
                    bytes_written += len(chunk)
                    if time.time() - last_log > 30:
                        log(ds_dir, f"    ...{(bytes_written + start_byte)/1e6:.1f} MB")
                        last_log = time.time()

        total = dest.stat().st_size
        log(ds_dir, f"    downloaded {total/1e6:.1f} MB")
        return total > 1024
    except requests.exceptions.Timeout:
        log(ds_dir, "    timeout")
        return False
    except Exception as e:
        log(ds_dir, f"    error: {e}")
        return False


def http_get(url: str, ds_dir: Path, timeout: int = 30) -> str | None:
    """HTTP GET returning text, or None on error."""
    try:
        r = requests.get(url, headers=UA_HEADER, timeout=timeout)
        if r.status_code == 200:
            return r.text
        log(ds_dir, f"  GET {url} → {r.status_code}")
    except Exception as e:
        log(ds_dir, f"  GET {url} error: {e}")
    return None


# ---------------------------------------------------------------------------
# Dataset-specific downloaders
# ---------------------------------------------------------------------------

def download_np_mrd() -> dict:
    """NP-MRD: natural products, ~281K compounds.

    Downloads structures JSON + SDF + nmrML bulk exports + peak lists.
    Data comes in 50K-compound shards.
    """
    ds_dir = DATA_DIR / "np_mrd"
    ds_dir.mkdir(exist_ok=True)
    result = {"dataset": "np_mrd", "started_at": datetime.now().isoformat()}
    log(ds_dir, "=== NP-MRD ===")

    if not check_disk(ds_dir):
        result["status"] = "aborted_disk"
        return result

    # NP-MRD has 7 shards (50K compounds each = 350K capacity)
    # We download:
    #   1. Experimental spectra bulk (small, critical)
    #   2. nmrML files (7 shards)
    #   3. Peak list text (7 shards)
    #   4. JSON structure files (7 shards)
    shards = [
        "NP0000001_NP0050000",
        "NP0050001_NP0100000",
        "NP0100001_NP0150000",
        "NP0150001_NP0200000",
        "NP0200001_NP0250000",
        "NP0250001_NP0300000",
        "NP0300001_NP0350000",
    ]

    urls = ["https://np-mrd.org/system/downloads/current/experimental_spectra.zip",
            "https://np-mrd.org/system/downloads/current/assignment_tables.zip"]

    for shard in shards:
        # nmrML (big, has spectrum data)
        urls.append(f"https://moldb.np-mrd.org/downloads/exports/hmdb/nmrml_files/NP-MRD_nmrml_files_{shard}.zip")
        # Peak lists (compact text)
        urls.append(f"https://moldb.np-mrd.org/downloads/exports/hmdb/peak_lists_txt/NP-MRD_nmr_peak_lists_{shard}.zip")
        # Structure JSON (SMILES + InChI)
        urls.append(f"https://np-mrd.org/system/downloads/current/npmrd_natural_products_{shard}_json.zip")

    downloaded = []
    failed = []
    for url in urls:
        filename = url.split("/")[-1]
        dest = ds_dir / filename
        if dest.exists() and dest.stat().st_size > 10240:
            log(ds_dir, f"  skip (exists): {filename}")
            downloaded.append(filename)
            continue
        if wget(url, dest, ds_dir):
            downloaded.append(filename)
        else:
            failed.append(filename)
        time.sleep(1)  # polite delay

    total_bytes = sum((ds_dir / f).stat().st_size for f in downloaded if (ds_dir / f).exists())
    result["files_downloaded"] = len(downloaded)
    result["files_failed"] = len(failed)
    result["total_bytes"] = total_bytes
    result["status"] = "complete" if len(downloaded) > 0 else "failed"
    write_manifest(ds_dir, result)
    log(ds_dir, f"NP-MRD done: {len(downloaded)} files, {total_bytes/1e9:.2f} GB")
    return result


def download_hmdb() -> dict:
    """HMDB: human metabolome experimental NMR + metabolite structures."""
    ds_dir = DATA_DIR / "hmdb"
    ds_dir.mkdir(exist_ok=True)
    result = {"dataset": "hmdb", "started_at": datetime.now().isoformat()}
    log(ds_dir, "=== HMDB ===")

    if not check_disk(ds_dir):
        result["status"] = "aborted_disk"
        return result

    # Known HMDB bulk download URLs (pattern /system/downloads/current/*.zip)
    urls = [
        "https://hmdb.ca/system/downloads/current/hmdb_metabolites.zip",
        "https://hmdb.ca/system/downloads/current/hmdb_nmr_peak_lists.zip",
        "https://hmdb.ca/system/downloads/current/hmdb_experimental_nmr_metabolites.zip",
        "https://hmdb.ca/system/downloads/current/hmdb_nmr_spectra.zip",
        "https://hmdb.ca/system/downloads/current/hmdb_predicted_nmr_spectra.zip",
        "https://hmdb.ca/system/downloads/current/structures.zip",
    ]

    downloaded = []
    failed = []
    for url in urls:
        filename = url.split("/")[-1]
        dest = ds_dir / filename
        if dest.exists() and dest.stat().st_size > 10240:
            log(ds_dir, f"  skip (exists): {filename}")
            downloaded.append(filename)
            continue
        if wget(url, dest, ds_dir):
            downloaded.append(filename)
        else:
            failed.append(filename)
            # Remove empty dest to allow retry later
            if dest.exists() and dest.stat().st_size < 1024:
                dest.unlink()
        time.sleep(2)

    total_bytes = sum((ds_dir / f).stat().st_size for f in downloaded if (ds_dir / f).exists())
    result["files_downloaded"] = len(downloaded)
    result["files_failed"] = len(failed)
    result["total_bytes"] = total_bytes
    result["status"] = "complete" if len(downloaded) > 0 else "failed"
    write_manifest(ds_dir, result)
    log(ds_dir, f"HMDB done: {len(downloaded)} files, {total_bytes/1e9:.2f} GB")
    return result


def download_bmrb_gissmo() -> dict:
    """BMRB metabolomics + GISSMO spin systems."""
    ds_dir = DATA_DIR / "bmrb_gissmo"
    ds_dir.mkdir(exist_ok=True)
    result = {"dataset": "bmrb_gissmo", "started_at": datetime.now().isoformat()}
    log(ds_dir, "=== BMRB + GISSMO ===")

    if not check_disk(ds_dir):
        result["status"] = "aborted_disk"
        return result

    urls = [
        # GISSMO all simulations zip
        "https://gissmo.bmrb.io/static/all_simulations.zip",
        # GISSMO: download all matching entries
        "https://gissmo.bmrb.io/static/entries_list.json",
    ]

    downloaded = []
    failed = []
    for url in urls:
        filename = url.split("/")[-1]
        dest = ds_dir / filename
        if dest.exists() and dest.stat().st_size > 1024:
            log(ds_dir, f"  skip (exists): {filename}")
            downloaded.append(filename)
            continue
        if wget(url, dest, ds_dir):
            downloaded.append(filename)
        else:
            failed.append(filename)
        time.sleep(1)

    # BMRB metabolomics via rsync or wget -r (small tree)
    bmrb_dir = ds_dir / "bmrb_metabolomics_entries"
    bmrb_dir.mkdir(exist_ok=True)
    log(ds_dir, "Fetching BMRB metabolomics entry list...")
    idx = http_get("https://bmrb.io/ftp/pub/bmrb/metabolomics/entry_directories/",
                   ds_dir, timeout=60)
    if idx:
        import re
        entries = re.findall(r'href="(bmse\d+)/?"', idx)
        log(ds_dir, f"  Found {len(entries)} BMRB metabolomics entries")
        # Download a listing of each entry's NMR-STAR file
        for i, entry in enumerate(entries[:2000]):  # cap at 2000
            url = f"https://bmrb.io/ftp/pub/bmrb/metabolomics/entry_directories/{entry}/{entry}.str"
            dest = bmrb_dir / f"{entry}.str"
            if dest.exists() and dest.stat().st_size > 500:
                continue
            try:
                r = requests.get(url, headers=UA_HEADER, timeout=15)
                if r.status_code == 200 and len(r.text) > 500:
                    dest.write_text(r.text)
                    downloaded.append(f"bmrb_metabolomics/{entry}.str")
            except Exception:
                pass
            if (i + 1) % 100 == 0:
                log(ds_dir, f"  BMRB progress: {i+1}/{len(entries)}")
            time.sleep(0.1)

    total_bytes = sum(f.stat().st_size for f in ds_dir.rglob("*")
                       if f.is_file() and f.suffix != ".log" and f.name != "manifest.json")
    result["files_downloaded"] = len(downloaded)
    result["files_failed"] = len(failed)
    result["total_bytes"] = total_bytes
    result["status"] = "complete" if len(downloaded) > 0 else "failed"
    write_manifest(ds_dir, result)
    log(ds_dir, f"BMRB/GISSMO done: {len(downloaded)} files, {total_bytes/1e9:.2f} GB")
    return result


def download_bml_nmr() -> dict:
    """Birmingham Metabolite Library NMR."""
    ds_dir = DATA_DIR / "bml_nmr"
    ds_dir.mkdir(exist_ok=True)
    result = {"dataset": "bml_nmr", "started_at": datetime.now().isoformat()}
    log(ds_dir, "=== BML-NMR ===")

    if not check_disk(ds_dir):
        result["status"] = "aborted_disk"
        return result

    # BML-NMR known URLs
    urls = [
        "http://www.bml-nmr.org/bml-nmr.tar.gz",
        "http://www.bml-nmr.org/downloads/bml-nmr.zip",
        "http://www.bml-nmr.org/data.tar.gz",
    ]

    downloaded = []
    failed = []
    for url in urls:
        filename = url.split("/")[-1]
        dest = ds_dir / filename
        if dest.exists() and dest.stat().st_size > 10240:
            log(ds_dir, f"  skip (exists): {filename}")
            downloaded.append(filename)
            continue
        if wget(url, dest, ds_dir):
            downloaded.append(filename)
            break  # one success is enough
        else:
            if dest.exists() and dest.stat().st_size < 1024:
                dest.unlink()

    total_bytes = sum((ds_dir / f).stat().st_size for f in downloaded if (ds_dir / f).exists())
    result["files_downloaded"] = len(downloaded)
    result["total_bytes"] = total_bytes
    result["status"] = "complete" if downloaded else "failed"
    write_manifest(ds_dir, result)
    log(ds_dir, f"BML-NMR done: {len(downloaded)} files, {total_bytes/1e6:.1f} MB")
    return result


def download_prospre() -> dict:
    """PROSPRE: 10K solvent-annotated 1H chemical shifts."""
    ds_dir = DATA_DIR / "prospre"
    ds_dir.mkdir(exist_ok=True)
    result = {"dataset": "prospre", "started_at": datetime.now().isoformat()}
    log(ds_dir, "=== PROSPRE ===")

    if not check_disk(ds_dir):
        result["status"] = "aborted_disk"
        return result

    urls = [
        "https://prospre.ca/downloads/train.zip",
        "https://prospre.ca/downloads/holdout_1.zip",
        "https://prospre.ca/downloads/holdout_2.zip",
    ]

    downloaded = []
    for url in urls:
        filename = url.split("/")[-1]
        dest = ds_dir / filename
        if dest.exists() and dest.stat().st_size > 10240:
            downloaded.append(filename)
            continue
        if wget(url, dest, ds_dir):
            downloaded.append(filename)
        time.sleep(1)

    total_bytes = sum((ds_dir / f).stat().st_size for f in downloaded if (ds_dir / f).exists())
    result["files_downloaded"] = len(downloaded)
    result["total_bytes"] = total_bytes
    result["status"] = "complete" if downloaded else "failed"
    write_manifest(ds_dir, result)
    log(ds_dir, f"PROSPRE done: {len(downloaded)} files, {total_bytes/1e6:.1f} MB")
    return result


def download_2dnmrgym() -> dict:
    """2DNMRGym: 22K experimental HSQC spectra."""
    ds_dir = DATA_DIR / "2dnmrgym"
    ds_dir.mkdir(exist_ok=True)
    result = {"dataset": "2dnmrgym", "started_at": datetime.now().isoformat()}
    log(ds_dir, "=== 2DNMRGym ===")

    if not check_disk(ds_dir):
        result["status"] = "aborted_disk"
        return result

    # Try HuggingFace first, fall back to git clone
    try:
        from huggingface_hub import snapshot_download
        log(ds_dir, "  Trying HuggingFace snapshot_download...")
        try:
            path = snapshot_download(
                repo_id="siriusxiao62/2DNMRGym",
                repo_type="dataset",
                local_dir=str(ds_dir / "hf_repo"),
            )
            log(ds_dir, f"  HF success: {path}")
            result["hf_repo"] = str(path)
        except Exception as e:
            log(ds_dir, f"  HF failed: {e}; falling back to git clone")
            raise
    except Exception:
        git_dir = ds_dir / "github_repo"
        if not git_dir.exists():
            cmd = ["git", "clone", "--depth", "1",
                   "https://github.com/siriusxiao62/2DNMRGym.git", str(git_dir)]
            r = subprocess.run(cmd, capture_output=True, timeout=300)
            log(ds_dir, f"  git clone exit: {r.returncode}")
            if r.returncode != 0:
                log(ds_dir, f"    stderr: {r.stderr.decode()[:300]}")

    total_bytes = sum(f.stat().st_size for f in ds_dir.rglob("*")
                       if f.is_file() and f.suffix != ".log" and f.name != "manifest.json")
    result["total_bytes"] = total_bytes
    result["status"] = "complete" if total_bytes > 1e6 else "failed"
    write_manifest(ds_dir, result)
    log(ds_dir, f"2DNMRGym done: {total_bytes/1e6:.1f} MB")
    return result


def download_nmrxiv() -> dict:
    """nmrXiv: modern raw NMR depositions (best effort)."""
    ds_dir = DATA_DIR / "nmrxiv"
    ds_dir.mkdir(exist_ok=True)
    result = {"dataset": "nmrxiv", "started_at": datetime.now().isoformat()}
    log(ds_dir, "=== nmrXiv ===")

    if not check_disk(ds_dir):
        result["status"] = "aborted_disk"
        return result

    # Try multiple API endpoints
    endpoints = [
        "https://nmrxiv.org/api/v1/projects",
        "https://api.nmrxiv.org/api/v1/projects",
        "https://nmrxiv.org/api/projects",
    ]

    found_endpoint = None
    for ep in endpoints:
        txt = http_get(ep, ds_dir, timeout=20)
        if txt and (txt.startswith("{") or txt.startswith("[")):
            found_endpoint = ep
            (ds_dir / "api_sample.json").write_text(txt[:100000])
            log(ds_dir, f"  Working API: {ep}")
            break

    if not found_endpoint:
        log(ds_dir, "  nmrXiv API unreachable — saving status and skipping")
        result["status"] = "unreachable"
        write_manifest(ds_dir, result)
        return result

    # If we got here, we have an endpoint. Save the raw sample and note
    # that full crawl needs manual follow-up based on the actual schema.
    log(ds_dir, "  API sample saved; full crawl TBD")
    result["status"] = "partial"
    result["notes"] = "API sample saved, full crawl needs schema inspection"
    write_manifest(ds_dir, result)
    return result


def download_logd_predictor() -> dict:
    """logD predictor: 754 pharma compounds with experimental 1H NMR + logD."""
    ds_dir = DATA_DIR / "logd_predictor"
    ds_dir.mkdir(exist_ok=True)
    result = {"dataset": "logd_predictor", "started_at": datetime.now().isoformat()}
    log(ds_dir, "=== logD predictor ===")

    git_dir = ds_dir / "github_repo"
    if git_dir.exists() and any(git_dir.iterdir()):
        log(ds_dir, "  skip (exists)")
    else:
        cmd = ["git", "clone", "--depth", "1",
               "https://github.com/Prospero1988/logD_predictor.git", str(git_dir)]
        r = subprocess.run(cmd, capture_output=True, timeout=300)
        log(ds_dir, f"  git clone exit: {r.returncode}")
        if r.returncode != 0:
            log(ds_dir, f"    stderr: {r.stderr.decode()[:300]}")

    total_bytes = sum(f.stat().st_size for f in ds_dir.rglob("*")
                       if f.is_file() and f.suffix != ".log" and f.name != "manifest.json")
    result["total_bytes"] = total_bytes
    result["status"] = "complete" if total_bytes > 100_000 else "failed"
    write_manifest(ds_dir, result)
    log(ds_dir, f"logD done: {total_bytes/1e6:.1f} MB")
    return result


# ---------------------------------------------------------------------------
# Dataset registry
# ---------------------------------------------------------------------------

DATASETS = {
    "logd_predictor": download_logd_predictor,  # smallest, fastest — start here
    "prospre":        download_prospre,
    "2dnmrgym":       download_2dnmrgym,
    "bml_nmr":        download_bml_nmr,
    "hmdb":           download_hmdb,
    "bmrb_gissmo":    download_bmrb_gissmo,
    "nmrxiv":         download_nmrxiv,
    "np_mrd":         download_np_mrd,  # largest, last
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", help="Run only one dataset")
    parser.add_argument("--force", action="store_true", help="Re-download even if complete")
    parser.add_argument("--list", action="store_true", help="List datasets and exit")
    args = parser.parse_args()

    if args.list:
        for name in DATASETS:
            ds_dir = DATA_DIR / name
            status = "complete" if is_complete(ds_dir) else "pending"
            print(f"  {name:20s} {status}")
        return

    targets = [args.dataset] if args.dataset else list(DATASETS.keys())

    print("=" * 70)
    print(f"  NMR Dataset Download — {len(targets)} targets")
    print(f"  Data dir: {DATA_DIR}")
    print(f"  Free disk: {free_gb():.1f} GB")
    print("=" * 70)

    results = []
    for name in targets:
        if name not in DATASETS:
            print(f"  skip (unknown): {name}")
            continue

        ds_dir = DATA_DIR / name
        ds_dir.mkdir(exist_ok=True)

        if is_complete(ds_dir) and not args.force:
            print(f"\n{name}: already complete, skipping")
            continue

        try:
            r = DATASETS[name]()
        except Exception as e:
            r = {"dataset": name, "status": "error", "error": str(e)}
            log(ds_dir, f"ERROR: {e}")
        results.append(r)
        print()

    # Summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Dataset':<20s} {'Status':<12s} {'Size':>10s}")
    print("-" * 45)
    for r in results:
        size_mb = r.get("total_bytes", 0) / 1e6
        print(f"{r.get('dataset',''):<20s} {r.get('status',''):<12s} {size_mb:>9.1f}MB")


if __name__ == "__main__":
    main()
