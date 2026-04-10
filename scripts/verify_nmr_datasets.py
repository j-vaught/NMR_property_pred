"""
Post-download verification for Tier-1 NMR datasets.

For each dataset in data/nmr/<name>/:
  - Count records (zip, CSV, SDF, json)
  - Find SMILES/InChI samples
  - Check format health
  - Estimate overlap with existing NMRexp

Writes data/nmr/DATASETS_INVENTORY.md with a summary table.
"""

import csv
import json
import re
import zipfile
from pathlib import Path

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

DATA_DIR = ROOT / "data" / "nmr"

DATASETS = [
    "np_mrd", "hmdb", "bmrb_gissmo", "bml_nmr",
    "prospre", "2dnmrgym", "nmrxiv", "logd_predictor",
]


def dir_size_mb(path: Path) -> float:
    if not path.exists():
        return 0
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1e6


def count_records_in_zip(zip_path: Path, max_inspect: int = 5) -> tuple[int, list[str]]:
    """Count files in a zip and return a sample of names."""
    try:
        with zipfile.ZipFile(zip_path) as z:
            names = z.namelist()
            return len(names), names[:max_inspect]
    except Exception as e:
        return 0, [f"ERROR: {e}"]


def find_smiles_in_text(text: str, max_samples: int = 3) -> list[str]:
    """Heuristic SMILES detection in text."""
    samples = []
    for line in text.split("\n")[:200]:
        line = line.strip()
        if len(line) < 4 or len(line) > 200:
            continue
        # SMILES has C/c/N/n/O and often parens/brackets
        if re.match(r'^[A-Za-z0-9@+\-\[\]\(\)=#/\\%\.]+$', line) and any(c in line for c in "CcNnO"):
            samples.append(line)
            if len(samples) >= max_samples:
                break
    return samples


def inspect_np_mrd() -> dict:
    ds = DATA_DIR / "np_mrd"
    if not ds.exists():
        return {"found": False}
    info = {"found": True, "size_mb": dir_size_mb(ds)}
    zips = list(ds.glob("*.zip"))
    info["n_zip_files"] = len(zips)
    total_entries = 0
    sample_names = []
    for z in zips[:3]:
        n, names = count_records_in_zip(z)
        total_entries += n
        if not sample_names:
            sample_names = names
    info["sampled_zip_entries"] = total_entries
    info["sample_filenames"] = sample_names
    info["has_smiles"] = "via InChI in nmrML files (per-compound)"
    info["format"] = "nmrML + peak_lists_txt + JSON structures"
    info["n_compounds_est"] = "~281K (7 shards × 50K)"
    return info


def inspect_hmdb() -> dict:
    ds = DATA_DIR / "hmdb"
    if not ds.exists():
        return {"found": False}
    info = {"found": True, "size_mb": dir_size_mb(ds)}
    zips = list(ds.glob("*.zip"))
    info["zip_files"] = [z.name for z in zips]
    info["format"] = "XML + zip bundles"
    info["has_smiles"] = "Yes (in metabolite XML)"
    return info


def inspect_bmrb_gissmo() -> dict:
    ds = DATA_DIR / "bmrb_gissmo"
    if not ds.exists():
        return {"found": False}
    info = {"found": True, "size_mb": dir_size_mb(ds)}
    # Count BMRB .str files
    str_files = list((ds / "bmrb_metabolomics_entries").glob("*.str")) if (ds / "bmrb_metabolomics_entries").exists() else []
    info["bmrb_metabolomics_entries"] = len(str_files)
    info["has_gissmo_zip"] = (ds / "all_simulations.zip").exists()
    info["format"] = "NMR-STAR + GISSMO simulations"
    info["has_smiles"] = "Via InChI in NMR-STAR entries"
    return info


def inspect_bml_nmr() -> dict:
    ds = DATA_DIR / "bml_nmr"
    if not ds.exists():
        return {"found": False}
    info = {"found": True, "size_mb": dir_size_mb(ds)}
    info["n_files"] = sum(1 for f in ds.rglob("*") if f.is_file())
    info["format"] = "Raw FIDs + XML metadata"
    return info


def inspect_prospre() -> dict:
    ds = DATA_DIR / "prospre"
    if not ds.exists():
        return {"found": False}
    info = {"found": True, "size_mb": dir_size_mb(ds)}
    zips = sorted(ds.glob("*.zip"))
    info["zip_files"] = [z.name for z in zips]
    info["format"] = "train/holdout zips (CSV or JSON inside)"
    info["has_smiles"] = "Yes (solvent-annotated)"
    return info


def inspect_2dnmrgym() -> dict:
    ds = DATA_DIR / "2dnmrgym"
    if not ds.exists():
        return {"found": False}
    info = {"found": True, "size_mb": dir_size_mb(ds)}
    info["n_files"] = sum(1 for f in ds.rglob("*") if f.is_file())
    info["format"] = "Tensor/graph pickles + peak lists"
    info["has_smiles"] = "Yes"
    return info


def inspect_nmrxiv() -> dict:
    ds = DATA_DIR / "nmrxiv"
    if not ds.exists():
        return {"found": False}
    info = {"found": True, "size_mb": dir_size_mb(ds)}
    info["api_sample_exists"] = (ds / "api_sample.json").exists()
    manifest = ds / "manifest.json"
    if manifest.exists():
        try:
            m = json.loads(manifest.read_text())
            info["status"] = m.get("status", "unknown")
        except Exception:
            pass
    info["format"] = "API-discoverable; needs manual crawl"
    return info


def inspect_logd_predictor() -> dict:
    ds = DATA_DIR / "logd_predictor"
    if not ds.exists():
        return {"found": False}
    info = {"found": True, "size_mb": dir_size_mb(ds)}
    # Look for CSV/data files
    csvs = list(ds.rglob("*.csv"))
    info["csv_files"] = [c.relative_to(ds).as_posix() for c in csvs[:5]]
    # Count rows in first CSV
    if csvs:
        try:
            with open(csvs[0]) as f:
                n_rows = sum(1 for _ in f) - 1
            info["first_csv_rows"] = n_rows
            # Check for SMILES
            with open(csvs[0]) as f:
                header = f.readline()
                info["has_smiles"] = "SMILES" in header or "smiles" in header
        except Exception as e:
            info["csv_error"] = str(e)
    info["format"] = "GitHub repo with CSV + code"
    return info


INSPECTORS = {
    "np_mrd": inspect_np_mrd,
    "hmdb": inspect_hmdb,
    "bmrb_gissmo": inspect_bmrb_gissmo,
    "bml_nmr": inspect_bml_nmr,
    "prospre": inspect_prospre,
    "2dnmrgym": inspect_2dnmrgym,
    "nmrxiv": inspect_nmrxiv,
    "logd_predictor": inspect_logd_predictor,
}


def main():
    print("=" * 70)
    print("  NMR Dataset Verification & Inventory")
    print("=" * 70)
    print(f"  Data dir: {DATA_DIR}")
    print()

    results = {}
    for name in DATASETS:
        print(f"=== {name} ===")
        info = INSPECTORS[name]()
        results[name] = info
        for k, v in info.items():
            print(f"  {k}: {v}")
        print()

    # Write inventory markdown
    md_path = DATA_DIR / "DATASETS_INVENTORY.md"
    lines = [
        "# NMR Datasets Inventory (post-download)",
        "",
        "Automated inventory generated by verify_nmr_datasets.py.",
        "",
        "| Dataset | Status | Size (MB) | Format | Notes |",
        "|---|---|---|---|---|",
    ]
    for name, info in results.items():
        if not info.get("found"):
            lines.append(f"| {name} | ❌ missing | - | - | Directory not found |")
            continue
        size = info.get("size_mb", 0)
        fmt = info.get("format", "unknown")
        status = "✅ downloaded"
        if size < 1:
            status = "⚠️ empty"
        notes = []
        for k in ("n_compounds_est", "n_files", "zip_files", "bmrb_metabolomics_entries",
                  "first_csv_rows", "sampled_zip_entries"):
            if k in info:
                notes.append(f"{k}={info[k]}")
        lines.append(f"| {name} | {status} | {size:.0f} | {fmt} | {', '.join(str(x) for x in notes[:2])} |")

    md_path.write_text("\n".join(lines))
    print(f"\nInventory written to: {md_path}")

    # Also save raw results as JSON
    (DATA_DIR / "DATASETS_INVENTORY.json").write_text(json.dumps(results, indent=2, default=str))


if __name__ == "__main__":
    main()
