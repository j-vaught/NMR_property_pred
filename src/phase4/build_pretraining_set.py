"""
Build the large NMR → descriptors pretraining set.

Processes all NMRexp 1H NMR compounds:
  1. Parse peak list from NMR_processed string
  2. Convert to continuous spectrum via Lorentzian broadening
  3. Extract spectrum_features (95) + peaklist_features (25)
  4. Compute RDKit descriptors from canonical SMILES (8 targets)

Output: data/nmr/phase4_pretraining_set.parquet

Uses multiprocessing with spawn context for RDKit safety.
Expected compounds: ~1.48M
Expected runtime: 30-60 min on comech-2422
"""

import ast
import multiprocessing as mp
import signal
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shared.config import Paths

paths = Paths()
OUT_PATH = paths.data / "nmr" / "phase4_pretraining_set.parquet"
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)


def _init_worker():
    """Initialize each worker process (called once per worker)."""
    # Suppress RDKit warnings
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")


def _process_row(args):
    """Process a single (canonical_smiles, n_peaks, nmr_processed_str) row.

    Returns dict or None on failure.
    """
    canon_smi, n_peaks_expected, nmr_str = args
    if n_peaks_expected > 100:
        return None  # pathological

    try:
        # Lazy imports (each worker)
        from rdkit import Chem
        from rdkit.Chem import Descriptors, Lipinski
        from phase2.spectrum_converter import parse_1h_peaks, peaks_to_spectrum_1h, _1H_GRID
        from phase2.nmr_features import extract_nmr_features, extract_peaklist_features

        mol = Chem.MolFromSmiles(canon_smi)
        if mol is None:
            return None

        peaks = parse_1h_peaks(nmr_str)
        if not peaks or len(peaks) > 100:
            return None

        # 5-second timeout per compound to avoid pathological ones hanging
        def _handler(signum, frame):
            raise TimeoutError()

        signal.signal(signal.SIGALRM, _handler)
        signal.alarm(5)
        try:
            spectrum = peaks_to_spectrum_1h(peaks, _1H_GRID)
            spec_feats = extract_nmr_features(spectrum)
            peak_feats = extract_peaklist_features(peaks)
        finally:
            signal.alarm(0)

        # Descriptors
        mol_with_h = Chem.AddHs(mol)
        h_count = sum(1 for a in mol_with_h.GetAtoms() if a.GetSymbol() == "H")

        return {
            "canonical_smiles": canon_smi,
            "spectrum_features": spec_feats.astype(np.float32).tolist(),
            "peaklist_features": peak_feats.astype(np.float32).tolist(),
            "mw": float(Descriptors.MolWt(mol)),
            "logp": float(Descriptors.MolLogP(mol)),
            "tpsa": float(Descriptors.TPSA(mol)),
            "n_heavy_atoms": int(mol.GetNumHeavyAtoms()),
            "h_count_molecular": int(h_count),
            "n_rotatable": int(Lipinski.NumRotatableBonds(mol)),
            "n_aromatic_rings": int(Lipinski.NumAromaticRings(mol)),
            "n_peaks": len(peaks),
        }
    except TimeoutError:
        return None
    except Exception:
        return None


def _canonicalize_batch(smiles_list):
    """Canonicalize a batch of SMILES (runs in a worker)."""
    from rdkit import Chem
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")

    out = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            out.append(None)
        else:
            out.append(Chem.MolToSmiles(mol))
    return out


def main():
    print("=" * 70)
    print("  Phase 4: Build NMR → descriptors pretraining set")
    print("=" * 70)

    # Resume support
    if OUT_PATH.exists():
        existing = pd.read_parquet(OUT_PATH, columns=["canonical_smiles"])
        existing_set = set(existing["canonical_smiles"])
        print(f"\nExisting parquet: {len(existing_set)} compounds. Resuming...")
    else:
        existing_set = set()

    # Step 1: Load NMRexp
    print(f"\nLoading NMRexp from {paths.nmrexp}")
    t0 = time.time()
    nmr = pd.read_parquet(
        paths.nmrexp,
        columns=["SMILES", "NMR_type", "NMR_processed"],
    )
    nmr = nmr[nmr["NMR_type"] == "1H NMR"].copy()
    nmr = nmr.dropna(subset=["NMR_processed", "SMILES"])
    print(f"  Loaded {len(nmr)} 1H NMR rows in {time.time()-t0:.1f}s")

    # Step 2: Count peaks per row (for deduplication)
    print("\nCounting peaks per row...")
    def _safe_count(s):
        try:
            return len(ast.literal_eval(s))
        except Exception:
            return 0
    nmr["n_peaks"] = nmr["NMR_processed"].apply(_safe_count)
    nmr = nmr[(nmr["n_peaks"] > 0) & (nmr["n_peaks"] <= 100)]
    print(f"  {len(nmr)} rows with valid peak counts (1-100 peaks)")

    # Step 3: Canonicalize SMILES (serial — LRU cache makes this fine)
    print("\nCanonicalizing SMILES...")
    from shared.data_utils import canonical_smiles
    t0 = time.time()
    unique_raw = nmr["SMILES"].unique()
    print(f"  {len(unique_raw)} unique raw SMILES")
    canon_map = {}
    for i, smi in enumerate(unique_raw):
        c = canonical_smiles(smi)
        if c is not None:
            canon_map[smi] = c
        if (i + 1) % 200000 == 0:
            print(f"    {i+1}/{len(unique_raw)} in {time.time()-t0:.1f}s")
    print(f"  Done: {len(canon_map)} canonical in {time.time()-t0:.1f}s")

    nmr["canonical_smiles"] = nmr["SMILES"].map(canon_map)
    nmr = nmr.dropna(subset=["canonical_smiles"])

    # Step 4: Keep entry with most peaks per canonical compound
    nmr = nmr.sort_values("n_peaks", ascending=False).drop_duplicates(
        subset=["canonical_smiles"], keep="first"
    )
    print(f"  {len(nmr)} unique canonical compounds after dedup")

    # Step 4b: Subsample for tractable runtime
    # 300K compounds is more than enough for MLP pretraining
    MAX_COMPOUNDS = 300_000
    if len(nmr) > MAX_COMPOUNDS:
        nmr = nmr.sample(n=MAX_COMPOUNDS, random_state=42).reset_index(drop=True)
        print(f"  Subsampled to {len(nmr)} compounds")

    # Step 5: Filter out already-processed (resume)
    if existing_set:
        nmr = nmr[~nmr["canonical_smiles"].isin(existing_set)]
        print(f"  {len(nmr)} to process (after filtering existing)")

    if len(nmr) == 0:
        print("Nothing to process. Exiting.")
        return

    # Step 6: Process each compound in parallel (spawn context)
    N_WORKERS = 24
    print(f"\nProcessing {len(nmr)} compounds with {N_WORKERS} workers (spawn)...")
    rows = list(zip(
        nmr["canonical_smiles"].tolist(),
        nmr["n_peaks"].tolist(),
        nmr["NMR_processed"].tolist(),
    ))
    del nmr

    # Free the big dataframe
    import gc
    gc.collect()

    ctx = mp.get_context("spawn")
    results = []
    t0 = time.time()
    batch_size = 10000  # process in batches to save intermediate parquets

    # Load existing results first
    if existing_set:
        existing_df = pd.read_parquet(OUT_PATH)
        results.append(existing_df)
        del existing_df

    with ctx.Pool(processes=N_WORKERS, initializer=_init_worker) as pool:
        batch_rows = []
        for i, result in enumerate(pool.imap_unordered(_process_row, rows, chunksize=1000)):
            if result is not None:
                batch_rows.append(result)

            if (i + 1) % 10000 == 0:
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed
                eta = (len(rows) - i - 1) / rate / 60
                print(f"  {i+1}/{len(rows)} processed, {len(batch_rows)} ok in this batch, "
                      f"{rate:.0f}/sec, ETA {eta:.1f} min", flush=True)

            # Save intermediate every 50k
            if len(batch_rows) >= 50000:
                batch_df = pd.DataFrame(batch_rows)
                results.append(batch_df)
                combined = pd.concat(results, ignore_index=True)
                combined.to_parquet(OUT_PATH, index=False)
                print(f"  → checkpoint saved: {len(combined)} compounds", flush=True)
                batch_rows = []

        # Final batch
        if batch_rows:
            batch_df = pd.DataFrame(batch_rows)
            results.append(batch_df)

    # Combine and save
    if results:
        combined = pd.concat(results, ignore_index=True)
        # Dedup in case resume picked up duplicates
        combined = combined.drop_duplicates(subset=["canonical_smiles"], keep="first")
        combined.to_parquet(OUT_PATH, index=False)

        elapsed = time.time() - t0
        print(f"\n{'='*70}")
        print(f"Done: {len(combined)} compounds in {elapsed/60:.1f} min")
        print(f"Saved to: {OUT_PATH}")
        print(f"File size: {OUT_PATH.stat().st_size / 1e6:.1f} MB")
        print(f"\nDescriptor stats:")
        for col in ["mw", "logp", "tpsa", "n_heavy_atoms", "h_count_molecular",
                    "n_rotatable", "n_aromatic_rings", "n_peaks"]:
            print(f"  {col:<22s} mean={combined[col].mean():8.2f}  "
                  f"std={combined[col].std():7.2f}  "
                  f"[{combined[col].min():.1f}, {combined[col].max():.1f}]")


if __name__ == "__main__":
    main()
