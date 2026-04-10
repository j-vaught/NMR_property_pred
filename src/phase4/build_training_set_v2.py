"""
Build expanded Phase 4 training set: real NMR overlaps + synthetic NMR.

v1 training set: 953 compounds (only NMRexp overlaps with 3 viscosity sources)
v2 training set: ALL viscosity compounds, using:
  1. Real NMR features where available (from pretraining set)
  2. Synthetic NMR features for compounds without experimental NMR

Viscosity sources:
  - ThermoML pure viscosity (CAS → SMILES via chemicals)
  - J Cheminf 2024 (direct SMILES)
  - chemicals GC expanded (direct SMILES)
  - gc_viscosity (direct SMILES, largest source)

Expected: ~2,000+ compounds with real NMR, ~5,000+ with synthetic NMR
→ ~6,000+ total (vs 953 in v1)
"""

import json
import sys
import signal
import time
from pathlib import Path
from multiprocessing import Pool, get_context

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shared.config import Paths
from rdkit import Chem
from rdkit.Chem import Descriptors

paths = Paths()
OUTPUT = paths.data / "nmr" / "phase4_training_set_v2.parquet"
PRETRAINING = paths.data / "nmr" / "phase4_pretraining_set.parquet"

DESCRIPTOR_COLS = [
    "mw", "logp", "tpsa", "n_heavy_atoms", "h_count_molecular",
    "n_rotatable", "n_aromatic_rings", "n_peaks",
]


def canon(s):
    try:
        m = Chem.MolFromSmiles(str(s))
        return Chem.MolToSmiles(m) if m else None
    except:
        return None


def compute_descriptors(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        mol_h = Chem.AddHs(mol)
        return {
            "mw": Descriptors.MolWt(mol),
            "logp": Descriptors.MolLogP(mol),
            "tpsa": Descriptors.TPSA(mol),
            "n_heavy_atoms": mol.GetNumHeavyAtoms(),
            "h_count_molecular": sum(1 for a in mol_h.GetAtoms() if a.GetAtomicNum() == 1),
            "n_rotatable": Descriptors.NumRotatableBonds(mol),
            "n_aromatic_rings": Descriptors.NumAromaticRings(mol),
            "is_hydrocarbon": all(a.GetAtomicNum() in (1, 6) for a in mol_h.GetAtoms()),
        }
    except:
        return None


def _process_synthetic(args):
    """Worker function for parallel synthetic NMR generation."""
    smiles, seed = args
    try:
        # Import inside worker to avoid fork issues
        from phase4.synthetic_nmr import synthetic_nmr_features
        result = synthetic_nmr_features(smiles, seed=seed)
        if result is not None:
            return smiles, result[0].tolist(), result[1].tolist()
    except:
        pass
    return None


def load_all_viscosity():
    """Load and merge all viscosity data sources."""
    rows = []

    # 1. GC viscosity (largest, 4968 compounds)
    gc = pd.read_csv(paths.data / "thermo" / "group_contribution" / "gc_viscosity.csv")
    gc["canonical_smiles"] = gc["SMILES"].apply(canon)
    gc = gc.dropna(subset=["canonical_smiles"])
    gc = gc[(gc["viscosity_Pas"] > 1e-7) & (gc["viscosity_Pas"] < 100)]
    for _, r in gc.iterrows():
        rows.append({
            "canonical_smiles": r["canonical_smiles"],
            "T_K": r["T_K"],
            "viscosity_Pa_s": r["viscosity_Pas"],
            "source_visc": "gc_viscosity",
        })
    print(f"  gc_viscosity: {len(gc)} rows, {gc.canonical_smiles.nunique()} compounds")

    # 2. Chemicals GC expanded
    cgc = pd.read_csv(paths.data / "thermo" / "group_contribution" / "chemicals_viscosity_expanded.csv")
    cgc["canonical_smiles"] = cgc["SMILES"].apply(canon)
    cgc = cgc.dropna(subset=["canonical_smiles"])
    cgc = cgc[(cgc["viscosity_Pas"] > 1e-7) & (cgc["viscosity_Pas"] < 100)]
    for _, r in cgc.iterrows():
        rows.append({
            "canonical_smiles": r["canonical_smiles"],
            "T_K": r["T_K"],
            "viscosity_Pa_s": r["viscosity_Pas"],
            "source_visc": "chemicals_GC",
        })
    print(f"  chemicals_GC: {len(cgc)} rows, {cgc.canonical_smiles.nunique()} compounds")

    # 3. J Cheminf 2024
    jc = pd.read_csv(paths.data / "viscosity_dataset" / "viscosity_smiles_3582.csv")
    jc["canonical_smiles"] = jc["CANON_SMILES"].apply(canon)
    jc = jc.dropna(subset=["canonical_smiles"])
    jc["viscosity_Pas"] = jc["Viscosity (cP)"] / 1000.0  # cP to Pa·s
    jc = jc[(jc["viscosity_Pas"] > 1e-7) & (jc["viscosity_Pas"] < 100)]
    for _, r in jc.iterrows():
        rows.append({
            "canonical_smiles": r["canonical_smiles"],
            "T_K": r["Temperature (K)"],
            "viscosity_Pa_s": r["viscosity_Pas"],
            "source_visc": "J_Cheminf_2024",
        })
    print(f"  J_Cheminf_2024: {len(jc)} rows, {jc.canonical_smiles.nunique()} compounds")

    # 4. ThermoML (CAS-based, need to resolve to SMILES)
    tv = pd.read_csv(paths.data / "thermo" / "thermoml_parsed" / "pure_viscosity.csv",
                      low_memory=False)
    tv = tv[(tv["viscosity_Pas"] > 1e-7) & (tv["viscosity_Pas"] < 100)]
    # Try to resolve CAS to SMILES via chemicals library
    cas_to_smiles = {}
    try:
        from chemicals.identifiers import search_chemical
        unique_cas = tv["CAS"].dropna().unique()
        print(f"  Resolving {len(unique_cas)} ThermoML CAS numbers to SMILES...")
        for cas in unique_cas:
            try:
                chem = search_chemical(str(cas))
                if chem and hasattr(chem, 'smiles') and chem.smiles:
                    c = canon(chem.smiles)
                    if c:
                        cas_to_smiles[cas] = c
            except:
                pass
        print(f"  Resolved {len(cas_to_smiles)}/{len(unique_cas)} CAS to SMILES")
    except ImportError:
        print("  chemicals library not available, skipping ThermoML CAS resolution")

    for _, r in tv.iterrows():
        cas = r.get("CAS")
        if cas in cas_to_smiles:
            rows.append({
                "canonical_smiles": cas_to_smiles[cas],
                "T_K": r["T_K"],
                "viscosity_Pa_s": r["viscosity_Pas"],
                "source_visc": "ThermoML",
            })
    tv_resolved = tv[tv["CAS"].isin(cas_to_smiles)]
    print(f"  ThermoML: {len(tv_resolved)} rows, "
          f"{tv_resolved['CAS'].nunique()} compounds with SMILES")

    df = pd.DataFrame(rows)
    # Deduplicate: keep first occurrence per (smiles, T_K rounded to 0.1K)
    df["T_round"] = df["T_K"].round(1)
    df = df.drop_duplicates(subset=["canonical_smiles", "T_round"], keep="first")
    df = df.drop(columns=["T_round"])

    print(f"\n  Total after dedup: {len(df)} rows, "
          f"{df['canonical_smiles'].nunique()} unique compounds")
    return df


def main():
    print("=" * 70)
    print("  Phase 4 v2: Build expanded training set")
    print("=" * 70)

    # Step 1: Load all viscosity data
    print("\n--- Step 1: Load all viscosity sources ---")
    visc_df = load_all_viscosity()

    # Step 2: Load real NMR features from pretraining set
    print("\n--- Step 2: Load real NMR features ---")
    pt = pd.read_parquet(PRETRAINING)
    nmr_lookup = {}
    for _, r in pt.iterrows():
        nmr_lookup[r["canonical_smiles"]] = {
            "spectrum_features": r["spectrum_features"],
            "peaklist_features": r["peaklist_features"],
            "n_peaks": r["n_peaks"],
        }
    print(f"  {len(nmr_lookup)} compounds with real NMR features")

    # Step 3: Check overlap
    visc_smiles = set(visc_df["canonical_smiles"].unique())
    nmr_smiles = set(nmr_lookup.keys())
    real_overlap = visc_smiles & nmr_smiles
    need_synthetic = visc_smiles - nmr_smiles
    print(f"\n  Viscosity compounds: {len(visc_smiles)}")
    print(f"  With real NMR: {len(real_overlap)}")
    print(f"  Need synthetic NMR: {len(need_synthetic)}")

    # Step 4: Generate synthetic NMR for missing compounds
    print(f"\n--- Step 3: Generate synthetic NMR ({len(need_synthetic)} compounds) ---")
    t0 = time.time()
    syn_lookup = {}

    # Use multiprocessing for speed
    args = [(s, hash(s) % 2**31) for s in need_synthetic]
    ctx = get_context("spawn")
    with ctx.Pool(processes=16) as pool:
        results = pool.map(_process_synthetic, args, chunksize=100)

    for r in results:
        if r is not None:
            smiles, spec, peak = r
            syn_lookup[smiles] = {
                "spectrum_features": spec,
                "peaklist_features": peak,
                "n_peaks": peak[1],  # peaklist_features[1] = n_peaks
            }

    print(f"  Generated {len(syn_lookup)}/{len(need_synthetic)} synthetic spectra "
          f"({time.time()-t0:.1f}s)")

    # Step 5: Build combined training set
    print(f"\n--- Step 4: Build combined training set ---")
    rows = []
    n_real = 0
    n_synth = 0
    n_skip = 0

    for _, r in visc_df.iterrows():
        smiles = r["canonical_smiles"]

        # Get NMR features (real or synthetic)
        if smiles in nmr_lookup:
            nmr = nmr_lookup[smiles]
            nmr_type = "real"
            n_real += 1
        elif smiles in syn_lookup:
            nmr = syn_lookup[smiles]
            nmr_type = "synthetic"
            n_synth += 1
        else:
            n_skip += 1
            continue

        # Compute descriptors
        desc = compute_descriptors(smiles)
        if desc is None:
            n_skip += 1
            continue

        rows.append({
            "canonical_smiles": smiles,
            "T_K": r["T_K"],
            "viscosity_Pa_s": r["viscosity_Pa_s"],
            "source_visc": r["source_visc"],
            "nmr_type": nmr_type,
            "spectrum_features": nmr["spectrum_features"],
            "peaklist_features": nmr["peaklist_features"],
            "n_peaks": nmr["n_peaks"],
            "log_viscosity": np.log(r["viscosity_Pa_s"]),
            **desc,
        })

    df_out = pd.DataFrame(rows)
    print(f"  Rows with real NMR: {n_real}")
    print(f"  Rows with synthetic NMR: {n_synth}")
    print(f"  Skipped (no NMR or bad SMILES): {n_skip}")
    print(f"\n  Final: {len(df_out)} rows, "
          f"{df_out['canonical_smiles'].nunique()} compounds")

    # Breakdown by NMR type
    for nmr_type in df_out["nmr_type"].unique():
        sub = df_out[df_out["nmr_type"] == nmr_type]
        print(f"    {nmr_type}: {len(sub)} rows, "
              f"{sub['canonical_smiles'].nunique()} compounds")

    # Temperature distribution
    print(f"\n  Temperature distribution:")
    for t_thresh in [240, 260, 280, 300, 320]:
        n = (df_out.T_K < t_thresh).sum()
        nc = df_out[df_out.T_K < t_thresh].canonical_smiles.nunique()
        print(f"    T < {t_thresh}K: {n} rows, {nc} compounds")

    # Save
    df_out.to_parquet(OUTPUT, index=False)
    print(f"\n  Saved to {OUTPUT} ({OUTPUT.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
