"""
Pre-compute NMR features AND Phase 1 features, save to disk.

Run this once to create cached feature files, then train_features.py
can load them instantly without holding the 3.37M-row NMR parquet in memory.

Produces cache/ directory with:
  - nmr_compound_cache.npz: spectra, spectrum features, peak-list features,
    Phase 1 Morgan FP, Phase 1 RDKit descriptors, HC mask
  - nmr_smiles_list.txt, labels_*.csv, *_feature_names.txt
"""

import ast
import gc
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shared.config import Paths
from shared.data_utils import canonical_smiles, inchi_to_canonical_smiles, is_hydrocarbon
from phase2.spectrum_converter import parse_1h_peaks, peaks_to_spectrum_1h, _1H_GRID
from phase2.nmr_features import (
    extract_nmr_features, get_feature_names,
    extract_peaklist_features, get_peaklist_feature_names,
)

paths = Paths()
CACHE_DIR = Path(__file__).resolve().parent / "cache"


def _load_labels(property_name: str) -> pd.DataFrame:
    """Load ThermoML + chemicals GC labels for a given property."""
    dfs = []

    if property_name == "viscosity":
        df = pd.read_csv(paths.thermoml_viscosity)
        val_col = [c for c in df.columns if "viscosity" in c.lower()][0]
    else:
        df = pd.read_csv(paths.thermoml_surface_tension)
        val_col = [c for c in df.columns if "surface" in c.lower() or "tension" in c.lower()][0]

    df = df.dropna(subset=["InChI"])
    df["canonical_smiles"] = df["InChI"].apply(inchi_to_canonical_smiles)
    df = df.dropna(subset=["canonical_smiles"])
    t_col = "T_K" if "T_K" in df.columns else [c for c in df.columns if "temp" in c.lower()][0]

    thermoml = pd.DataFrame({
        "canonical_smiles": df["canonical_smiles"],
        "T_K": pd.to_numeric(df[t_col], errors="coerce"),
        "value": pd.to_numeric(df[val_col], errors="coerce"),
        "source": "thermoml",
    }).dropna()

    if "P_kPa" in df.columns:
        p = pd.to_numeric(df.loc[thermoml.index, "P_kPa"], errors="coerce")
        thermoml = thermoml[p.isna() | (p <= 110)]

    dfs.append(thermoml)

    if property_name == "viscosity":
        gc_path = paths.thermo / "group_contribution" / "chemicals_viscosity_expanded.csv"
    else:
        gc_path = paths.thermo / "group_contribution" / "chemicals_surface_tension_expanded.csv"

    if gc_path.exists():
        gc_df = pd.read_csv(gc_path)
        gc_df["canonical_smiles"] = gc_df["SMILES"].apply(canonical_smiles)
        gc_df = gc_df.dropna(subset=["canonical_smiles"])
        gc_val = [c for c in gc_df.columns if "viscosity" in c.lower() or "surface" in c.lower()][0]
        gc_out = pd.DataFrame({
            "canonical_smiles": gc_df["canonical_smiles"],
            "T_K": pd.to_numeric(gc_df["T_K"], errors="coerce"),
            "value": pd.to_numeric(gc_df[gc_val], errors="coerce"),
            "source": "gc",
        }).dropna()
        dfs.append(gc_out)

    combined = pd.concat(dfs, ignore_index=True)
    combined = combined[(combined["T_K"] >= 200) & (combined["T_K"] <= 500)]

    if property_name == "viscosity":
        combined = combined[(combined["value"] > 0) & (combined["value"] < 100)]
    else:
        combined = combined[(combined["value"] > 0) & (combined["value"] < 1)]

    combined["T_round"] = combined["T_K"].round(1)
    combined["priority"] = combined["source"].map({"thermoml": 0, "gc": 1})
    combined = combined.sort_values("priority").drop_duplicates(
        subset=["canonical_smiles", "T_round"], keep="first"
    ).drop(columns=["T_round", "priority"])

    return combined


def _load_phase1_features(smiles_list: list[str]) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    """Load Phase 1 Morgan FP + RDKit descriptors for the given compounds.

    For compounds not in the pre-computed CSVs, generates features on-the-fly.

    Returns
    -------
    morgan : np.ndarray, shape (n_compounds, 2048)
    rdkit_desc : np.ndarray, shape (n_compounds, n_desc)
    morgan_names : list of feature names
    rdkit_names : list of feature names
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors

    # Load pre-computed files
    morgan_df = pd.read_csv(paths.morgan_r2)
    rdkit_df = pd.read_csv(paths.features / "rdkit_descriptors.csv")

    # Canonicalize SMILES in Phase 1 files
    morgan_df["canon"] = morgan_df["SMILES"].apply(canonical_smiles)
    rdkit_df["canon"] = rdkit_df["SMILES"].apply(canonical_smiles)

    morgan_cols = [c for c in morgan_df.columns if c.startswith("morgan_")]
    rdkit_cols = [c for c in rdkit_df.columns if c not in ("CAS", "SMILES", "canon")]

    morgan_lookup = morgan_df.dropna(subset=["canon"]).drop_duplicates(subset=["canon"]).set_index("canon")[morgan_cols]
    rdkit_lookup = rdkit_df.dropna(subset=["canon"]).drop_duplicates(subset=["canon"]).set_index("canon")[rdkit_cols]

    # Get the RDKit descriptor calculator names for on-the-fly generation
    calc_names = [name for name, _ in Descriptors.descList]
    # Map rdkit_cols to calculator names (they should match)
    rdkit_col_set = set(rdkit_cols)

    print(f"  Phase 1 features: {len(morgan_lookup)} Morgan, {len(rdkit_lookup)} RDKit")

    morgan_arr = np.zeros((len(smiles_list), len(morgan_cols)), dtype=np.float32)
    rdkit_arr = np.zeros((len(smiles_list), len(rdkit_cols)), dtype=np.float32)
    n_from_csv = 0
    n_generated = 0

    for i, smi in enumerate(smiles_list):
        # Try CSV lookup first
        if smi in morgan_lookup.index:
            morgan_arr[i] = morgan_lookup.loc[smi].values
            n_from_csv += 1
        else:
            # Generate Morgan FP on-the-fly
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
                morgan_arr[i] = np.array(fp, dtype=np.float32)
                n_generated += 1

        if smi in rdkit_lookup.index:
            rdkit_arr[i] = rdkit_lookup.loc[smi].values
        else:
            # Generate RDKit descriptors on-the-fly
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                # Build descriptor name -> calculator mapping
                desc_map = {name: func for name, func in Descriptors.descList}
                for j, col in enumerate(rdkit_cols):
                    if col in desc_map:
                        try:
                            val = desc_map[col](mol)
                            if val is None or not np.isfinite(val):
                                val = 0.0
                        except Exception:
                            val = 0.0
                        rdkit_arr[i, j] = val

    # Clean NaN/inf
    morgan_arr = np.nan_to_num(morgan_arr, nan=0.0, posinf=0.0, neginf=0.0)
    rdkit_arr = np.nan_to_num(rdkit_arr, nan=0.0, posinf=0.0, neginf=0.0)

    print(f"  Loaded {n_from_csv} from CSV, generated {n_generated} on-the-fly")
    print(f"  Morgan shape: {morgan_arr.shape}, RDKit shape: {rdkit_arr.shape}")

    return morgan_arr, rdkit_arr, morgan_cols, list(rdkit_cols)


def precompute():
    """Pre-compute and cache all features for NMR-matched compounds."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: Collect all target SMILES from both properties
    print("Loading labels...")
    visc_labels = _load_labels("viscosity")
    st_labels = _load_labels("surface_tension")
    all_target_smiles = set(visc_labels["canonical_smiles"].unique()) | set(st_labels["canonical_smiles"].unique())
    print(f"  Total target SMILES: {len(all_target_smiles)}")

    # Step 2: Load NMR data and match SMILES
    print("Loading NMR parquet (1H NMR only)...")
    nmr = pd.read_parquet(
        paths.nmrexp,
        columns=["SMILES", "NMR_type", "NMR_processed"],
    )
    nmr = nmr[nmr["NMR_type"] == "1H NMR"].copy()
    nmr = nmr.dropna(subset=["NMR_processed", "SMILES"])
    print(f"  {len(nmr)} 1H NMR rows, {nmr['SMILES'].nunique()} unique SMILES")

    # Direct SMILES match
    mask_direct = nmr["SMILES"].isin(all_target_smiles)
    print(f"  {mask_direct.sum()} rows match targets by raw SMILES")

    # Canonicalize unmatched NMR SMILES
    remaining_targets = all_target_smiles - set(nmr.loc[mask_direct, "SMILES"].unique())
    print(f"  {len(remaining_targets)} target SMILES still unmatched")

    unmatched_raw = nmr.loc[~mask_direct, "SMILES"].unique()
    print(f"  Canonicalizing {len(unmatched_raw)} unmatched NMR SMILES...")

    canon_map = {}
    for i, smi in enumerate(unmatched_raw):
        c = canonical_smiles(smi)
        if c is not None and c in remaining_targets:
            canon_map[smi] = c
        if (i + 1) % 200000 == 0:
            print(f"    {i+1}/{len(unmatched_raw)}, {len(canon_map)} matches...", flush=True)

    print(f"  {len(canon_map)} additional matches from canonicalization")

    # Build canonical column
    mask_mapped = nmr["SMILES"].isin(canon_map)
    nmr["canonical_smiles"] = nmr["SMILES"].copy()
    nmr.loc[mask_mapped, "canonical_smiles"] = nmr.loc[mask_mapped, "SMILES"].map(canon_map)
    nmr = nmr[mask_direct | mask_mapped]
    nmr = nmr.dropna(subset=["canonical_smiles"])

    # Keep best entry per compound (most peaks)
    def _count_peaks(s):
        try:
            return len(ast.literal_eval(s))
        except Exception:
            return 0

    nmr["n_peaks"] = nmr["NMR_processed"].apply(_count_peaks)
    nmr = nmr.sort_values("n_peaks", ascending=False).drop_duplicates(
        subset=["canonical_smiles"], keep="first"
    )
    print(f"  {len(nmr)} unique compounds with matched 1H NMR")

    # Step 3: Parse peaks, extract spectrum + peak-list features
    print("Extracting spectrum and peak-list features...")
    compound_data = []

    for i, (_, row) in enumerate(nmr.iterrows()):
        smi = row["canonical_smiles"]

        # Parse peak list FIRST (preserves absolute integrals)
        peaks = parse_1h_peaks(row["NMR_processed"])
        if peaks is None:
            continue

        # Convert peaks to spectrum
        spec = peaks_to_spectrum_1h(peaks, _1H_GRID)

        # Extract both feature types
        spectrum_feats = extract_nmr_features(spec)
        peaklist_feats = extract_peaklist_features(peaks)
        hc = is_hydrocarbon(smi)

        compound_data.append({
            "canonical_smiles": smi,
            "spectrum": spec,
            "spectrum_features": spectrum_feats,
            "peaklist_features": peaklist_feats,
            "is_hydrocarbon": hc,
        })

        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(nmr)} compounds...", flush=True)

    print(f"  {len(compound_data)} compounds with valid features")

    # Free NMR dataframe AND canonical_smiles LRU cache to reclaim ~30GB
    del nmr
    canonical_smiles.cache_clear()
    gc.collect()
    print("  Freed NMR data and SMILES cache")

    # Step 4: Load Phase 1 features (Morgan FP + RDKit descriptors)
    smiles_list = [d["canonical_smiles"] for d in compound_data]
    print(f"\nLoading Phase 1 features for {len(smiles_list)} compounds...")
    morgan_arr, rdkit_arr, morgan_names, rdkit_names = _load_phase1_features(smiles_list)

    # Step 5: Save everything
    spectra = np.stack([d["spectrum"] for d in compound_data])
    spectrum_features = np.stack([d["spectrum_features"] for d in compound_data])
    peaklist_features = np.stack([d["peaklist_features"] for d in compound_data])
    hc_mask = np.array([d["is_hydrocarbon"] for d in compound_data])

    np.savez_compressed(
        CACHE_DIR / "nmr_compound_cache.npz",
        spectra=spectra,
        spectrum_features=spectrum_features,
        peaklist_features=peaklist_features,
        phase1_morgan=morgan_arr,
        phase1_rdkit=rdkit_arr,
        hc_mask=hc_mask,
    )

    with open(CACHE_DIR / "nmr_smiles_list.txt", "w") as f:
        for smi in smiles_list:
            f.write(smi + "\n")

    visc_labels.to_csv(CACHE_DIR / "labels_viscosity.csv", index=False)
    st_labels.to_csv(CACHE_DIR / "labels_surface_tension.csv", index=False)

    spectrum_names = get_feature_names()
    peaklist_names = get_peaklist_feature_names()

    for fname, names in [
        ("spectrum_feature_names.txt", spectrum_names),
        ("peaklist_feature_names.txt", peaklist_names),
        ("morgan_feature_names.txt", morgan_names),
        ("rdkit_feature_names.txt", rdkit_names),
    ]:
        with open(CACHE_DIR / fname, "w") as f:
            for name in names:
                f.write(name + "\n")

    print(f"\nCache saved to {CACHE_DIR}/")
    print(f"  Compounds: {len(smiles_list)}")
    print(f"  Spectrum features: {spectrum_features.shape}")
    print(f"  Peak-list features: {peaklist_features.shape}")
    print(f"  Morgan FP: {morgan_arr.shape}")
    print(f"  RDKit descriptors: {rdkit_arr.shape}")
    print(f"  Hydrocarbons: {hc_mask.sum()}")


if __name__ == "__main__":
    precompute()
