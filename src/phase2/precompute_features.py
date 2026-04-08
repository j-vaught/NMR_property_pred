"""
Pre-compute NMR features and save to disk.

Run this once to create cached feature files, then train_features.py
can load them instantly without holding the 3.37M-row NMR parquet in memory.
"""

import gc
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shared.config import Paths
from shared.data_utils import canonical_smiles, inchi_to_canonical_smiles, is_hydrocarbon
from phase2.spectrum_converter import convert_compound
from phase2.nmr_features import extract_nmr_features, get_feature_names

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
        gc = pd.read_csv(gc_path)
        gc["canonical_smiles"] = gc["SMILES"].apply(canonical_smiles)
        gc = gc.dropna(subset=["canonical_smiles"])
        gc_val = [c for c in gc.columns if "viscosity" in c.lower() or "surface" in c.lower()][0]
        gc_df = pd.DataFrame({
            "canonical_smiles": gc["canonical_smiles"],
            "T_K": pd.to_numeric(gc["T_K"], errors="coerce"),
            "value": pd.to_numeric(gc[gc_val], errors="coerce"),
            "source": "gc",
        }).dropna()
        dfs.append(gc_df)

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


def precompute():
    """Pre-compute and cache NMR features for all compounds with labels."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: Collect all target SMILES from both properties
    print("Loading labels...")
    visc_labels = _load_labels("viscosity")
    st_labels = _load_labels("surface_tension")
    all_target_smiles = set(visc_labels["canonical_smiles"].unique()) | set(st_labels["canonical_smiles"].unique())
    print(f"  Total target SMILES: {len(all_target_smiles)}")

    # Step 2: Load NMR data, match SMILES, convert spectra, extract features
    print("Loading NMR parquet (1H NMR only, needed columns)...")
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
    import ast
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

    # Convert to spectra and extract features
    print("Converting spectra and extracting features...")
    feature_names = get_feature_names()
    compound_data = []

    for i, (_, row) in enumerate(nmr.iterrows()):
        smi = row["canonical_smiles"]
        spec = convert_compound(row["NMR_processed"], nmr_type="1H NMR")
        if spec is not None:
            features = extract_nmr_features(spec)
            hc = is_hydrocarbon(smi)
            compound_data.append({
                "canonical_smiles": smi,
                "spectrum": spec,
                "features": features,
                "is_hydrocarbon": hc,
            })
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(nmr)} compounds...", flush=True)

    print(f"  {len(compound_data)} compounds with valid spectra and features")

    # Free NMR dataframe
    del nmr
    gc.collect()

    # Save
    smiles_list = [d["canonical_smiles"] for d in compound_data]
    spectra = np.stack([d["spectrum"] for d in compound_data])
    features = np.stack([d["features"] for d in compound_data])
    hc_mask = np.array([d["is_hydrocarbon"] for d in compound_data])

    np.savez_compressed(
        CACHE_DIR / "nmr_compound_cache.npz",
        spectra=spectra,
        features=features,
        hc_mask=hc_mask,
    )

    # Save SMILES list separately (can't store strings in npz easily)
    with open(CACHE_DIR / "nmr_smiles_list.txt", "w") as f:
        for smi in smiles_list:
            f.write(smi + "\n")

    # Save labels
    visc_labels.to_csv(CACHE_DIR / "labels_viscosity.csv", index=False)
    st_labels.to_csv(CACHE_DIR / "labels_surface_tension.csv", index=False)

    # Save feature names
    with open(CACHE_DIR / "feature_names.txt", "w") as f:
        for name in feature_names:
            f.write(name + "\n")

    print(f"\nCache saved to {CACHE_DIR}/")
    print(f"  Compounds: {len(smiles_list)}")
    print(f"  Spectra shape: {spectra.shape}")
    print(f"  Features shape: {features.shape}")
    print(f"  Hydrocarbons: {hc_mask.sum()}")
    print(f"  Feature names: {len(feature_names)}")


if __name__ == "__main__":
    precompute()
