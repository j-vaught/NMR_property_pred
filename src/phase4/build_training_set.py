"""
Build the Phase 4 training set: NMR features + viscosity measurements.

Joins:
  - NMR peak lists from: NMRexp (primary), NMRShiftDB2, NMRBank
  - Viscosity measurements from: ThermoML, J Cheminf 2024, chemicals GC

Output: data/nmr/phase4_training_set.parquet with columns:
  canonical_smiles, T_K, viscosity_Pa_s, log_viscosity, source_visc,
  spectrum_features (95), peaklist_features (25), is_hydrocarbon,
  mw, logp, n_heavy_atoms, n_h, h_count_nmr

The spectrum_features and peaklist_features are stored as list columns
(parquet array). Features computed via existing Phase 2 pipeline.
"""

import ast
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shared.config import Paths
from shared.data_utils import canonical_smiles, inchi_to_canonical_smiles, is_hydrocarbon
from phase2.spectrum_converter import parse_1h_peaks, peaks_to_spectrum_1h, _1H_GRID
from phase2.nmr_features import (
    extract_nmr_features,
    extract_peaklist_features,
)

paths = Paths()
OUT_PATH = paths.data / "nmr" / "phase4_training_set.parquet"
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

OVERLAP_FILE = Path(__file__).parent / "outputs" / "nmr_visc_overlap_smiles.txt"


def load_overlap_smiles() -> set[str]:
    """Load the pre-computed set of NMR∩viscosity canonical SMILES."""
    with open(OVERLAP_FILE) as f:
        return set(line.strip() for line in f if line.strip())


def collect_nmr_peaks(target_smiles: set[str]) -> dict[str, list]:
    """Return dict canonical_smiles → peak list from the best available source.

    Priority: NMRexp (peak lists) > NMRShiftDB2 (SDF) > NMRBank (text)
    """
    peaks_map: dict[str, list] = {}

    # --- 1. NMRexp (structured peak lists) ---
    print("Loading NMRexp peak lists...")
    t0 = time.time()
    nmr = pd.read_parquet(
        paths.nmrexp,
        columns=["SMILES", "NMR_type", "NMR_processed"],
    )
    nmr = nmr[nmr["NMR_type"] == "1H NMR"]
    nmr = nmr.dropna(subset=["NMR_processed", "SMILES"])
    print(f"  {len(nmr)} 1H NMR rows")

    # Fast filter: direct SMILES match first
    mask = nmr["SMILES"].isin(target_smiles)
    print(f"  {mask.sum()} direct SMILES matches")

    # Canonicalize unmatched
    unmatched_smiles = nmr.loc[~mask, "SMILES"].unique()
    print(f"  Canonicalizing {len(unmatched_smiles)} remaining SMILES...")
    canon_map = {}
    for i, smi in enumerate(unmatched_smiles):
        c = canonical_smiles(smi)
        if c is not None and c in target_smiles:
            canon_map[smi] = c
        if (i + 1) % 200000 == 0:
            print(f"    {i+1}/{len(unmatched_smiles)}, {len(canon_map)} matches")
    print(f"  → {len(canon_map)} additional canonical matches")

    nmr["canonical"] = nmr["SMILES"].copy()
    mapped_mask = nmr["SMILES"].isin(canon_map)
    nmr.loc[mapped_mask, "canonical"] = nmr.loc[mapped_mask, "SMILES"].map(canon_map)
    nmr = nmr[mask | mapped_mask]

    # Count peaks per entry to pick best
    def count_peaks_safe(s):
        try:
            return len(ast.literal_eval(s))
        except Exception:
            return 0

    nmr["n_peaks"] = nmr["NMR_processed"].apply(count_peaks_safe)
    nmr = nmr.sort_values("n_peaks", ascending=False).drop_duplicates(
        subset=["canonical"], keep="first"
    )

    for _, row in nmr.iterrows():
        smi = row["canonical"]
        if smi in peaks_map:
            continue
        peaks = parse_1h_peaks(row["NMR_processed"])
        if peaks:
            peaks_map[smi] = peaks

    del nmr
    import gc
    gc.collect()
    print(f"  NMRexp: got peaks for {len(peaks_map)} compounds in {time.time()-t0:.1f}s")

    # --- 2. NMRShiftDB2 (SDF with spectrum properties) ---
    print("\nLoading NMRShiftDB2 peaks...")
    try:
        from rdkit import Chem
        from rdkit import RDLogger
        RDLogger.DisableLog("rdApp.*")

        sd_path = paths.data / "nmr" / "nmrshiftdb2" / "nmrshiftdb2withsignals.sd"
        if sd_path.exists():
            suppl = Chem.SDMolSupplier(str(sd_path))
            added = 0
            for mol in suppl:
                if mol is None:
                    continue
                smi = Chem.MolToSmiles(mol)
                if not smi or smi not in target_smiles or smi in peaks_map:
                    continue
                # Find 1H spectrum prop
                props = mol.GetPropsAsDict()
                h_prop_keys = [k for k in props if "Spectrum 1H" in k]
                if not h_prop_keys:
                    continue
                spec_str = str(props[h_prop_keys[0]])
                # Format: shift;intensity;multiplicity;atom | shift;... |
                peaks = []
                for entry in spec_str.split("|"):
                    parts = entry.split(";")
                    if len(parts) >= 1:
                        try:
                            shift = float(parts[0])
                            mult = parts[2] if len(parts) > 2 else "s"
                            peaks.append({
                                "center_ppm": shift,
                                "integral": 1.0,  # not in SDF format
                                "multiplicity": mult.lower().strip() or "s",
                                "j_couplings_hz": [],
                                "shift_high": shift,
                                "shift_low": shift,
                            })
                        except ValueError:
                            continue
                if peaks:
                    peaks_map[smi] = peaks
                    added += 1
            print(f"  NMRShiftDB2: added {added} compounds (total {len(peaks_map)})")
    except Exception as e:
        print(f"  NMRShiftDB2 error: {e}")

    # --- 3. NMRBank (CSV with chemical shift text) ---
    print("\nLoading NMRBank peaks...")
    nb_path = paths.data / "nmr" / "nmrbank" / "NMRBank_full" / "NMRBank_data_with_unique_SMILES_149135_in_156621.csv"
    if nb_path.exists():
        try:
            nb = pd.read_csv(nb_path, usecols=["SMILES", "1H NMR chemical shifts"])
            nb = nb.dropna(subset=["SMILES", "1H NMR chemical shifts"])
            added = 0
            for _, row in nb.iterrows():
                smi_raw = row["SMILES"]
                smi = canonical_smiles(smi_raw)
                if not smi or smi not in target_smiles or smi in peaks_map:
                    continue
                shifts_str = str(row["1H NMR chemical shifts"])
                # Parse rough: extract floats from text
                import re
                nums = re.findall(r"(\d+\.\d+)", shifts_str)
                if len(nums) < 1:
                    continue
                peaks = [
                    {
                        "center_ppm": float(x),
                        "integral": 1.0,
                        "multiplicity": "m",
                        "j_couplings_hz": [],
                        "shift_high": float(x),
                        "shift_low": float(x),
                    }
                    for x in nums[:50]  # cap at 50 peaks per compound
                ]
                if peaks:
                    peaks_map[smi] = peaks
                    added += 1
            print(f"  NMRBank: added {added} compounds (total {len(peaks_map)})")
        except Exception as e:
            print(f"  NMRBank error: {e}")

    return peaks_map


def load_viscosity_measurements(target_smiles: set[str]) -> pd.DataFrame:
    """Return DataFrame with columns [canonical_smiles, T_K, viscosity_Pa_s, source_visc]."""
    rows = []

    # 1. ThermoML
    print("\nLoading ThermoML viscosity...")
    tml = pd.read_csv(paths.thermoml_viscosity)
    tml = tml.dropna(subset=["InChI"])
    tml["canonical"] = tml["InChI"].apply(inchi_to_canonical_smiles)
    tml = tml.dropna(subset=["canonical"])
    tml = tml[tml["canonical"].isin(target_smiles)]

    # Detect columns
    t_col = "T_K" if "T_K" in tml.columns else [c for c in tml.columns if "temp" in c.lower()][0]
    v_col = [c for c in tml.columns if "viscosity" in c.lower()][0]

    for _, row in tml.iterrows():
        T = pd.to_numeric(row[t_col], errors="coerce")
        v = pd.to_numeric(row[v_col], errors="coerce")
        if pd.isna(T) or pd.isna(v) or v <= 0 or v > 100:
            continue
        if T < 200 or T > 500:
            continue
        # Pressure filter
        if "P_kPa" in tml.columns:
            p = pd.to_numeric(row.get("P_kPa"), errors="coerce")
            if not pd.isna(p) and p > 110:
                continue
        rows.append({
            "canonical_smiles": row["canonical"],
            "T_K": float(T),
            "viscosity_Pa_s": float(v),  # ThermoML is already in Pa·s
            "source_visc": "thermoml",
        })
    print(f"  ThermoML: {len(rows)} rows so far")

    # 2. J Cheminf 2024 viscosity dataset (cP → Pa·s)
    print("\nLoading J Cheminf 2024 viscosity...")
    jc_path = paths.data / "viscosity_dataset" / "viscosity_smiles_3582.csv"
    if jc_path.exists():
        jc = pd.read_csv(jc_path)
        for _, row in jc.iterrows():
            smi_raw = row.get("CANON_SMILES")
            if not smi_raw:
                continue
            smi = canonical_smiles(smi_raw)
            if not smi or smi not in target_smiles:
                continue
            T = row.get("Temperature (K)")
            v_cp = row.get("Viscosity (cP)")
            if pd.isna(T) or pd.isna(v_cp) or v_cp <= 0:
                continue
            if T < 200 or T > 500:
                continue
            rows.append({
                "canonical_smiles": smi,
                "T_K": float(T),
                "viscosity_Pa_s": float(v_cp) * 1e-3,  # 1 cP = 1 mPa·s = 1e-3 Pa·s
                "source_visc": "jcheminf2024",
            })
    jc_count = sum(1 for r in rows if r["source_visc"] == "jcheminf2024")
    print(f"  J Cheminf: {jc_count} rows")

    # 3. chemicals group contribution
    print("\nLoading chemicals GC viscosity...")
    gc_path = paths.thermo / "group_contribution" / "chemicals_viscosity_expanded.csv"
    if gc_path.exists():
        gc = pd.read_csv(gc_path)
        for _, row in gc.iterrows():
            smi_raw = row.get("SMILES")
            if not smi_raw:
                continue
            smi = canonical_smiles(smi_raw)
            if not smi or smi not in target_smiles:
                continue
            T = row.get("T_K")
            val = row.get("viscosity_Pas") if "viscosity_Pas" in gc.columns else row.get("viscosity")
            if pd.isna(T) or pd.isna(val) or val <= 0 or val > 100:
                continue
            if T < 200 or T > 500:
                continue
            rows.append({
                "canonical_smiles": smi,
                "T_K": float(T),
                "viscosity_Pa_s": float(val),
                "source_visc": "chemicals_gc",
            })
    gc_count = sum(1 for r in rows if r["source_visc"] == "chemicals_gc")
    print(f"  chemicals_GC: {gc_count} rows")

    df = pd.DataFrame(rows)
    return df


def build_features_for_peaks(peaks_map: dict, target_smiles: set) -> dict[str, dict]:
    """Compute NMR features for each compound.

    Includes a per-compound timeout to avoid pathological peak lists
    (e.g., >100 peaks that cause Lorentzian broadening to hang).
    """
    import signal

    def handler(signum, frame):
        raise TimeoutError("feature computation timeout")

    print("\nComputing NMR features...")
    feature_map = {}
    t0 = time.time()
    n_timeout = 0
    n_error = 0

    # Skip compounds with > 100 peaks (pathological, probably bad parse)
    items = [(s, p) for s, p in peaks_map.items() if s in target_smiles]
    print(f"  {len(items)} compounds to process")

    for i, (smi, peaks) in enumerate(items):
        if len(peaks) > 100:
            n_error += 1
            continue
        try:
            # 5-second timeout per compound
            signal.signal(signal.SIGALRM, handler)
            signal.alarm(5)
            spectrum = peaks_to_spectrum_1h(peaks, _1H_GRID)
            spec_feats = extract_nmr_features(spectrum)
            peak_feats = extract_peaklist_features(peaks)
            signal.alarm(0)

            feature_map[smi] = {
                "spectrum_features": spec_feats.astype(np.float32).tolist(),
                "peaklist_features": peak_feats.astype(np.float32).tolist(),
                "n_peaks": len(peaks),
            }
        except TimeoutError:
            n_timeout += 1
            signal.alarm(0)
        except Exception as e:
            n_error += 1
            signal.alarm(0)

        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(items)} in {time.time()-t0:.1f}s "
                  f"(ok={len(feature_map)}, timeout={n_timeout}, err={n_error})",
                  flush=True)

    print(f"  Got features for {len(feature_map)} compounds in {time.time()-t0:.1f}s "
          f"(timeout={n_timeout}, err={n_error})")
    return feature_map


def add_rdkit_descriptors(df: pd.DataFrame) -> pd.DataFrame:
    """Add molecular descriptors computed from SMILES."""
    print("\nAdding RDKit descriptors...")
    from rdkit import Chem
    from rdkit.Chem import Descriptors, Lipinski
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")

    # Compute per-compound, then merge
    unique_smiles = df["canonical_smiles"].unique()
    desc_rows = []
    for smi in unique_smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        formula_h = sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == "H")
        # Add implicit Hs
        mol_with_h = Chem.AddHs(mol)
        h_count = sum(1 for atom in mol_with_h.GetAtoms() if atom.GetSymbol() == "H")
        desc_rows.append({
            "canonical_smiles": smi,
            "mw": float(Descriptors.MolWt(mol)),
            "logp": float(Descriptors.MolLogP(mol)),
            "tpsa": float(Descriptors.TPSA(mol)),
            "n_heavy_atoms": int(mol.GetNumHeavyAtoms()),
            "h_count_molecular": int(h_count),
            "n_rotatable": int(Lipinski.NumRotatableBonds(mol)),
            "n_aromatic_rings": int(Lipinski.NumAromaticRings(mol)),
            "is_hydrocarbon": is_hydrocarbon(smi),
        })
    desc_df = pd.DataFrame(desc_rows)
    return df.merge(desc_df, on="canonical_smiles", how="left")


def main():
    print("=" * 70)
    print("  Phase 4: Build viscosity training set")
    print("=" * 70)

    target_smiles = load_overlap_smiles()
    print(f"\nTarget SMILES (NMR ∩ viscosity): {len(target_smiles)}")

    # 1. Get NMR peak lists
    peaks_map = collect_nmr_peaks(target_smiles)
    print(f"\n→ Got peaks for {len(peaks_map)} / {len(target_smiles)} target compounds")

    # 2. Compute NMR features
    feature_map = build_features_for_peaks(peaks_map, target_smiles)

    # 3. Load viscosity measurements
    visc_df = load_viscosity_measurements(target_smiles)
    print(f"\n→ Loaded {len(visc_df)} viscosity measurements "
          f"({visc_df['canonical_smiles'].nunique()} compounds, "
          f"{visc_df['source_visc'].value_counts().to_dict()})")

    # 4. Keep only compounds where we have NMR features
    visc_df = visc_df[visc_df["canonical_smiles"].isin(feature_map)]
    print(f"\n→ After NMR filter: {len(visc_df)} rows, "
          f"{visc_df['canonical_smiles'].nunique()} compounds")

    # 5. Attach NMR features
    visc_df["spectrum_features"] = visc_df["canonical_smiles"].map(
        lambda s: feature_map[s]["spectrum_features"]
    )
    visc_df["peaklist_features"] = visc_df["canonical_smiles"].map(
        lambda s: feature_map[s]["peaklist_features"]
    )
    visc_df["n_peaks"] = visc_df["canonical_smiles"].map(
        lambda s: feature_map[s]["n_peaks"]
    )

    # 6. Add log viscosity + RDKit descriptors
    visc_df["log_viscosity"] = np.log(visc_df["viscosity_Pa_s"])
    visc_df = add_rdkit_descriptors(visc_df)

    # 7. Dedup measurements at same (SMILES, T) — prefer experimental
    priority = {"thermoml": 0, "jcheminf2024": 1, "chemicals_gc": 2}
    visc_df["prio"] = visc_df["source_visc"].map(priority)
    visc_df["T_round"] = visc_df["T_K"].round(1)
    visc_df = visc_df.sort_values("prio").drop_duplicates(
        subset=["canonical_smiles", "T_round"], keep="first"
    )
    visc_df = visc_df.drop(columns=["prio", "T_round"])

    print(f"\n→ Final training set: {len(visc_df)} rows, "
          f"{visc_df['canonical_smiles'].nunique()} compounds")
    print(f"  hydrocarbons: {visc_df['is_hydrocarbon'].sum()} rows, "
          f"{visc_df[visc_df['is_hydrocarbon']]['canonical_smiles'].nunique()} compounds")
    print(f"  T range: {visc_df['T_K'].min():.0f} - {visc_df['T_K'].max():.0f} K")
    print(f"  visc range: {visc_df['viscosity_Pa_s'].min():.6f} - "
          f"{visc_df['viscosity_Pa_s'].max():.4f} Pa·s")
    print(f"  sources: {visc_df['source_visc'].value_counts().to_dict()}")

    # 8. Save
    visc_df.to_parquet(OUT_PATH, index=False)
    print(f"\nSaved to {OUT_PATH}")
    print(f"  size: {OUT_PATH.stat().st_size/1e6:.2f} MB")


if __name__ == "__main__":
    main()
