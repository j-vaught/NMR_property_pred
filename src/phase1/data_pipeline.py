import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shared.config import Paths, DataConfig, SplitConfig
from shared.data_utils import (
    canonical_smiles,
    inchi_to_canonical_smiles,
    is_hydrocarbon,
    fit_arrhenius,
    fit_linear_st,
    make_temperature_features,
)


def load_fingerprints(paths: Paths, data_config: DataConfig) -> pd.DataFrame:
    fp_path = paths.morgan_r2 if data_config.fingerprint == "morgan_r2" else paths.morgan_r3
    df = pd.read_csv(fp_path)

    smiles_col = "SMILES"
    bit_cols = [c for c in df.columns if c not in ("CAS", "SMILES")]

    df["canonical_smiles"] = df[smiles_col].apply(canonical_smiles)
    df = df.dropna(subset=["canonical_smiles"])
    fp_df = df.set_index("canonical_smiles")[bit_cols].astype(np.float32)
    fp_df = fp_df[~fp_df.index.duplicated(keep="first")]

    # Remove zero-variance columns (bits that are never set)
    nonzero_mask = fp_df.std() > 0
    fp_df = fp_df.loc[:, nonzero_mask]

    # Add RDKit descriptors if available
    desc_path = paths.features / "rdkit_descriptors.csv"
    if desc_path.exists():
        desc_df = pd.read_csv(desc_path)
        desc_df["canonical_smiles"] = desc_df["SMILES"].apply(canonical_smiles)
        desc_df = desc_df.dropna(subset=["canonical_smiles"])
        desc_cols = [c for c in desc_df.columns if c not in ("CAS", "SMILES", "canonical_smiles")]
        desc_df = desc_df.set_index("canonical_smiles")[desc_cols].astype(np.float32)
        desc_df = desc_df[~desc_df.index.duplicated(keep="first")]

        # Replace NaN/inf with 0, then standardize descriptors to same scale as FP
        desc_df = desc_df.replace([np.inf, -np.inf], np.nan).fillna(0)
        desc_std = desc_df.std()
        desc_df = desc_df.loc[:, desc_std > 0]
        desc_mean = desc_df.mean()
        desc_std = desc_df.std()
        desc_df = (desc_df - desc_mean) / desc_std.replace(0, 1)

        # Merge FP + descriptors
        common_idx = fp_df.index.intersection(desc_df.index)
        fp_df = pd.concat([fp_df.loc[common_idx], desc_df.loc[common_idx]], axis=1)
        print(f"[FP+Desc] {len(fp_df)} compounds, {fp_df.shape[1]} features "
              f"({nonzero_mask.sum()} FP bits + {len(desc_df.columns)} descriptors)")
    else:
        print(f"[FP] Loaded {len(fp_df)} compounds with {fp_df.shape[1]} features from {fp_path.name}")

    return fp_df


def _load_thermoml_labels(path: Path, property_name: str, data_config: DataConfig) -> pd.DataFrame:
    df = pd.read_csv(path)

    if "InChI" not in df.columns:
        for col in df.columns:
            if "inchi" in col.lower():
                df = df.rename(columns={col: "InChI"})
                break

    df = df.dropna(subset=["InChI"])
    df["canonical_smiles"] = df["InChI"].apply(inchi_to_canonical_smiles)
    df = df.dropna(subset=["canonical_smiles"])

    if "T_K" in df.columns:
        t_col = "T_K"
    elif "temperature" in "".join(df.columns).lower():
        t_col = [c for c in df.columns if "temp" in c.lower()][0]
    else:
        t_col = "T_K"

    if property_name == "viscosity":
        val_col = [c for c in df.columns if "viscosity" in c.lower()][0]
    else:
        val_col = [c for c in df.columns if "surface" in c.lower() or "tension" in c.lower()][0]

    result = pd.DataFrame({
        "canonical_smiles": df["canonical_smiles"],
        "T_K": pd.to_numeric(df[t_col], errors="coerce"),
        "value": pd.to_numeric(df[val_col], errors="coerce"),
        "source": "thermoml",
    })

    result = result.dropna()

    if "P_kPa" in df.columns or any("pressure" in c.lower() for c in df.columns):
        p_col = "P_kPa" if "P_kPa" in df.columns else [c for c in df.columns if "pressure" in c.lower()][0]
        p_vals = pd.to_numeric(df.loc[result.index, p_col], errors="coerce")
        result = result[p_vals.isna() | (p_vals <= data_config.P_max_kPa)]

    result = result[
        (result["T_K"] >= data_config.T_min) & (result["T_K"] <= data_config.T_max)
    ]

    if property_name == "viscosity":
        result = result[result["value"] > 0]
        result = result[result["value"] < 100]

    if property_name == "surface_tension":
        result = result[result["value"] > 0]
        result = result[result["value"] < 1.0]

    n_compounds = result["canonical_smiles"].nunique()
    print(f"[ThermoML {property_name}] {len(result)} rows, {n_compounds} compounds")
    return result


def _load_gc_labels(path: Path, property_name: str, data_config: DataConfig) -> pd.DataFrame:
    df = pd.read_csv(path)

    df["canonical_smiles"] = df["SMILES"].apply(canonical_smiles)
    df = df.dropna(subset=["canonical_smiles"])

    val_col = [c for c in df.columns if "viscosity" in c.lower() or "surface" in c.lower()][0]

    result = pd.DataFrame({
        "canonical_smiles": df["canonical_smiles"],
        "T_K": pd.to_numeric(df["T_K"], errors="coerce"),
        "value": pd.to_numeric(df[val_col], errors="coerce"),
        "source": "chemicals_gc",
    })
    result = result.dropna()
    result = result[
        (result["T_K"] >= data_config.T_min) & (result["T_K"] <= data_config.T_max)
    ]

    if property_name == "viscosity":
        result = result[result["value"] > 0]

    n_compounds = result["canonical_smiles"].nunique()
    print(f"[GC {property_name}] {len(result)} rows, {n_compounds} compounds")
    return result


def _load_nist_webbook_labels(paths: Paths, data_config: DataConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    from shared.config import Paths as _P  # noqa: avoid circular
    from chemicals.identifiers import search_chemical

    nist_dir = paths.nist_webbook_dir
    visc_rows = []
    st_rows = []

    for csv_path in sorted(nist_dir.glob("C*.csv")):
        cas_raw = csv_path.stem
        try:
            result = search_chemical(cas_raw)
            smi = canonical_smiles(result.smiles) if result and result.smiles else None
        except Exception:
            smi = None

        if smi is None:
            continue

        try:
            df = pd.read_csv(csv_path)
        except Exception:
            continue

        t_col = [c for c in df.columns if "temperature" in c.lower() or "T_K" in c][0] if any(
            "temperature" in c.lower() or "T_K" in c for c in df.columns
        ) else df.columns[0]

        for _, row in df.iterrows():
            T = float(row[t_col]) if pd.notna(row[t_col]) else None
            if T is None or T < data_config.T_min or T > data_config.T_max:
                continue

            visc_col = [c for c in df.columns if "viscosity" in c.lower()]
            if visc_col and pd.notna(row[visc_col[0]]):
                v = float(row[visc_col[0]])
                v_pas = v * 1e-6  # uPa.s to Pa.s
                if v_pas > 0:
                    visc_rows.append({"canonical_smiles": smi, "T_K": T, "value": v_pas, "source": "nist_webbook"})

            st_col = [c for c in df.columns if "surface" in c.lower() or "tension" in c.lower()]
            if st_col and pd.notna(row[st_col[0]]):
                s = float(row[st_col[0]])
                if 0 < s < 1.0:
                    st_rows.append({"canonical_smiles": smi, "T_K": T, "value": s, "source": "nist_webbook"})

    visc_df = pd.DataFrame(visc_rows)
    st_df = pd.DataFrame(st_rows)

    print(f"[NIST WebBook] viscosity: {len(visc_df)} rows, {visc_df['canonical_smiles'].nunique() if len(visc_df) > 0 else 0} compounds")
    print(f"[NIST WebBook] surface tension: {len(st_df)} rows, {st_df['canonical_smiles'].nunique() if len(st_df) > 0 else 0} compounds")
    return visc_df, st_df


def build_unified_labels(paths: Paths, data_config: DataConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    thermoml_visc = _load_thermoml_labels(paths.thermoml_viscosity, "viscosity", data_config)
    thermoml_st = _load_thermoml_labels(paths.thermoml_surface_tension, "surface_tension", data_config)

    gc_visc = _load_gc_labels(paths.gc_viscosity, "viscosity", data_config)
    gc_st = _load_gc_labels(paths.gc_surface_tension, "surface_tension", data_config)

    nist_visc, nist_st = _load_nist_webbook_labels(paths, data_config)

    source_priority = {"thermoml": 0, "nist_webbook": 1, "chemicals_gc": 2}

    def merge_and_dedup(dfs: list[pd.DataFrame]) -> pd.DataFrame:
        combined = pd.concat([df for df in dfs if len(df) > 0], ignore_index=True)
        combined["T_K_round"] = combined["T_K"].round(1)
        combined["priority"] = combined["source"].map(source_priority)
        combined = combined.sort_values("priority")
        combined = combined.drop_duplicates(subset=["canonical_smiles", "T_K_round"], keep="first")
        combined = combined.drop(columns=["T_K_round", "priority"])
        return combined

    visc_unified = merge_and_dedup([thermoml_visc, nist_visc, gc_visc])
    st_unified = merge_and_dedup([thermoml_st, nist_st, gc_st])

    print(f"\n[Unified] Viscosity: {len(visc_unified)} rows, {visc_unified['canonical_smiles'].nunique()} compounds")
    print(f"[Unified] Surface tension: {len(st_unified)} rows, {st_unified['canonical_smiles'].nunique()} compounds")

    return visc_unified, st_unified


def join_features_labels(
    fp_df: pd.DataFrame, visc_df: pd.DataFrame, st_df: pd.DataFrame,
    data_config: DataConfig = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    visc_joined = visc_df[visc_df["canonical_smiles"].isin(fp_df.index)]
    st_joined = st_df[st_df["canonical_smiles"].isin(fp_df.index)]

    if data_config and data_config.hydrocarbons_only:
        hc_smiles = {smi for smi in fp_df.index if is_hydrocarbon(smi)}
        visc_before = visc_joined["canonical_smiles"].nunique()
        st_before = st_joined["canonical_smiles"].nunique()
        visc_joined = visc_joined[visc_joined["canonical_smiles"].isin(hc_smiles)]
        st_joined = st_joined[st_joined["canonical_smiles"].isin(hc_smiles)]
        print(f"\n[HC filter] Viscosity: {visc_before} -> {visc_joined['canonical_smiles'].nunique()} compounds")
        print(f"[HC filter] Surface tension: {st_before} -> {st_joined['canonical_smiles'].nunique()} compounds")

    # Tag hydrocarbons for evaluation filtering later
    hc_set = {smi for smi in fp_df.index if is_hydrocarbon(smi)}
    visc_joined = visc_joined.copy()
    visc_joined["is_hydrocarbon"] = visc_joined["canonical_smiles"].isin(hc_set)
    st_joined = st_joined.copy()
    st_joined["is_hydrocarbon"] = st_joined["canonical_smiles"].isin(hc_set)
    n_hc_v = visc_joined[visc_joined["is_hydrocarbon"]]["canonical_smiles"].nunique()
    n_hc_s = st_joined[st_joined["is_hydrocarbon"]]["canonical_smiles"].nunique()
    print(f"[HC tag] {n_hc_v} visc hydrocarbons, {n_hc_s} ST hydrocarbons (for eval reporting)")

    print(f"[Joined] Viscosity: {len(visc_joined)} rows, {visc_joined['canonical_smiles'].nunique()} compounds")
    print(f"[Joined] Surface tension: {len(st_joined)} rows, {st_joined['canonical_smiles'].nunique()} compounds")

    all_compounds = set(visc_joined["canonical_smiles"]) | set(st_joined["canonical_smiles"])
    both = set(visc_joined["canonical_smiles"]) & set(st_joined["canonical_smiles"])
    print(f"[Joined] Total unique compounds: {len(all_compounds)}, with both properties: {len(both)}")

    return visc_joined, st_joined


def fit_arrhenius_targets(visc_df: pd.DataFrame, data_config: DataConfig) -> pd.DataFrame:
    records = []
    for smi, group in visc_df.groupby("canonical_smiles"):
        if len(group) < data_config.min_points_per_compound:
            continue
        T = group["T_K"].values
        eta = group["value"].values
        A, B, r2 = fit_arrhenius(T, eta)
        if r2 >= data_config.arrhenius_r2_threshold:
            records.append({"canonical_smiles": smi, "A": A, "B": B, "r2": r2, "n_points": len(group)})

    result = pd.DataFrame(records)
    print(f"[Arrhenius viscosity] {len(result)} compounds (filtered from {visc_df['canonical_smiles'].nunique()}, "
          f"min_pts={data_config.min_points_per_compound}, r2>={data_config.arrhenius_r2_threshold})")
    return result


def fit_linear_st_targets(st_df: pd.DataFrame, data_config: DataConfig) -> pd.DataFrame:
    records = []
    for smi, group in st_df.groupby("canonical_smiles"):
        if len(group) < data_config.min_points_per_compound:
            continue
        T = group["T_K"].values
        sigma = group["value"].values
        A, B, r2 = fit_linear_st(T, sigma)
        if r2 >= data_config.arrhenius_r2_threshold:
            records.append({"canonical_smiles": smi, "A": A, "B": B, "r2": r2, "n_points": len(group)})

    result = pd.DataFrame(records)
    print(f"[Linear ST] {len(result)} compounds (filtered from {st_df['canonical_smiles'].nunique()})")
    return result


def split_by_compound(
    compounds: list[str],
    visc_smiles: set[str],
    st_smiles: set[str],
    split_config: SplitConfig,
) -> dict[str, list[str]]:
    rng = np.random.RandomState(split_config.seed)
    compounds = list(compounds)

    strata = []
    for smi in compounds:
        has_v = smi in visc_smiles
        has_s = smi in st_smiles
        if has_v and has_s:
            strata.append("both")
        elif has_v:
            strata.append("visc_only")
        else:
            strata.append("st_only")

    indices = np.arange(len(compounds))
    rng.shuffle(indices)

    n = len(compounds)
    n_test = max(1, int(n * split_config.test_frac))
    n_val = max(1, int(n * split_config.val_frac))

    test_idx = indices[:n_test]
    val_idx = indices[n_test:n_test + n_val]
    train_idx = indices[n_test + n_val:]

    splits = {
        "train": [compounds[i] for i in train_idx],
        "val": [compounds[i] for i in val_idx],
        "test": [compounds[i] for i in test_idx],
    }

    for name, smis in splits.items():
        n_v = sum(1 for s in smis if s in visc_smiles)
        n_s = sum(1 for s in smis if s in st_smiles)
        print(f"[Split] {name}: {len(smis)} compounds ({n_v} visc, {n_s} ST)")

    return splits


def build_dataset(paths: Paths = None, data_config: DataConfig = None, split_config: SplitConfig = None):
    if paths is None:
        paths = Paths()
    if data_config is None:
        data_config = DataConfig()
    if split_config is None:
        split_config = SplitConfig()

    print("=" * 60)
    print("BUILDING DATASET")
    print("=" * 60)

    fp_df = load_fingerprints(paths, data_config)
    visc_df, st_df = build_unified_labels(paths, data_config)
    visc_joined, st_joined = join_features_labels(fp_df, visc_df, st_df, data_config)

    all_compounds = sorted(set(visc_joined["canonical_smiles"]) | set(st_joined["canonical_smiles"]))
    splits = split_by_compound(
        all_compounds,
        set(visc_joined["canonical_smiles"]),
        set(st_joined["canonical_smiles"]),
        split_config,
    )

    return {
        "fp_df": fp_df,
        "visc_df": visc_joined,
        "st_df": st_joined,
        "splits": splits,
        "data_config": data_config,
    }


if __name__ == "__main__":
    dataset = build_dataset()
    print("\nDone. Ready for training.")
