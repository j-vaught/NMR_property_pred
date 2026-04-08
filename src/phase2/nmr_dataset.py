"""
PyTorch dataset that joins NMRexp spectra with ThermoML property labels.

Provides direct-prediction and Arrhenius-prediction datasets where the
molecular representation is a 1H NMR spectrum (1200-point Lorentzian-broadened
continuous signal) instead of fingerprints.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import KFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shared.config import Paths
from shared.data_utils import (
    canonical_smiles,
    inchi_to_canonical_smiles,
    is_hydrocarbon,
    make_temperature_features,
    fit_arrhenius,
)
from phase2.spectrum_converter import convert_compound


paths = Paths()


def _canon_one(smi):
    """Canonicalize a single SMILES string (module-level for pickling)."""
    from rdkit import Chem
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol)


# ---------------------------------------------------------------------------
# Label loading (mirrors phase1/train_direct_optimal.py load_labels)
# ---------------------------------------------------------------------------

def _load_labels(property_name: str) -> pd.DataFrame:
    """Load ThermoML + chemicals GC labels for a given property."""
    dfs = []

    # ThermoML
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

    # GC expanded
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

    # Dedup: prefer thermoml
    combined["T_round"] = combined["T_K"].round(1)
    combined["priority"] = combined["source"].map({"thermoml": 0, "gc": 1})
    combined = combined.sort_values("priority").drop_duplicates(
        subset=["canonical_smiles", "T_round"], keep="first"
    ).drop(columns=["T_round", "priority"])

    return combined


# ---------------------------------------------------------------------------
# NMR loading and spectrum conversion
# ---------------------------------------------------------------------------

def _load_nmr_spectra(target_smiles: set[str] | None = None) -> dict[str, np.ndarray]:
    """Load NMRexp 1H NMR rows, canonicalize SMILES, pick best entry per
    compound (most peaks), and convert each to a spectrum array.

    Parameters
    ----------
    target_smiles : set of canonical SMILES to filter to (optional).
        If provided, only NMR rows whose raw SMILES canonicalize to a
        member of this set are kept. This avoids canonicalizing 1.4M
        SMILES when we only need a few thousand.

    Returns dict mapping canonical_smiles -> spectrum (1200,).
    """
    print("[NMR] Loading NMRexp parquet...")
    df = pd.read_parquet(paths.nmrexp, columns=["SMILES", "NMR_type", "NMR_processed"])
    df = df[df["NMR_type"] == "1H NMR"].copy()
    df = df.dropna(subset=["NMR_processed", "SMILES"])
    print(f"[NMR] {len(df)} 1H NMR rows, {df['SMILES'].nunique()} unique raw SMILES")

    # Fast pre-filter: only keep NMR rows whose SMILES match targets
    if target_smiles is not None:
        # Pass 1: direct raw SMILES match (most NMRexp SMILES are canonical)
        mask_direct = df["SMILES"].isin(target_smiles)
        print(f"[NMR] {mask_direct.sum()} rows match targets by raw SMILES")

        # Pass 2: canonicalize unmatched NMR SMILES in parallel to find
        # non-canonical forms of target molecules.
        remaining_targets = target_smiles - set(df.loc[mask_direct, "SMILES"].unique())
        if remaining_targets:
            unmatched_raw = df.loc[~mask_direct, "SMILES"].unique()
            print(f"[NMR] Canonicalizing {len(unmatched_raw)} unmatched SMILES (parallel)...")

            from multiprocessing import Pool

            with Pool(processes=8) as pool:
                results = pool.map(_canon_one, unmatched_raw, chunksize=10000)

            canon_map = {}
            for raw_smi, canon in zip(unmatched_raw, results):
                if canon is not None and canon in remaining_targets:
                    canon_map[raw_smi] = canon

            print(f"[NMR] {len(canon_map)} additional matches from full canonicalization")
            mask_mapped = df["SMILES"].isin(canon_map)
        else:
            mask_mapped = pd.Series(False, index=df.index)
            canon_map = {}

        # Build canonical column
        df["canonical_smiles"] = df["SMILES"].copy()
        df.loc[mask_mapped, "canonical_smiles"] = df.loc[mask_mapped, "SMILES"].map(canon_map)
        df = df[mask_direct | mask_mapped]
        df = df.dropna(subset=["canonical_smiles"])
    else:
        # No filter: canonicalize everything (slow!)
        df["canonical_smiles"] = df["SMILES"].apply(canonical_smiles)
        df = df.dropna(subset=["canonical_smiles"])

    # Count peaks per entry to select the best one per compound
    def _count_peaks(s):
        try:
            import ast as _ast
            return len(_ast.literal_eval(s))
        except Exception:
            return 0

    df["n_peaks"] = df["NMR_processed"].apply(_count_peaks)

    # Keep entry with most peaks per compound
    df = df.sort_values("n_peaks", ascending=False).drop_duplicates(
        subset=["canonical_smiles"], keep="first"
    )

    print(f"[NMR] {len(df)} unique compounds with matched 1H NMR data")

    # Convert to spectra
    spectra = {}
    n_fail = 0
    for i, (_, row) in enumerate(df.iterrows()):
        smi = row["canonical_smiles"]
        spec = convert_compound(row["NMR_processed"], nmr_type="1H NMR")
        if spec is not None:
            spectra[smi] = spec
        else:
            n_fail += 1
        if (i + 1) % 100 == 0:
            print(f"  Converting spectra: {i+1}/{len(df)}...", flush=True)

    print(f"[NMR] {len(spectra)} spectra converted, {n_fail} failures")
    return spectra


# ---------------------------------------------------------------------------
# Build joined dataset
# ---------------------------------------------------------------------------

def build_nmr_property_dataset(property_name: str = "viscosity") -> dict:
    """Build dataset joining NMR spectra with property labels.

    Parameters
    ----------
    property_name : 'viscosity' or 'surface_tension'

    Returns
    -------
    dict with keys:
        spectra : np.ndarray of shape (n_compounds, 1200)
        labels_df : pd.DataFrame with columns [canonical_smiles, T_K, value, source]
        smiles_list : list of canonical SMILES (compound order matches spectra rows)
        hc_mask : np.ndarray of bool, True for hydrocarbons
    """
    # Load property labels FIRST (small, fast)
    labels = _load_labels(property_name)
    print(f"[Labels {property_name}] {len(labels)} rows, "
          f"{labels['canonical_smiles'].nunique()} compounds")

    # Load NMR spectra, filtering to only compounds with labels
    target_smiles = set(labels["canonical_smiles"].unique())
    spectra_dict = _load_nmr_spectra(target_smiles=target_smiles)

    # Inner join: keep only compounds that have both NMR and labels
    nmr_smiles = set(spectra_dict.keys())
    label_smiles = set(labels["canonical_smiles"].unique())
    common_smiles = sorted(nmr_smiles & label_smiles)

    labels_joined = labels[labels["canonical_smiles"].isin(common_smiles)].copy()
    labels_joined = labels_joined.reset_index(drop=True)

    # Build aligned arrays
    smiles_list = common_smiles
    smi_to_idx = {s: i for i, s in enumerate(smiles_list)}
    spectra_arr = np.stack([spectra_dict[s] for s in smiles_list])
    hc_mask = np.array([is_hydrocarbon(s) for s in smiles_list])

    print(f"[Joined] {len(smiles_list)} compounds with both NMR + {property_name}")
    print(f"  Hydrocarbons: {hc_mask.sum()}")
    print(f"  Measurement rows: {len(labels_joined)}")

    return {
        "spectra": spectra_arr,
        "labels_df": labels_joined,
        "smiles_list": smiles_list,
        "hc_mask": hc_mask,
        "smi_to_idx": smi_to_idx,
    }


# ---------------------------------------------------------------------------
# PyTorch Datasets
# ---------------------------------------------------------------------------

class NMRDirectDataset(Dataset):
    """Dataset for direct property prediction: one sample per (compound, T) measurement.

    Returns (spectrum_tensor [1, 1200], t_features [3], target [1]).
    """

    def __init__(
        self,
        spectra: np.ndarray,
        labels_df: pd.DataFrame,
        smi_to_idx: dict[str, int],
        property_name: str = "viscosity",
        training: bool = False,
    ):
        self.spectra = spectra
        self.smi_to_idx = smi_to_idx
        self.property_name = property_name
        self.training = training

        # Build per-row arrays
        self.compound_indices = []
        self.t_features = []
        self.targets = []

        for _, row in labels_df.iterrows():
            smi = row["canonical_smiles"]
            if smi not in smi_to_idx:
                continue
            cidx = smi_to_idx[smi]
            T_K = row["T_K"]
            value = row["value"]

            self.compound_indices.append(cidx)
            t_feat = make_temperature_features(np.array([T_K]))[0]
            self.t_features.append(t_feat)

            if property_name == "viscosity":
                target = np.log(value)
            else:
                target = value
            self.targets.append(target)

        self.compound_indices = np.array(self.compound_indices)
        self.t_features = np.array(self.t_features, dtype=np.float32)
        self.targets = np.array(self.targets, dtype=np.float32).reshape(-1, 1)

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        cidx = self.compound_indices[idx]
        spec = self.spectra[cidx].copy()

        # Data augmentation during training
        if self.training:
            spec = self._augment(spec)

        # Add channel dimension: (1200,) -> (1, 1200)
        spectrum_tensor = torch.from_numpy(spec).float().unsqueeze(0)
        t_features = torch.from_numpy(self.t_features[idx]).float()
        target = torch.from_numpy(self.targets[idx]).float()

        return spectrum_tensor, t_features, target

    @staticmethod
    def _augment(spec: np.ndarray) -> np.ndarray:
        """Spectrum augmentation: shift + noise + scaling + baseline drift.

        Aggressive augmentation to compensate for small dataset size.
        """
        # Random horizontal shift (up to 5 grid points ~ 0.05 ppm)
        shift = np.random.randint(-5, 6)
        if shift != 0:
            spec = np.roll(spec, shift)
            if shift > 0:
                spec[:shift] = 0
            else:
                spec[shift:] = 0

        # Additive Gaussian noise (stronger)
        noise_std = np.random.uniform(0.005, 0.02)
        noise = np.random.normal(0, noise_std, spec.shape)
        spec = spec + noise

        # Random intensity scaling (wider range)
        scale = np.random.uniform(0.8, 1.2)
        spec = spec * scale

        # Random baseline drift (low-frequency polynomial)
        if np.random.random() < 0.5:
            x = np.linspace(-1, 1, len(spec))
            degree = np.random.randint(1, 4)
            coeffs = np.random.randn(degree + 1) * 0.015
            drift = np.polyval(coeffs, x) * (np.max(np.abs(spec)) + 1e-10)
            spec = spec + drift

        # Random peak dropout (zero a small ppm region)
        if np.random.random() < 0.2:
            center = np.random.randint(0, len(spec))
            width = np.random.randint(5, 20)
            lo = max(0, center - width // 2)
            hi = min(len(spec), center + width // 2)
            spec[lo:hi] = 0

        # Re-clip and normalize
        spec = np.clip(spec, 0, None)
        peak = spec.max()
        if peak > 0:
            spec /= peak

        return spec.astype(np.float32)


class NMRArrheniusDataset(Dataset):
    """Dataset for Arrhenius prediction: one sample per compound.

    Fits ln(eta) = A + B/T to each compound's viscosity data.
    Returns (spectrum_tensor [1, 1200], target_AB [2]).
    """

    def __init__(
        self,
        spectra: np.ndarray,
        labels_df: pd.DataFrame,
        smi_to_idx: dict[str, int],
        min_points: int = 5,
        r2_threshold: float = 0.95,
    ):
        self.spectra = spectra
        self.compound_indices = []
        self.targets = []
        self.smiles = []

        for smi, group in labels_df.groupby("canonical_smiles"):
            if smi not in smi_to_idx:
                continue
            if len(group) < min_points:
                continue

            T_K = group["T_K"].values
            eta = group["value"].values

            if np.any(eta <= 0):
                continue

            A, B, r2 = fit_arrhenius(T_K, eta)
            if r2 < r2_threshold:
                continue

            self.compound_indices.append(smi_to_idx[smi])
            self.targets.append([A, B])
            self.smiles.append(smi)

        self.compound_indices = np.array(self.compound_indices)
        self.targets = np.array(self.targets, dtype=np.float32)
        print(f"[Arrhenius] {len(self.targets)} compounds pass filters "
              f"(min_points={min_points}, r2>={r2_threshold})")

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        cidx = self.compound_indices[idx]
        spec = self.spectra[cidx]

        spectrum_tensor = torch.from_numpy(spec).float().unsqueeze(0)
        target_AB = torch.from_numpy(self.targets[idx]).float()

        return spectrum_tensor, target_AB


# ---------------------------------------------------------------------------
# Cross-validation splitting
# ---------------------------------------------------------------------------

def split_compounds(
    smiles_list: list[str],
    hc_mask: np.ndarray,
    n_folds: int = 5,
    seed: int = 42,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """KFold split at compound level.

    Parameters
    ----------
    smiles_list : list of canonical SMILES
    hc_mask : boolean mask (not used for splitting, but available for stratification)
    n_folds : number of folds
    seed : random seed

    Returns
    -------
    list of (train_idx, test_idx) tuples (indices into smiles_list)
    """
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    folds = []
    for train_idx, test_idx in kf.split(smiles_list):
        folds.append((train_idx, test_idx))
    return folds


# ---------------------------------------------------------------------------
# Main: build and print statistics
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    for prop in ["viscosity", "surface_tension"]:
        print(f"\n{'='*60}")
        print(f"  {prop.upper()}")
        print(f"{'='*60}")

        data = build_nmr_property_dataset(prop)
        spectra = data["spectra"]
        labels_df = data["labels_df"]
        smiles_list = data["smiles_list"]
        hc_mask = data["hc_mask"]
        smi_to_idx = data["smi_to_idx"]

        print(f"\n  Compounds with NMR + {prop}: {len(smiles_list)}")
        print(f"  Hydrocarbons: {hc_mask.sum()}")
        print(f"  Non-hydrocarbons: {(~hc_mask).sum()}")
        print(f"  Total measurement rows: {len(labels_df)}")
        print(f"  Spectra shape: {spectra.shape}")

        # Test direct dataset
        ds = NMRDirectDataset(spectra, labels_df, smi_to_idx, prop)
        print(f"\n  NMRDirectDataset: {len(ds)} samples")
        spec_t, t_feat, target = ds[0]
        print(f"    spectrum: {spec_t.shape}, t_features: {t_feat.shape}, target: {target.shape}")

        # Test Arrhenius dataset (viscosity only)
        if prop == "viscosity":
            arr_ds = NMRArrheniusDataset(spectra, labels_df, smi_to_idx)
            print(f"  NMRArrheniusDataset: {len(arr_ds)} samples")
            if len(arr_ds) > 0:
                spec_t, ab = arr_ds[0]
                print(f"    spectrum: {spec_t.shape}, target_AB: {ab.shape}")

        # Test splitting
        folds = split_compounds(smiles_list, hc_mask)
        print(f"\n  5-fold split:")
        for i, (tr, te) in enumerate(folds):
            n_hc_te = hc_mask[te].sum()
            print(f"    Fold {i+1}: train={len(tr)}, test={len(te)} ({n_hc_te} HC)")
