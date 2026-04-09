"""
Phase 3: Load and process fuel mixture NMR spectra.

Fuel NMR data comes from two sources:
  - Burnett: 3 fuels (JP-5, Jet-A, HRJ-Camelina) with 32K-65K point spectra
  - Parker 2025: 6 fuels with ~59K point spectra each

All spectra are resampled to the same 1200-point 0-12 ppm grid used in Phase 2
so that the same feature extraction pipeline can be applied.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shared.config import Paths
from phase2.nmr_features import extract_nmr_features, extract_peaklist_features, _REGIONS
from phase2.spectrum_converter import _1H_GRID

paths = Paths()
FUEL_DIR = paths.fuels


# Jameel 2021 NMR functional group shift ranges (ppm)
# These map 1H NMR chemical shift regions to specific functional groups
# and work for both pure compounds AND fuel mixtures.
FUNCTIONAL_GROUPS = {
    "aromatic_H":     (6.42, 8.99),
    "olefinic_H":     (4.50, 6.42),
    "alpha_CH":       (2.88, 3.40),
    "alpha_CH2":      (2.64, 2.88),
    "alpha_CH3":      (2.04, 2.64),
    "naphthenic_H":   (1.57, 1.96),
    "paraffinic_CH":  (1.39, 1.57),
    "paraffinic_CH2": (0.94, 1.39),
    "paraffinic_CH3": (0.25, 0.94),
}


# POSF number mapping for cross-referencing between NMR and AFRL data
# NMR fuel name -> POSF number
FUEL_POSF = {
    "JP-5 POSF 10289": "10289",
    "Jet-A POSF 10325": "10325",
    "HRJ-Camelina POSF 7720": "7720",
    "Shell-SPK POSF 5729": "5729",
    "Sasol-IPK POSF 7629": "7629",
    "Gevo-ATJ POSF 10151": "10151",
}

# Approximate POSF cross-references (same fuel type, different batch)
POSF_APPROX_MATCH = {
    "10151": "11498",   # Gevo ATJ: NMR batch vs AFRL batch
    "7720": "10301",    # HRJ Camelina: Burnett vs AFRL UOP camelina
}


def extract_functional_groups(spectrum: np.ndarray,
                               ppm_grid: np.ndarray = None) -> np.ndarray:
    """Extract Jameel 2021 functional group fractions from a 1H NMR spectrum.

    These features are physically meaningful for both pure compounds and
    fuel mixtures, as they represent the hydrogen distribution across
    chemical environments.

    Parameters
    ----------
    spectrum : max-normalized spectrum (1200 points)
    ppm_grid : ppm grid (default: 0-12 ppm, 1200 points)

    Returns
    -------
    features : array of shape (9,) — fractional integral in each FG region
    """
    if ppm_grid is None:
        ppm_grid = _1H_GRID

    total = spectrum.sum()
    if total <= 0:
        return np.zeros(len(FUNCTIONAL_GROUPS), dtype=np.float32)

    features = []
    for name, (lo, hi) in FUNCTIONAL_GROUPS.items():
        mask = (ppm_grid >= lo) & (ppm_grid <= hi)
        region_integral = spectrum[mask].sum() if mask.any() else 0.0
        features.append(region_integral / total)

    return np.array(features, dtype=np.float32)


def get_functional_group_names() -> list[str]:
    return [f"fg_{name}" for name in FUNCTIONAL_GROUPS]


def _resample_spectrum(ppm: np.ndarray, intensity: np.ndarray,
                       target_grid: np.ndarray = None) -> np.ndarray:
    """Resample a raw NMR spectrum to the standard 1200-point grid."""
    if target_grid is None:
        target_grid = _1H_GRID

    if ppm[0] > ppm[-1]:
        ppm = ppm[::-1]
        intensity = intensity[::-1]

    resampled = np.interp(target_grid, ppm, intensity, left=0, right=0)
    resampled = np.clip(resampled, 0, None)

    peak = resampled.max()
    if peak > 0:
        resampled /= peak

    return resampled.astype(np.float32)


def _extract_pseudo_peaklist(spectrum: np.ndarray, ppm_grid: np.ndarray = None,
                              threshold: float = 0.02) -> list[dict]:
    """Extract a pseudo peak list from a continuous spectrum.

    Since fuel NMR spectra are continuous (not discrete peak lists), we
    identify peaks as contiguous regions above threshold and estimate
    integrals from the area under each region.
    """
    if ppm_grid is None:
        ppm_grid = _1H_GRID

    above = spectrum > threshold
    if not above.any():
        return [{"center_ppm": 1.0, "integral": 1.0, "multiplicity": "m",
                 "j_couplings_hz": [], "shift_high": 1.0, "shift_low": 1.0}]

    transitions = np.diff(above.astype(int))
    starts = np.where(transitions == 1)[0] + 1
    ends = np.where(transitions == -1)[0] + 1

    if above[0]:
        starts = np.concatenate([[0], starts])
    if above[-1]:
        ends = np.concatenate([ends, [len(spectrum)]])

    peaks = []
    ppm_step = abs(ppm_grid[1] - ppm_grid[0]) if len(ppm_grid) > 1 else 0.01

    for s, e in zip(starts, ends):
        region = spectrum[s:e]
        region_ppm = ppm_grid[s:e]

        total_int = region.sum()
        if total_int > 0:
            center = np.sum(region_ppm * region) / total_int
        else:
            center = region_ppm.mean()

        integral = total_int * ppm_step

        peaks.append({
            "center_ppm": float(center),
            "integral": float(integral),
            "multiplicity": "m",
            "j_couplings_hz": [],
            "shift_high": float(region_ppm.max()),
            "shift_low": float(region_ppm.min()),
        })

    # Scale integrals so total ~ 20 (typical fuel H-count proxy)
    total = sum(p["integral"] for p in peaks)
    if total > 0:
        scale = 20.0 / total
        for p in peaks:
            p["integral"] *= scale

    return peaks


def load_burnett_fuels() -> dict[str, dict]:
    """Load Burnett fuel NMR spectra and properties."""
    burnett_dir = FUEL_DIR / "burnett"
    fuels = {}

    fuel_files = {
        "JP-5 POSF 10289": "JP5_NMR_spectrum.csv",
        "Jet-A POSF 10325": "JetA_POSF10325_NMR_spectrum.csv",
        "HRJ-Camelina POSF 7720": "HRJ_POSF7720_NMR_spectrum.csv",
    }

    for fuel_name, nmr_file in fuel_files.items():
        path = burnett_dir / nmr_file
        if not path.exists():
            continue

        df = pd.read_csv(path)
        ppm = df.iloc[:, 0].values
        intensity = df.iloc[:, 1].values

        spectrum = _resample_spectrum(ppm, intensity)
        spectrum_feats = extract_nmr_features(spectrum)
        pseudo_peaks = _extract_pseudo_peaklist(spectrum)
        peaklist_feats = extract_peaklist_features(pseudo_peaks)
        fg_feats = extract_functional_groups(spectrum)

        fuel = {
            "spectrum": spectrum,
            "spectrum_features": spectrum_feats,
            "peaklist_features": peaklist_feats,
            "fg_features": fg_feats,
            "source": "burnett",
        }

        # Load viscosity data if available
        visc_file = burnett_dir / nmr_file.replace("_NMR_spectrum.csv", "_experimental_viscosity.csv")
        if visc_file.exists():
            vd = pd.read_csv(visc_file)
            fuel["measured_viscosity"] = {
                "T_K": vd.iloc[:, 0].values,
                "viscosity_cSt": vd.iloc[:, 1].values,
                "source": "burnett_experimental",
            }

        fuels[fuel_name] = fuel

    return fuels


def load_parker_fuels() -> dict[str, dict]:
    """Load Parker 2025 fuel NMR spectra."""
    nmr_path = FUEL_DIR / "parker2025_extracted" / "nmr_spectra.csv"
    if not nmr_path.exists():
        return {}

    df = pd.read_csv(nmr_path)
    fuels = {}

    for fuel_name in df["fuel_name"].unique():
        fuel_df = df[df["fuel_name"] == fuel_name].sort_values("chemical_shift_ppm")
        ppm = fuel_df["chemical_shift_ppm"].values
        intensity = fuel_df["intensity_rel"].values

        spectrum = _resample_spectrum(ppm, intensity)
        spectrum_feats = extract_nmr_features(spectrum)
        pseudo_peaks = _extract_pseudo_peaklist(spectrum)
        peaklist_feats = extract_peaklist_features(pseudo_peaks)
        fg_feats = extract_functional_groups(spectrum)

        fuels[fuel_name] = {
            "spectrum": spectrum,
            "spectrum_features": spectrum_feats,
            "peaklist_features": peaklist_feats,
            "fg_features": fg_feats,
            "source": "parker2025",
        }

    return fuels


def _match_afrl_by_posf(fuel_posf: str, afrl_df: pd.DataFrame) -> pd.DataFrame | None:
    """Match AFRL data rows by POSF number (exact or approximate)."""
    # Try exact match
    mask = afrl_df["POSF_number"].astype(str).str.strip() == fuel_posf
    if mask.any():
        return afrl_df[mask]

    # Try float match (some POSF stored as float)
    try:
        posf_float = float(fuel_posf)
        mask = afrl_df["POSF_number"] == posf_float
        if mask.any():
            return afrl_df[mask]
    except ValueError:
        pass

    # Try approximate match
    approx = POSF_APPROX_MATCH.get(fuel_posf)
    if approx:
        return _match_afrl_by_posf(approx, afrl_df)

    return None


def load_afrl_properties() -> dict:
    """Load all AFRL measured properties, indexed by POSF."""
    afrl_dir = FUEL_DIR / "afrl_extracted"
    data = {}

    for prop_name, filename, val_col in [
        ("viscosity", "afrl_viscosity.csv", "viscosity_cSt"),
        ("surface_tension", "afrl_surface_tension.csv", "surface_tension_mNm"),
        ("density", "afrl_density.csv", "density_kgm3"),
    ]:
        path = afrl_dir / filename
        if path.exists():
            data[prop_name] = pd.read_csv(path)

    return data


def attach_measured_properties(fuels: dict[str, dict]) -> None:
    """Attach AFRL measured properties to fuels by POSF matching."""
    afrl = load_afrl_properties()

    for fuel_name, fuel_data in fuels.items():
        posf = FUEL_POSF.get(fuel_name)
        if not posf:
            continue

        # Viscosity
        if "viscosity" in afrl:
            matched = _match_afrl_by_posf(posf, afrl["viscosity"])
            if matched is not None and len(matched) > 0:
                fuel_data["measured_viscosity_afrl"] = {
                    "T_K": matched["T_K"].values,
                    "viscosity_cSt": matched["viscosity_cSt"].values,
                    "source": "AFRL_Edwards_2020",
                }

        # Surface tension
        if "surface_tension" in afrl:
            matched = _match_afrl_by_posf(posf, afrl["surface_tension"])
            if matched is not None and len(matched) > 0:
                fuel_data["measured_surface_tension"] = {
                    "T_K": matched["T_K"].values,
                    "surface_tension_mNm": matched["surface_tension_mNm"].values,
                    "source": "AFRL_Edwards_2020",
                }

        # Density (needed for viscosity unit conversion)
        if "density" in afrl:
            matched = _match_afrl_by_posf(posf, afrl["density"])
            if matched is not None and len(matched) > 0:
                fuel_data["measured_density"] = {
                    "T_K": matched["T_K"].values,
                    "density_kgm3": matched["density_kgm3"].values,
                    "source": "AFRL_Edwards_2020",
                }


def load_all_fuels() -> dict[str, dict]:
    """Load all fuel NMR spectra and attach measured properties."""
    burnett = load_burnett_fuels()
    parker = load_parker_fuels()

    all_fuels = {}
    all_fuels.update(burnett)
    all_fuels.update(parker)  # Parker overwrites for overlapping fuels

    attach_measured_properties(all_fuels)

    return all_fuels


if __name__ == "__main__":
    print("=" * 60)
    print("  Phase 3: Fuel NMR Data Loading")
    print("=" * 60)

    fuels = load_all_fuels()
    for name, data in fuels.items():
        posf = FUEL_POSF.get(name, "?")
        print(f"\n  {name} (POSF {posf}):")
        print(f"    Source: {data['source']}")
        print(f"    Spectrum: {data['spectrum'].shape}")
        print(f"    FG fractions: {dict(zip(get_functional_group_names(), data['fg_features']))}")

        for key in data:
            if "measured" in key:
                mdata = data[key]
                print(f"    {key}: {len(mdata[list(mdata.keys())[0]])} T points ({mdata['source']})")
