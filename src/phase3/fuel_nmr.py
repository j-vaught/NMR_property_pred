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


# POSF number mapping for cross-referencing between data sources
FUEL_POSF = {
    "JP-5 POSF 10289": "10289",
    "Jet-A POSF 10325": "10325",
    "HRJ-Camelina POSF 7720": "7720",
    "Shell-SPK POSF 5729": "5729",
    "Sasol-IPK POSF 7629": "7629",
    "Gevo-ATJ POSF 10151": "10151",
}


def _resample_spectrum(ppm: np.ndarray, intensity: np.ndarray,
                       target_grid: np.ndarray = None) -> np.ndarray:
    """Resample a raw NMR spectrum to the standard 1200-point grid.

    Parameters
    ----------
    ppm : array of chemical shift values (may be descending)
    intensity : array of intensity values
    target_grid : target ppm grid (default: 0-12 ppm, 1200 points)

    Returns
    -------
    resampled : array of shape (1200,), max-normalized to [0, 1]
    """
    if target_grid is None:
        target_grid = _1H_GRID

    # Ensure ascending order
    if ppm[0] > ppm[-1]:
        ppm = ppm[::-1]
        intensity = intensity[::-1]

    # Interpolate to target grid
    resampled = np.interp(target_grid, ppm, intensity, left=0, right=0)

    # Clip negative values (noise/background)
    resampled = np.clip(resampled, 0, None)

    # Max-normalize
    peak = resampled.max()
    if peak > 0:
        resampled /= peak

    return resampled.astype(np.float32)


def _extract_pseudo_peaklist(spectrum: np.ndarray, ppm_grid: np.ndarray = None,
                              threshold: float = 0.02) -> list[dict]:
    """Extract a pseudo peak list from a continuous spectrum.

    Since fuel NMR spectra are continuous (not peak lists), we identify
    peaks as local maxima above a threshold and estimate integrals from
    the area under each peak region.

    Parameters
    ----------
    spectrum : max-normalized spectrum (1200 points)
    ppm_grid : ppm grid
    threshold : minimum intensity to consider as a peak

    Returns
    -------
    peaks : list of dicts compatible with extract_peaklist_features()
    """
    if ppm_grid is None:
        ppm_grid = _1H_GRID

    # Find regions above threshold
    above = spectrum > threshold
    if not above.any():
        return [{"center_ppm": 1.0, "integral": 1.0, "multiplicity": "m",
                 "j_couplings_hz": [], "shift_high": 1.0, "shift_low": 1.0}]

    # Find contiguous regions
    transitions = np.diff(above.astype(int))
    starts = np.where(transitions == 1)[0] + 1
    ends = np.where(transitions == -1)[0] + 1

    # Handle edge cases
    if above[0]:
        starts = np.concatenate([[0], starts])
    if above[-1]:
        ends = np.concatenate([ends, [len(spectrum)]])

    peaks = []
    for s, e in zip(starts, ends):
        region = spectrum[s:e]
        region_ppm = ppm_grid[s:e]

        # Peak center: intensity-weighted mean ppm
        total_int = region.sum()
        if total_int > 0:
            center = np.sum(region_ppm * region) / total_int
        else:
            center = region_ppm.mean()

        # Integral: proportional to area under curve, scaled by ppm width
        ppm_step = abs(ppm_grid[1] - ppm_grid[0]) if len(ppm_grid) > 1 else 0.01
        integral = total_int * ppm_step

        # Scale integral to approximate H-count (normalize so total ~ 20 for a typical fuel)
        peaks.append({
            "center_ppm": float(center),
            "integral": float(integral),
            "multiplicity": "m",  # mixtures always appear as complex multiplets
            "j_couplings_hz": [],
            "shift_high": float(region_ppm.max()),
            "shift_low": float(region_ppm.min()),
        })

    # Scale integrals so total ≈ 20 (typical fuel average H-count proxy)
    total = sum(p["integral"] for p in peaks)
    if total > 0:
        scale = 20.0 / total
        for p in peaks:
            p["integral"] *= scale

    return peaks


def load_burnett_fuels() -> dict[str, dict]:
    """Load Burnett fuel NMR spectra and properties.

    Returns dict mapping fuel_name -> {spectrum, spectrum_features, peaklist_features,
                                        viscosity_data, source}
    """
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

        # Resample to standard grid
        spectrum = _resample_spectrum(ppm, intensity)
        spectrum_feats = extract_nmr_features(spectrum)

        # Extract pseudo peak list for peaklist features
        pseudo_peaks = _extract_pseudo_peaklist(spectrum)
        peaklist_feats = extract_peaklist_features(pseudo_peaks)

        fuel = {
            "spectrum": spectrum,
            "spectrum_features": spectrum_feats,
            "peaklist_features": peaklist_feats,
            "source": "burnett",
        }

        # Load viscosity data if available
        visc_file = burnett_dir / nmr_file.replace("_NMR_spectrum.csv", "_experimental_viscosity.csv")
        if visc_file.exists():
            visc_df = pd.read_csv(visc_file)
            fuel["viscosity_data"] = visc_df.values  # [[T_K, visc_cSt], ...]

        fuels[fuel_name] = fuel

    return fuels


def load_parker_fuels() -> dict[str, dict]:
    """Load Parker 2025 fuel NMR spectra.

    Returns dict mapping fuel_name -> {spectrum, spectrum_features, peaklist_features, source}
    """
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

        fuels[fuel_name] = {
            "spectrum": spectrum,
            "spectrum_features": spectrum_feats,
            "peaklist_features": peaklist_feats,
            "source": "parker2025",
        }

    return fuels


def load_fuel_properties() -> dict[str, dict]:
    """Load measured fuel properties from AFRL + Burnett for validation.

    Returns dict mapping fuel_name -> {viscosity: DataFrame, surface_tension: DataFrame,
                                        density: DataFrame}
    """
    props = {}

    # AFRL viscosity
    visc_path = FUEL_DIR / "afrl_extracted" / "afrl_viscosity.csv"
    if visc_path.exists():
        visc = pd.read_csv(visc_path)
        for name in visc["fuel_name"].unique():
            props.setdefault(name, {})["viscosity_afrl"] = visc[visc["fuel_name"] == name]

    # AFRL surface tension
    st_path = FUEL_DIR / "afrl_extracted" / "afrl_surface_tension.csv"
    if st_path.exists():
        st = pd.read_csv(st_path)
        for name in st["fuel_name"].unique():
            props.setdefault(name, {})["surface_tension_afrl"] = st[st["fuel_name"] == name]

    # AFRL density
    dens_path = FUEL_DIR / "afrl_extracted" / "afrl_density.csv"
    if dens_path.exists():
        dens = pd.read_csv(dens_path)
        for name in dens["fuel_name"].unique():
            props.setdefault(name, {})["density_afrl"] = dens[dens["fuel_name"] == name]

    # Burnett JP-5 viscosity
    jp5_visc = FUEL_DIR / "burnett" / "JP5_experimental_viscosity.csv"
    if jp5_visc.exists():
        df = pd.read_csv(jp5_visc)
        props.setdefault("JP-5 POSF 10289", {})["viscosity_burnett"] = df

    return props


def load_all_fuels() -> dict[str, dict]:
    """Load all fuel NMR spectra from both sources.

    Merges Burnett and Parker fuels, preferring Parker for duplicates
    (higher spectral resolution). Attaches measured properties for validation.
    """
    burnett = load_burnett_fuels()
    parker = load_parker_fuels()

    # Merge (Parker takes priority for overlapping fuels)
    all_fuels = {}
    all_fuels.update(burnett)
    all_fuels.update(parker)  # overwrites JP-5, Jet-A, HRJ if present

    # Attach measured properties
    props = load_fuel_properties()
    for fuel_name, fuel_data in all_fuels.items():
        posf = FUEL_POSF.get(fuel_name, "")
        # Match by POSF number in property data
        for prop_name, prop_data in props.items():
            if posf and posf in prop_name:
                fuel_data.update(prop_data)
                break

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
        print(f"    Spectrum: {data['spectrum'].shape}, max={data['spectrum'].max():.3f}")
        print(f"    Spectrum features: {data['spectrum_features'].shape}")
        print(f"    Peaklist features: {data['peaklist_features'].shape}")
        print(f"    Total H proxy: {data['peaklist_features'][0]:.1f}")
        if "viscosity_burnett" in data:
            print(f"    Burnett viscosity: {len(data['viscosity_burnett'])} T points")
        if "viscosity_data" in data:
            print(f"    Burnett viscosity (raw): {data['viscosity_data'].shape}")
        for key in data:
            if "afrl" in key:
                print(f"    {key}: {len(data[key])} rows")
