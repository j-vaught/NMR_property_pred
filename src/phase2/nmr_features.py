"""
Extract fixed-length feature vectors from NMR spectra for use with
traditional ML models (Ridge, XGBoost, etc.).

The raw 1200-point spectrum is too high-dimensional for ~512 compounds.
Instead, we compute domain-informed features:
  - Spectral bins (mean signal in ~30 ppm regions)
  - Peak statistics (count, mean position, spread)
  - Region integrals (aromatic, olefinic, oxygenated, aliphatic)
  - Spectral shape descriptors (moments, entropy)

This parallels Phase 1's feature engineering approach, which worked well
with Lasso feature selection + Ridge/NN models.
"""

import numpy as np
from phase2.spectrum_converter import _1H_GRID


# PPM regions of chemical significance for 1H NMR
_REGIONS = {
    "alkyl_high_field": (0.0, 1.0),      # methyl, shielded CH
    "alkyl_mid": (1.0, 2.0),             # methylene, CH
    "alkyl_alpha_hetero": (2.0, 3.0),    # CH near O/N/halogen
    "oxygenated": (3.0, 4.5),            # OCH, NCH
    "olefinic_low": (4.5, 5.5),          # vinyl, allyl
    "olefinic_high": (5.5, 6.5),         # vinyl
    "aromatic_low": (6.5, 7.5),          # aromatic
    "aromatic_high": (7.5, 8.5),         # aromatic, downfield
    "aldehyde_downfield": (8.5, 12.0),   # aldehyde, carboxylic
}


def extract_nmr_features(spectrum: np.ndarray, ppm_grid: np.ndarray = None) -> np.ndarray:
    """Extract a fixed-length feature vector from a single 1H NMR spectrum.

    Parameters
    ----------
    spectrum : np.ndarray, shape (1200,)
        Max-normalized spectrum on the standard 0-12 ppm grid.
    ppm_grid : np.ndarray, optional
        PPM grid corresponding to spectrum points. Defaults to _1H_GRID.

    Returns
    -------
    features : np.ndarray, shape (n_features,)
        Feature vector. Current implementation yields ~70 features.
    """
    if ppm_grid is None:
        ppm_grid = _1H_GRID

    features = []

    # --- 1. Fine spectral bins (30 bins of 0.4 ppm each, covering 0-12 ppm) ---
    n_bins = 30
    bin_edges = np.linspace(0, 12, n_bins + 1)
    for i in range(n_bins):
        mask = (ppm_grid >= bin_edges[i]) & (ppm_grid < bin_edges[i + 1])
        if mask.any():
            features.append(spectrum[mask].mean())
            features.append(spectrum[mask].max())
        else:
            features.append(0.0)
            features.append(0.0)

    # --- 2. Region integrals and peak counts ---
    for name, (lo, hi) in _REGIONS.items():
        mask = (ppm_grid >= lo) & (ppm_grid < hi)
        region = spectrum[mask] if mask.any() else np.zeros(1)

        # Integral (sum of signal, proportional to number of protons)
        features.append(region.sum())

        # Fraction of total integral
        total = spectrum.sum()
        features.append(region.sum() / (total + 1e-10))

        # Peak count (local maxima above threshold)
        threshold = 0.05
        above = region > threshold
        # Count transitions from below to above threshold
        if len(region) > 1:
            crossings = np.diff(above.astype(int))
            n_peaks = max(0, (crossings == 1).sum())
        else:
            n_peaks = 0
        features.append(float(n_peaks))

    # --- 3. Global spectral statistics ---
    # Spectral centroid (center of mass)
    total = spectrum.sum()
    if total > 0:
        centroid = np.sum(ppm_grid * spectrum) / total
    else:
        centroid = 6.0  # midpoint
    features.append(centroid)

    # Spectral spread (weighted std)
    if total > 0:
        spread = np.sqrt(np.sum((ppm_grid - centroid) ** 2 * spectrum) / total)
    else:
        spread = 0.0
    features.append(spread)

    # Spectral skewness
    if total > 0 and spread > 1e-6:
        skew = np.sum((ppm_grid - centroid) ** 3 * spectrum) / (total * spread ** 3)
    else:
        skew = 0.0
    features.append(skew)

    # Spectral kurtosis
    if total > 0 and spread > 1e-6:
        kurt = np.sum((ppm_grid - centroid) ** 4 * spectrum) / (total * spread ** 4)
    else:
        kurt = 0.0
    features.append(kurt)

    # Total number of significant peaks (above 5% of max)
    threshold = 0.05
    above = spectrum > threshold
    if len(spectrum) > 1:
        crossings = np.diff(above.astype(int))
        total_peaks = max(0, (crossings == 1).sum())
    else:
        total_peaks = 0
    features.append(float(total_peaks))

    # Spectral entropy (normalized)
    spec_pos = spectrum.clip(min=1e-10)
    spec_norm = spec_pos / spec_pos.sum()
    entropy = -np.sum(spec_norm * np.log(spec_norm))
    max_entropy = np.log(len(spectrum))
    features.append(entropy / max_entropy)

    # Fraction of spectrum that is non-zero (above noise floor)
    features.append((spectrum > 0.01).mean())

    # Max peak height (should be ~1.0 for normalized spectra, but varies)
    features.append(spectrum.max())

    return np.array(features, dtype=np.float32)


# ---------------------------------------------------------------------------
# Peak-list features (extracted BEFORE max-normalization)
# ---------------------------------------------------------------------------

_MULT_CATEGORIES = {"s": 0, "d": 1, "t": 2, "q": 3, "m": 4}
# dd, dt, dq, td, tt, ddd, etc. all map to "complex" (5)
# everything else maps to "other" (6)
_N_MULT = 7


def extract_peaklist_features(peaks: list[dict]) -> np.ndarray:
    """Extract features from a parsed 1H NMR peak list (before normalization).

    These features preserve absolute hydrogen count information that is
    lost during max-normalization of the continuous spectrum. Total H count
    is a strong proxy for molecular weight, which is critical for viscosity.

    Parameters
    ----------
    peaks : list of dict
        Output of parse_1h_peaks(). Each dict has keys: center_ppm,
        integral, multiplicity, j_couplings_hz, shift_high, shift_low.

    Returns
    -------
    features : np.ndarray, shape (n_peaklist_features,)
    """
    if not peaks:
        return np.zeros(_n_peaklist_features(), dtype=np.float32)

    features = []
    integrals = [p["integral"] for p in peaks]
    centers = [p["center_ppm"] for p in peaks]
    total_h = sum(integrals)

    # --- Size proxies (3) ---
    features.append(total_h)                       # total hydrogen count
    features.append(float(len(peaks)))             # number of peaks
    features.append(max(integrals))                # max single integral

    # --- Absolute H count per chemical shift region (9) ---
    for name, (lo, hi) in _REGIONS.items():
        region_h = sum(
            p["integral"] for p in peaks
            if lo <= p["center_ppm"] < hi
        )
        features.append(region_h)

    # --- Multiplicity distribution (7) ---
    mult_counts = np.zeros(_N_MULT, dtype=np.float32)
    for p in peaks:
        m = p.get("multiplicity", "s").lower().strip()
        if m in _MULT_CATEGORIES:
            mult_counts[_MULT_CATEGORIES[m]] += 1
        elif len(m) > 1 and m[0] in "dtq":
            mult_counts[5] += 1  # complex
        else:
            mult_counts[6] += 1  # other
    n_peaks = len(peaks)
    for i in range(_N_MULT):
        features.append(mult_counts[i] / n_peaks)

    # --- J-coupling statistics (3) ---
    all_j = []
    for p in peaks:
        all_j.extend(p.get("j_couplings_hz", []))
    if all_j:
        features.append(np.mean(all_j))            # mean J
        features.append(max(all_j))                 # max J
    else:
        features.append(0.0)
        features.append(0.0)
    features.append(
        sum(1 for p in peaks if p.get("j_couplings_hz")) / n_peaks
    )  # fraction with J info

    # --- Derived ratios (3) ---
    alkyl_h = sum(
        p["integral"] for p in peaks if p["center_ppm"] < 2.0
    )
    aromatic_h = sum(
        p["integral"] for p in peaks if 6.5 <= p["center_ppm"] < 8.5
    )
    ch3_region_h = sum(
        p["integral"] for p in peaks if p["center_ppm"] < 1.0
    )
    ch2_region_h = sum(
        p["integral"] for p in peaks if 1.0 <= p["center_ppm"] < 2.0
    )

    features.append(alkyl_h / (total_h + 1e-10))        # alkyl H fraction
    features.append(aromatic_h / (total_h + 1e-10))      # aromatic H fraction
    features.append(
        ch2_region_h / (ch3_region_h + ch2_region_h + 1e-10)
    )  # CH2/CH3 proxy (higher = longer chains)

    return np.array(features, dtype=np.float32)


def _n_peaklist_features() -> int:
    """Number of features returned by extract_peaklist_features."""
    return 3 + len(_REGIONS) + _N_MULT + 3 + 3  # 25


def get_peaklist_feature_names() -> list[str]:
    """Return human-readable names for peak-list features."""
    names = [
        "total_h_count",
        "n_peaks",
        "max_single_integral",
    ]
    for region_name in _REGIONS:
        names.append(f"h_count_{region_name}")
    names.extend([
        "frac_singlet", "frac_doublet", "frac_triplet", "frac_quartet",
        "frac_multiplet", "frac_complex", "frac_other",
    ])
    names.extend([
        "j_coupling_mean", "j_coupling_max", "frac_with_j",
    ])
    names.extend([
        "alkyl_h_fraction", "aromatic_h_fraction", "ch2_ch3_proxy",
    ])
    return names


def extract_features_batch(spectra: np.ndarray, ppm_grid: np.ndarray = None) -> np.ndarray:
    """Extract features for an array of spectra.

    Parameters
    ----------
    spectra : np.ndarray, shape (n_compounds, 1200)
    ppm_grid : np.ndarray, optional

    Returns
    -------
    features : np.ndarray, shape (n_compounds, n_features)
    """
    return np.stack([extract_nmr_features(s, ppm_grid) for s in spectra])


def get_feature_names() -> list[str]:
    """Return human-readable feature names for the extracted features."""
    names = []

    # Fine bins (30 bins x 2 features)
    bin_edges = np.linspace(0, 12, 31)
    for i in range(30):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        names.append(f"bin_{lo:.1f}_{hi:.1f}_mean")
        names.append(f"bin_{lo:.1f}_{hi:.1f}_max")

    # Region features (9 regions x 3 features)
    for name in _REGIONS:
        names.append(f"region_{name}_integral")
        names.append(f"region_{name}_frac")
        names.append(f"region_{name}_peaks")

    # Global stats
    names.extend([
        "spectral_centroid",
        "spectral_spread",
        "spectral_skewness",
        "spectral_kurtosis",
        "total_peaks",
        "spectral_entropy",
        "nonzero_fraction",
        "max_peak_height",
    ])

    return names


if __name__ == "__main__":
    # Quick test
    from phase2.spectrum_converter import convert_compound

    test_1h = "[('m', [], '2H', 7.27, 7.24), ('d', ['3.8Hz'], '2H', 3.68, 3.68), ('s', [], '9H', 1.33, 1.33)]"
    spec = convert_compound(test_1h, "1H NMR")

    features = extract_nmr_features(spec)
    names = get_feature_names()
    print(f"Feature vector: {features.shape[0]} features")
    print(f"Feature names: {len(names)}")
    assert len(names) == len(features), f"Mismatch: {len(names)} names vs {len(features)} features"

    print("\nNon-zero features:")
    for name, val in zip(names, features):
        if abs(val) > 1e-6:
            print(f"  {name:40s} = {val:.4f}")
