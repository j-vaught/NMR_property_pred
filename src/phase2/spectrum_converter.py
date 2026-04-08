"""
Convert NMRexp peak lists to fixed-length continuous spectra via Lorentzian broadening.

NMRexp data format (from NMRexp_1H_13C.parquet, column NMR_processed):
  1H NMR: string repr of list of tuples
    ('multiplicity', ['J_couplings'], 'integral_str', shift_high_ppm, shift_low_ppm)
  13C NMR: string repr of list of tuples
    (shift_ppm, multiplicity_or_None, J_or_None)
"""

import ast
import re

import numpy as np


# ---------------------------------------------------------------------------
# Multiplicity -> number of lines
# ---------------------------------------------------------------------------
_MULT_LINES = {
    "s": 1, "d": 2, "t": 3, "q": 4, "quint": 5, "sext": 6, "sept": 7,
    "dd": 4, "dt": 6, "dq": 8, "td": 6, "tt": 9, "ddd": 8,
    "m": None,  # handled separately
    "br s": 1, "br d": 2, "brs": 1, "brd": 2,
}


def _parse_integral(integral_str: str) -> float:
    """Parse integral string like '2H' -> 2.0, '0.5H' -> 0.5."""
    match = re.match(r"([0-9]*\.?[0-9]+)\s*[Hh]", integral_str.strip())
    if match:
        return float(match.group(1))
    return 1.0


def _parse_j_value(j_str: str) -> float:
    """Parse J-coupling string like '3.8Hz' -> 3.8."""
    match = re.match(r"([0-9]*\.?[0-9]+)", j_str.strip())
    if match:
        return float(match.group(1))
    return 0.0


# ---------------------------------------------------------------------------
# 1H NMR parsing
# ---------------------------------------------------------------------------

def parse_1h_peaks(nmr_processed_str: str) -> list[dict] | None:
    """Parse 1H NMR_processed string into list of peak dicts.

    Returns list of dicts with keys:
        center_ppm, integral, multiplicity, j_couplings_hz
    Returns None on parse failure.
    """
    try:
        raw = ast.literal_eval(nmr_processed_str)
    except (ValueError, SyntaxError):
        return None

    if not isinstance(raw, list) or len(raw) == 0:
        return None

    peaks = []
    for entry in raw:
        try:
            mult, j_list, integral_str, shift_high, shift_low = entry
            center = (float(shift_high) + float(shift_low)) / 2.0
            integral = _parse_integral(str(integral_str))
            j_hz = [_parse_j_value(j) for j in j_list] if j_list else []
            peaks.append({
                "center_ppm": center,
                "integral": integral,
                "multiplicity": str(mult).lower().strip(),
                "j_couplings_hz": j_hz,
                "shift_high": float(shift_high),
                "shift_low": float(shift_low),
            })
        except (ValueError, TypeError):
            continue

    return peaks if peaks else None


# ---------------------------------------------------------------------------
# 13C NMR parsing
# ---------------------------------------------------------------------------

def parse_13c_peaks(nmr_processed_str: str) -> list[dict] | None:
    """Parse 13C NMR_processed string into list of peak dicts.

    Returns list of dicts with keys:
        shift_ppm, multiplicity, j_hz
    Returns None on parse failure.
    """
    try:
        raw = ast.literal_eval(nmr_processed_str)
    except (ValueError, SyntaxError):
        return None

    if not isinstance(raw, list) or len(raw) == 0:
        return None

    peaks = []
    for entry in raw:
        try:
            shift, mult, j = entry
            peaks.append({
                "shift_ppm": float(shift),
                "multiplicity": str(mult).lower().strip() if mult else None,
                "j_hz": float(j) if j is not None else None,
            })
        except (ValueError, TypeError):
            continue

    return peaks if peaks else None


# ---------------------------------------------------------------------------
# Spectrum generation: 1H
# ---------------------------------------------------------------------------

def _lorentzian(x: np.ndarray, center: float, gamma: float, intensity: float = 1.0) -> np.ndarray:
    """Lorentzian lineshape: L(x) = intensity * (gamma/2)^2 / ((x - center)^2 + (gamma/2)^2)."""
    half_gamma = gamma / 2.0
    return intensity * half_gamma ** 2 / ((x - center) ** 2 + half_gamma ** 2)


def _gaussian(x: np.ndarray, center: float, sigma: float, intensity: float = 1.0) -> np.ndarray:
    """Gaussian lineshape for multiplet envelopes."""
    return intensity * np.exp(-0.5 * ((x - center) / sigma) ** 2)


def peaks_to_spectrum_1h(
    peaks: list[dict],
    ppm_grid: np.ndarray,
    gamma: float = 0.01,
    spec_freq_mhz: float = 400.0,
) -> np.ndarray:
    """Generate 1H spectrum from parsed peaks via Lorentzian broadening.

    Parameters
    ----------
    peaks : list of dicts from parse_1h_peaks
    ppm_grid : 1-D array of ppm values
    gamma : Lorentzian linewidth in ppm
    spec_freq_mhz : spectrometer frequency for J-coupling -> ppm conversion

    Returns
    -------
    np.ndarray : spectrum values on ppm_grid, max-normalized to [0, 1]
    """
    spectrum = np.zeros_like(ppm_grid, dtype=np.float64)

    for peak in peaks:
        center = peak["center_ppm"]
        intensity = peak["integral"]
        mult = peak["multiplicity"]
        j_hz = peak["j_couplings_hz"]

        if mult == "m":
            # Multiplet with chemical shift range: use Gaussian envelope
            shift_range = abs(peak.get("shift_high", center) - peak.get("shift_low", center))
            if shift_range > 1e-4:
                sigma = max(shift_range / 2.0, gamma)
                spectrum += _gaussian(ppm_grid, center, sigma, intensity)
            else:
                # Degenerate multiplet, treat as broad singlet
                spectrum += _lorentzian(ppm_grid, center, gamma * 3, intensity)
            continue

        # Determine number of lines from multiplicity
        n_lines = _MULT_LINES.get(mult, 1)
        if n_lines is None:
            n_lines = 1

        # Simple first-order splitting patterns
        if n_lines == 1 or not j_hz:
            # Singlet or no J-coupling info
            spectrum += _lorentzian(ppm_grid, center, gamma, intensity)
        elif len(j_hz) == 1:
            # Single J-coupling: equally split lines with Pascal's triangle intensities
            j_ppm = j_hz[0] / spec_freq_mhz
            # Pascal's triangle coefficients for n_lines
            coeffs = [1]
            for _ in range(n_lines - 1):
                coeffs = [1] + [coeffs[k] + coeffs[k + 1] for k in range(len(coeffs) - 1)] + [1]
            total_coeff = sum(coeffs)
            for i, c in enumerate(coeffs):
                offset = (i - (n_lines - 1) / 2.0) * j_ppm
                line_intensity = intensity * c / total_coeff
                spectrum += _lorentzian(ppm_grid, center + offset, gamma, line_intensity)
        else:
            # Multiple J-couplings (dd, dt, dq, etc.): successive splitting
            # Start with single line at center
            lines = [(center, intensity)]
            for j_val in j_hz:
                j_ppm = j_val / spec_freq_mhz
                new_lines = []
                for pos, inten in lines:
                    # Split into doublet around each existing line
                    new_lines.append((pos - j_ppm / 2.0, inten / 2.0))
                    new_lines.append((pos + j_ppm / 2.0, inten / 2.0))
                lines = new_lines
            for pos, inten in lines:
                spectrum += _lorentzian(ppm_grid, pos, gamma, inten)

    # Max-normalize to [0, 1]
    peak_max = spectrum.max()
    if peak_max > 0:
        spectrum /= peak_max

    return spectrum.astype(np.float32)


# ---------------------------------------------------------------------------
# Spectrum generation: 13C
# ---------------------------------------------------------------------------

def peaks_to_spectrum_13c(
    peaks: list[dict],
    ppm_grid: np.ndarray,
    gamma: float = 0.5,
) -> np.ndarray:
    """Generate 13C spectrum from parsed peaks via Lorentzian broadening.

    Each peak has unit intensity (13C is typically not integrated quantitatively).

    Parameters
    ----------
    peaks : list of dicts from parse_13c_peaks
    ppm_grid : 1-D array of ppm values
    gamma : Lorentzian linewidth in ppm

    Returns
    -------
    np.ndarray : spectrum values on ppm_grid, max-normalized to [0, 1]
    """
    spectrum = np.zeros_like(ppm_grid, dtype=np.float64)

    for peak in peaks:
        shift = peak["shift_ppm"]
        spectrum += _lorentzian(ppm_grid, shift, gamma, intensity=1.0)

    peak_max = spectrum.max()
    if peak_max > 0:
        spectrum /= peak_max

    return spectrum.astype(np.float32)


# ---------------------------------------------------------------------------
# High-level converter
# ---------------------------------------------------------------------------

# Default grids
_1H_GRID = np.linspace(0, 12, 1200)
_13C_GRID = np.linspace(0, 220, 1100)


def convert_compound(
    nmr_processed_str: str,
    nmr_type: str = "1H NMR",
    spec_freq_mhz: float = 400.0,
) -> np.ndarray | None:
    """Convert a single NMR_processed string to a fixed-length spectrum array.

    Parameters
    ----------
    nmr_processed_str : the NMR_processed column value (string repr of list)
    nmr_type : '1H NMR' or '13C NMR'
    spec_freq_mhz : spectrometer frequency (used for 1H J-coupling splitting)

    Returns
    -------
    np.ndarray of shape (1200,) for 1H or (1100,) for 13C, or None on failure.
    """
    if not isinstance(nmr_processed_str, str) or len(nmr_processed_str.strip()) == 0:
        return None

    try:
        if nmr_type == "1H NMR":
            peaks = parse_1h_peaks(nmr_processed_str)
            if peaks is None:
                return None
            return peaks_to_spectrum_1h(peaks, _1H_GRID, spec_freq_mhz=spec_freq_mhz)
        elif nmr_type == "13C NMR":
            peaks = parse_13c_peaks(nmr_processed_str)
            if peaks is None:
                return None
            return peaks_to_spectrum_13c(peaks, _13C_GRID)
        else:
            return None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Test 1H parsing and conversion
    test_1h = "[('m', [], '2H', 7.27, 7.24), ('d', ['3.8Hz'], '2H', 3.68, 3.68), ('s', [], '9H', 1.33, 1.33)]"
    peaks = parse_1h_peaks(test_1h)
    print("1H peaks:", peaks)
    spec = convert_compound(test_1h, "1H NMR")
    print(f"1H spectrum: shape={spec.shape}, min={spec.min():.4f}, max={spec.max():.4f}")
    print(f"  Non-zero points: {(spec > 1e-4).sum()}")

    # Test 13C parsing and conversion
    test_13c = "[(180.0, 'd', 300.8), (167.6, 'd', 22.7), (133.7, None, None)]"
    peaks = parse_13c_peaks(test_13c)
    print("\n13C peaks:", peaks)
    spec = convert_compound(test_13c, "13C NMR")
    print(f"13C spectrum: shape={spec.shape}, min={spec.min():.4f}, max={spec.max():.4f}")
    print(f"  Non-zero points: {(spec > 1e-4).sum()}")
