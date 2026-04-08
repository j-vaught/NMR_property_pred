"""
Phase 2: NMR-specific data augmentation for training.

Augmentations operate on peak lists (before spectrum generation) or on
generated spectra (after). All augmentations are stochastic and controlled
by per-transform probabilities so they can be disabled at eval time.
"""

import numpy as np


# Common NMR solvent regions (ppm) to zero out during augmentation
SOLVENT_REGIONS = {
    "CDCl3": (7.24, 7.28),
    "DMSO": (2.49, 2.51),
    "D2O": (4.65, 4.85),
}


class PeakAugmentation:
    """Augmentations applied to peak lists before spectrum generation.

    Each peak is represented as a dict with at least:
        - 'center': chemical shift in ppm
        - 'intensity': relative intensity (0-1)
    Additional keys (multiplicity, J-coupling, etc.) are preserved.

    Parameters
    ----------
    p_shift : float
        Probability of applying shift perturbation.
    p_linewidth : float
        Probability of returning a randomized linewidth.
    p_intensity : float
        Probability of applying intensity jitter.
    p_dropout : float
        Probability of applying peak dropout.
    """

    def __init__(self, p_shift=0.8, p_linewidth=0.8, p_intensity=0.8, p_dropout=0.3):
        self.p_shift = p_shift
        self.p_linewidth = p_linewidth
        self.p_intensity = p_intensity
        self.p_dropout = p_dropout

    def shift_perturbation(self, peaks, max_shift=0.05):
        """Add random +/- max_shift ppm to all peak centers."""
        augmented = []
        for peak in peaks:
            p = dict(peak)
            p["center"] = p["center"] + np.random.uniform(-max_shift, max_shift)
            augmented.append(p)
        return augmented

    def linewidth_variation(self):
        """Return a random gamma sampled from U(0.005, 0.02) instead of fixed 0.01."""
        return np.random.uniform(0.005, 0.02)

    def intensity_jitter(self, peaks, factor=0.2):
        """Multiply each peak intensity by U(1 - factor, 1 + factor)."""
        augmented = []
        for peak in peaks:
            p = dict(peak)
            jitter = np.random.uniform(1.0 - factor, 1.0 + factor)
            p["intensity"] = p["intensity"] * jitter
            augmented.append(p)
        return augmented

    def peak_dropout(self, peaks, p=0.1):
        """Randomly drop peaks with probability p. Always keep at least one peak."""
        if len(peaks) <= 1:
            return list(peaks)
        kept = [pk for pk in peaks if np.random.random() > p]
        if len(kept) == 0:
            # Keep at least one peak (random)
            kept = [peaks[np.random.randint(len(peaks))]]
        return kept

    def __call__(self, peaks):
        """Apply all augmentations with configured probabilities.

        Parameters
        ----------
        peaks : list of dict
            Peak list with 'center' and 'intensity' keys.

        Returns
        -------
        augmented_peaks : list of dict
            Augmented peak list.
        gamma : float
            Linewidth to use for spectrum generation.
        """
        aug = list(peaks)

        if np.random.random() < self.p_dropout:
            aug = self.peak_dropout(aug)

        if np.random.random() < self.p_shift:
            aug = self.shift_perturbation(aug)

        if np.random.random() < self.p_intensity:
            aug = self.intensity_jitter(aug)

        gamma = self.linewidth_variation() if np.random.random() < self.p_linewidth else 0.01

        return aug, gamma


class SpectrumAugmentation:
    """Augmentations applied to generated 1D spectra (numpy arrays).

    Parameters
    ----------
    p_noise : float
        Probability of applying Gaussian noise.
    p_baseline : float
        Probability of applying baseline drift.
    p_solvent : float
        Probability of applying solvent region dropout.
    """

    def __init__(self, p_noise=0.8, p_baseline=0.5, p_solvent=0.3):
        self.p_noise = p_noise
        self.p_baseline = p_baseline
        self.p_solvent = p_solvent

    def gaussian_noise(self, spectrum, sigma_frac=0.01):
        """Add Gaussian noise with std = sigma_frac * max(|spectrum|)."""
        sigma = sigma_frac * np.max(np.abs(spectrum) + 1e-10)
        noise = np.random.normal(0.0, sigma, size=spectrum.shape)
        return spectrum + noise

    def baseline_drift(self, spectrum, max_degree=3):
        """Add a random low-order polynomial baseline drift.

        The polynomial degree is drawn from U(1, max_degree).
        Coefficients are scaled so the drift magnitude is at most
        ~5% of the spectrum's max amplitude.
        """
        n = len(spectrum)
        x = np.linspace(-1, 1, n)
        degree = np.random.randint(1, max_degree + 1)
        coeffs = np.random.randn(degree + 1) * 0.02
        drift = np.polyval(coeffs, x)
        # Scale drift relative to spectrum magnitude
        spec_max = np.max(np.abs(spectrum)) + 1e-10
        drift = drift * spec_max
        return spectrum + drift

    def solvent_dropout(self, spectrum, ppm_grid):
        """Zero out common solvent regions in the spectrum.

        Solvent regions: CDCl3 (7.24-7.28), DMSO (2.49-2.51), D2O (4.65-4.85).
        """
        spectrum = spectrum.copy()
        for name, (lo, hi) in SOLVENT_REGIONS.items():
            mask = (ppm_grid >= lo) & (ppm_grid <= hi)
            spectrum[mask] = 0.0
        return spectrum

    def __call__(self, spectrum, ppm_grid):
        """Apply all spectrum augmentations with configured probabilities.

        Parameters
        ----------
        spectrum : np.ndarray, shape (N,)
            Generated 1D NMR spectrum.
        ppm_grid : np.ndarray, shape (N,)
            Chemical shift grid in ppm.

        Returns
        -------
        augmented : np.ndarray, shape (N,)
        """
        aug = spectrum.copy()

        if np.random.random() < self.p_noise:
            aug = self.gaussian_noise(aug)

        if np.random.random() < self.p_baseline:
            aug = self.baseline_drift(aug)

        if np.random.random() < self.p_solvent:
            aug = self.solvent_dropout(aug, ppm_grid)

        return aug


def generate_lorentzian_spectrum(peaks, ppm_grid, gamma=0.01):
    """Generate a 1D NMR spectrum from a peak list using Lorentzian line shapes.

    Parameters
    ----------
    peaks : list of dict
        Each dict must have 'center' (ppm) and 'intensity' (relative).
    ppm_grid : np.ndarray, shape (N,)
        Chemical shift grid.
    gamma : float
        Half-width at half-maximum (HWHM) of the Lorentzian in ppm.

    Returns
    -------
    spectrum : np.ndarray, shape (N,)
    """
    spectrum = np.zeros_like(ppm_grid, dtype=np.float64)
    for peak in peaks:
        center = peak["center"]
        intensity = peak["intensity"]
        lorentz = intensity * (gamma ** 2) / ((ppm_grid - center) ** 2 + gamma ** 2)
        spectrum += lorentz
    return spectrum.astype(np.float32)


class NMRAugmentor:
    """Combined peak + spectrum augmentor for NMR training pipelines.

    Parameters
    ----------
    training : bool
        If False, no augmentation is applied (passthrough mode).
    peak_aug_kwargs : dict
        Keyword arguments for PeakAugmentation.
    spec_aug_kwargs : dict
        Keyword arguments for SpectrumAugmentation.
    """

    def __init__(self, training=True, peak_aug_kwargs=None, spec_aug_kwargs=None):
        self.training = training
        self.peak_aug = PeakAugmentation(**(peak_aug_kwargs or {}))
        self.spec_aug = SpectrumAugmentation(**(spec_aug_kwargs or {}))

    def __call__(self, peaks, ppm_grid, gamma=0.01):
        """Apply augmentations and generate spectrum.

        Parameters
        ----------
        peaks : list of dict
            Peak list with 'center' and 'intensity' keys.
        ppm_grid : np.ndarray, shape (N,)
            Chemical shift grid.
        gamma : float
            Default linewidth (HWHM). May be overridden by peak augmentation.

        Returns
        -------
        spectrum : np.ndarray, shape (N,)
            Augmented 1D spectrum.
        """
        if self.training:
            aug_peaks, aug_gamma = self.peak_aug(peaks)
            spectrum = generate_lorentzian_spectrum(aug_peaks, ppm_grid, gamma=aug_gamma)
            spectrum = self.spec_aug(spectrum, ppm_grid)
        else:
            spectrum = generate_lorentzian_spectrum(peaks, ppm_grid, gamma=gamma)

        return spectrum


if __name__ == "__main__":
    # Quick sanity check
    ppm = np.linspace(0, 12, 1200)
    peaks = [
        {"center": 1.2, "intensity": 1.0},
        {"center": 3.5, "intensity": 0.6},
        {"center": 7.26, "intensity": 0.8},
        {"center": 8.1, "intensity": 0.3},
    ]

    print("=== PeakAugmentation ===")
    pa = PeakAugmentation()
    for i in range(3):
        aug_peaks, gamma = pa(peaks)
        print(f"  Trial {i+1}: {len(aug_peaks)} peaks, gamma={gamma:.4f}")
        for p in aug_peaks:
            print(f"    center={p['center']:.3f}, intensity={p['intensity']:.3f}")

    print("\n=== SpectrumAugmentation ===")
    sa = SpectrumAugmentation()
    base = generate_lorentzian_spectrum(peaks, ppm, gamma=0.01)
    print(f"  Base spectrum: shape={base.shape}, max={base.max():.4f}")
    for i in range(3):
        aug = sa(base, ppm)
        diff = np.abs(aug - base).mean()
        print(f"  Trial {i+1}: mean|diff|={diff:.6f}, max={aug.max():.4f}")

    print("\n=== NMRAugmentor (training=True) ===")
    augmentor = NMRAugmentor(training=True)
    for i in range(3):
        spec = augmentor(peaks, ppm)
        print(f"  Trial {i+1}: shape={spec.shape}, max={spec.max():.4f}, min={spec.min():.4f}")

    print("\n=== NMRAugmentor (training=False) ===")
    augmentor_eval = NMRAugmentor(training=False)
    spec_eval = augmentor_eval(peaks, ppm)
    print(f"  Eval: shape={spec_eval.shape}, max={spec_eval.max():.4f}")
    spec_eval2 = augmentor_eval(peaks, ppm)
    print(f"  Deterministic check: identical={np.allclose(spec_eval, spec_eval2)}")
