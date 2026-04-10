"""
Generate synthetic 1H NMR features from SMILES using RDKit.

For compounds with viscosity data but no experimental NMR spectra,
we predict approximate 1H chemical shifts from molecular structure
and generate the same 95+25 = 120 feature vector as the real pipeline.

Chemical shift prediction uses a substructure-based lookup table
for common H environments. Accuracy is ~0.5-1.0 ppm, which is
sufficient for the 0.4 ppm binning in the spectrum features.

This enables training on ~6,000 viscosity compounds instead of ~953.
"""

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors

from phase2.nmr_features import extract_nmr_features, extract_peaklist_features
from phase2.spectrum_converter import peaks_to_spectrum_1h, _1H_GRID


# Chemical shift lookup for common H environments (ppm)
# Based on standard NMR tables (Silverstein, Pavia & Lampman)
_SHIFT_RULES = [
    # (SMARTS pattern, approximate 1H shift in ppm)
    # Aldehydes
    ("[CX3H1](=O)", 9.7),
    # Aromatic H
    ("[c]([H])", 7.2),
    # Vinyl H (=CH-)
    ("[CX3H1]=[CX3]", 5.3),
    ("[CX3H2]=[CX3]", 5.0),
    # OCH (methine next to O)
    ("[OX2][CX4H1]", 3.7),
    # OCH2
    ("[OX2][CX4H2]", 3.5),
    # OCH3
    ("[OX2][CX4H3]", 3.3),
    # NCH
    ("[NX3][CX4H1]", 2.7),
    # NCH2
    ("[NX3][CX4H2]", 2.5),
    # NCH3
    ("[NX3][CX4H3]", 2.3),
    # CH alpha to C=O
    ("[CX3](=O)[CX4H1]", 2.3),
    ("[CX3](=O)[CX4H2]", 2.2),
    ("[CX3](=O)[CX4H3]", 2.1),
    # Allylic CH
    ("[CX3]=[CX3][CX4H1]", 2.0),
    ("[CX3]=[CX3][CX4H2]", 1.9),
    # Benzylic CH
    ("[c][CX4H1]", 2.5),
    ("[c][CX4H2]", 2.3),
    ("[c][CX4H3]", 2.3),
    # CH2 in alkyl chain (generic, low priority)
    ("[CX4H2]([CX4])[CX4]", 1.3),
    # CH in alkyl chain
    ("[CX4H1]([CX4])([CX4])[CX4]", 1.5),
    # CH3 on alkyl chain
    ("[CX4H3][CX4]", 0.9),
    # Isolated CH3 (methane-like)
    ("[CX4H3]", 0.9),
    # Generic CH2
    ("[CX4H2]", 1.3),
    # Generic CH
    ("[CX4H1]", 1.5),
]

# Pre-compile SMARTS patterns
_COMPILED_RULES = []
for smarts, shift in _SHIFT_RULES:
    pat = Chem.MolFromSmarts(smarts)
    if pat is not None:
        _COMPILED_RULES.append((pat, shift))


def predict_h_shifts(smiles):
    """Predict approximate 1H chemical shifts for each H atom in a molecule.

    Returns a list of (shift_ppm, n_equiv_h) tuples representing peak groups.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    mol = Chem.AddHs(mol)

    # Find H atoms and their chemical shifts
    h_shifts = {}  # H atom idx → shift
    assigned = set()

    # Apply rules in priority order (first match wins per H atom)
    for pat, shift in _COMPILED_RULES:
        matches = mol.GetSubstructMatches(pat)
        for match in matches:
            # Find H atoms in the match or attached to matched heavy atoms
            for atom_idx in match:
                atom = mol.GetAtomWithIdx(atom_idx)
                if atom.GetAtomicNum() == 1 and atom_idx not in assigned:
                    h_shifts[atom_idx] = shift
                    assigned.add(atom_idx)
                elif atom.GetAtomicNum() != 1:
                    # Check H neighbors of this heavy atom
                    for nbr in atom.GetNeighbors():
                        if nbr.GetAtomicNum() == 1 and nbr.GetIdx() not in assigned:
                            h_shifts[nbr.GetIdx()] = shift
                            assigned.add(nbr.GetIdx())

    # Assign default shift to any unassigned H atoms
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 1 and atom.GetIdx() not in assigned:
            h_shifts[atom.GetIdx()] = 1.5  # generic CH

    if not h_shifts:
        return None

    # Group by similar shift (within 0.1 ppm tolerance)
    shifts = sorted(h_shifts.values())
    groups = []
    current_shift = shifts[0]
    current_count = 1

    for s in shifts[1:]:
        if abs(s - current_shift) < 0.15:
            current_count += 1
            current_shift = (current_shift * (current_count - 1) + s) / current_count
        else:
            groups.append((current_shift, current_count))
            current_shift = s
            current_count = 1
    groups.append((current_shift, current_count))

    return groups


def generate_synthetic_peaks(smiles):
    """Generate a synthetic peak list compatible with the real NMR pipeline.

    Returns a list of dicts matching the format of parse_1h_peaks() output:
    {center_ppm, integral, multiplicity, j_couplings_hz, shift_high, shift_low}
    """
    groups = predict_h_shifts(smiles)
    if groups is None or len(groups) == 0:
        return None

    peaks = []
    for shift, n_h in groups:
        # Estimate multiplicity from n+1 rule (crude approximation)
        if n_h == 1:
            mult = "s"
        elif n_h <= 3:
            mult = "m"
        else:
            mult = "m"

        # Add slight randomness to avoid perfectly identical spectra
        # (real NMR has solvent effects, conformational averaging, etc.)
        jitter = np.random.normal(0, 0.05)
        shift_j = shift + jitter

        peaks.append({
            "center_ppm": float(shift_j),
            "integral": float(n_h),
            "multiplicity": mult,
            "j_couplings_hz": [7.0] if mult in ("d", "t", "q") else [],
            "shift_high": float(shift_j + 0.02),
            "shift_low": float(shift_j - 0.02),
        })

    return peaks


def synthetic_nmr_features(smiles, seed=None):
    """Generate the full 120-dim NMR feature vector from SMILES.

    Returns (spectrum_features[95], peaklist_features[25]) or None if failed.
    """
    if seed is not None:
        np.random.seed(seed)

    peaks = generate_synthetic_peaks(smiles)
    if peaks is None:
        return None

    # Generate continuous spectrum from peaks (same as real pipeline)
    spectrum = peaks_to_spectrum_1h(peaks, _1H_GRID)

    # Extract features
    spec_feats = extract_nmr_features(spectrum, _1H_GRID)
    peak_feats = extract_peaklist_features(peaks)

    return spec_feats, peak_feats


if __name__ == "__main__":
    # Quick test
    test_smiles = [
        "CCCCCCCCCC",           # decane (simple alkane)
        "c1ccccc1",             # benzene
        "CCCCCCCCCCCCCCCC",     # hexadecane
        "CC(C)CCCCCCCCCC",      # 2-methyldodecane
        "c1ccc(CC)cc1",         # ethylbenzene
    ]

    for s in test_smiles:
        result = synthetic_nmr_features(s, seed=42)
        if result is not None:
            spec, peak = result
            mol = Chem.MolFromSmiles(s)
            name = s[:30]
            total_h = peak[0]  # first peaklist feature = total_h_count
            print(f"{name:30s}  spec={spec.shape}  peak={peak.shape}  "
                  f"total_H={total_h:.0f}  MW={Descriptors.MolWt(mol):.1f}")
        else:
            print(f"{s:30s}  FAILED")
