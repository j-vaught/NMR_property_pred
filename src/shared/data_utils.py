import functools
import numpy as np
from rdkit import Chem
from rdkit.Chem.inchi import MolFromInchi


@functools.lru_cache(maxsize=500_000)
def canonical_smiles(smi: str) -> str | None:
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol)


@functools.lru_cache(maxsize=500_000)
def inchi_to_canonical_smiles(inchi: str) -> str | None:
    mol = MolFromInchi(inchi, sanitize=True, treatWarningAsError=False)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol)


def viscosity_to_pas(value: float, unit: str) -> float:
    conversions = {
        "pas": 1.0,
        "pa*s": 1.0,
        "pa.s": 1.0,
        "mpas": 1e-3,
        "mpa*s": 1e-3,
        "mpa.s": 1e-3,
        "upas": 1e-6,
        "upa*s": 1e-6,
        "cp": 1e-3,
        "p": 0.1,
    }
    factor = conversions.get(unit.lower().strip())
    if factor is None:
        raise ValueError(f"Unknown viscosity unit: {unit}")
    return value * factor


def surface_tension_to_nm(value: float, unit: str) -> float:
    conversions = {
        "n/m": 1.0,
        "nm": 1.0,
        "mn/m": 1e-3,
        "mnm": 1e-3,
        "dyn/cm": 1e-3,
    }
    factor = conversions.get(unit.lower().strip().replace(" ", ""))
    if factor is None:
        raise ValueError(f"Unknown surface tension unit: {unit}")
    return value * factor


def fit_arrhenius(T_K: np.ndarray, viscosity_Pas: np.ndarray) -> tuple[float, float, float]:
    """Fit ln(eta) = A + B/T via OLS. Returns (A, B, r2)."""
    ln_eta = np.log(viscosity_Pas)
    inv_T = 1.0 / T_K
    X = np.column_stack([np.ones_like(inv_T), inv_T])
    coeffs, residuals, _, _ = np.linalg.lstsq(X, ln_eta, rcond=None)
    A, B = coeffs[0], coeffs[1]

    ss_res = np.sum((ln_eta - X @ coeffs) ** 2)
    ss_tot = np.sum((ln_eta - np.mean(ln_eta)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return A, B, r2


def fit_linear_st(T_K: np.ndarray, st_Nm: np.ndarray) -> tuple[float, float, float]:
    """Fit sigma = A + B*T via OLS. Returns (A, B, r2)."""
    X = np.column_stack([np.ones_like(T_K), T_K])
    coeffs, _, _, _ = np.linalg.lstsq(X, st_Nm, rcond=None)
    A, B = coeffs[0], coeffs[1]

    ss_res = np.sum((st_Nm - X @ coeffs) ** 2)
    ss_tot = np.sum((st_Nm - np.mean(st_Nm)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return A, B, r2


def make_temperature_features(T_K: np.ndarray) -> np.ndarray:
    """Create temperature feature vector: [T/1000, 1/T, (T/1000)^2]."""
    T_scaled = T_K / 1000.0
    return np.column_stack([T_scaled, 1.0 / T_K, T_scaled ** 2])
