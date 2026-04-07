from dataclasses import dataclass, field
from pathlib import Path


import os

def _find_root() -> Path:
    """Find project root — works on both local Mac and Comech server."""
    # Check environment variable first
    env_root = os.environ.get("NMR_PROJECT_ROOT")
    if env_root:
        return Path(env_root)
    # Check common locations
    candidates = [
        Path("/Users/user/Downloads/untitled folder"),  # Mac
        Path.home() / "NMR_property_pred",  # Comech server
    ]
    for p in candidates:
        if (p / "src").exists():
            return p
    return Path.cwd()

ROOT = _find_root()


@dataclass
class Paths:
    root: Path = ROOT
    data: Path = ROOT / "data"
    features: Path = ROOT / "data" / "features"
    thermo: Path = ROOT / "data" / "thermo"
    fuels: Path = ROOT / "data" / "fuels"
    outputs: Path = ROOT / "src" / "phase1" / "outputs"

    # Feature files
    morgan_r2: Path = features / "morgan_r2_2048.csv"
    morgan_r3: Path = features / "morgan_r3_2048.csv"
    compound_index: Path = features / "compound_smiles_index.csv"

    # Label files — ThermoML
    thermoml_viscosity: Path = thermo / "thermoml_parsed" / "pure_viscosity.csv"
    thermoml_surface_tension: Path = thermo / "thermoml_parsed" / "pure_surface_tension.csv"

    # Label files — chemicals library expanded
    gc_viscosity: Path = thermo / "group_contribution" / "chemicals_viscosity_expanded.csv"
    gc_surface_tension: Path = thermo / "group_contribution" / "chemicals_surface_tension_expanded.csv"

    # Label files — NIST WebBook
    nist_webbook_dir: Path = thermo / "nist_webbook"

    # NMR data
    nmrexp: Path = ROOT / "data" / "nmr" / "nmrexp" / "NMRexp_1H_13C.parquet"

    # Additional properties
    flash_boiling: Path = thermo / "additional_properties" / "flash_boiling_points.csv"


@dataclass
class DataConfig:
    fingerprint: str = "morgan_r2"
    fp_bits: int = 2048
    T_min: float = 200.0
    T_max: float = 500.0
    P_max_kPa: float = 110.0
    min_points_per_compound: int = 5
    arrhenius_r2_threshold: float = 0.95
    hydrocarbons_only: bool = False
    eval_hydrocarbons_only: bool = True


@dataclass
class SplitConfig:
    train_frac: float = 0.8
    val_frac: float = 0.1
    test_frac: float = 0.1
    n_folds: int = 5
    seed: int = 42


@dataclass
class ModelConfig:
    encoder_layers: list = field(default_factory=lambda: [2048, 512, 256, 128])
    dropout: float = 0.3
    use_batch_norm: bool = True
    head_layers: list = field(default_factory=lambda: [128, 64])
    head_dropout: float = 0.2
    t_feature_dim: int = 3


@dataclass
class TrainConfig:
    approach: str = "direct"
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 300
    patience: int = 30
    lr_scheduler: str = "cosine"
    lr_min: float = 1e-6
    viscosity_weight: float = 1.0
    surface_tension_weight: float = 1.0
    seed: int = 42


@dataclass
class JP5Validation:
    temperatures_K: list = field(default_factory=lambda: [
        288.15, 298.15, 308.15, 318.15, 328.15, 338.15, 348.15
    ])
    viscosities_cP: list = field(default_factory=lambda: [
        1.78, 1.61, 1.46, 1.34, 1.14, 0.978, 0.853
    ])
