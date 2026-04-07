# NMR Property Prediction

Predicting thermophysical properties (viscosity, surface tension) of organic compounds and fuel mixtures from NMR spectra and molecular features.

## Project Overview

This project builds ML models that predict temperature-dependent viscosity and surface tension from molecular structure. The work is organized in three phases.

**Phase 1** (current): Molecular fingerprint and descriptor-based models. Predict properties from Morgan fingerprints (2048-bit) and 195 RDKit molecular descriptors using neural networks. Trained on 856 compounds from NIST ThermoML and the chemicals library.

**Phase 2** (planned): NMR spectrum-based models. Predict properties directly from experimental 1H and 13C NMR spectra. Uses 3.37M spectra from NMRexp and 64K compounds from NMRShiftDB2. Three architectures: 1D CNN, 1D ResNet, and Transformer on peak lists.

**Phase 3** (planned): Fuel mixture prediction. Predict properties of complex fuel mixtures from their NMR spectra. Validated against JP-5, Jet-A, and alternative aviation fuels.

## Phase 1 Results

Best results from 5-fold compound-level cross-validation with Lasso feature selection, Huber loss, and 5-model ensemble.

| Metric | Viscosity | Surface Tension |
|--------|-----------|-----------------|
| R² (log/standardized, all compounds) | 0.638 +/- 0.088 | 0.558 +/- 0.248 |
| R² (hydrocarbons only, real units) | 0.703 +/- 0.046 | 0.623 +/- 0.284 |
| Training compounds | 856 | 713 |
| Features selected (Lasso) | 22-38 per fold | 24-30 per fold |

Evaluated with proper compound-level splits (no temperature data leakage) and cross-validation for reliable estimates.

## Data Sources

All data sources are documented in `data/DATA_DOCUMENTATION.pdf`. Key sources include NIST ThermoML (1.16M experimental data points), NMRexp (3.37M NMR spectra), NMRShiftDB2 (64K compounds), ILThermo (33K ionic liquid data points), and the chemicals/thermo Python libraries.

Fuel-specific data from the AFRL Edwards 2020 report, Parker et al. 2025 (NMR atom types for aviation fuels), and Abdul Jameel et al. 2018 (281 fuel NMR dataset).

The original modeling approach is based on work by [Burnett](https://github.com/Matthew-Burnett-450/Physical-Property-Modeling-Using-NMR-Spectra-and-Functional-Group-Representation) (MIT license), which predicted viscosity and surface tension for 65 hydrocarbons using simulated NMR spectra and functional group counts. This project extends that approach with 13x more compounds, experimental NMR data, proper evaluation methodology, and multi-phase architecture.

## Project Structure

```
src/
├── shared/               # Common utilities
│   ├── config.py         # All paths and hyperparameters
│   ├── data_utils.py     # SMILES handling, Arrhenius fitting, HC filter
│   └── metrics.py        # R², RMSE, MAPE
├── phase1/               # Morgan FP models
│   ├── data_pipeline.py  # Multi-source data joining
│   ├── model.py          # NN architectures (single-task, multi-task, Arrhenius)
│   ├── physics_loss.py   # Physics-informed loss functions
│   ├── train_direct_optimal.py   # Best direct approach
│   ├── train_optimal.py          # Best Arrhenius approach
│   └── outputs/          # Training logs and results
├── phase2/               # NMR spectrum models (planned)
└── phase3/               # Mixture prediction (planned)

data/
├── nmr/                  # NMR spectral databases
├── thermo/               # Thermophysical property data
├── features/             # Precomputed fingerprints and descriptors
├── fuels/                # Aviation fuel data and extracted tables
├── papers/               # Source papers and supplementary info
└── DATA_DOCUMENTATION.pdf
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The code auto-detects the project root, so it runs on both local machines and remote servers.

## Training

```bash
# Phase 1 direct optimal (recommended)
python -u src/phase1/train_direct_optimal.py

# Phase 1 Arrhenius approach
python -u src/phase1/train_optimal.py

# Model comparison (XGBoost vs NN vs Ridge vs RF)
python -u src/phase1/quick_model_compare.py
```

## Key Findings

1. **RDKit molecular descriptors carry 50-60x more predictive information than Morgan fingerprint bits** for thermophysical property prediction. Lasso feature selection from 1696 to ~30 features dramatically improves performance.

2. **Neural networks outperform tree-based models** (XGBoost, LightGBM, Random Forest) on this problem due to sparse binary fingerprint features that require learned compression.

3. **The Arrhenius approach (predicting [A,B] coefficients) has higher apparent R² but suffers from exponential error amplification** when reconstructing viscosity. The direct approach with temperature as input is more reliable on diverse compound sets.

4. **Hydrocarbon-focused evaluation is essential** for the fuel application. Models trained on diverse chemistry (alcohols, acids, etc.) perform differently on hydrocarbons specifically.

5. **Group contribution (GC) property data should not be mixed with experimental data for Arrhenius fitting** — GC data spans wider temperature ranges where the 2-parameter Andrade equation breaks down, poisoning the fits.

## Author

J.C. Vaught
