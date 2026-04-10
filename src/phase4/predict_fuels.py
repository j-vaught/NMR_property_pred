"""
Phase 4 fuel viscosity prediction.

Two-stage pipeline:
  Stage 1: NMR features → RDKit descriptors (trained on 943 viscosity compounds)
  Stage 2: NMR features + predicted descriptors + T → log viscosity
           (reuses Phase 4 +descriptors model structure)

For fuels (no SMILES available):
  - Extract NMR features from fuel spectrum (reusing phase3.fuel_nmr)
  - Predict "effective descriptors" via Stage 1 model
  - Feed into Stage 2 model to get viscosity

Validates against:
  - Burnett JP-5 experimental viscosity (7 T points)
  - AFRL measured viscosity for Shell-SPK, Sasol-IPK, etc.

Target: beat Phase 3's 54% MAPE on fuel viscosity.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shared.config import Paths
from shared.data_utils import make_temperature_features
from shared.metrics import compute_all_metrics
from phase3.fuel_nmr import load_all_fuels, FUEL_POSF

paths = Paths()
TRAINING_PARQUET = paths.data / "nmr" / "phase4_training_set.parquet"
OUTPUT_DIR = Path(__file__).parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

DESCRIPTOR_COLS = [
    "mw", "logp", "tpsa", "n_heavy_atoms", "h_count_molecular",
    "n_rotatable", "n_aromatic_rings", "n_peaks",
]

# Approximate fuel densities (kg/m³) for Pa·s ↔ cSt conversion
FUEL_DENSITIES = {
    "10289": 827.0,  # JP-5
    "10325": 803.0,  # Jet A
    "7720":  757.0,  # HRJ Camelina
    "5729":  737.0,  # Shell SPK
    "7629":  769.0,  # Sasol IPK
    "10151": 757.0,  # Gevo ATJ
}


def pas_to_cst(pas, density_kgm3):
    """Convert dynamic viscosity (Pa·s) to kinematic viscosity (cSt)."""
    return pas / (1e-6 * density_kgm3)


# ---------------------------------------------------------------------------
# Stage 1: NMR features → RDKit descriptors
# ---------------------------------------------------------------------------

def train_descriptors_predictor(X, Y_desc):
    """Train one XGBoost per descriptor (multi-output via separate models)."""
    from xgboost import XGBRegressor

    scaler_x = StandardScaler().fit(X)
    X_scaled = scaler_x.transform(X)

    models = []
    scalers_y = []
    for j in range(Y_desc.shape[1]):
        y = Y_desc[:, j]
        scy = StandardScaler().fit(y.reshape(-1, 1))
        y_std = scy.transform(y.reshape(-1, 1)).flatten()

        model = XGBRegressor(
            n_estimators=500,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            min_child_weight=5,
            verbosity=0,
        )
        model.fit(X_scaled, y_std)
        models.append(model)
        scalers_y.append(scy)

    return models, scaler_x, scalers_y


def predict_descriptors(models, scaler_x, scalers_y, X):
    """Predict all descriptors for a batch."""
    X_scaled = scaler_x.transform(X)
    Y_pred = np.zeros((X.shape[0], len(models)))
    for j, (model, scy) in enumerate(zip(models, scalers_y)):
        y_std = model.predict(X_scaled)
        Y_pred[:, j] = scy.inverse_transform(y_std.reshape(-1, 1)).flatten()
    return Y_pred


# ---------------------------------------------------------------------------
# Stage 2: NMR + descriptors + T → log viscosity
# ---------------------------------------------------------------------------

def train_viscosity_model(X, y_log):
    """Train XGBoost viscosity model on entire training set (no CV)."""
    from xgboost import XGBRegressor

    scaler_x = StandardScaler().fit(X)
    scaler_y = StandardScaler().fit(y_log.reshape(-1, 1))

    X_scaled = scaler_x.transform(X)
    y_std = scaler_y.transform(y_log.reshape(-1, 1)).flatten()

    # Split off 10% val
    rng = np.random.RandomState(42)
    n_val = max(100, int(0.1 * len(X_scaled)))
    idx = rng.permutation(len(X_scaled))
    val_idx = idx[:n_val]
    tr_idx = idx[n_val:]

    model = XGBRegressor(
        n_estimators=1000,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        min_child_weight=5,
        early_stopping_rounds=30,
        verbosity=0,
    )
    model.fit(X_scaled[tr_idx], y_std[tr_idx],
              eval_set=[(X_scaled[val_idx], y_std[val_idx])], verbose=False)
    return model, scaler_x, scaler_y


def predict_viscosity(model, scaler_x, scaler_y, X):
    X_scaled = scaler_x.transform(X)
    y_std = model.predict(X_scaled)
    return scaler_y.inverse_transform(y_std.reshape(-1, 1)).flatten()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("  Phase 4 FUEL VALIDATION")
    print("  NMR → descriptors → viscosity pipeline")
    print("=" * 70)

    # Load training set
    print(f"\nLoading {TRAINING_PARQUET}")
    df = pd.read_parquet(TRAINING_PARQUET)
    df = df[df["viscosity_Pa_s"] <= 30.0]
    df = df[df["viscosity_Pa_s"] >= 1e-5]
    df = df.dropna(subset=DESCRIPTOR_COLS)
    df["log_viscosity"] = np.log(df["viscosity_Pa_s"])
    print(f"  {len(df)} rows, {df['canonical_smiles'].nunique()} compounds")

    spectrum_feats = np.stack([np.array(s, dtype=np.float32) for s in df["spectrum_features"]])
    peaklist_feats = np.stack([np.array(s, dtype=np.float32) for s in df["peaklist_features"]])
    T_feats = make_temperature_features(df["T_K"].values).astype(np.float32)

    # Stage 1 training: NMR features → descriptors
    # Deduplicate compounds first (one row per compound)
    print("\n--- Stage 1: Train NMR → descriptors ---")
    compound_df = df.drop_duplicates("canonical_smiles").reset_index(drop=True)
    spec_cpd = np.stack([np.array(s, dtype=np.float32) for s in compound_df["spectrum_features"]])
    peak_cpd = np.stack([np.array(s, dtype=np.float32) for s in compound_df["peaklist_features"]])
    X_stage1 = np.hstack([spec_cpd, peak_cpd])  # NMR features only (no T, no descriptors)
    Y_desc = compound_df[DESCRIPTOR_COLS].values.astype(np.float32)

    print(f"  X shape: {X_stage1.shape}, Y shape: {Y_desc.shape}")
    desc_models, desc_scx, desc_scys = train_descriptors_predictor(X_stage1, Y_desc)

    # Sanity check: predict descriptors on training and report in-sample R²
    Y_pred_train = predict_descriptors(desc_models, desc_scx, desc_scys, X_stage1)
    print(f"  In-sample descriptor prediction R²:")
    for j, name in enumerate(DESCRIPTOR_COLS):
        m = compute_all_metrics(Y_desc[:, j], Y_pred_train[:, j])
        print(f"    {name:<22s} R²={m['r2']:.3f}  MAPE={m['mape']:.1f}%")

    # Stage 2 training: NMR + descriptors + T → log viscosity
    print("\n--- Stage 2: Train NMR + descriptors + T → log viscosity ---")
    X_stage2 = np.hstack([spectrum_feats, peaklist_feats,
                           df[DESCRIPTOR_COLS].values.astype(np.float32),
                           T_feats])
    y_log = df["log_viscosity"].values.astype(np.float32)
    print(f"  X shape: {X_stage2.shape}, y shape: {y_log.shape}")

    visc_model, visc_scx, visc_scy = train_viscosity_model(X_stage2, y_log)

    # ---------------- Fuel prediction ----------------
    print("\n--- Loading fuels ---")
    fuels = load_all_fuels()
    print(f"  {len(fuels)} fuels loaded")

    results = {}
    for fuel_name, fuel_data in fuels.items():
        posf = FUEL_POSF.get(fuel_name, "")
        density = FUEL_DENSITIES.get(posf, 800.0)
        print(f"\n=== {fuel_name} (POSF {posf}, ρ={density:.0f} kg/m³) ===")

        fuel_spec_feats = np.array(fuel_data["spectrum_features"], dtype=np.float32)
        fuel_peak_feats = np.array(fuel_data["peaklist_features"], dtype=np.float32)
        fuel_nmr = np.hstack([fuel_spec_feats, fuel_peak_feats]).reshape(1, -1)

        # Stage 1: predict descriptors for this fuel
        pred_desc = predict_descriptors(desc_models, desc_scx, desc_scys, fuel_nmr)[0]
        print(f"  Predicted descriptors:")
        for name, val in zip(DESCRIPTOR_COLS, pred_desc):
            print(f"    {name:<22s} {val:.2f}")

        # Find measured viscosity
        measured = None
        if "measured_viscosity" in fuel_data:
            m = fuel_data["measured_viscosity"]
            measured = {"T_K": m["T_K"], "cSt": m["viscosity_cSt"], "source": m["source"]}
        elif "measured_viscosity_afrl" in fuel_data:
            m = fuel_data["measured_viscosity_afrl"]
            measured = {"T_K": m["T_K"], "cSt": m["viscosity_cSt"], "source": m["source"]}

        if measured is None:
            print(f"  (no measured viscosity data)")
            results[fuel_name] = {"posf": posf, "predicted_descriptors": pred_desc.tolist()}
            continue

        # Stage 2: predict viscosity at measured temperatures
        T_K_arr = np.array(measured["T_K"], dtype=np.float32)
        T_feat = make_temperature_features(T_K_arr).astype(np.float32)

        # Build stage 2 feature matrix: one row per temperature
        n = len(T_K_arr)
        X_fuel = np.hstack([
            np.tile(fuel_spec_feats, (n, 1)),
            np.tile(fuel_peak_feats, (n, 1)),
            np.tile(pred_desc.astype(np.float32), (n, 1)),
            T_feat,
        ])

        y_pred_log = predict_viscosity(visc_model, visc_scx, visc_scy, X_fuel)
        y_pred_pas = np.exp(y_pred_log)  # Pa·s
        y_pred_cst = pas_to_cst(y_pred_pas, density)

        # Compare to measured cSt
        y_true_cst = np.array(measured["cSt"], dtype=np.float32)
        errors = np.abs(y_pred_cst - y_true_cst) / y_true_cst * 100

        print(f"\n  {'T (K)':>8s} {'Meas (cSt)':>12s} {'Pred (cSt)':>12s} {'Error':>8s}")
        for t, m, p, err in zip(T_K_arr, y_true_cst, y_pred_cst, errors):
            print(f"  {t:8.1f} {m:12.2f} {p:12.2f} {err:7.1f}%")
        mape = errors.mean()
        print(f"  MAPE: {mape:.1f}%")

        results[fuel_name] = {
            "posf": posf,
            "density_used": density,
            "predicted_descriptors": pred_desc.tolist(),
            "T_K": T_K_arr.tolist(),
            "measured_cSt": y_true_cst.tolist(),
            "predicted_cSt": y_pred_cst.tolist(),
            "predicted_Pa_s": y_pred_pas.tolist(),
            "mape": float(mape),
            "source": measured["source"],
        }

    # Overall summary
    print(f"\n{'='*70}")
    print(f"  SUMMARY: Phase 4 fuel viscosity")
    print(f"{'='*70}")
    print(f"\n  {'Fuel':<30s} {'MAPE':>10s}")
    print(f"  {'-'*45}")
    mapes = []
    for name, r in results.items():
        if "mape" in r:
            print(f"  {name:<30s} {r['mape']:>9.1f}%")
            mapes.append(r["mape"])
    if mapes:
        print(f"  {'-'*45}")
        print(f"  {'AVERAGE':<30s} {np.mean(mapes):>9.1f}%")

    print(f"\n  Phase 3 baseline average: 53.8% (functional_groups)")
    print(f"  Phase 4 average: {np.mean(mapes):.1f}%")
    if mapes:
        improvement = 53.8 - np.mean(mapes)
        print(f"  Improvement: {improvement:+.1f} percentage points")

    # Save results
    def _s(o):
        if isinstance(o, dict): return {k: _s(v) for k, v in o.items()}
        if isinstance(o, list): return [_s(v) for v in o]
        if isinstance(o, (np.floating, float)): return float(o)
        if isinstance(o, (np.integer, int)): return int(o)
        if isinstance(o, np.ndarray): return o.tolist()
        return o

    with open(OUTPUT_DIR / "phase4_fuel_predictions.json", "w") as f:
        json.dump(_s(results), f, indent=2)
    print(f"\nSaved to {OUTPUT_DIR / 'phase4_fuel_predictions.json'}")


if __name__ == "__main__":
    main()
