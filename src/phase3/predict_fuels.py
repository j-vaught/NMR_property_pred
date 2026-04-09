"""
Phase 3: Predict fuel mixture properties from NMR spectra.

Strategy:
  1. Train property models on pure compounds (Phase 2 data) using NMR features
  2. Extract identical features from fuel mixture NMR spectra
  3. Predict fuel properties and validate against AFRL/Burnett measured data

Three feature sets compared:
  - NMR spectral features (95 dims) — Phase 2 baseline
  - NMR full (spectral + peak-list, 120 dims) — includes H count info
  - Functional groups (9 dims) — Jameel 2021 shift-to-FG mapping

Unit handling:
  - Model trained on viscosity in Pa·s (dynamic viscosity)
  - AFRL/Burnett measured data in cSt (kinematic viscosity)
  - Conversion: dynamic_Pa_s = kinematic_cSt × 1e-6 × density_kg_m3
  - Surface tension: model in N/m, AFRL in mN/m → multiply by 1000
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
from phase3.fuel_nmr import (
    load_all_fuels, extract_functional_groups, get_functional_group_names,
    FUEL_POSF, _1H_GRID,
)

paths = Paths()
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
CACHE_DIR = Path(__file__).resolve().parents[1] / "phase2" / "cache"


def _load_pure_compound_data():
    """Load Phase 2 cached features and labels."""
    cache = np.load(CACHE_DIR / "nmr_compound_cache.npz")
    return {
        "spectrum_features": cache["spectrum_features"],
        "peaklist_features": cache["peaklist_features"],
        "spectra": cache["spectra"],
    }


def _build_training_data(feature_matrix, smiles_list, property_name):
    """Build X, y from pure compounds."""
    smi_to_idx = {s: i for i, s in enumerate(smiles_list)}
    labels = pd.read_csv(CACHE_DIR / f"labels_{property_name}.csv")
    labels = labels[labels["canonical_smiles"].isin(smi_to_idx)]

    X_rows, y_rows = [], []
    for _, row in labels.iterrows():
        cidx = smi_to_idx[row["canonical_smiles"]]
        t_feat = make_temperature_features(np.array([row["T_K"]]))[0]
        X_rows.append(np.concatenate([feature_matrix[cidx], t_feat]))

        if property_name == "viscosity":
            y_rows.append(np.log(row["value"]))  # Pa·s in log space
        else:
            y_rows.append(row["value"])  # N/m

    return np.array(X_rows, dtype=np.float32), np.array(y_rows, dtype=np.float32)


def _train_xgboost(X_train, y_train):
    """Train XGBoost on full training data."""
    from xgboost import XGBRegressor

    scaler_x = StandardScaler().fit(X_train)
    scaler_y = StandardScaler().fit(y_train.reshape(-1, 1))

    X_tr = scaler_x.transform(X_train)
    y_tr = scaler_y.transform(y_train.reshape(-1, 1)).flatten()

    n_val = max(1, int(0.15 * len(X_tr)))
    perm = np.random.RandomState(42).permutation(len(X_tr))
    val_idx, tr_idx = perm[:n_val], perm[n_val:]

    model = XGBRegressor(
        n_estimators=500, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
        min_child_weight=5, early_stopping_rounds=30, verbosity=0,
    )
    model.fit(X_tr[tr_idx], y_tr[tr_idx],
              eval_set=[(X_tr[val_idx], y_tr[val_idx])], verbose=False)

    return model, scaler_x, scaler_y


def _predict(model, scaler_x, scaler_y, fuel_features, T_K_values, property_name):
    """Predict fuel property at given temperatures."""
    X = np.array([
        np.concatenate([fuel_features, make_temperature_features(np.array([T]))[0]])
        for T in T_K_values
    ], dtype=np.float32)

    y_pred_std = model.predict(scaler_x.transform(X))
    y_pred = scaler_y.inverse_transform(y_pred_std.reshape(-1, 1)).flatten()

    if property_name == "viscosity":
        return np.exp(y_pred)  # Pa·s
    return y_pred  # N/m


def _cst_to_pas(cst, density_kgm3):
    """Convert kinematic viscosity (cSt) to dynamic viscosity (Pa·s)."""
    return cst * 1e-6 * density_kgm3


def _pas_to_cst(pas, density_kgm3):
    """Convert dynamic viscosity (Pa·s) to kinematic viscosity (cSt)."""
    return pas / (1e-6 * density_kgm3)


# Approximate fuel densities at 15-20C (kg/m³) for unit conversion
# From AFRL Edwards 2020 Table 5 and 10
FUEL_DENSITIES = {
    "10289": 827.0,   # JP-5
    "10325": 803.0,   # Jet A
    "7720":  757.0,   # HRJ Camelina (approx from HEFA range)
    "5729":  737.0,   # Shell SPK
    "7629":  769.0,   # Sasol IPK
    "10151": 757.0,   # Gevo ATJ (approx)
}


def _serialize(obj):
    if isinstance(obj, dict):
        return {k: _serialize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_serialize(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        return float(obj)
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def main():
    print("=" * 70)
    print("  PHASE 3: FUEL MIXTURE PROPERTY PREDICTION FROM NMR")
    print("=" * 70)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load fuel NMR data
    print("\nLoading fuel NMR spectra...")
    fuels = load_all_fuels()
    print(f"  {len(fuels)} fuels loaded")

    # Load pure compound data
    print("Loading pure compound training data...")
    compound_data = _load_pure_compound_data()
    with open(CACHE_DIR / "nmr_smiles_list.txt") as f:
        smiles_list = [line.strip() for line in f]

    # Extract FG features for pure compounds too
    spectra = compound_data["spectra"]
    fg_features_pure = np.stack([extract_functional_groups(s) for s in spectra])

    # Define feature sets
    feature_configs = {
        "nmr_spectrum": compound_data["spectrum_features"],
        "nmr_full": np.hstack([compound_data["spectrum_features"],
                                compound_data["peaklist_features"]]),
        "functional_groups": fg_features_pure,
    }

    results = {}

    for property_name in ["viscosity", "surface_tension"]:
        print(f"\n{'#'*70}")
        print(f"  {property_name.upper()}")
        print(f"{'#'*70}")

        prop_results = {}

        for fs_name, feat_matrix in feature_configs.items():
            print(f"\n  --- {fs_name} ({feat_matrix.shape[1]} features) ---")

            X_train, y_train = _build_training_data(
                feat_matrix, smiles_list, property_name)
            print(f"  Training on {X_train.shape[0]} pure compound samples...")

            model, scaler_x, scaler_y = _train_xgboost(X_train, y_train)

            fs_results = {}

            for fuel_name, fuel_data in fuels.items():
                posf = FUEL_POSF.get(fuel_name, "")
                density = FUEL_DENSITIES.get(posf, 800.0)

                # Build fuel feature vector
                if fs_name == "nmr_spectrum":
                    fuel_feats = fuel_data["spectrum_features"]
                elif fs_name == "nmr_full":
                    fuel_feats = np.concatenate([
                        fuel_data["spectrum_features"],
                        fuel_data["peaklist_features"],
                    ])
                elif fs_name == "functional_groups":
                    fuel_feats = fuel_data["fg_features"]

                fuel_result = {"fuel_name": fuel_name, "posf": posf}

                # Find measured data
                measured = None
                if property_name == "viscosity":
                    # Check Burnett data first, then AFRL
                    if "measured_viscosity" in fuel_data:
                        m = fuel_data["measured_viscosity"]
                        measured = {
                            "T_K": m["T_K"],
                            "value_cSt": m["viscosity_cSt"],
                            "source": m["source"],
                        }
                    elif "measured_viscosity_afrl" in fuel_data:
                        m = fuel_data["measured_viscosity_afrl"]
                        measured = {
                            "T_K": m["T_K"],
                            "value_cSt": m["viscosity_cSt"],
                            "source": m["source"],
                        }
                elif property_name == "surface_tension":
                    if "measured_surface_tension" in fuel_data:
                        m = fuel_data["measured_surface_tension"]
                        measured = {
                            "T_K": m["T_K"],
                            "value_mNm": m["surface_tension_mNm"],
                            "source": m["source"],
                        }

                if measured is not None:
                    T_K = measured["T_K"]
                    pred_raw = _predict(model, scaler_x, scaler_y,
                                        fuel_feats, T_K, property_name)

                    if property_name == "viscosity":
                        # Model outputs Pa·s, measured is cSt
                        pred_cSt = _pas_to_cst(pred_raw, density)
                        meas_vals = measured["value_cSt"]
                        errors = np.abs(pred_cSt - meas_vals) / meas_vals * 100

                        print(f"\n    {fuel_name} (density≈{density:.0f} kg/m³):")
                        print(f"    {'T (K)':>8s} {'Meas (cSt)':>12s} {'Pred (cSt)':>12s} {'Error':>8s}")
                        for t, m_val, p_val, err in zip(T_K, meas_vals, pred_cSt, errors):
                            print(f"    {t:8.1f} {m_val:12.2f} {p_val:12.2f} {err:7.1f}%")
                        mape = errors.mean()
                        print(f"    MAPE: {mape:.1f}%")

                        fuel_result["measured_T_K"] = T_K.tolist()
                        fuel_result["measured_cSt"] = meas_vals.tolist()
                        fuel_result["predicted_cSt"] = pred_cSt.tolist()
                        fuel_result["mape"] = float(mape)

                    else:  # surface_tension
                        # Model outputs N/m, measured is mN/m
                        pred_mNm = pred_raw * 1000
                        meas_vals = measured["value_mNm"]
                        errors = np.abs(pred_mNm - meas_vals) / meas_vals * 100

                        print(f"\n    {fuel_name}:")
                        print(f"    {'T (K)':>8s} {'Meas (mN/m)':>12s} {'Pred (mN/m)':>12s} {'Error':>8s}")
                        for t, m_val, p_val, err in zip(T_K, meas_vals, pred_mNm, errors):
                            print(f"    {t:8.1f} {m_val:12.2f} {p_val:12.2f} {err:7.1f}%")
                        mape = errors.mean()
                        print(f"    MAPE: {mape:.1f}%")

                        fuel_result["measured_T_K"] = T_K.tolist()
                        fuel_result["measured_mNm"] = meas_vals.tolist()
                        fuel_result["predicted_mNm"] = pred_mNm.tolist()
                        fuel_result["mape"] = float(mape)

                else:
                    # No measured data — predict at standard temperatures
                    T_std = np.array([233.15, 253.15, 273.15, 293.15, 313.15, 333.15])
                    pred_raw = _predict(model, scaler_x, scaler_y,
                                        fuel_feats, T_std, property_name)

                    if property_name == "viscosity":
                        pred_cSt = _pas_to_cst(pred_raw, density)
                        fuel_result["predicted_T_K"] = T_std.tolist()
                        fuel_result["predicted_cSt"] = pred_cSt.tolist()
                    else:
                        pred_mNm = pred_raw * 1000
                        fuel_result["predicted_T_K"] = T_std.tolist()
                        fuel_result["predicted_mNm"] = pred_mNm.tolist()

                fs_results[fuel_name] = fuel_result

            prop_results[fs_name] = fs_results

        results[property_name] = prop_results

    # Summary table
    print(f"\n{'='*70}")
    print("  VALIDATION SUMMARY (fuels with measured data)")
    print(f"{'='*70}")

    for property_name in results:
        print(f"\n  {property_name.upper()}:")
        print(f"  {'Feature Set':<20s} {'Fuel':<30s} {'MAPE':>8s}")
        print(f"  {'-'*60}")

        for fs_name in results[property_name]:
            for fuel_name, fr in results[property_name][fs_name].items():
                if "mape" in fr:
                    print(f"  {fs_name:<20s} {fuel_name:<30s} {fr['mape']:>7.1f}%")

    # Save
    with open(OUTPUT_DIR / "phase3_fuel_predictions.json", "w") as f:
        json.dump(_serialize(results), f, indent=2)

    print(f"\nResults saved to {OUTPUT_DIR / 'phase3_fuel_predictions.json'}")


if __name__ == "__main__":
    main()
