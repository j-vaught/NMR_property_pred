"""
Phase 3: Predict fuel mixture properties from NMR spectra.

Strategy: train property models on pure compounds (Phase 2 data), then
apply to fuel mixture NMR spectra. Fuel NMR = weighted sum of component
spectra, so spectral features capture the composite composition.

Uses the same NMR spectral features + Phase 1 FP/desc features as Phase 2.
For fuel mixtures, only NMR features are available (no SMILES/fingerprints),
so we train NMR-only models on pure compounds and apply to fuels.

Validation against AFRL measured data and Burnett experimental viscosity.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shared.config import Paths
from shared.data_utils import make_temperature_features
from shared.metrics import compute_all_metrics
from phase3.fuel_nmr import load_all_fuels, FUEL_POSF

paths = Paths()
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
CACHE_DIR = Path(__file__).resolve().parents[1] / "phase2" / "cache"


def _load_pure_compound_data():
    """Load Phase 2 cached pure compound features and labels."""
    cache = np.load(CACHE_DIR / "nmr_compound_cache.npz")
    spectrum_features = cache["spectrum_features"]
    peaklist_features = cache["peaklist_features"]

    with open(CACHE_DIR / "nmr_smiles_list.txt") as f:
        smiles_list = [line.strip() for line in f]

    return spectrum_features, peaklist_features, smiles_list


def _build_training_data(
    spectrum_features, peaklist_features, smiles_list,
    property_name, use_peaklist=True,
):
    """Build X, y arrays from pure compound data for a given property."""
    smi_to_idx = {s: i for i, s in enumerate(smiles_list)}

    labels = pd.read_csv(CACHE_DIR / f"labels_{property_name}.csv")
    labels = labels[labels["canonical_smiles"].isin(smi_to_idx)]

    X_rows, y_rows = [], []
    for _, row in labels.iterrows():
        smi = row["canonical_smiles"]
        cidx = smi_to_idx[smi]
        T_K = row["T_K"]
        value = row["value"]

        feats = [spectrum_features[cidx]]
        if use_peaklist:
            feats.append(peaklist_features[cidx])
        feats.append(make_temperature_features(np.array([T_K]))[0])
        X_rows.append(np.concatenate(feats))

        if property_name == "viscosity":
            y_rows.append(np.log(value))
        else:
            y_rows.append(value)

    return np.array(X_rows, dtype=np.float32), np.array(y_rows, dtype=np.float32)


def _train_xgboost(X_train, y_train):
    """Train XGBoost model on all training data (no CV — using full dataset)."""
    from xgboost import XGBRegressor

    scaler_x = StandardScaler().fit(X_train)
    scaler_y = StandardScaler().fit(y_train.reshape(-1, 1))

    X_tr = scaler_x.transform(X_train)
    y_tr = scaler_y.transform(y_train.reshape(-1, 1)).flatten()

    # Holdout 15% for early stopping
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


def _predict_fuel(model, scaler_x, scaler_y, fuel_features, T_K_values,
                  property_name):
    """Predict fuel property at given temperatures.

    Parameters
    ----------
    fuel_features : NMR features for the fuel (no temperature component)
    T_K_values : array of temperatures
    property_name : 'viscosity' or 'surface_tension'

    Returns
    -------
    predictions : array of predicted property values in real units
    """
    X_fuel = []
    for T_K in T_K_values:
        t_feat = make_temperature_features(np.array([T_K]))[0]
        x = np.concatenate([fuel_features, t_feat])
        X_fuel.append(x)

    X_fuel = np.array(X_fuel, dtype=np.float32)
    X_scaled = scaler_x.transform(X_fuel)
    y_pred_std = model.predict(X_scaled)
    y_pred = scaler_y.inverse_transform(y_pred_std.reshape(-1, 1)).flatten()

    if property_name == "viscosity":
        return np.exp(y_pred)  # Convert from log to real Pa·s
    return y_pred


def main():
    print("=" * 70)
    print("  PHASE 3: FUEL MIXTURE PROPERTY PREDICTION FROM NMR")
    print("=" * 70)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load fuel NMR data
    print("\nLoading fuel NMR spectra...")
    fuels = load_all_fuels()
    print(f"  {len(fuels)} fuels loaded")
    for name in fuels:
        print(f"    {name} ({fuels[name]['source']})")

    # Load pure compound training data
    print("\nLoading pure compound training data...")
    spectrum_features, peaklist_features, smiles_list = _load_pure_compound_data()
    print(f"  {len(smiles_list)} compounds, {spectrum_features.shape[1]}+{peaklist_features.shape[1]} features")

    results = {}

    for property_name in ["viscosity", "surface_tension"]:
        print(f"\n{'#'*70}")
        print(f"  {property_name.upper()}")
        print(f"{'#'*70}")

        # Build training data from pure compounds
        X_train, y_train = _build_training_data(
            spectrum_features, peaklist_features, smiles_list,
            property_name, use_peaklist=True,
        )
        print(f"  Training: {X_train.shape[0]} samples, {X_train.shape[1]} features")

        # Train XGBoost on full pure compound dataset
        print("  Training XGBoost on pure compounds...")
        model, scaler_x, scaler_y = _train_xgboost(X_train, y_train)

        # Predict for each fuel
        prop_results = {}
        print(f"\n  Fuel Predictions:")

        for fuel_name, fuel_data in fuels.items():
            # Build fuel feature vector (NMR features only, no temperature)
            fuel_feats = np.concatenate([
                fuel_data["spectrum_features"],
                fuel_data["peaklist_features"],
            ])

            fuel_result = {"fuel_name": fuel_name, "source": fuel_data["source"]}

            # Find measured data for validation
            measured_T = []
            measured_val = []

            if property_name == "viscosity":
                # Burnett experimental viscosity
                if "viscosity_data" in fuel_data:
                    vd = fuel_data["viscosity_data"]
                    measured_T = vd[:, 0]
                    measured_val = vd[:, 1]  # cSt
                elif "viscosity_burnett" in fuel_data:
                    vd = fuel_data["viscosity_burnett"]
                    measured_T = vd["T"].values
                    measured_val = vd["Viscosity"].values

                # Standard temperature range for prediction
                T_predict = np.array([253.15, 273.15, 293.15, 313.15, 333.15, 348.15])
                unit = "Pa·s"
                # Note: Burnett viscosity is in cSt, model predicts Pa·s
                # For comparison at similar T values, we'll predict at measured T if available

            else:  # surface_tension
                T_predict = np.array([233.15, 253.15, 273.15, 293.15, 313.15, 333.15, 353.15])
                unit = "N/m"

            if len(measured_T) > 0:
                # Predict at measured temperatures for direct comparison
                predictions = _predict_fuel(model, scaler_x, scaler_y,
                                            fuel_feats, measured_T, property_name)
                fuel_result["measured_T_K"] = measured_T.tolist()
                fuel_result["measured_values"] = measured_val.tolist()
                fuel_result["predicted_values"] = predictions.tolist()

                print(f"\n  {fuel_name}:")
                print(f"    T (K)     Measured     Predicted     Error")
                for t, m, p in zip(measured_T, measured_val, predictions):
                    err = abs(p - m) / m * 100 if m != 0 else 0
                    print(f"    {t:7.1f}    {m:10.4f}    {p:10.4f}    {err:5.1f}%")

            # Also predict at standard temperatures
            predictions_std = _predict_fuel(model, scaler_x, scaler_y,
                                            fuel_feats, T_predict, property_name)
            fuel_result["standard_T_K"] = T_predict.tolist()
            fuel_result["standard_predictions"] = predictions_std.tolist()

            if len(measured_T) == 0:
                print(f"\n  {fuel_name}: (no measured {property_name} data)")
                print(f"    Predictions at standard T:")
                for t, p in zip(T_predict, predictions_std):
                    print(f"      {t:.1f} K: {p:.4f} {unit}")

            prop_results[fuel_name] = fuel_result

        results[property_name] = prop_results

    # Save results
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
        if isinstance(obj, pd.DataFrame):
            return obj.to_dict(orient="records")
        return obj

    with open(OUTPUT_DIR / "phase3_fuel_predictions.json", "w") as f:
        json.dump(_serialize(results), f, indent=2)

    print(f"\n{'='*70}")
    print(f"  Results saved to {OUTPUT_DIR / 'phase3_fuel_predictions.json'}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
