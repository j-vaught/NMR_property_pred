"""
Phase 4 v5: Trained on expanded dataset (6,223 compounds, real + synthetic NMR).

Same model as v3 (NMR + MLP-desc + T → XGBoost → log viscosity),
but trained on the v2 training set which has:
  - 6,223 compounds (vs 953 in v1)
  - 34,460 rows (vs 10,633 in v1)
  - 5,442 rows below 260K (vs 355 in v1)
  - 1,144 compounds with real NMR, 5,079 with synthetic NMR

The fine-tuned MLP was trained on real NMR data only. For synthetic
NMR compounds, the MLP descriptor predictions will be noisier but
still capture rough molecular properties from the synthetic spectrum.

Baselines:
  Phase 3 (functional groups):    53.8%
  Phase 4 v1 (XGBoost Stage 1):   39.4%
  Phase 4 v3 (FT-MLP, 953 cpd):   29.3%
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shared.config import Paths
from shared.data_utils import make_temperature_features
from phase3.fuel_nmr import load_all_fuels, FUEL_POSF

paths = Paths()
TRAINING_V2 = paths.data / "nmr" / "phase4_training_set_v2.parquet"
OUTPUT_DIR = Path(__file__).parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

FINETUNED_MODEL = OUTPUT_DIR / "phase4_descriptors_finetuned.pt"

DESCRIPTOR_COLS = [
    "mw", "logp", "tpsa", "n_heavy_atoms", "h_count_molecular",
    "n_rotatable", "n_aromatic_rings", "n_peaks",
]

FUEL_DENSITIES = {
    "10289": 827.0, "10325": 803.0, "7720": 757.0,
    "5729": 737.0, "7629": 769.0, "10151": 757.0,
}


def pas_to_cst(pas, density_kgm3):
    return pas / (1e-6 * density_kgm3)


# ---------------------------------------------------------------------------
# Descriptor MLP
# ---------------------------------------------------------------------------

def build_mlp(input_dim, n_outputs, hidden, dropout=0.2):
    import torch.nn as nn
    layers = []
    in_d = input_dim
    for h in hidden:
        layers.extend([
            nn.Linear(in_d, h),
            nn.BatchNorm1d(h),
            nn.ReLU(),
            nn.Dropout(dropout),
        ])
        in_d = h
    layers.append(nn.Linear(in_d, n_outputs))
    return nn.Sequential(*layers)


def load_descriptor_mlp(device):
    import torch
    ckpt = torch.load(FINETUNED_MODEL, map_location="cpu", weights_only=False)
    model = build_mlp(
        input_dim=ckpt["input_dim"],
        n_outputs=ckpt["n_outputs"],
        hidden=tuple(ckpt["hidden"]),
    )
    model.load_state_dict(ckpt["state_dict"])
    model = model.to(device)
    model.eval()
    scalers = {k: ckpt[k] for k in ["X_mean", "X_std", "Y_mean", "Y_std"]}
    return model, scalers


def predict_descriptors(model, scalers, X_raw, device):
    import torch
    X_scaled = (X_raw - scalers["X_mean"]) / scalers["X_std"]
    model.eval()
    with torch.no_grad():
        xt = torch.from_numpy(X_scaled.astype(np.float32)).to(device)
        y_scaled = model(xt).cpu().numpy()
    return y_scaled * scalers["Y_std"] + scalers["Y_mean"]


# ---------------------------------------------------------------------------
# XGBoost (v3 hyperparameters)
# ---------------------------------------------------------------------------

def train_viscosity_model(X, y_log, sample_weight=None):
    from xgboost import XGBRegressor

    scaler_x = StandardScaler().fit(X)
    scaler_y = StandardScaler().fit(y_log.reshape(-1, 1))
    X_scaled = scaler_x.transform(X)
    y_std = scaler_y.transform(y_log.reshape(-1, 1)).flatten()

    rng = np.random.RandomState(42)
    n_val = max(100, int(0.1 * len(X_scaled)))
    idx = rng.permutation(len(X_scaled))
    val_idx, tr_idx = idx[:n_val], idx[n_val:]

    model = XGBRegressor(
        n_estimators=1500,
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

    fit_kwargs = {}
    if sample_weight is not None:
        fit_kwargs["sample_weight"] = sample_weight[tr_idx]
        fit_kwargs["sample_weight_eval_set"] = [sample_weight[val_idx]]

    model.fit(X_scaled[tr_idx], y_std[tr_idx],
              eval_set=[(X_scaled[val_idx], y_std[val_idx])],
              verbose=False, **fit_kwargs)

    y_val_pred = scaler_y.inverse_transform(
        model.predict(X_scaled[val_idx]).reshape(-1, 1)).flatten()
    y_val_true = y_log[val_idx]
    r2 = 1 - ((y_val_true - y_val_pred) ** 2).sum() / \
             ((y_val_true - y_val_true.mean()) ** 2).sum()
    print(f"    Val log R² = {r2:.3f}, n_trees = {model.best_iteration}")

    return model, scaler_x, scaler_y


def predict_viscosity(model, sx, sy, X):
    return sy.inverse_transform(model.predict(sx.transform(X)).reshape(-1, 1)).flatten()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import torch

    print("=" * 70)
    print("  Phase 4 v5: Expanded dataset (real + synthetic NMR)")
    print("=" * 70)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load expanded training set
    print(f"\n--- Loading v2 training set ---")
    df = pd.read_parquet(TRAINING_V2)
    df = df[df["viscosity_Pa_s"] <= 30.0]
    df = df[df["viscosity_Pa_s"] >= 1e-5]
    df = df.dropna(subset=DESCRIPTOR_COLS)
    if "log_viscosity" not in df.columns:
        df["log_viscosity"] = np.log(df["viscosity_Pa_s"])
    print(f"  {len(df)} rows, {df['canonical_smiles'].nunique()} compounds")
    print(f"  Real NMR: {(df['nmr_type']=='real').sum()} rows, "
          f"{df[df['nmr_type']=='real']['canonical_smiles'].nunique()} compounds")
    print(f"  Synthetic NMR: {(df['nmr_type']=='synthetic').sum()} rows, "
          f"{df[df['nmr_type']=='synthetic']['canonical_smiles'].nunique()} compounds")
    print(f"  T < 260K: {(df.T_K < 260).sum()} rows")

    # Extract NMR features
    spectrum_feats = np.stack([np.array(s, dtype=np.float32) for s in df["spectrum_features"]])
    peaklist_feats = np.stack([np.array(s, dtype=np.float32) for s in df["peaklist_features"]])
    nmr_feats = np.hstack([spectrum_feats, peaklist_feats])
    T_feats = make_temperature_features(df["T_K"].values).astype(np.float32)

    # Predict descriptors via MLP
    print(f"\n--- Predicting MLP descriptors ---")
    desc_model, desc_scalers = load_descriptor_mlp(device)
    desc_feats = predict_descriptors(desc_model, desc_scalers, nmr_feats, device)

    # Build feature matrix
    X = np.hstack([nmr_feats, desc_feats, T_feats])
    y_log = df["log_viscosity"].values.astype(np.float32)
    print(f"  Feature matrix: {X.shape}")

    # Sample weights: real NMR gets 3x weight (more trustworthy)
    weights = np.ones(len(df), dtype=np.float32)
    real_mask = (df["nmr_type"] == "real").values
    weights[real_mask] = 3.0
    weights /= weights.mean()

    # Train model A: with sample weights
    print(f"\n--- Training model A: weighted (real NMR 3x) ---")
    model_A, scx_A, scy_A = train_viscosity_model(X, y_log, weights)

    # Train model B: unweighted (let volume of synthetic help)
    print(f"\n--- Training model B: unweighted ---")
    model_B, scx_B, scy_B = train_viscosity_model(X, y_log, None)

    # Train model C: real NMR only (same as v3 but more compounds)
    print(f"\n--- Training model C: real NMR only ---")
    real_idx = df["nmr_type"] == "real"
    X_real = X[real_idx]
    y_real = y_log[real_idx]
    model_C, scx_C, scy_C = train_viscosity_model(X_real, y_real, None)

    # Fuel prediction
    print(f"\n--- Fuel prediction ---")
    fuels = load_all_fuels()

    models = {
        "A_weighted": (model_A, scx_A, scy_A),
        "B_unweighted": (model_B, scx_B, scy_B),
        "C_real_only": (model_C, scx_C, scy_C),
    }

    results = {}
    for fuel_name, fuel_data in fuels.items():
        posf = FUEL_POSF.get(fuel_name, "")
        density = FUEL_DENSITIES.get(posf, 800.0)

        fuel_spec = np.array(fuel_data["spectrum_features"], dtype=np.float32)
        fuel_peak = np.array(fuel_data["peaklist_features"], dtype=np.float32)
        fuel_nmr = np.hstack([fuel_spec, fuel_peak]).reshape(1, -1)
        fuel_desc = predict_descriptors(desc_model, desc_scalers, fuel_nmr, device)

        measured = None
        if "measured_viscosity" in fuel_data:
            m = fuel_data["measured_viscosity"]
            measured = {"T_K": m["T_K"], "cSt": m["viscosity_cSt"], "source": m["source"]}
        elif "measured_viscosity_afrl" in fuel_data:
            m = fuel_data["measured_viscosity_afrl"]
            measured = {"T_K": m["T_K"], "cSt": m["viscosity_cSt"], "source": m["source"]}

        if measured is None:
            continue

        T_K_arr = np.array(measured["T_K"], dtype=np.float32)
        y_true_cst = np.array(measured["cSt"], dtype=np.float32)
        n = len(T_K_arr)
        T_feat = make_temperature_features(T_K_arr).astype(np.float32)
        X_fuel = np.hstack([
            np.tile(fuel_nmr, (n, 1)),
            np.tile(fuel_desc, (n, 1)),
            T_feat,
        ])

        print(f"\n=== {fuel_name} (POSF {posf}, ρ={density:.0f}) ===")
        print(f"  {'Model':<16s} {'T (K)':>8s} {'Meas':>10s} {'Pred':>10s} {'Err':>8s}")

        fuel_results = {}
        for model_name, (model, sx, sy) in models.items():
            y_pred_log = predict_viscosity(model, sx, sy, X_fuel)
            y_pred_pas = np.exp(y_pred_log)
            y_pred_cst = pas_to_cst(y_pred_pas, density)
            errors = np.abs(y_pred_cst - y_true_cst) / y_true_cst * 100
            mape = errors.mean()

            for t, mv, p, err in zip(T_K_arr, y_true_cst, y_pred_cst, errors):
                print(f"  {model_name:<16s} {t:8.1f} {mv:10.2f} {p:10.2f} {err:7.1f}%")

            fuel_results[model_name] = {
                "mape": float(mape),
                "predicted_cSt": y_pred_cst.tolist(),
            }

        results[fuel_name] = {
            "posf": posf,
            "measured_cSt": y_true_cst.tolist(),
            "T_K": T_K_arr.tolist(),
            **fuel_results,
        }

    # Summary
    print(f"\n{'='*70}")
    print(f"  SUMMARY")
    print(f"{'='*70}")

    for model_name in models:
        mapes = [r[model_name]["mape"] for r in results.values() if model_name in r]
        avg = np.mean(mapes) if mapes else float("nan")
        print(f"\n  {model_name}:")
        for fuel_name, r in results.items():
            if model_name in r:
                print(f"    {fuel_name:<30s} {r[model_name]['mape']:>8.1f}%")
        print(f"    {'AVERAGE':<30s} {avg:>8.1f}%")

    print(f"\n  v3 baseline (953 compounds):  29.3%")

    def _s(o):
        if isinstance(o, dict): return {k: _s(v) for k, v in o.items()}
        if isinstance(o, list): return [_s(v) for v in o]
        if isinstance(o, (np.floating, float)): return float(o)
        if isinstance(o, (np.integer, int)): return int(o)
        if isinstance(o, np.ndarray): return o.tolist()
        return o

    with open(OUTPUT_DIR / "phase4_v5_fuel_predictions.json", "w") as f:
        json.dump(_s(results), f, indent=2)
    print(f"\nSaved to {OUTPUT_DIR / 'phase4_v5_fuel_predictions.json'}")


if __name__ == "__main__":
    main()
