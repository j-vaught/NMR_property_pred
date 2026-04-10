"""
Phase 4 v4: v3 + sample weighting for hydrocarbons and cold temperatures.

Same features as v3 (NMR + MLP-descriptors + T = 131 dims), but
XGBoost training uses sample weights to focus on:
  1. Hydrocarbon compounds (fuels are hydrocarbons)
  2. Cold temperatures (fuel predictions at 233-253K)

This addresses distribution mismatch without changing the data:
the model pays more attention to the compounds and temperatures
that matter for fuel prediction.

Baselines:
  Phase 3 (functional groups):    53.8%
  Phase 4 v1 (XGBoost Stage 1):   39.4%
  Phase 4 v3 (FT-MLP):            29.3%
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
TRAINING_PARQUET = paths.data / "nmr" / "phase4_training_set.parquet"
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
# Descriptor MLP (same as v3)
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
# Weighted XGBoost training
# ---------------------------------------------------------------------------

def compute_sample_weights(df):
    """Upweight hydrocarbons and cold temperatures."""
    weights = np.ones(len(df), dtype=np.float32)

    # Hydrocarbon boost: 3x weight
    hc_mask = df["is_hydrocarbon"].values.astype(bool)
    weights[hc_mask] *= 3.0

    # Cold temperature boost: linear ramp from 1x at 300K to 5x at 220K
    T = df["T_K"].values
    cold_boost = np.clip((300 - T) / 80, 0, 1) * 4.0 + 1.0  # 1x at 300K, 5x at 220K
    weights *= cold_boost

    # Normalize to mean=1
    weights /= weights.mean()

    print(f"  Weight range: [{weights.min():.2f}, {weights.max():.2f}]")
    print(f"  HC rows: mean_weight={weights[hc_mask].mean():.2f}")
    print(f"  T<260K rows: mean_weight={weights[T < 260].mean():.2f}")
    print(f"  Non-HC T>300K: mean_weight={weights[(~hc_mask) & (T > 300)].mean():.2f}")
    return weights


def train_viscosity_model(X, y_log, weights):
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
    model.fit(
        X_scaled[tr_idx], y_std[tr_idx],
        sample_weight=weights[tr_idx],
        eval_set=[(X_scaled[val_idx], y_std[val_idx])],
        sample_weight_eval_set=[weights[val_idx]],
        verbose=False,
    )

    y_val_pred = scaler_y.inverse_transform(
        model.predict(X_scaled[val_idx]).reshape(-1, 1)).flatten()
    y_val_true = y_log[val_idx]
    r2 = 1 - ((y_val_true - y_val_pred) ** 2).sum() / \
             ((y_val_true - y_val_true.mean()) ** 2).sum()
    print(f"  Val log R² = {r2:.3f}, n_trees = {model.best_iteration}")

    return model, scaler_x, scaler_y


def predict_viscosity(model, sx, sy, X):
    return sy.inverse_transform(model.predict(sx.transform(X)).reshape(-1, 1)).flatten()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import torch

    print("=" * 70)
    print("  Phase 4 v4: v3 + HC/cold-T sample weighting")
    print("=" * 70)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load data
    print(f"\n--- Loading training data ---")
    df = pd.read_parquet(TRAINING_PARQUET)
    df = df[df["viscosity_Pa_s"] <= 30.0]
    df = df[df["viscosity_Pa_s"] >= 1e-5]
    df = df.dropna(subset=DESCRIPTOR_COLS)
    df["log_viscosity"] = np.log(df["viscosity_Pa_s"])
    print(f"  {len(df)} rows, {df['canonical_smiles'].nunique()} compounds")

    spectrum_feats = np.stack([np.array(s, dtype=np.float32) for s in df["spectrum_features"]])
    peaklist_feats = np.stack([np.array(s, dtype=np.float32) for s in df["peaklist_features"]])
    nmr_feats = np.hstack([spectrum_feats, peaklist_feats])
    T_feats = make_temperature_features(df["T_K"].values).astype(np.float32)

    # MLP descriptors
    print(f"\n--- Predicting descriptors ---")
    desc_model, desc_scalers = load_descriptor_mlp(device)
    desc_feats = predict_descriptors(desc_model, desc_scalers, nmr_feats, device)

    X = np.hstack([nmr_feats, desc_feats, T_feats])
    y_log = df["log_viscosity"].values.astype(np.float32)
    print(f"  Features: {X.shape}")

    # Sample weights
    print(f"\n--- Computing sample weights ---")
    weights = compute_sample_weights(df)

    # Train
    print(f"\n--- Training weighted XGBoost ---")
    visc_model, scx, scy = train_viscosity_model(X, y_log, weights)

    # Fuel prediction
    print(f"\n--- Fuel prediction ---")
    fuels = load_all_fuels()

    results = {}
    for fuel_name, fuel_data in fuels.items():
        posf = FUEL_POSF.get(fuel_name, "")
        density = FUEL_DENSITIES.get(posf, 800.0)
        print(f"\n=== {fuel_name} (POSF {posf}, ρ={density:.0f} kg/m³) ===")

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
            print("  (no measured viscosity data)")
            results[fuel_name] = {"posf": posf}
            continue

        T_K_arr = np.array(measured["T_K"], dtype=np.float32)
        n = len(T_K_arr)
        T_feat = make_temperature_features(T_K_arr).astype(np.float32)
        X_fuel = np.hstack([
            np.tile(fuel_nmr, (n, 1)),
            np.tile(fuel_desc, (n, 1)),
            T_feat,
        ])

        y_pred_log = predict_viscosity(visc_model, scx, scy, X_fuel)
        y_pred_pas = np.exp(y_pred_log)
        y_pred_cst = pas_to_cst(y_pred_pas, density)
        y_true_cst = np.array(measured["cSt"], dtype=np.float32)
        errors = np.abs(y_pred_cst - y_true_cst) / y_true_cst * 100

        print(f"\n  {'T (K)':>8s} {'Meas (cSt)':>12s} {'Pred (cSt)':>12s} {'Error':>8s}")
        for t, mv, p, err in zip(T_K_arr, y_true_cst, y_pred_cst, errors):
            print(f"  {t:8.1f} {mv:12.2f} {p:12.2f} {err:7.1f}%")
        mape = errors.mean()
        print(f"  MAPE: {mape:.1f}%")

        results[fuel_name] = {
            "posf": posf,
            "density_used": density,
            "T_K": T_K_arr.tolist(),
            "measured_cSt": y_true_cst.tolist(),
            "predicted_cSt": y_pred_cst.tolist(),
            "mape": float(mape),
            "source": measured["source"],
        }

    # Summary
    print(f"\n{'='*70}")
    print(f"  SUMMARY: Phase 4 v4 (HC/cold-T weighted)")
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

    print(f"\n  Phase 3 baseline:          53.8%")
    print(f"  Phase 4 v1 (XGBoost S1):   39.4%")
    print(f"  Phase 4 v3 (unweighted):    29.3%")
    print(f"  Phase 4 v4 (HC weighted):  {np.mean(mapes):.1f}%")

    def _s(o):
        if isinstance(o, dict): return {k: _s(v) for k, v in o.items()}
        if isinstance(o, list): return [_s(v) for v in o]
        if isinstance(o, (np.floating, float)): return float(o)
        if isinstance(o, (np.integer, int)): return int(o)
        if isinstance(o, np.ndarray): return o.tolist()
        return o

    with open(OUTPUT_DIR / "phase4_v4_fuel_predictions.json", "w") as f:
        json.dump(_s(results), f, indent=2)
    print(f"\nSaved to {OUTPUT_DIR / 'phase4_v4_fuel_predictions.json'}")


if __name__ == "__main__":
    main()
