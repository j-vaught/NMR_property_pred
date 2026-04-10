"""
Phase 4 v2: Fuel viscosity prediction using the GPU-pretrained NMR→descriptors MLP.

Pipeline:
  Stage 1: GPU-pretrained PyTorch MLP (trained on 296K NMRexp compounds)
           NMR features (120) → 8 RDKit descriptors
  Stage 2: XGBoost (retrained on 943 viscosity compounds)
           NMR features + descriptors + T → log viscosity

This replaces v1 (predict_fuels.py) which trained a *per-fuel-set* XGBoost
Stage 1 on only the 943 viscosity compounds. v2 uses the pretrained MLP
which saw 296K compounds and learned much stronger NMR→descriptor mappings.

Baselines to beat:
  Phase 3 functional groups: 53.8% average MAPE
  Phase 4 v1 (XGBoost Stage 1): 39.4% average MAPE
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
from phase3.fuel_nmr import load_all_fuels, FUEL_POSF

paths = Paths()
TRAINING_PARQUET = paths.data / "nmr" / "phase4_training_set.parquet"
OUTPUT_DIR = Path(__file__).parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

MODEL_PATH = OUTPUT_DIR / "phase4_descriptors_model.pt"
SCALERS_PATH = OUTPUT_DIR / "phase4_descriptors_scalers.npz"

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
    return pas / (1e-6 * density_kgm3)


# ---------------------------------------------------------------------------
# Stage 1: Load GPU-pretrained PyTorch MLP
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


def load_descriptors_model():
    import torch
    ckpt = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    model = build_mlp(
        input_dim=ckpt["input_dim"],
        n_outputs=ckpt["n_outputs"],
        hidden=tuple(ckpt["hidden"]),
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    scalers = np.load(SCALERS_PATH)
    return model, {
        "X_mean": scalers["X_mean"],
        "X_std": scalers["X_std"],
        "Y_mean": scalers["Y_mean"],
        "Y_std": scalers["Y_std"],
    }


def predict_descriptors_mlp(model, scalers, X):
    """Predict descriptors from NMR features via the pretrained MLP."""
    import torch
    X_scaled = (X - scalers["X_mean"]) / scalers["X_std"]
    with torch.no_grad():
        xt = torch.from_numpy(X_scaled.astype(np.float32))
        y_scaled = model(xt).numpy()
    Y = y_scaled * scalers["Y_std"] + scalers["Y_mean"]
    return Y


# ---------------------------------------------------------------------------
# Stage 2: NMR + descriptors + T → log viscosity
# ---------------------------------------------------------------------------

def train_viscosity_model(X, y_log):
    from xgboost import XGBRegressor

    scaler_x = StandardScaler().fit(X)
    scaler_y = StandardScaler().fit(y_log.reshape(-1, 1))
    X_scaled = scaler_x.transform(X)
    y_std = scaler_y.transform(y_log.reshape(-1, 1)).flatten()

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
    print("  Phase 4 v2: GPU-pretrained descriptors → fuel viscosity")
    print("=" * 70)

    # Load Stage 1: pretrained MLP
    print(f"\nLoading pretrained MLP: {MODEL_PATH}")
    desc_model, desc_scalers = load_descriptors_model()
    print(f"  MLP loaded, X_mean shape={desc_scalers['X_mean'].shape}, "
          f"Y_mean shape={desc_scalers['Y_mean'].shape}")

    # Load training set for Stage 2
    print(f"\nLoading viscosity training set: {TRAINING_PARQUET}")
    df = pd.read_parquet(TRAINING_PARQUET)
    df = df[df["viscosity_Pa_s"] <= 30.0]
    df = df[df["viscosity_Pa_s"] >= 1e-5]
    df = df.dropna(subset=DESCRIPTOR_COLS)
    df["log_viscosity"] = np.log(df["viscosity_Pa_s"])
    print(f"  {len(df)} rows, {df['canonical_smiles'].nunique()} compounds")

    spectrum_feats = np.stack([np.array(s, dtype=np.float32) for s in df["spectrum_features"]])
    peaklist_feats = np.stack([np.array(s, dtype=np.float32) for s in df["peaklist_features"]])
    T_feats = make_temperature_features(df["T_K"].values).astype(np.float32)
    nmr_feats = np.hstack([spectrum_feats, peaklist_feats])  # (N, 120)

    # Sanity check: MLP in-distribution R² on the 943 compounds
    print("\n--- Sanity check: MLP on 943 viscosity compounds ---")
    compound_df = df.drop_duplicates("canonical_smiles").reset_index(drop=True)
    spec_cpd = np.stack([np.array(s, dtype=np.float32) for s in compound_df["spectrum_features"]])
    peak_cpd = np.stack([np.array(s, dtype=np.float32) for s in compound_df["peaklist_features"]])
    nmr_cpd = np.hstack([spec_cpd, peak_cpd])
    Y_true = compound_df[DESCRIPTOR_COLS].values.astype(np.float32)
    Y_pred = predict_descriptors_mlp(desc_model, desc_scalers, nmr_cpd)
    print("  Per-descriptor R² on viscosity compounds:")
    for j, name in enumerate(DESCRIPTOR_COLS):
        ss_res = ((Y_true[:, j] - Y_pred[:, j]) ** 2).sum()
        ss_tot = ((Y_true[:, j] - Y_true[:, j].mean()) ** 2).sum()
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        mae = np.abs(Y_true[:, j] - Y_pred[:, j]).mean()
        print(f"    {name:<22s} R²={r2:+.3f}  MAE={mae:8.2f}")

    # Stage 2: viscosity model
    # IMPORTANT: use MLP-predicted descriptors for training data so that
    # the distribution matches what the fuels will see at inference.
    print("\n--- Stage 2: Train NMR + MLP-descriptors + T → log viscosity ---")
    # Predict descriptors for *each row* in training set
    desc_pred_train = predict_descriptors_mlp(desc_model, desc_scalers, nmr_feats)
    X_stage2 = np.hstack([nmr_feats, desc_pred_train, T_feats])
    y_log = df["log_viscosity"].values.astype(np.float32)
    print(f"  X shape: {X_stage2.shape}, y shape: {y_log.shape}")

    visc_model, visc_scx, visc_scy = train_viscosity_model(X_stage2, y_log)

    # Load fuels
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

        # Stage 1: MLP-predicted descriptors
        pred_desc = predict_descriptors_mlp(desc_model, desc_scalers, fuel_nmr)[0]
        print("  Predicted descriptors (MLP):")
        for name, val in zip(DESCRIPTOR_COLS, pred_desc):
            print(f"    {name:<22s} {val:.2f}")

        # Measured viscosity
        measured = None
        if "measured_viscosity" in fuel_data:
            m = fuel_data["measured_viscosity"]
            measured = {"T_K": m["T_K"], "cSt": m["viscosity_cSt"], "source": m["source"]}
        elif "measured_viscosity_afrl" in fuel_data:
            m = fuel_data["measured_viscosity_afrl"]
            measured = {"T_K": m["T_K"], "cSt": m["viscosity_cSt"], "source": m["source"]}

        if measured is None:
            print("  (no measured viscosity data)")
            results[fuel_name] = {"posf": posf, "predicted_descriptors": pred_desc.tolist()}
            continue

        T_K_arr = np.array(measured["T_K"], dtype=np.float32)
        T_feat = make_temperature_features(T_K_arr).astype(np.float32)

        n = len(T_K_arr)
        X_fuel = np.hstack([
            np.tile(fuel_spec_feats, (n, 1)),
            np.tile(fuel_peak_feats, (n, 1)),
            np.tile(pred_desc.astype(np.float32), (n, 1)),
            T_feat,
        ])

        y_pred_log = predict_viscosity(visc_model, visc_scx, visc_scy, X_fuel)
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
            "predicted_descriptors": pred_desc.tolist(),
            "T_K": T_K_arr.tolist(),
            "measured_cSt": y_true_cst.tolist(),
            "predicted_cSt": y_pred_cst.tolist(),
            "predicted_Pa_s": y_pred_pas.tolist(),
            "mape": float(mape),
            "source": measured["source"],
        }

    print(f"\n{'='*70}")
    print(f"  SUMMARY: Phase 4 v2 (GPU-pretrained descriptors)")
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

    print(f"\n  Phase 3 baseline:      53.8%")
    print(f"  Phase 4 v1 (XGBoost):  39.4%")
    print(f"  Phase 4 v2 (MLP GPU):  {np.mean(mapes):.1f}%")

    def _s(o):
        if isinstance(o, dict): return {k: _s(v) for k, v in o.items()}
        if isinstance(o, list): return [_s(v) for v in o]
        if isinstance(o, (np.floating, float)): return float(o)
        if isinstance(o, (np.integer, int)): return int(o)
        if isinstance(o, np.ndarray): return o.tolist()
        return o

    with open(OUTPUT_DIR / "phase4_v2_fuel_predictions.json", "w") as f:
        json.dump(_s(results), f, indent=2)
    print(f"\nSaved to {OUTPUT_DIR / 'phase4_v2_fuel_predictions.json'}")


if __name__ == "__main__":
    main()
