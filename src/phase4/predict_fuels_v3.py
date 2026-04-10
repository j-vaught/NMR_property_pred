"""
Phase 4 v3: Fine-tuned pretrained MLP → fuel viscosity prediction.

v2 failed (44.9% MAPE) because the pretrained MLP has catastrophic
distribution shift on hydrocarbons: trained on 296K drug-like NMRexp
compounds, it predicts nonzero TPSA/aromatic-rings for pure alkanes.

v3 fixes this by:
  1. Loading the pretrained MLP (296K compound prior)
  2. Fine-tuning it on the 943 viscosity compounds (hydrocarbon-rich)
  3. Using the fine-tuned MLP to predict fuel descriptors
  4. Feeding them + NMR + T into XGBoost viscosity model

Baselines:
  Phase 3 (functional groups):    53.8%
  Phase 4 v1 (XGBoost Stage 1):   39.4%
  Phase 4 v2 (raw pretrained):    44.9%  ← distribution shift
  Phase 4 v3 (fine-tuned MLP):    ?
"""

import copy
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

PRETRAINED_MODEL = OUTPUT_DIR / "phase4_descriptors_model.pt"
PRETRAINED_SCALERS = OUTPUT_DIR / "phase4_descriptors_scalers.npz"
FINETUNED_MODEL = OUTPUT_DIR / "phase4_descriptors_finetuned.pt"

DESCRIPTOR_COLS = [
    "mw", "logp", "tpsa", "n_heavy_atoms", "h_count_molecular",
    "n_rotatable", "n_aromatic_rings", "n_peaks",
]

FUEL_DENSITIES = {
    "10289": 827.0, "10325": 803.0, "7720":  757.0,
    "5729":  737.0, "7629":  769.0, "10151": 757.0,
}


def pas_to_cst(pas, density_kgm3):
    return pas / (1e-6 * density_kgm3)


# ---------------------------------------------------------------------------
# MLP build/load/finetune
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


def load_pretrained():
    import torch
    ckpt = torch.load(PRETRAINED_MODEL, map_location="cpu", weights_only=False)
    model = build_mlp(
        input_dim=ckpt["input_dim"],
        n_outputs=ckpt["n_outputs"],
        hidden=tuple(ckpt["hidden"]),
    )
    model.load_state_dict(ckpt["state_dict"])
    s = np.load(PRETRAINED_SCALERS)
    scalers = {k: s[k] for k in ["X_mean", "X_std", "Y_mean", "Y_std"]}
    return model, scalers, ckpt


def finetune_mlp(model, scalers, X_raw, Y_raw, device, epochs=100, lr=1e-4):
    """Fine-tune on viscosity compounds with compound-aware scaling.

    We re-fit new target scalers on the viscosity distribution so the MLP
    output head adapts to hydrocarbon statistics instead of drug-like ones.
    """
    import torch
    import torch.nn as nn

    # Apply *pretrained* X scaling (NMR feature statistics don't shift much)
    X_scaled = (X_raw - scalers["X_mean"]) / scalers["X_std"]

    # Re-fit Y scalers on the viscosity distribution
    Y_mean = Y_raw.mean(axis=0)
    Y_std = Y_raw.std(axis=0) + 1e-8
    Y_scaled = (Y_raw - Y_mean) / Y_std

    # Replace the last Linear's bias/weight scaling space by recomputing
    # targets in the *new* scaler space. We keep the same model weights —
    # the output head will learn to map them to the new Y scalers via
    # gradient descent. Because this is a small mismatch, SGD handles it.
    new_scalers = {
        "X_mean": scalers["X_mean"],
        "X_std": scalers["X_std"],
        "Y_mean": Y_mean,
        "Y_std": Y_std,
    }

    model = model.to(device)
    model.train()

    # Train/val split by compound (already deduplicated upstream)
    rng = np.random.RandomState(42)
    idx = rng.permutation(len(X_scaled))
    n_val = max(50, int(0.15 * len(X_scaled)))
    val_idx = idx[:n_val]
    tr_idx = idx[n_val:]

    X_tr = torch.from_numpy(X_scaled[tr_idx].astype(np.float32)).to(device)
    Y_tr = torch.from_numpy(Y_scaled[tr_idx].astype(np.float32)).to(device)
    X_va = torch.from_numpy(X_scaled[val_idx].astype(np.float32)).to(device)
    Y_va = torch.from_numpy(Y_scaled[val_idx].astype(np.float32)).to(device)

    # Lower LR for backbone, higher for output head
    backbone_params = []
    head_params = []
    for name, p in model.named_parameters():
        # Last Linear is at the highest index in Sequential
        if name.startswith(f"{len(model) - 1}."):
            head_params.append(p)
        else:
            backbone_params.append(p)

    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": lr},
        {"params": head_params, "lr": lr * 10},
    ], weight_decay=1e-4)

    best_val = float("inf")
    best_state = None
    batch_size = 256

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(X_tr), device=device)
        total_loss = 0.0
        n_batches = 0
        for start in range(0, len(X_tr), batch_size):
            batch_idx = perm[start:start + batch_size]
            xb, yb = X_tr[batch_idx], Y_tr[batch_idx]
            optimizer.zero_grad()
            pred = model(xb)
            loss = ((pred - yb) ** 2).mean()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        model.eval()
        with torch.no_grad():
            val_pred = model(X_va)
            val_loss = ((val_pred - Y_va) ** 2).mean().item()

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch+1:3d}: train_loss={total_loss/n_batches:.4f} "
                  f"val_loss={val_loss:.4f}")

    model.load_state_dict(best_state)
    model.eval()
    print(f"    Best val loss: {best_val:.4f}")
    return model, new_scalers


def predict_descriptors(model, scalers, X, device):
    import torch
    model.eval()
    X_scaled = (X - scalers["X_mean"]) / scalers["X_std"]
    with torch.no_grad():
        xt = torch.from_numpy(X_scaled.astype(np.float32)).to(device)
        y_scaled = model(xt).cpu().numpy()
    return y_scaled * scalers["Y_std"] + scalers["Y_mean"]


# ---------------------------------------------------------------------------
# Stage 2: Viscosity XGBoost
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
    val_idx, tr_idx = idx[:n_val], idx[n_val:]

    model = XGBRegressor(
        n_estimators=1000, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=1.0, min_child_weight=5,
        early_stopping_rounds=30, verbosity=0,
    )
    model.fit(X_scaled[tr_idx], y_std[tr_idx],
              eval_set=[(X_scaled[val_idx], y_std[val_idx])], verbose=False)
    return model, scaler_x, scaler_y


def predict_viscosity(model, sx, sy, X):
    return sy.inverse_transform(model.predict(sx.transform(X)).reshape(-1, 1)).flatten()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import torch

    print("=" * 70)
    print("  Phase 4 v3: Fine-tuned pretrained MLP → fuel viscosity")
    print("=" * 70)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load pretrained MLP
    print(f"\n--- Stage 1a: Load pretrained MLP ---")
    print(f"  {PRETRAINED_MODEL}")
    model, pretrained_scalers, ckpt = load_pretrained()
    model = model.to(device)

    # Load viscosity training set
    print(f"\n--- Stage 1b: Load 943-compound viscosity training set ---")
    df = pd.read_parquet(TRAINING_PARQUET)
    df = df[df["viscosity_Pa_s"] <= 30.0]
    df = df[df["viscosity_Pa_s"] >= 1e-5]
    df = df.dropna(subset=DESCRIPTOR_COLS)
    df["log_viscosity"] = np.log(df["viscosity_Pa_s"])
    print(f"  {len(df)} rows, {df['canonical_smiles'].nunique()} compounds")

    spectrum_feats = np.stack([np.array(s, dtype=np.float32) for s in df["spectrum_features"]])
    peaklist_feats = np.stack([np.array(s, dtype=np.float32) for s in df["peaklist_features"]])
    T_feats = make_temperature_features(df["T_K"].values).astype(np.float32)
    nmr_feats = np.hstack([spectrum_feats, peaklist_feats])

    compound_df = df.drop_duplicates("canonical_smiles").reset_index(drop=True)
    spec_cpd = np.stack([np.array(s, dtype=np.float32) for s in compound_df["spectrum_features"]])
    peak_cpd = np.stack([np.array(s, dtype=np.float32) for s in compound_df["peaklist_features"]])
    nmr_cpd = np.hstack([spec_cpd, peak_cpd])
    Y_cpd = compound_df[DESCRIPTOR_COLS].values.astype(np.float32)

    # Report pretrained MLP performance on viscosity compounds
    print(f"\n  Pretrained MLP R² on viscosity compounds (BEFORE fine-tuning):")
    Y_pred_before = predict_descriptors(model, pretrained_scalers, nmr_cpd, device)
    for j, name in enumerate(DESCRIPTOR_COLS):
        ss_res = ((Y_cpd[:, j] - Y_pred_before[:, j]) ** 2).sum()
        ss_tot = ((Y_cpd[:, j] - Y_cpd[:, j].mean()) ** 2).sum()
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        print(f"    {name:<22s} R²={r2:+.3f}")

    # Fine-tune
    print(f"\n--- Stage 1c: Fine-tune MLP on 943 viscosity compounds ---")
    t0 = time.time()
    model, ft_scalers = finetune_mlp(
        model, pretrained_scalers, nmr_cpd, Y_cpd, device,
        epochs=150, lr=1e-4,
    )
    print(f"  Fine-tuning done in {time.time()-t0:.1f}s")

    print(f"\n  Fine-tuned MLP R² on viscosity compounds (AFTER fine-tuning):")
    Y_pred_after = predict_descriptors(model, ft_scalers, nmr_cpd, device)
    for j, name in enumerate(DESCRIPTOR_COLS):
        ss_res = ((Y_cpd[:, j] - Y_pred_after[:, j]) ** 2).sum()
        ss_tot = ((Y_cpd[:, j] - Y_cpd[:, j].mean()) ** 2).sum()
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        mae = np.abs(Y_cpd[:, j] - Y_pred_after[:, j]).mean()
        print(f"    {name:<22s} R²={r2:+.3f}  MAE={mae:8.2f}")

    # Save fine-tuned model
    torch.save({
        "state_dict": model.state_dict(),
        "input_dim": ckpt["input_dim"],
        "hidden": ckpt["hidden"],
        "n_outputs": ckpt["n_outputs"],
        "descriptor_cols": DESCRIPTOR_COLS,
        "X_mean": ft_scalers["X_mean"],
        "X_std": ft_scalers["X_std"],
        "Y_mean": ft_scalers["Y_mean"],
        "Y_std": ft_scalers["Y_std"],
    }, FINETUNED_MODEL)
    print(f"  Saved fine-tuned model to {FINETUNED_MODEL}")

    # Stage 2: viscosity model trained on MLP-predicted descriptors
    # (match distribution to fuel inference)
    print(f"\n--- Stage 2: XGBoost NMR + FT-MLP-descriptors + T → log viscosity ---")
    desc_pred_train = predict_descriptors(model, ft_scalers, nmr_feats, device)
    X_stage2 = np.hstack([nmr_feats, desc_pred_train, T_feats])
    y_log = df["log_viscosity"].values.astype(np.float32)
    print(f"  X shape: {X_stage2.shape}")

    visc_model, visc_scx, visc_scy = train_viscosity_model(X_stage2, y_log)

    # Fuel prediction
    print(f"\n--- Loading fuels ---")
    fuels = load_all_fuels()
    print(f"  {len(fuels)} fuels loaded")

    results = {}
    for fuel_name, fuel_data in fuels.items():
        posf = FUEL_POSF.get(fuel_name, "")
        density = FUEL_DENSITIES.get(posf, 800.0)
        print(f"\n=== {fuel_name} (POSF {posf}, ρ={density:.0f} kg/m³) ===")

        fuel_spec = np.array(fuel_data["spectrum_features"], dtype=np.float32)
        fuel_peak = np.array(fuel_data["peaklist_features"], dtype=np.float32)
        fuel_nmr = np.hstack([fuel_spec, fuel_peak]).reshape(1, -1)

        pred_desc = predict_descriptors(model, ft_scalers, fuel_nmr, device)[0]
        print("  Predicted descriptors (FT-MLP):")
        for name, val in zip(DESCRIPTOR_COLS, pred_desc):
            print(f"    {name:<22s} {val:.2f}")

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
            np.tile(fuel_spec, (n, 1)),
            np.tile(fuel_peak, (n, 1)),
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
    print(f"  SUMMARY: Phase 4 v3 (fine-tuned pretrained MLP)")
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
    print(f"  Phase 4 v2 (raw MLP):      44.9%")
    print(f"  Phase 4 v3 (FT-MLP):       {np.mean(mapes):.1f}%")

    def _s(o):
        if isinstance(o, dict): return {k: _s(v) for k, v in o.items()}
        if isinstance(o, list): return [_s(v) for v in o]
        if isinstance(o, (np.floating, float)): return float(o)
        if isinstance(o, (np.integer, int)): return int(o)
        if isinstance(o, np.ndarray): return o.tolist()
        return o

    with open(OUTPUT_DIR / "phase4_v3_fuel_predictions.json", "w") as f:
        json.dump(_s(results), f, indent=2)
    print(f"\nSaved to {OUTPUT_DIR / 'phase4_v3_fuel_predictions.json'}")


if __name__ == "__main__":
    main()
