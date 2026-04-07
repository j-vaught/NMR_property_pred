"""
Phase 1 Optimal: Combines all 5 research findings into one training pipeline.
1. Lasso feature selection (~100 features from 1696)
2. ThermoML-only Arrhenius fitting (recovers 112 compounds)
3. Hybrid physics loss (lambda=0.5)
4. 5-fold compound-level CV
5. 5-model ensemble per fold
"""

import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shared.config import Paths, DataConfig
from shared.data_utils import (
    canonical_smiles, inchi_to_canonical_smiles,
    fit_arrhenius, fit_linear_st, is_hydrocarbon,
)
from shared.metrics import compute_all_metrics
from phase1.model import ArrheniusModel, count_parameters
from phase1.physics_loss import HybridArrheniusLoss

paths = Paths()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ── Data Loading ─────────────────────────────────────────────────────────────

def load_features():
    """Load Morgan FP + RDKit descriptors, return DataFrame indexed by canonical SMILES."""
    fp = pd.read_csv(paths.morgan_r2)
    fp["canonical_smiles"] = fp["SMILES"].apply(canonical_smiles)
    fp = fp.dropna(subset=["canonical_smiles"])
    bit_cols = [c for c in fp.columns if c not in ("CAS", "SMILES", "canonical_smiles")]
    fp = fp.set_index("canonical_smiles")[bit_cols].astype(np.float32)
    fp = fp[~fp.index.duplicated(keep="first")]
    fp = fp.loc[:, fp.std() > 0]

    desc = pd.read_csv(paths.features / "rdkit_descriptors.csv")
    desc["canonical_smiles"] = desc["SMILES"].apply(canonical_smiles)
    desc = desc.dropna(subset=["canonical_smiles"])
    desc_cols = [c for c in desc.columns if c not in ("CAS", "SMILES", "canonical_smiles")]
    desc = desc.set_index("canonical_smiles")[desc_cols].astype(np.float32)
    desc = desc[~desc.index.duplicated(keep="first")]
    desc = desc.replace([np.inf, -np.inf], np.nan).fillna(0)
    desc = desc.loc[:, desc.std() > 0]
    desc = (desc - desc.mean()) / desc.std().replace(0, 1)

    common = fp.index.intersection(desc.index)
    features = pd.concat([fp.loc[common], desc.loc[common]], axis=1)
    print(f"[Features] {features.shape[0]} compounds, {features.shape[1]} raw features")
    return features


def load_thermoml_labels(property_name):
    """Load ONLY ThermoML labels (not GC) to avoid poisoning Arrhenius fits."""
    if property_name == "viscosity":
        df = pd.read_csv(paths.thermoml_viscosity)
        val_col = [c for c in df.columns if "viscosity" in c.lower()][0]
    else:
        df = pd.read_csv(paths.thermoml_surface_tension)
        val_col = [c for c in df.columns if "surface" in c.lower() or "tension" in c.lower()][0]

    df = df.dropna(subset=["InChI"])
    df["canonical_smiles"] = df["InChI"].apply(inchi_to_canonical_smiles)
    df = df.dropna(subset=["canonical_smiles"])

    t_col = "T_K" if "T_K" in df.columns else [c for c in df.columns if "temp" in c.lower()][0]

    result = pd.DataFrame({
        "canonical_smiles": df["canonical_smiles"],
        "T_K": pd.to_numeric(df[t_col], errors="coerce"),
        "value": pd.to_numeric(df[val_col], errors="coerce"),
    }).dropna()

    result = result[(result["T_K"] >= 200) & (result["T_K"] <= 500)]

    if "P_kPa" in df.columns:
        p_vals = pd.to_numeric(df.loc[result.index, "P_kPa"], errors="coerce")
        result = result[p_vals.isna() | (p_vals <= 110)]

    if property_name == "viscosity":
        result = result[(result["value"] > 0) & (result["value"] < 100)]
    else:
        result = result[(result["value"] > 0) & (result["value"] < 1)]

    print(f"[ThermoML {property_name}] {len(result)} rows, {result['canonical_smiles'].nunique()} compounds")
    return result


def build_arrhenius_targets(features, labels, property_name, min_pts=5, r2_thresh=0.95):
    """Fit Arrhenius/linear per compound. Return X, Y, smiles, hc_mask."""
    joined = labels[labels["canonical_smiles"].isin(features.index)]

    X_list, Y_list, smiles_list = [], [], []
    for smi, grp in joined.groupby("canonical_smiles"):
        if len(grp) < min_pts:
            continue
        T = grp["T_K"].values
        vals = grp["value"].values
        if property_name == "viscosity":
            A, B, r2 = fit_arrhenius(T, vals)
        else:
            A, B, r2 = fit_linear_st(T, vals)
        if r2 >= r2_thresh:
            X_list.append(features.loc[smi].values)
            Y_list.append([A, B])
            smiles_list.append(smi)

    X = np.array(X_list, dtype=np.float32)
    Y = np.array(Y_list, dtype=np.float32)
    hc_mask = np.array([is_hydrocarbon(s) for s in smiles_list])
    print(f"[Arrhenius {property_name}] {len(X)} compounds pass filter ({hc_mask.sum()} HC)")
    return X, Y, smiles_list, hc_mask


# ── Lasso Feature Selection ─────────────────────────────────────────────────

def select_features_lasso(X_train, Y_train, alpha=0.1):
    """Use Lasso to select informative features. Returns selected column indices."""
    from sklearn.linear_model import Lasso

    scaler_x = StandardScaler().fit(X_train)
    scaler_y = StandardScaler().fit(Y_train)
    X_s = scaler_x.transform(X_train)
    Y_s = scaler_y.transform(Y_train)

    # Fit Lasso on each target, union selected features
    selected = set()
    for col in range(Y_s.shape[1]):
        lasso = Lasso(alpha=alpha, max_iter=10000)
        lasso.fit(X_s, Y_s[:, col])
        nonzero = np.where(np.abs(lasso.coef_) > 1e-6)[0]
        selected.update(nonzero)

    selected = sorted(selected)
    print(f"[Lasso] alpha={alpha}: {len(selected)} features selected from {X_train.shape[1]}")
    return selected


# ── Training ─────────────────────────────────────────────────────────────────

def train_single_model(X_train, Y_train, X_val, Y_val, scaler_y, seed,
                        property_name, epochs=500, lr=3e-4, patience=50):
    """Train one ArrheniusModel with hybrid physics loss."""
    set_seed(seed)
    device = get_device()

    X_tr_t = torch.tensor(X_train, dtype=torch.float32)
    Y_tr_t = torch.tensor(scaler_y.transform(Y_train), dtype=torch.float32)
    X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)
    Y_val_t = torch.tensor(scaler_y.transform(Y_val), dtype=torch.float32).to(device)

    ds = TensorDataset(X_tr_t, Y_tr_t)
    loader = DataLoader(ds, batch_size=min(64, len(X_tr_t)), shuffle=True)

    model = ArrheniusModel(fp_dim=X_train.shape[1], dropout=0.3).to(device)

    # Hybrid physics loss
    if property_name == "viscosity":
        loss_fn = HybridArrheniusLoss(
            lam=0.5,
            scaler_mean=scaler_y.mean_.tolist(),
            scaler_scale=scaler_y.scale_.tolist(),
        ).to(device)
    else:
        # For surface tension, just use Huber on coefficients (linear model, no exp amplification)
        loss_fn = lambda pred, true: nn.functional.huber_loss(pred, true, delta=1.0)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    warmup = 10
    def lr_lambda(ep):
        if ep < warmup:
            return (ep + 1) / warmup
        progress = (ep - warmup) / max(1, epochs - warmup)
        return max(1e-6 / lr, 0.5 * (1 + np.cos(np.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    best_val = float("inf")
    best_state = None
    wait = 0

    for epoch in range(epochs):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        scheduler.step()

        model.eval()
        with torch.no_grad():
            pred_val = model(X_val_t)
            val_loss = loss_fn(pred_val, Y_val_t).item()

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1

        if wait >= patience:
            break

    model.load_state_dict(best_state)
    return model, best_val


def evaluate_arrhenius(models, X_test, Y_test, smiles_test, hc_mask_test,
                        scaler_y, labels_df, property_name, device):
    """Evaluate ensemble on test set."""
    X_t = torch.tensor(X_test, dtype=torch.float32).to(device)

    # Ensemble prediction (average in coefficient space)
    all_preds = []
    for model in models:
        model.eval()
        with torch.no_grad():
            pred_s = model(X_t).cpu().numpy()
        pred = scaler_y.inverse_transform(pred_s)
        all_preds.append(pred)

    ensemble_pred = np.mean(all_preds, axis=0)

    # [A,B] metrics
    ab_metrics = compute_all_metrics(Y_test.flatten(), ensemble_pred.flatten())

    # Reconstruct at actual temperatures
    all_true, all_pred, all_hc = [], [], []
    for i, smi in enumerate(smiles_test):
        A_p, B_p = ensemble_pred[i]
        cpd = labels_df[labels_df["canonical_smiles"] == smi]
        hc = hc_mask_test[i]
        for _, row in cpd.iterrows():
            T = row["T_K"]
            if property_name == "viscosity":
                pv = np.exp(A_p + B_p / T)
            else:
                pv = max(A_p + B_p * T, 1e-6)
            all_true.append(row["value"])
            all_pred.append(pv)
            all_hc.append(hc)

    all_true, all_pred, all_hc = np.array(all_true), np.array(all_pred), np.array(all_hc)
    recon_all = compute_all_metrics(all_true, all_pred)

    recon_hc = None
    if all_hc.any():
        recon_hc = compute_all_metrics(all_true[all_hc], all_pred[all_hc])

    return ab_metrics, recon_all, recon_hc


# ── Main Pipeline ────────────────────────────────────────────────────────────

import pandas as pd


def run_property(property_name, n_folds=5, n_ensemble=5, lasso_alpha=0.1):
    """Full pipeline for one property: Lasso selection + k-fold CV + ensemble."""
    print(f"\n{'#'*60}")
    print(f"  {property_name.upper()} — Optimal Pipeline")
    print(f"  Lasso α={lasso_alpha}, {n_folds}-fold CV, {n_ensemble}-model ensemble")
    print(f"{'#'*60}")

    features = load_features()
    labels = load_thermoml_labels(property_name)
    X, Y, smiles, hc_mask = build_arrhenius_targets(features, labels, property_name)

    if len(X) < 30:
        print(f"Too few compounds ({len(X)}), skipping")
        return None

    device = get_device()
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    fold_results = []
    t0 = time.time()

    for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
        print(f"\n--- Fold {fold+1}/{n_folds} (train={len(train_idx)}, test={len(test_idx)}) ---")

        X_train_raw, X_test_raw = X[train_idx], X[test_idx]
        Y_train, Y_test = Y[train_idx], Y[test_idx]
        smiles_test = [smiles[i] for i in test_idx]
        hc_test = hc_mask[test_idx]

        # Lasso feature selection on training data only
        selected = select_features_lasso(X_train_raw, Y_train, alpha=lasso_alpha)
        if len(selected) < 5:
            print(f"  Too few features selected ({len(selected)}), using all")
            selected = list(range(X_train_raw.shape[1]))

        X_train = X_train_raw[:, selected]
        X_test = X_test_raw[:, selected]

        # Standardize features
        scaler_x = StandardScaler().fit(X_train)
        X_train_s = scaler_x.transform(X_train).astype(np.float32)
        X_test_s = scaler_x.transform(X_test).astype(np.float32)

        # Standardize targets
        scaler_y = StandardScaler().fit(Y_train)

        # Split train into train/val for early stopping (90/10)
        n_val = max(1, int(0.1 * len(X_train_s)))
        perm = np.random.RandomState(42 + fold).permutation(len(X_train_s))
        val_idx = perm[:n_val]
        tr_idx = perm[n_val:]

        X_tr, X_vl = X_train_s[tr_idx], X_train_s[val_idx]
        Y_tr, Y_vl = Y_train[tr_idx], Y_train[val_idx]

        # Train ensemble
        models = []
        for e in range(n_ensemble):
            seed = 42 + fold * 1000 + e * 111
            model, val_loss = train_single_model(
                X_tr, Y_tr, X_vl, Y_vl, scaler_y, seed, property_name,
                epochs=500, lr=3e-4, patience=50,
            )
            models.append(model)

        # Evaluate on test fold
        ab_m, recon_all, recon_hc = evaluate_arrhenius(
            models, X_test_s, Y_test, smiles_test, hc_test,
            scaler_y, labels, property_name, device,
        )

        n_hc_test = int(hc_test.sum())
        print(f"  [A,B] R²={ab_m['r2']:.3f} | Recon R²={recon_all['r2']:.3f} MAPE={recon_all['mape']:.1f}%", end="")
        if recon_hc:
            print(f" | HC R²={recon_hc['r2']:.3f} MAPE={recon_hc['mape']:.1f}% ({n_hc_test} cpds)")
        else:
            print(f" | HC: 0 cpds")

        fold_results.append({
            "fold": fold,
            "ab": ab_m,
            "recon_all": recon_all,
            "recon_hc": recon_hc,
            "n_test": len(test_idx),
            "n_hc_test": n_hc_test,
            "n_features": len(selected),
        })

    # Aggregate
    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"  {property_name.upper()} SUMMARY ({elapsed:.0f}s)")
    print(f"{'='*60}")

    for key_group, label in [("ab", "[A,B] coefficients"), ("recon_all", "Reconstructed (ALL)"), ("recon_hc", "Reconstructed (HC)")]:
        vals = {}
        for metric in ["r2", "mape", "rmse"]:
            fold_vals = [fr[key_group][metric] for fr in fold_results if fr[key_group] is not None and metric in fr[key_group]]
            if fold_vals:
                vals[metric] = (np.mean(fold_vals), np.std(fold_vals))

        if vals:
            print(f"\n  {label}:")
            for metric, (mean, std) in vals.items():
                print(f"    {metric}: {mean:.4f} ± {std:.4f}")

    return fold_results


def main():
    print("=" * 60)
    print("  PHASE 1 OPTIMAL: All 5 improvements combined")
    print("  1. Lasso feature selection")
    print("  2. ThermoML-only (no GC data)")
    print("  3. Hybrid physics loss (λ=0.5)")
    print("  4. 5-fold compound-level CV")
    print("  5. 5-model ensemble")
    print("=" * 60)

    results = {}
    results["viscosity"] = run_property("viscosity")
    results["surface_tension"] = run_property("surface_tension")

    out = paths.root / "src" / "phase1" / "outputs"
    out.mkdir(parents=True, exist_ok=True)

    # Serialize (convert numpy types)
    def to_serializable(obj):
        if isinstance(obj, dict):
            return {k: to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [to_serializable(v) for v in obj]
        elif isinstance(obj, (np.floating, float)):
            return float(obj)
        elif isinstance(obj, (np.integer, int)):
            return int(obj)
        return obj

    with open(out / "optimal_results.json", "w") as f:
        json.dump(to_serializable(results), f, indent=2)

    print(f"\nResults saved to {out / 'optimal_results.json'}")


if __name__ == "__main__":
    main()
