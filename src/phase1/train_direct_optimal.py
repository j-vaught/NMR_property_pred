"""
Phase 1 Direct Optimal: Predict ln(viscosity) and surface_tension directly
with temperature as input. Uses ALL 856/714 compounds (no Arrhenius filtering).
Combines: Lasso feature selection + Huber loss + 5-fold CV + ensemble.
"""

import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shared.config import Paths
from shared.data_utils import (
    canonical_smiles, inchi_to_canonical_smiles,
    is_hydrocarbon, make_temperature_features,
)
from shared.metrics import compute_all_metrics
from phase1.model import SingleTaskModel, count_parameters

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


def load_features():
    fp = pd.read_csv(paths.morgan_r2)
    fp["canonical_smiles"] = fp["SMILES"].apply(canonical_smiles)
    fp = fp.dropna(subset=["canonical_smiles"]).set_index("canonical_smiles")
    bit_cols = [c for c in fp.columns if c not in ("CAS", "SMILES")]
    fp = fp[bit_cols].astype(np.float32)
    fp = fp[~fp.index.duplicated(keep="first")]
    fp = fp.loc[:, fp.std() > 0]

    desc = pd.read_csv(paths.features / "rdkit_descriptors.csv")
    desc["canonical_smiles"] = desc["SMILES"].apply(canonical_smiles)
    desc = desc.dropna(subset=["canonical_smiles"]).set_index("canonical_smiles")
    desc_cols = [c for c in desc.columns if c not in ("CAS", "SMILES")]
    desc = desc[desc_cols].astype(np.float32)
    desc = desc[~desc.index.duplicated(keep="first")]
    desc = desc.replace([np.inf, -np.inf], np.nan).fillna(0)
    desc = desc.loc[:, desc.std() > 0]

    common = fp.index.intersection(desc.index)
    features = pd.concat([fp.loc[common], desc.loc[common]], axis=1)
    print(f"[Features] {features.shape[0]} compounds, {features.shape[1]} dims")
    return features


def load_labels(property_name):
    """Load ThermoML + chemicals GC labels."""
    dfs = []

    # ThermoML
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

    thermoml = pd.DataFrame({
        "canonical_smiles": df["canonical_smiles"],
        "T_K": pd.to_numeric(df[t_col], errors="coerce"),
        "value": pd.to_numeric(df[val_col], errors="coerce"),
        "source": "thermoml",
    }).dropna()

    if "P_kPa" in df.columns:
        p = pd.to_numeric(df.loc[thermoml.index, "P_kPa"], errors="coerce")
        thermoml = thermoml[p.isna() | (p <= 110)]

    dfs.append(thermoml)

    # GC expanded
    if property_name == "viscosity":
        gc_path = paths.thermo / "group_contribution" / "chemicals_viscosity_expanded.csv"
    else:
        gc_path = paths.thermo / "group_contribution" / "chemicals_surface_tension_expanded.csv"

    if gc_path.exists():
        gc = pd.read_csv(gc_path)
        gc["canonical_smiles"] = gc["SMILES"].apply(canonical_smiles)
        gc = gc.dropna(subset=["canonical_smiles"])
        gc_val = [c for c in gc.columns if "viscosity" in c.lower() or "surface" in c.lower()][0]
        gc_df = pd.DataFrame({
            "canonical_smiles": gc["canonical_smiles"],
            "T_K": pd.to_numeric(gc["T_K"], errors="coerce"),
            "value": pd.to_numeric(gc[gc_val], errors="coerce"),
            "source": "gc",
        }).dropna()
        dfs.append(gc_df)

    combined = pd.concat(dfs, ignore_index=True)
    combined = combined[(combined["T_K"] >= 200) & (combined["T_K"] <= 500)]

    if property_name == "viscosity":
        combined = combined[(combined["value"] > 0) & (combined["value"] < 100)]
    else:
        combined = combined[(combined["value"] > 0) & (combined["value"] < 1)]

    # Dedup: prefer thermoml
    combined["T_round"] = combined["T_K"].round(1)
    combined["priority"] = combined["source"].map({"thermoml": 0, "gc": 1})
    combined = combined.sort_values("priority").drop_duplicates(
        subset=["canonical_smiles", "T_round"], keep="first"
    ).drop(columns=["T_round", "priority"])

    print(f"[Labels {property_name}] {len(combined)} rows, {combined['canonical_smiles'].nunique()} compounds")
    return combined


def build_direct_data(features, labels, compound_list, property_name, scaler_x=None, scaler_y=None, selected_cols=None):
    """Build (FP+desc, T_features, target) tensors for direct prediction."""
    sub = labels[labels["canonical_smiles"].isin(compound_list)]
    sub = sub[sub["canonical_smiles"].isin(features.index)]

    X_mol, T_feats, targets = [], [], []
    for _, row in sub.iterrows():
        smi = row["canonical_smiles"]
        fp = features.loc[smi].values
        t = make_temperature_features(np.array([row["T_K"]]))[0]
        val = np.log(row["value"]) if property_name == "viscosity" else row["value"]
        X_mol.append(fp)
        T_feats.append(t)
        targets.append(val)

    X_mol = np.array(X_mol, dtype=np.float32)
    T_feats = np.array(T_feats, dtype=np.float32)
    targets = np.array(targets, dtype=np.float32).reshape(-1, 1)

    # Apply feature selection
    if selected_cols is not None:
        X_mol = X_mol[:, selected_cols]

    if scaler_x is None:
        scaler_x = StandardScaler().fit(X_mol)
    X_mol = scaler_x.transform(X_mol).astype(np.float32)

    if scaler_y is None:
        scaler_y = StandardScaler().fit(targets)
    targets = scaler_y.transform(targets).astype(np.float32)

    return (
        torch.tensor(X_mol), torch.tensor(T_feats), torch.tensor(targets),
        scaler_x, scaler_y,
    )


def select_features_lasso(X, Y, alpha=0.1):
    sx = StandardScaler().fit(X)
    sy = StandardScaler().fit(Y)
    Xs, Ys = sx.transform(X), sy.transform(Y)

    selected = set()
    for col in range(Ys.shape[1]):
        lasso = Lasso(alpha=alpha, max_iter=10000)
        lasso.fit(Xs, Ys[:, col])
        nonzero = np.where(np.abs(lasso.coef_) > 1e-6)[0]
        selected.update(nonzero)

    selected = sorted(selected)
    print(f"[Lasso] {len(selected)} features selected from {X.shape[1]}")
    return selected


def train_one_model(X_tr, T_tr, Y_tr, X_vl, T_vl, Y_vl, seed, epochs=500, lr=3e-4, patience=50):
    set_seed(seed)
    device = get_device()

    ds = TensorDataset(X_tr, T_tr, Y_tr)
    loader = DataLoader(ds, batch_size=min(256, len(X_tr)), shuffle=True)

    model = SingleTaskModel(
        fp_dim=X_tr.shape[1], t_feature_dim=T_tr.shape[1], dropout=0.3
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    warmup = 10

    def lr_lambda(ep):
        if ep < warmup:
            return (ep + 1) / warmup
        progress = (ep - warmup) / max(1, epochs - warmup)
        return max(1e-6 / lr, 0.5 * (1 + np.cos(np.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    X_vl_d, T_vl_d, Y_vl_d = X_vl.to(device), T_vl.to(device), Y_vl.to(device)

    best_val = float("inf")
    best_state = None
    wait = 0

    for epoch in range(epochs):
        model.train()
        for xb, tb, yb in loader:
            xb, tb, yb = xb.to(device), tb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb, tb)
            loss = nn.functional.huber_loss(pred, yb, delta=1.0)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            val_loss = nn.functional.huber_loss(model(X_vl_d, T_vl_d), Y_vl_d).item()

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
        if wait >= patience:
            break

    model.load_state_dict(best_state)
    return model


def run_property(property_name, n_folds=5, n_ensemble=5, lasso_alpha=0.1):
    print(f"\n{'#'*60}")
    print(f"  {property_name.upper()} — Direct Optimal")
    print(f"  {n_folds}-fold CV, {n_ensemble} ensemble, Lasso α={lasso_alpha}")
    print(f"{'#'*60}")

    features = load_features()
    labels = load_labels(property_name)

    # Get all compounds that have both features and labels
    joined = labels[labels["canonical_smiles"].isin(features.index)]
    all_compounds = sorted(joined["canonical_smiles"].unique())
    hc_set = {s for s in all_compounds if is_hydrocarbon(s)}
    print(f"[Joined] {len(all_compounds)} compounds ({len(hc_set)} HC), {len(joined)} rows")

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    fold_results = []
    t0 = time.time()

    for fold, (train_cpd_idx, test_cpd_idx) in enumerate(kf.split(all_compounds)):
        train_cpds = [all_compounds[i] for i in train_cpd_idx]
        test_cpds = [all_compounds[i] for i in test_cpd_idx]

        print(f"\n--- Fold {fold+1}/{n_folds} (train={len(train_cpds)} cpds, test={len(test_cpds)} cpds) ---")

        # Get training data for Lasso selection
        train_labels = joined[joined["canonical_smiles"].isin(train_cpds)]
        X_for_lasso = np.array([features.loc[s].values for s in train_cpds if s in features.index], dtype=np.float32)

        # Lasso targets: mean property per compound
        Y_for_lasso = []
        lasso_cpds = []
        for s in train_cpds:
            if s not in features.index:
                continue
            cpd = train_labels[train_labels["canonical_smiles"] == s]
            if len(cpd) == 0:
                continue
            val = np.log(cpd["value"].mean()) if property_name == "viscosity" else cpd["value"].mean()
            Y_for_lasso.append(val)
            lasso_cpds.append(s)

        X_for_lasso = np.array([features.loc[s].values for s in lasso_cpds], dtype=np.float32)
        Y_for_lasso = np.array(Y_for_lasso, dtype=np.float32).reshape(-1, 1)

        selected = select_features_lasso(X_for_lasso, Y_for_lasso, alpha=lasso_alpha)
        if len(selected) < 10:
            selected = list(range(features.shape[1]))
            print(f"  Lasso too aggressive, using all {len(selected)} features")

        # Build tensors with selected features
        train_data = build_direct_data(features, joined, train_cpds, property_name, selected_cols=selected)
        X_tr, T_tr, Y_tr, scaler_x, scaler_y = train_data

        test_data = build_direct_data(features, joined, test_cpds, property_name, scaler_x, scaler_y, selected)
        X_te, T_te, Y_te, _, _ = test_data

        # Split train into train/val (90/10)
        n_val = max(1, int(0.1 * len(X_tr)))
        perm = np.random.RandomState(42 + fold).permutation(len(X_tr))
        vi, ti = perm[:n_val], perm[n_val:]

        # Train ensemble
        models = []
        for e in range(n_ensemble):
            model = train_one_model(
                X_tr[ti], T_tr[ti], Y_tr[ti],
                X_tr[vi], T_tr[vi], Y_tr[vi],
                seed=42 + fold * 1000 + e * 111,
            )
            models.append(model)

        # Ensemble predict on test
        device = get_device()
        all_preds = []
        for model in models:
            model.eval()
            with torch.no_grad():
                pred = model(X_te.to(device), T_te.to(device)).cpu().numpy()
            all_preds.append(scaler_y.inverse_transform(pred))
        ensemble_pred = np.mean(all_preds, axis=0).flatten()
        y_true = scaler_y.inverse_transform(Y_te.numpy()).flatten()

        # Metrics in standardized space
        m_std = compute_all_metrics(y_true, ensemble_pred)

        # Metrics in real units
        if property_name == "viscosity":
            y_true_real = np.exp(y_true)
            y_pred_real = np.exp(ensemble_pred)
        else:
            y_true_real = y_true
            y_pred_real = np.clip(ensemble_pred, 1e-6, None)
        m_real = compute_all_metrics(y_true_real, y_pred_real)

        # HC-only metrics
        test_labels = joined[joined["canonical_smiles"].isin(test_cpds)]
        hc_indices = []
        idx = 0
        for _, row in test_labels.iterrows():
            if row["canonical_smiles"] in features.index:
                if row["canonical_smiles"] in hc_set:
                    hc_indices.append(idx)
                idx += 1

        m_hc = None
        if hc_indices:
            hc_idx = np.array(hc_indices)
            if property_name == "viscosity":
                m_hc = compute_all_metrics(np.exp(y_true[hc_idx]), np.exp(ensemble_pred[hc_idx]))
            else:
                m_hc = compute_all_metrics(y_true[hc_idx], np.clip(ensemble_pred[hc_idx], 1e-6, None))

        n_hc_test = len(hc_indices)
        unit = "log" if property_name == "viscosity" else "N/m"
        print(f"  {unit} R²={m_std['r2']:.3f} | Real R²={m_real['r2']:.3f} MAPE={m_real['mape']:.1f}%", end="")
        if m_hc:
            print(f" | HC R²={m_hc['r2']:.3f} MAPE={m_hc['mape']:.1f}% ({n_hc_test} pts)")
        else:
            print(f" | HC: 0 pts")

        fold_results.append({
            "fold": fold,
            "n_features": len(selected),
            "n_train_cpds": len(train_cpds),
            "n_test_cpds": len(test_cpds),
            "n_hc_test_pts": n_hc_test,
            "std_metrics": m_std,
            "real_metrics": m_real,
            "hc_metrics": m_hc,
        })

    # Summary
    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"  {property_name.upper()} SUMMARY ({elapsed:.0f}s)")
    print(f"{'='*60}")

    for key, label in [("std_metrics", "Standardized"), ("real_metrics", "Real units"), ("hc_metrics", "HC only")]:
        vals = {}
        for metric in ["r2", "mape", "rmse"]:
            fold_vals = [fr[key][metric] for fr in fold_results if fr[key] is not None and metric in fr[key]]
            if fold_vals:
                vals[metric] = (np.mean(fold_vals), np.std(fold_vals))
        if vals:
            print(f"\n  {label}:")
            for metric, (mean, std) in vals.items():
                print(f"    {metric}: {mean:.4f} ± {std:.4f}")

    return fold_results


def main():
    print("=" * 60)
    print("  PHASE 1 DIRECT OPTIMAL")
    print("  All compounds, T as input, Lasso + Huber + 5-fold + ensemble")
    print("=" * 60)

    results = {}
    results["viscosity"] = run_property("viscosity")
    results["surface_tension"] = run_property("surface_tension")

    out = paths.root / "src" / "phase1" / "outputs"
    out.mkdir(parents=True, exist_ok=True)

    def to_ser(obj):
        if isinstance(obj, dict):
            return {k: to_ser(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [to_ser(v) for v in obj]
        if isinstance(obj, (np.floating, float)):
            return float(obj)
        if isinstance(obj, (np.integer, int)):
            return int(obj)
        return obj

    with open(out / "direct_optimal_results.json", "w") as f:
        json.dump(to_ser(results), f, indent=2)
    print(f"\nResults saved to {out / 'direct_optimal_results.json'}")


if __name__ == "__main__":
    main()
