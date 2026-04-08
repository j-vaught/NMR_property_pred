"""
Phase 2 Feature-Based Training: NMR features + Ridge/XGBoost -> property prediction.

Instead of end-to-end CNN on raw spectra (which fails with 512 compounds),
extract fixed NMR features and use traditional ML models that are proven
for small datasets. Mirrors Phase 1's successful approach.

Models:
  - Ridge regression with Lasso feature selection
  - XGBoost with early stopping
  - Small neural network (same architecture as Phase 1)

All use 5-fold compound-level CV with the same splits as Phase 1/Phase 2 CNN.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge, LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shared.config import Paths
from shared.data_utils import is_hydrocarbon, make_temperature_features
from shared.metrics import compute_all_metrics
from phase2.nmr_dataset import build_nmr_property_dataset
from phase2.nmr_features import extract_features_batch, get_feature_names

paths = Paths()
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"


def _build_xy(
    nmr_features: np.ndarray,
    labels_df,
    smi_to_idx: dict,
    compound_set: set,
    property_name: str,
):
    """Build X (NMR features + temperature) and y arrays for a set of compounds."""
    X_rows = []
    y_rows = []

    for _, row in labels_df.iterrows():
        smi = row["canonical_smiles"]
        if smi not in compound_set or smi not in smi_to_idx:
            continue

        cidx = smi_to_idx[smi]
        T_K = row["T_K"]
        value = row["value"]

        # NMR features + temperature features
        t_feat = make_temperature_features(np.array([T_K]))[0]
        x = np.concatenate([nmr_features[cidx], t_feat])
        X_rows.append(x)

        if property_name == "viscosity":
            y_rows.append(np.log(value))
        else:
            y_rows.append(value)

    return np.array(X_rows, dtype=np.float32), np.array(y_rows, dtype=np.float32)


def train_ridge_with_lasso_selection(
    X_train, y_train, X_test, y_test, alpha_ridge=1.0, lasso_alpha=0.01,
):
    """Train Ridge regression with Lasso feature selection.

    Mirrors Phase 1's successful approach: Lasso selects features,
    then Ridge predicts using selected features.
    """
    # Standardize features
    scaler_x = StandardScaler().fit(X_train)
    scaler_y = StandardScaler().fit(y_train.reshape(-1, 1))

    X_tr = scaler_x.transform(X_train)
    X_te = scaler_x.transform(X_test)
    y_tr = scaler_y.transform(y_train.reshape(-1, 1)).flatten()

    # Lasso feature selection
    lasso = LassoCV(alphas=np.logspace(-3, 1, 20), cv=3, max_iter=5000)
    lasso.fit(X_tr, y_tr)
    selected = np.abs(lasso.coef_) > 1e-6

    n_selected = selected.sum()
    if n_selected < 3:
        # Fall back to all features if Lasso is too aggressive
        selected = np.ones(X_tr.shape[1], dtype=bool)
        n_selected = selected.sum()

    # Ridge on selected features
    ridge = Ridge(alpha=alpha_ridge)
    ridge.fit(X_tr[:, selected], y_tr)
    y_pred_std = ridge.predict(X_te[:, selected])

    y_pred = scaler_y.inverse_transform(y_pred_std.reshape(-1, 1)).flatten()

    return y_pred, n_selected, lasso.alpha_


def train_xgboost(X_train, y_train, X_test, y_test):
    """Train XGBoost with early stopping."""
    try:
        from xgboost import XGBRegressor
    except ImportError:
        print("  XGBoost not installed, skipping")
        return None

    scaler_y = StandardScaler().fit(y_train.reshape(-1, 1))
    y_tr = scaler_y.transform(y_train.reshape(-1, 1)).flatten()

    # Small XGBoost for small data
    model = XGBRegressor(
        n_estimators=500,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        min_child_weight=5,
        early_stopping_rounds=30,
        verbosity=0,
    )

    # Split off validation for early stopping
    n_val = max(1, int(0.15 * len(X_train)))
    perm = np.random.RandomState(42).permutation(len(X_train))
    val_idx = perm[:n_val]
    tr_idx = perm[n_val:]

    model.fit(
        X_train[tr_idx], y_tr[tr_idx],
        eval_set=[(X_train[val_idx], y_tr[val_idx])],
        verbose=False,
    )

    y_pred_std = model.predict(X_test)
    y_pred = scaler_y.inverse_transform(y_pred_std.reshape(-1, 1)).flatten()

    return y_pred


def train_small_nn(X_train, y_train, X_test, y_test, epochs=300, patience=30):
    """Train a small neural network (Phase 1 style)."""
    import torch
    import torch.nn as nn

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    scaler_x = StandardScaler().fit(X_train)
    scaler_y = StandardScaler().fit(y_train.reshape(-1, 1))

    X_tr = scaler_x.transform(X_train).astype(np.float32)
    X_te = scaler_x.transform(X_test).astype(np.float32)
    y_tr = scaler_y.transform(y_train.reshape(-1, 1)).flatten().astype(np.float32)

    # Lasso feature selection first
    lasso = LassoCV(alphas=np.logspace(-3, 1, 20), cv=3, max_iter=5000)
    lasso.fit(X_tr, y_tr)
    selected = np.abs(lasso.coef_) > 1e-6
    if selected.sum() < 3:
        selected = np.ones(X_tr.shape[1], dtype=bool)

    n_feat = int(selected.sum())
    X_tr = X_tr[:, selected]
    X_te = X_te[:, selected]

    # Small NN
    model = nn.Sequential(
        nn.Linear(n_feat, 64),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(32, 1),
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    # Train/val split
    n_val = max(1, int(0.1 * len(X_tr)))
    perm = np.random.RandomState(42).permutation(len(X_tr))
    val_idx = perm[:n_val]
    tr_idx = perm[n_val:]

    X_tr_t = torch.from_numpy(X_tr[tr_idx]).to(device)
    y_tr_t = torch.from_numpy(y_tr[tr_idx]).unsqueeze(1).to(device)
    X_val_t = torch.from_numpy(X_tr[val_idx]).to(device)
    y_val_t = torch.from_numpy(y_tr[val_idx]).unsqueeze(1).to(device)

    best_val = float("inf")
    best_state = None
    wait = 0

    for epoch in range(epochs):
        model.train()
        pred = model(X_tr_t)
        loss = nn.functional.huber_loss(pred, y_tr_t, delta=1.0)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_t)
            val_loss = nn.functional.huber_loss(val_pred, y_val_t, delta=1.0).item()

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1

        if wait >= patience:
            break

    model.load_state_dict(best_state)
    model.eval()

    X_te_t = torch.from_numpy(X_te).to(device)
    with torch.no_grad():
        y_pred_std = model(X_te_t).cpu().numpy().flatten()

    y_pred = scaler_y.inverse_transform(y_pred_std.reshape(-1, 1)).flatten()
    return y_pred, n_feat


def _to_serializable(obj):
    """Convert numpy types to Python types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_serializable(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        return float(obj)
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if obj is None:
        return None
    return obj


def main():
    print("=" * 60)
    print("  PHASE 2: NMR FEATURES -> PROPERTY PREDICTION")
    print("  Ridge / XGBoost / NN with extracted NMR features")
    print("=" * 60)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results = {}

    for property_name in ["viscosity", "surface_tension"]:
        print(f"\n{'#'*60}")
        print(f"  {property_name.upper()}")
        print(f"{'#'*60}")

        # Load data
        data = build_nmr_property_dataset(property_name)
        spectra = data["spectra"]
        labels_df = data["labels_df"]
        smiles_list = data["smiles_list"]
        hc_mask = data["hc_mask"]
        smi_to_idx = data["smi_to_idx"]

        # Extract NMR features
        print(f"  Extracting NMR features from {len(spectra)} spectra...")
        nmr_features = extract_features_batch(spectra)
        feature_names = get_feature_names()
        print(f"  Feature matrix: {nmr_features.shape} ({len(feature_names)} features)")

        # HC set for evaluation
        hc_set = {s for i, s in enumerate(smiles_list) if hc_mask[i]}

        # 5-fold CV
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        all_compounds = np.array(smiles_list)

        prop_results = {}
        t0 = time.time()

        for model_name, train_fn in [
            ("ridge", "ridge"),
            ("xgboost", "xgboost"),
            ("nn", "nn"),
        ]:
            fold_results = []
            print(f"\n  --- {model_name.upper()} ---")

            for fold, (train_cpd_idx, test_cpd_idx) in enumerate(kf.split(all_compounds)):
                train_cpds = set(all_compounds[train_cpd_idx])
                test_cpds = set(all_compounds[test_cpd_idx])

                X_train, y_train = _build_xy(nmr_features, labels_df, smi_to_idx, train_cpds, property_name)
                X_test, y_test = _build_xy(nmr_features, labels_df, smi_to_idx, test_cpds, property_name)

                if len(X_train) == 0 or len(X_test) == 0:
                    fold_results.append(None)
                    continue

                # Train model
                if train_fn == "ridge":
                    y_pred, n_sel, lasso_alpha = train_ridge_with_lasso_selection(
                        X_train, y_train, X_test, y_test
                    )
                    if fold == 0:
                        print(f"    Lasso alpha={lasso_alpha:.4f}, {n_sel} features selected")
                elif train_fn == "xgboost":
                    y_pred = train_xgboost(X_train, y_train, X_test, y_test)
                    if y_pred is None:
                        fold_results.append(None)
                        continue
                elif train_fn == "nn":
                    y_pred, n_feat = train_small_nn(X_train, y_train, X_test, y_test)
                    if fold == 0:
                        print(f"    NN using {n_feat} features")

                # Compute metrics
                m_std = compute_all_metrics(y_test, y_pred)

                if property_name == "viscosity":
                    y_true_real = np.exp(y_test)
                    y_pred_real = np.exp(y_pred)
                else:
                    y_true_real = y_test
                    y_pred_real = np.clip(y_pred, 1e-6, None)
                m_real = compute_all_metrics(y_true_real, y_pred_real)

                # HC-only
                test_labels = labels_df[labels_df["canonical_smiles"].isin(test_cpds)]
                hc_indices = []
                idx = 0
                for _, row in test_labels.iterrows():
                    smi = row["canonical_smiles"]
                    if smi in smi_to_idx:
                        if smi in hc_set:
                            hc_indices.append(idx)
                        idx += 1

                m_hc = None
                if hc_indices:
                    hc_idx = np.array(hc_indices)
                    if max(hc_idx) < len(y_test):
                        if property_name == "viscosity":
                            m_hc = compute_all_metrics(np.exp(y_test[hc_idx]), np.exp(y_pred[hc_idx]))
                        else:
                            m_hc = compute_all_metrics(y_test[hc_idx], np.clip(y_pred[hc_idx], 1e-6, None))

                unit = "log" if property_name == "viscosity" else "N/m"
                print(f"    F{fold+1}: {unit} R2={m_std['r2']:.3f} | Real R2={m_real['r2']:.3f}", end="")
                if m_hc:
                    print(f" | HC R2={m_hc['r2']:.3f} ({len(hc_indices)} pts)")
                else:
                    print(f" | HC: 0 pts")

                fold_results.append({
                    "fold": fold,
                    "std_metrics": m_std,
                    "real_metrics": m_real,
                    "hc_metrics": m_hc,
                    "n_hc_test_pts": len(hc_indices),
                })

            # Summary for this model
            valid = [fr for fr in fold_results if fr is not None]
            if valid:
                for key, label in [("std_metrics", "Standardized"), ("real_metrics", "Real units"), ("hc_metrics", "HC only")]:
                    r2_vals = [fr[key]["r2"] for fr in valid if fr.get(key)]
                    if r2_vals:
                        print(f"    {label}: R2={np.mean(r2_vals):.4f} +/- {np.std(r2_vals):.4f}")

            prop_results[f"{model_name}_direct"] = fold_results

        elapsed = time.time() - t0
        print(f"\n  {property_name.upper()} total time: {elapsed:.0f}s")
        results[property_name] = prop_results

        # Save intermediate
        with open(OUTPUT_DIR / "phase2_features_results.json", "w") as f:
            json.dump(_to_serializable(results), f, indent=2)

    # Final comparison with Phase 1
    print(f"\n{'='*60}")
    print("  PHASE 1 vs PHASE 2 FEATURE-BASED COMPARISON")
    print(f"{'='*60}")

    p1_path = paths.root / "src" / "phase1" / "outputs" / "direct_optimal_results.json"
    if p1_path.exists():
        with open(p1_path) as f:
            p1 = json.load(f)

        for prop in ["viscosity", "surface_tension"]:
            if prop not in p1 or prop not in results:
                continue
            print(f"\n  {prop.upper()}:")
            print(f"  {'Method':<35s} {'log/std R2':>10s} {'Real R2':>10s} {'HC R2':>10s}")
            print(f"  {'-'*65}")

            # Phase 1
            p1_folds = p1[prop]
            p1_std = np.mean([f["std_metrics"]["r2"] for f in p1_folds])
            p1_real = np.mean([f["real_metrics"]["r2"] for f in p1_folds])
            p1_hc = np.mean([f["hc_metrics"]["r2"] for f in p1_folds if f["hc_metrics"]])
            print(f"  {'Phase1 FP+desc':<35s} {p1_std:>10.3f} {p1_real:>10.3f} {p1_hc:>10.3f}")

            # Phase 2 feature models
            for model_key in results[prop]:
                folds = [f for f in results[prop][model_key] if f is not None]
                if not folds:
                    continue
                std = np.mean([f["std_metrics"]["r2"] for f in folds])
                real = np.mean([f["real_metrics"]["r2"] for f in folds])
                hc_vals = [f["hc_metrics"]["r2"] for f in folds if f.get("hc_metrics")]
                hc = np.mean(hc_vals) if hc_vals else float("nan")
                print(f"  {'NMR ' + model_key:<35s} {std:>10.3f} {real:>10.3f} {hc:>10.3f}")

    print(f"\nResults saved to {OUTPUT_DIR / 'phase2_features_results.json'}")


if __name__ == "__main__":
    main()
