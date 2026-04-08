"""
Phase 2 Feature-Based Training: compare NMR-only vs combined NMR+FP+desc.

Trains Ridge/XGBoost/NN on four feature combinations to isolate the
contribution of NMR features, peak-list features, and Phase 1 descriptors.

Requires running precompute_features.py first to create the cache.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shared.config import Paths
from shared.data_utils import make_temperature_features
from shared.metrics import compute_all_metrics

paths = Paths()
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
CACHE_DIR = Path(__file__).resolve().parent / "cache"


def _build_xy(
    feature_matrix: np.ndarray,
    labels_df,
    smi_to_idx: dict,
    compound_set: set,
    property_name: str,
):
    """Build X (features + temperature) and y arrays for a set of compounds."""
    X_rows = []
    y_rows = []

    for _, row in labels_df.iterrows():
        smi = row["canonical_smiles"]
        if smi not in compound_set or smi not in smi_to_idx:
            continue

        cidx = smi_to_idx[smi]
        T_K = row["T_K"]
        value = row["value"]

        t_feat = make_temperature_features(np.array([T_K]))[0]
        x = np.concatenate([feature_matrix[cidx], t_feat])
        X_rows.append(x)

        if property_name == "viscosity":
            y_rows.append(np.log(value))
        else:
            y_rows.append(value)

    return np.array(X_rows, dtype=np.float32), np.array(y_rows, dtype=np.float32)


def train_ridge(X_train, y_train, X_test, y_test):
    """Ridge with Lasso feature selection."""
    scaler_x = StandardScaler().fit(X_train)
    scaler_y = StandardScaler().fit(y_train.reshape(-1, 1))

    X_tr = scaler_x.transform(X_train)
    X_te = scaler_x.transform(X_test)
    y_tr = scaler_y.transform(y_train.reshape(-1, 1)).flatten()

    lasso = LassoCV(alphas=np.logspace(-3, 1, 20), cv=3, max_iter=5000)
    lasso.fit(X_tr, y_tr)
    selected = np.abs(lasso.coef_) > 1e-6
    if selected.sum() < 3:
        selected = np.ones(X_tr.shape[1], dtype=bool)

    ridge = Ridge(alpha=1.0)
    ridge.fit(X_tr[:, selected], y_tr)
    y_pred_std = ridge.predict(X_te[:, selected])
    y_pred = scaler_y.inverse_transform(y_pred_std.reshape(-1, 1)).flatten()

    return y_pred, int(selected.sum())


def train_xgboost(X_train, y_train, X_test, y_test):
    """XGBoost with early stopping."""
    try:
        from xgboost import XGBRegressor
    except ImportError:
        return None

    scaler_y = StandardScaler().fit(y_train.reshape(-1, 1))
    y_tr = scaler_y.transform(y_train.reshape(-1, 1)).flatten()

    model = XGBRegressor(
        n_estimators=500, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
        min_child_weight=5, early_stopping_rounds=30, verbosity=0,
    )

    n_val = max(1, int(0.15 * len(X_train)))
    perm = np.random.RandomState(42).permutation(len(X_train))
    val_idx, tr_idx = perm[:n_val], perm[n_val:]

    model.fit(X_train[tr_idx], y_tr[tr_idx],
              eval_set=[(X_train[val_idx], y_tr[val_idx])], verbose=False)

    y_pred_std = model.predict(X_test)
    return scaler_y.inverse_transform(y_pred_std.reshape(-1, 1)).flatten()


def _to_serializable(obj):
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
    return obj


def main():
    print("=" * 70)
    print("  PHASE 2: NMR FEATURE COMPARISON")
    print("  4 feature sets x 2 models x 2 properties")
    print("=" * 70)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load cache
    print("Loading pre-computed cache...")
    cache = np.load(CACHE_DIR / "nmr_compound_cache.npz")
    spectrum_features = cache["spectrum_features"]
    peaklist_features = cache["peaklist_features"]
    phase1_morgan = cache["phase1_morgan"]
    phase1_rdkit = cache["phase1_rdkit"]
    hc_mask_all = cache["hc_mask"]

    with open(CACHE_DIR / "nmr_smiles_list.txt") as f:
        all_smiles = [line.strip() for line in f]
    smi_to_idx = {s: i for i, s in enumerate(all_smiles)}

    print(f"  {len(all_smiles)} compounds")
    print(f"  Spectrum features: {spectrum_features.shape[1]}")
    print(f"  Peak-list features: {peaklist_features.shape[1]}")
    print(f"  Morgan FP: {phase1_morgan.shape[1]}")
    print(f"  RDKit desc: {phase1_rdkit.shape[1]}")

    # Define feature sets
    feature_sets = {
        "nmr_spectrum": spectrum_features,
        "nmr_full": np.hstack([spectrum_features, peaklist_features]),
        "phase1_matched": np.hstack([phase1_morgan, phase1_rdkit]),
        "combined": np.hstack([spectrum_features, peaklist_features, phase1_morgan, phase1_rdkit]),
    }

    results = {}

    for property_name in ["viscosity", "surface_tension"]:
        print(f"\n{'#'*70}")
        print(f"  {property_name.upper()}")
        print(f"{'#'*70}")

        labels_df = pd.read_csv(CACHE_DIR / f"labels_{property_name}.csv")
        nmr_smiles_set = set(all_smiles)
        labels_df = labels_df[labels_df["canonical_smiles"].isin(nmr_smiles_set)].copy()
        smiles_list = sorted(set(labels_df["canonical_smiles"].unique()) & nmr_smiles_set)
        hc_mask = np.array([hc_mask_all[smi_to_idx[s]] for s in smiles_list])
        hc_set = {s for i, s in enumerate(smiles_list) if hc_mask[i]}

        print(f"  {len(smiles_list)} compounds, {len(labels_df)} rows, {hc_mask.sum()} HC")

        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        all_compounds = np.array(smiles_list)

        prop_results = {}

        for fs_name, feat_matrix in feature_sets.items():
            fs_results = {}
            print(f"\n  === {fs_name.upper()} ({feat_matrix.shape[1]} features) ===")

            for model_name, train_fn in [("ridge", train_ridge), ("xgboost", train_xgboost)]:
                fold_results = []

                for fold, (train_cpd_idx, test_cpd_idx) in enumerate(kf.split(all_compounds)):
                    train_cpds = set(all_compounds[train_cpd_idx])
                    test_cpds = set(all_compounds[test_cpd_idx])

                    X_train, y_train = _build_xy(feat_matrix, labels_df, smi_to_idx, train_cpds, property_name)
                    X_test, y_test = _build_xy(feat_matrix, labels_df, smi_to_idx, test_cpds, property_name)

                    if len(X_train) == 0 or len(X_test) == 0:
                        fold_results.append(None)
                        continue

                    if model_name == "ridge":
                        result = train_fn(X_train, y_train, X_test, y_test)
                        y_pred, n_sel = result
                        if fold == 0:
                            print(f"    {model_name}: {n_sel} features selected")
                    else:
                        y_pred = train_fn(X_train, y_train, X_test, y_test)
                        if y_pred is None:
                            fold_results.append(None)
                            continue

                    # Metrics
                    m_std = compute_all_metrics(y_test, y_pred)
                    if property_name == "viscosity":
                        m_real = compute_all_metrics(np.exp(y_test), np.exp(y_pred))
                    else:
                        m_real = compute_all_metrics(y_test, np.clip(y_pred, 1e-6, None))

                    # HC metrics
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

                    fold_results.append({
                        "fold": fold,
                        "std_metrics": m_std,
                        "real_metrics": m_real,
                        "hc_metrics": m_hc,
                        "n_hc_test_pts": len(hc_indices),
                    })

                # Print summary for this model
                valid = [f for f in fold_results if f is not None]
                if valid:
                    std_r2 = np.mean([f["std_metrics"]["r2"] for f in valid])
                    real_r2 = np.mean([f["real_metrics"]["r2"] for f in valid])
                    hc_vals = [f["hc_metrics"]["r2"] for f in valid if f.get("hc_metrics")]
                    hc_r2 = np.mean(hc_vals) if hc_vals else float("nan")
                    print(f"    {model_name}: std R2={std_r2:.3f}  real R2={real_r2:.3f}  HC R2={hc_r2:.3f}")

                fs_results[model_name] = fold_results

            prop_results[fs_name] = fs_results

        results[property_name] = prop_results

        # Save intermediate
        with open(OUTPUT_DIR / "phase2_comparison_results.json", "w") as f:
            json.dump(_to_serializable(results), f, indent=2)

    # Final comparison table
    print(f"\n{'='*70}")
    print("  FINAL COMPARISON TABLE")
    print(f"{'='*70}")

    # Load Phase 1 results
    p1_path = paths.root / "src" / "phase1" / "outputs" / "direct_optimal_results.json"
    p1 = None
    if p1_path.exists():
        with open(p1_path) as f:
            p1 = json.load(f)

    for prop in ["viscosity", "surface_tension"]:
        print(f"\n  {prop.upper()}:")
        print(f"  {'Feature Set':<25s} {'Model':<10s} {'std R2':>8s} {'real R2':>8s} {'HC R2':>8s}")
        print(f"  {'-'*60}")

        if p1 and prop in p1:
            folds = p1[prop]
            std = np.mean([f["std_metrics"]["r2"] for f in folds])
            real = np.mean([f["real_metrics"]["r2"] for f in folds])
            hc = np.mean([f["hc_metrics"]["r2"] for f in folds if f.get("hc_metrics")])
            print(f"  {'Phase1 (856 cpds)':<25s} {'ensemble':<10s} {std:>8.3f} {real:>8.3f} {hc:>8.3f}")

        if prop in results:
            for fs_name in results[prop]:
                for model_name in results[prop][fs_name]:
                    folds = [f for f in results[prop][fs_name][model_name] if f]
                    if not folds:
                        continue
                    std = np.mean([f["std_metrics"]["r2"] for f in folds])
                    real = np.mean([f["real_metrics"]["r2"] for f in folds])
                    hc_vals = [f["hc_metrics"]["r2"] for f in folds if f.get("hc_metrics")]
                    hc = np.mean(hc_vals) if hc_vals else float("nan")
                    print(f"  {fs_name:<25s} {model_name:<10s} {std:>8.3f} {real:>8.3f} {hc:>8.3f}")

    print(f"\nResults saved to {OUTPUT_DIR / 'phase2_comparison_results.json'}")


if __name__ == "__main__":
    main()
