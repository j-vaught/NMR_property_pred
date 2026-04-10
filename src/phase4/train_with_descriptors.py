"""
Phase 4 with RDKit descriptors: NMR + descriptors → log_viscosity.

Baseline showed NMR alone can't beat Phase 2 (HC R² -0.777 vs 0.694).
Add the RDKit descriptors we computed in build_training_set.py and
see if the 2x dataset with descriptors beats Phase 2 combined.

Features (131 dims):
  - spectrum_features (95)
  - peaklist_features (25)
  - RDKit descriptors (8): mw, logp, tpsa, n_heavy_atoms, h_count,
                           n_rotatable, n_aromatic_rings, n_peaks
  - T features (3)

Target: log(viscosity_Pa_s).
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shared.config import Paths
from shared.data_utils import make_temperature_features
from shared.metrics import compute_all_metrics

paths = Paths()
TRAINING_PARQUET = paths.data / "nmr" / "phase4_training_set.parquet"
OUTPUT_DIR = Path(__file__).parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

DESCRIPTOR_COLS = [
    "mw", "logp", "tpsa", "n_heavy_atoms", "h_count_molecular",
    "n_rotatable", "n_aromatic_rings", "n_peaks",
]


def load_data():
    """Load and build feature matrix including descriptors."""
    print(f"Loading {TRAINING_PARQUET}")
    df = pd.read_parquet(TRAINING_PARQUET)
    print(f"  {len(df)} rows, {df['canonical_smiles'].nunique()} compounds")

    # Filter
    before = len(df)
    df = df[df["viscosity_Pa_s"] <= 30.0].copy()
    df = df[df["viscosity_Pa_s"] >= 1e-5].copy()
    df = df.dropna(subset=DESCRIPTOR_COLS)
    print(f"  After filters: {len(df)} rows (dropped {before - len(df)})")

    df["log_viscosity"] = np.log(df["viscosity_Pa_s"])

    # Build feature matrix
    spectrum_feats = np.stack([np.array(s, dtype=np.float32) for s in df["spectrum_features"]])
    peaklist_feats = np.stack([np.array(s, dtype=np.float32) for s in df["peaklist_features"]])
    desc_feats = df[DESCRIPTOR_COLS].values.astype(np.float32)
    T_feats = make_temperature_features(df["T_K"].values).astype(np.float32)

    X = np.hstack([spectrum_feats, peaklist_feats, desc_feats, T_feats])
    y = df["log_viscosity"].values.astype(np.float32)

    print(f"  X shape: {X.shape}")
    print(f"  y shape: {y.shape}, range: [{y.min():.2f}, {y.max():.2f}]")
    print(f"  feature dims: spectrum=95, peaklist=25, desc={len(DESCRIPTOR_COLS)}, T=3 = {X.shape[1]}")

    return df, X, y


def train_fold(X_train, y_train, X_val, y_val):
    """Train XGBoost with early stopping."""
    from xgboost import XGBRegressor

    scaler_x = StandardScaler().fit(X_train)
    scaler_y = StandardScaler().fit(y_train.reshape(-1, 1))

    X_tr = scaler_x.transform(X_train)
    X_va = scaler_x.transform(X_val)
    y_tr = scaler_y.transform(y_train.reshape(-1, 1)).flatten()
    y_va = scaler_y.transform(y_val.reshape(-1, 1)).flatten()

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
    model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    return model, scaler_x, scaler_y


def predict(model, scaler_x, scaler_y, X):
    X_scaled = scaler_x.transform(X)
    y_pred_std = model.predict(X_scaled)
    return scaler_y.inverse_transform(y_pred_std.reshape(-1, 1)).flatten()


def evaluate_fold(y_true_log, y_pred_log, is_hc):
    log_metrics = compute_all_metrics(y_true_log, y_pred_log)
    y_pred_log_clipped = np.clip(y_pred_log, y_true_log.min() - 2, y_true_log.max() + 2)
    y_true_real = np.exp(y_true_log)
    y_pred_real = np.exp(y_pred_log_clipped)
    real_metrics = compute_all_metrics(y_true_real, y_pred_real)

    hc_metrics = None
    if is_hc.sum() > 0:
        hc_log = compute_all_metrics(y_true_log[is_hc], y_pred_log[is_hc])
        hc_real = compute_all_metrics(y_true_real[is_hc], y_pred_real[is_hc])
        hc_metrics = {"log": hc_log, "real": hc_real}

    return {
        "log": log_metrics,
        "real": real_metrics,
        "hc": hc_metrics,
        "n_test": len(y_true_log),
        "n_test_hc": int(is_hc.sum()),
    }


def main():
    print("=" * 70)
    print("  Phase 4 WITH DESCRIPTORS: XGBoost NMR + RDKit desc → log_viscosity")
    print("=" * 70)

    df, X, y = load_data()

    compounds = df["canonical_smiles"].unique()
    print(f"\n{len(compounds)} unique compounds for 5-fold CV")

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_results = []
    t0 = time.time()

    for fold, (train_idx, test_idx) in enumerate(kf.split(compounds)):
        train_cpds = set(compounds[train_idx])
        test_cpds = set(compounds[test_idx])

        tr_mask = df["canonical_smiles"].isin(train_cpds).values
        te_mask = df["canonical_smiles"].isin(test_cpds).values

        X_tr, y_tr = X[tr_mask], y[tr_mask]
        X_te, y_te = X[te_mask], y[te_mask]

        rng = np.random.RandomState(42 + fold)
        n_val = max(100, int(0.1 * len(X_tr)))
        val_idx = rng.choice(len(X_tr), size=n_val, replace=False)
        tr_only = np.ones(len(X_tr), dtype=bool)
        tr_only[val_idx] = False

        print(f"\n--- Fold {fold+1}/5 ---")
        print(f"  train: {tr_only.sum()}, val: {n_val}, test: {len(X_te)}")

        model, scx, scy = train_fold(X_tr[tr_only], y_tr[tr_only], X_tr[~tr_only], y_tr[~tr_only])
        y_pred = predict(model, scx, scy, X_te)
        is_hc = df.loc[te_mask, "is_hydrocarbon"].values

        metrics = evaluate_fold(y_te, y_pred, is_hc)
        fold_results.append(metrics)

        log_r2 = metrics["log"]["r2"]
        real_r2 = metrics["real"]["r2"]
        hc_str = ""
        if metrics["hc"]:
            hc_str = f" | HC log R²={metrics['hc']['log']['r2']:.3f}  real R²={metrics['hc']['real']['r2']:.3f}  MAPE={metrics['hc']['real']['mape']:.1f}%"
        print(f"  log R²={log_r2:.3f}  real R²={real_r2:.3f}{hc_str}")

    # Summary
    print(f"\n{'='*70}")
    print(f"  SUMMARY ({time.time() - t0:.0f}s)")
    print(f"{'='*70}")

    def agg(path):
        vals = []
        for f in fold_results:
            d = f
            try:
                for k in path:
                    d = d[k]
                vals.append(d)
            except (KeyError, TypeError):
                pass
        return (np.mean(vals), np.std(vals)) if vals else (float("nan"), float("nan"))

    print(f"\n  {'Metric':<35s} {'Mean':>10s} {'Std':>10s}")
    print(f"  {'-'*60}")
    for label, path in [
        ("Log space R²", ("log", "r2")),
        ("Log space MAPE", ("log", "mape")),
        ("Real space R²", ("real", "r2")),
        ("Real space MAPE", ("real", "mape")),
        ("HC Log R²", ("hc", "log", "r2")),
        ("HC Log MAPE", ("hc", "log", "mape")),
        ("HC Real R²", ("hc", "real", "r2")),
        ("HC Real MAPE", ("hc", "real", "mape")),
    ]:
        m, s = agg(path)
        print(f"  {label:<35s} {m:>10.4f} {s:>10.4f}")

    hc_real_r2, _ = agg(("hc", "real", "r2"))
    print(f"\n  Phase 2 combined best:  HC real R² = 0.694")
    print(f"  Phase 4 baseline:       HC real R² = -0.777")
    print(f"  Phase 4 +descriptors:   HC real R² = {hc_real_r2:.4f}")

    def _s(o):
        if isinstance(o, dict): return {k: _s(v) for k, v in o.items()}
        if isinstance(o, list): return [_s(v) for v in o]
        if isinstance(o, (np.floating, float)): return float(o)
        if isinstance(o, (np.integer, int)): return int(o)
        if isinstance(o, np.ndarray): return o.tolist()
        return o

    with open(OUTPUT_DIR / "phase4_with_descriptors_results.json", "w") as f:
        json.dump(_s(fold_results), f, indent=2)
    print(f"\nSaved to {OUTPUT_DIR / 'phase4_with_descriptors_results.json'}")


if __name__ == "__main__":
    main()
