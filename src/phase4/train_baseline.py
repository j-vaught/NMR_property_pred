"""
Phase 4 baseline: XGBoost on spectrum + peak-list + T features → log_viscosity.

5-fold compound-level CV on the 953-compound training set.
Evaluates in BOTH log space (training loss) and real space (physical).

Target: beat Phase 2's HC R² = 0.694 (combined features, 512 compounds).
Phase 4 advantage: 2x more compounds + larger temperature range.
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


def load_data():
    """Load training set and build feature matrix."""
    print(f"Loading {TRAINING_PARQUET}")
    df = pd.read_parquet(TRAINING_PARQUET)
    print(f"  {len(df)} rows, {df['canonical_smiles'].nunique()} compounds")

    # Clip outrageous viscosity values to avoid log-space issues
    # Cut at 0.01 Pa·s (10 mPa·s) lower bound isn't needed (real fluids can be that low)
    # But cut the extreme upper tail: >30 Pa·s is polymer territory
    before = len(df)
    df = df[df["viscosity_Pa_s"] <= 30.0].copy()
    df = df[df["viscosity_Pa_s"] >= 1e-5].copy()
    print(f"  After viscosity filter [1e-5, 30] Pa·s: {len(df)} rows (dropped {before - len(df)})")

    # Recompute log viscosity (in case NaN)
    df["log_viscosity"] = np.log(df["viscosity_Pa_s"])

    # Build feature matrix
    print("Building feature matrix...")
    spectrum_feats = np.stack([np.array(s, dtype=np.float32) for s in df["spectrum_features"]])
    peaklist_feats = np.stack([np.array(s, dtype=np.float32) for s in df["peaklist_features"]])
    T_feats = make_temperature_features(df["T_K"].values).astype(np.float32)

    X = np.hstack([spectrum_feats, peaklist_feats, T_feats])
    y = df["log_viscosity"].values.astype(np.float32)

    print(f"  X shape: {X.shape}")
    print(f"  y shape: {y.shape}, range: [{y.min():.2f}, {y.max():.2f}]")

    return df, X, y


def train_fold(X_train, y_train, X_val, y_val):
    """Train a single XGBoost model with early stopping."""
    from xgboost import XGBRegressor

    # Standardize features and targets
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
    """Predict log-viscosity and inverse-transform."""
    X_scaled = scaler_x.transform(X)
    y_pred_std = model.predict(X_scaled)
    y_pred = scaler_y.inverse_transform(y_pred_std.reshape(-1, 1)).flatten()
    return y_pred


def evaluate_fold(y_true_log, y_pred_log, is_hc):
    """Compute metrics in log space, real space, and HC subset."""
    # Log space
    log_metrics = compute_all_metrics(y_true_log, y_pred_log)

    # Real space — clip predictions to avoid exp() overflow
    y_pred_log_clipped = np.clip(y_pred_log, y_true_log.min() - 2, y_true_log.max() + 2)
    y_true_real = np.exp(y_true_log)
    y_pred_real = np.exp(y_pred_log_clipped)
    real_metrics = compute_all_metrics(y_true_real, y_pred_real)

    # HC-only metrics
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
    print("  Phase 4 Baseline: XGBoost NMR → log_viscosity")
    print("=" * 70)

    df, X, y = load_data()

    # Compound-level 5-fold CV
    compounds = df["canonical_smiles"].unique()
    n_compounds = len(compounds)
    print(f"\n{n_compounds} unique compounds for 5-fold CV")

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

        # Split train into train/val for early stopping (10% val)
        rng = np.random.RandomState(42 + fold)
        n_val = max(100, int(0.1 * len(X_tr)))
        val_idx = rng.choice(len(X_tr), size=n_val, replace=False)
        tr_only_mask = np.ones(len(X_tr), dtype=bool)
        tr_only_mask[val_idx] = False
        X_tr_only = X_tr[tr_only_mask]
        y_tr_only = y_tr[tr_only_mask]
        X_val = X_tr[~tr_only_mask]
        y_val = y_tr[~tr_only_mask]

        print(f"\n--- Fold {fold+1}/5 ---")
        print(f"  train: {len(X_tr_only)}, val: {len(X_val)}, test: {len(X_te)}")
        print(f"  train cpds: {len(train_cpds)}, test cpds: {len(test_cpds)}")

        model, scx, scy = train_fold(X_tr_only, y_tr_only, X_val, y_val)

        y_pred = predict(model, scx, scy, X_te)
        is_hc = df.loc[te_mask, "is_hydrocarbon"].values

        metrics = evaluate_fold(y_te, y_pred, is_hc)
        fold_results.append(metrics)

        # Print
        log_r2 = metrics["log"]["r2"]
        real_r2 = metrics["real"]["r2"]
        real_mape = metrics["real"]["mape"]
        hc_str = ""
        if metrics["hc"]:
            hc_str = f" | HC log R²={metrics['hc']['log']['r2']:.3f}  HC real R²={metrics['hc']['real']['r2']:.3f}  HC MAPE={metrics['hc']['real']['mape']:.1f}%  ({metrics['n_test_hc']} pts)"
        print(f"  log R²={log_r2:.3f}  real R²={real_r2:.3f}  MAPE={real_mape:.1f}%{hc_str}")

    # Summary
    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"  SUMMARY ({elapsed:.0f}s)")
    print(f"{'='*70}")

    def agg(path):
        """Aggregate a metric across folds."""
        vals = []
        for f in fold_results:
            d = f
            try:
                for k in path:
                    d = d[k]
                vals.append(d)
            except (KeyError, TypeError):
                pass
        if vals:
            return np.mean(vals), np.std(vals)
        return float("nan"), float("nan")

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

    # Compare to Phase 2
    print(f"\n  {'='*60}")
    print(f"  Phase 2 best (combined features, 512 cpds):")
    print(f"    HC real R² = 0.694 (combined XGBoost, viscosity)")
    print(f"    HC real MAPE = not directly comparable (different data)")
    hc_real_r2, _ = agg(("hc", "real", "r2"))
    diff = hc_real_r2 - 0.694
    symbol = "↑" if diff > 0 else "↓"
    print(f"\n  Phase 4 baseline HC real R² = {hc_real_r2:.4f} ({symbol}{abs(diff):.4f} vs Phase 2)")

    # Save
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

    with open(OUTPUT_DIR / "phase4_baseline_results.json", "w") as f:
        json.dump(_to_serializable(fold_results), f, indent=2)
    print(f"\nResults saved to {OUTPUT_DIR / 'phase4_baseline_results.json'}")


if __name__ == "__main__":
    main()
