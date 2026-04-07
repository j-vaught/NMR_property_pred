"""
Cross-validation analysis for Arrhenius/linear-ST Ridge regression models.
Implements k-fold CV, LOOCV, and statistical power analysis.
"""
import sys
import json
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shared.config import Paths, DataConfig, SplitConfig
from phase1.data_pipeline import (
    load_fingerprints, build_unified_labels, join_features_labels,
    fit_arrhenius_targets, fit_linear_st_targets,
)
from shared.data_utils import is_hydrocarbon, fit_arrhenius, fit_linear_st
from shared.metrics import compute_all_metrics

warnings.filterwarnings("ignore")

# ── 1. Load data and build Arrhenius/linear-ST targets ──────────────────────

print("=" * 70)
print("STEP 1: Loading data and fitting Arrhenius / linear-ST targets")
print("=" * 70)

paths = Paths()
data_config = DataConfig()

fp_df = load_fingerprints(paths, data_config)
visc_df, st_df = build_unified_labels(paths, data_config)
visc_joined, st_joined = join_features_labels(fp_df, visc_df, st_df, data_config)

visc_targets = fit_arrhenius_targets(visc_joined, data_config)
st_targets = fit_linear_st_targets(st_joined, data_config)

# Tag HC
visc_targets["is_hc"] = visc_targets["canonical_smiles"].apply(is_hydrocarbon)
st_targets["is_hc"] = st_targets["canonical_smiles"].apply(is_hydrocarbon)

print(f"\nViscosity Arrhenius targets: {len(visc_targets)} total, {visc_targets['is_hc'].sum()} HC")
print(f"Surface tension linear targets: {len(st_targets)} total, {st_targets['is_hc'].sum()} HC")

# ── 2. Prepare features and targets ────────────────────────────────────────

def prepare_Xy(targets_df, fp_df):
    """Return X (fingerprints), Y ([A,B]), smiles list, is_hc array, n_points array."""
    common = targets_df["canonical_smiles"].isin(fp_df.index)
    tdf = targets_df[common].copy()
    smiles = tdf["canonical_smiles"].values
    X = fp_df.loc[smiles].values.astype(np.float64)
    Y = tdf[["A", "B"]].values
    is_hc = tdf["is_hc"].values
    n_pts = tdf["n_points"].values
    return X, Y, smiles, is_hc, n_pts


X_visc, Y_visc, smi_visc, hc_visc, npts_visc = prepare_Xy(visc_targets, fp_df)
X_st, Y_st, smi_st, hc_st, npts_st = prepare_Xy(st_targets, fp_df)

print(f"\nFeature-matched viscosity: {len(X_visc)} compounds ({hc_visc.sum()} HC)")
print(f"Feature-matched surface tension: {len(X_st)} compounds ({hc_st.sum()} HC)")

# ── 3. Ridge regression with k-fold CV ─────────────────────────────────────

from sklearn.linear_model import Ridge
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler


def make_strata(is_hc, n_pts):
    """Create stratification bins: HC flag x n_points quartile."""
    quartiles = pd.qcut(n_pts, q=4, labels=False, duplicates="drop")
    return np.array([f"{int(h)}_{int(q)}" for h, q in zip(is_hc, quartiles)])


def reconstruct_viscosity(A, B, T_array):
    """Reconstruct eta from Arrhenius: eta = exp(A + B/T)."""
    return np.exp(A + B / T_array)


def reconstruct_surface_tension(A, B, T_array):
    """Reconstruct sigma from linear: sigma = A + B*T."""
    return A + B * T_array


def evaluate_fold(Y_true, Y_pred, raw_df, smiles_test, prop_type, is_hc_test):
    """Evaluate a single fold. Returns dict of metrics."""
    results = {}

    # A,B parameter R2
    from shared.metrics import r2_score, mape as mape_fn, rmse as rmse_fn
    r2_A = r2_score(Y_true[:, 0], Y_pred[:, 0])
    r2_B = r2_score(Y_true[:, 1], Y_pred[:, 1])
    results["param_r2_A"] = r2_A
    results["param_r2_B"] = r2_B

    # Reconstruct on raw data points for each test compound
    all_true_vals = []
    all_pred_vals = []
    hc_true_vals = []
    hc_pred_vals = []

    for i, smi in enumerate(smiles_test):
        compound_data = raw_df[raw_df["canonical_smiles"] == smi]
        if len(compound_data) == 0:
            continue
        T = compound_data["T_K"].values
        true_vals = compound_data["value"].values

        if prop_type == "viscosity":
            pred_vals = reconstruct_viscosity(Y_pred[i, 0], Y_pred[i, 1], T)
        else:
            pred_vals = reconstruct_surface_tension(Y_pred[i, 0], Y_pred[i, 1], T)

        all_true_vals.extend(true_vals)
        all_pred_vals.extend(pred_vals)

        if is_hc_test[i]:
            hc_true_vals.extend(true_vals)
            hc_pred_vals.extend(pred_vals)

    all_true_vals = np.array(all_true_vals)
    all_pred_vals = np.array(all_pred_vals)

    if len(all_true_vals) > 0:
        all_metrics = compute_all_metrics(all_true_vals, all_pred_vals)
        results["recon_r2_all"] = all_metrics["r2"]
        results["recon_mape_all"] = all_metrics["mape"]
        results["recon_rmse_all"] = all_metrics["rmse"]
        results["n_test_compounds"] = len(smiles_test)
        results["n_test_points"] = len(all_true_vals)

    if len(hc_true_vals) > 0:
        hc_true_vals = np.array(hc_true_vals)
        hc_pred_vals = np.array(hc_pred_vals)
        hc_metrics = compute_all_metrics(hc_true_vals, hc_pred_vals)
        results["recon_r2_hc"] = hc_metrics["r2"]
        results["recon_mape_hc"] = hc_metrics["mape"]
        results["recon_rmse_hc"] = hc_metrics["rmse"]
        results["n_hc_test_compounds"] = int(is_hc_test.sum())
        results["n_hc_test_points"] = len(hc_true_vals)
    else:
        results["recon_r2_hc"] = None
        results["recon_mape_hc"] = None
        results["recon_rmse_hc"] = None
        results["n_hc_test_compounds"] = 0
        results["n_hc_test_points"] = 0

    return results


def run_kfold_cv(X, Y, smiles, is_hc, n_pts, raw_df, prop_type, n_folds, seed=42, alpha=1.0):
    """Run stratified k-fold CV at compound level."""
    strata = make_strata(is_hc, n_pts)

    # Handle case where some strata have fewer members than n_folds
    unique_strata, counts = np.unique(strata, return_counts=True)
    min_count = counts.min()
    effective_folds = min(n_folds, min_count)
    if effective_folds < n_folds:
        print(f"  Warning: reducing folds from {n_folds} to {effective_folds} (smallest stratum has {min_count} members)")
        # Fallback: use HC flag only for stratification
        strata = is_hc.astype(int).astype(str)
        unique_strata2, counts2 = np.unique(strata, return_counts=True)
        effective_folds = min(n_folds, counts2.min())
        if effective_folds < n_folds:
            print(f"  Warning: even HC-only stratification needs reduction to {effective_folds} folds")

    skf = StratifiedKFold(n_splits=effective_folds, shuffle=True, random_state=seed)
    fold_results = []

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, strata)):
        X_train, X_test = X[train_idx], X[test_idx]
        Y_train, Y_test = Y[train_idx], Y[test_idx]
        smi_test = smiles[test_idx]
        hc_test = is_hc[test_idx]

        # Standardize features
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        # Train Ridge for A and B jointly
        model = Ridge(alpha=alpha)
        model.fit(X_train_s, Y_train)
        Y_pred = model.predict(X_test_s)

        fold_res = evaluate_fold(Y_test, Y_pred, raw_df, smi_test, prop_type, hc_test)
        fold_res["fold"] = fold_idx
        fold_results.append(fold_res)

        print(f"  Fold {fold_idx}: {fold_res['n_test_compounds']} compounds "
              f"({fold_res['n_hc_test_compounds']} HC), "
              f"recon R2={fold_res['recon_r2_all']:.4f}, "
              f"HC R2={fold_res.get('recon_r2_hc', 'N/A')}")

    return fold_results


def summarize_folds(fold_results, metric_keys=None):
    """Compute mean, std, and 95% CI for each metric across folds."""
    if metric_keys is None:
        metric_keys = ["param_r2_A", "param_r2_B", "recon_r2_all", "recon_mape_all",
                        "recon_r2_hc", "recon_mape_hc", "n_hc_test_compounds"]

    summary = {}
    for key in metric_keys:
        vals = [r[key] for r in fold_results if r.get(key) is not None]
        if len(vals) == 0:
            continue
        vals = np.array(vals, dtype=float)
        mean = float(np.mean(vals))
        std = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        n = len(vals)
        se = std / np.sqrt(n) if n > 0 else 0.0
        t_crit = stats.t.ppf(0.975, df=max(1, n - 1))
        ci_low = mean - t_crit * se
        ci_high = mean + t_crit * se
        summary[key] = {
            "mean": round(mean, 6),
            "std": round(std, 6),
            "ci95_low": round(ci_low, 6),
            "ci95_high": round(ci_high, 6),
            "n_folds": n,
            "values": [round(v, 6) for v in vals],
        }
    return summary


# ── Run 5-fold and 10-fold CV ──────────────────────────────────────────────

all_results = {}

for n_folds in [5, 10]:
    print(f"\n{'=' * 70}")
    print(f"STEP 3: {n_folds}-FOLD CV")
    print(f"{'=' * 70}")

    for prop_type, X, Y, smiles, is_hc, n_pts, raw_df in [
        ("viscosity", X_visc, Y_visc, smi_visc, hc_visc, npts_visc, visc_joined),
        ("surface_tension", X_st, Y_st, smi_st, hc_st, npts_st, st_joined),
    ]:
        print(f"\n--- {prop_type.upper()} ({n_folds}-fold) ---")
        fold_results = run_kfold_cv(X, Y, smiles, is_hc, n_pts, raw_df, prop_type, n_folds)
        summary = summarize_folds(fold_results)

        key = f"{prop_type}_{n_folds}fold"
        all_results[key] = {
            "fold_results": fold_results,
            "summary": summary,
        }

        print(f"\n  Summary for {prop_type} ({n_folds}-fold):")
        for metric, vals in summary.items():
            print(f"    {metric}: {vals['mean']:.4f} +/- {vals['std']:.4f} "
                  f"  95% CI: [{vals['ci95_low']:.4f}, {vals['ci95_high']:.4f}]")

# ── 4. LOOCV for HC compounds ──────────────────────────────────────────────

print(f"\n{'=' * 70}")
print("STEP 4: LOOCV on HC compounds")
print("=" * 70)

for prop_type, X, Y, smiles, is_hc, raw_df in [
    ("viscosity", X_visc, Y_visc, smi_visc, hc_visc, visc_joined),
    ("surface_tension", X_st, Y_st, smi_st, hc_st, st_joined),
]:
    hc_mask = is_hc.astype(bool)
    X_hc = X[hc_mask]
    Y_hc = Y[hc_mask]
    smi_hc = smiles[hc_mask]
    n_hc = len(X_hc)

    if n_hc < 3:
        print(f"  {prop_type}: only {n_hc} HC compounds, skipping LOOCV")
        continue

    print(f"\n--- {prop_type.upper()} LOOCV ({n_hc} HC compounds) ---")

    loo_preds = np.zeros_like(Y_hc)
    loo_recon_true = []
    loo_recon_pred = []

    for i in range(n_hc):
        train_mask = np.ones(n_hc, dtype=bool)
        train_mask[i] = False

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_hc[train_mask])
        X_test_s = scaler.transform(X_hc[i:i+1])

        model = Ridge(alpha=1.0)
        model.fit(X_train_s, Y_hc[train_mask])
        pred = model.predict(X_test_s)[0]
        loo_preds[i] = pred

        # Reconstruct
        compound_data = raw_df[raw_df["canonical_smiles"] == smi_hc[i]]
        if len(compound_data) > 0:
            T = compound_data["T_K"].values
            true_vals = compound_data["value"].values
            if prop_type == "viscosity":
                pred_vals = reconstruct_viscosity(pred[0], pred[1], T)
            else:
                pred_vals = reconstruct_surface_tension(pred[0], pred[1], T)
            loo_recon_true.extend(true_vals)
            loo_recon_pred.extend(pred_vals)

    from shared.metrics import r2_score, mape as mape_fn

    param_r2_A = r2_score(Y_hc[:, 0], loo_preds[:, 0])
    param_r2_B = r2_score(Y_hc[:, 1], loo_preds[:, 1])

    loo_recon_true = np.array(loo_recon_true)
    loo_recon_pred = np.array(loo_recon_pred)
    recon_metrics = compute_all_metrics(loo_recon_true, loo_recon_pred)

    loo_summary = {
        "n_compounds": n_hc,
        "param_r2_A": round(param_r2_A, 6),
        "param_r2_B": round(param_r2_B, 6),
        "recon_r2": round(recon_metrics["r2"], 6),
        "recon_mape": round(recon_metrics["mape"], 6),
        "recon_rmse": round(recon_metrics["rmse"], 6),
        "n_recon_points": len(loo_recon_true),
    }

    all_results[f"{prop_type}_loocv_hc"] = loo_summary

    print(f"  Param R2(A)={param_r2_A:.4f}, R2(B)={param_r2_B:.4f}")
    print(f"  Reconstructed R2={recon_metrics['r2']:.4f}, MAPE={recon_metrics['mape']:.2f}%, "
          f"RMSE={recon_metrics['rmse']:.6f}")

# ── 5. Statistical Power Analysis ──────────────────────────────────────────

print(f"\n{'=' * 70}")
print("STEP 5: Statistical Power Analysis")
print("=" * 70)

power_results = {}

for prop_type, n_compounds in [("viscosity", len(X_visc)), ("surface_tension", len(X_st))]:
    print(f"\n--- {prop_type.upper()} (n={n_compounds}) ---")

    # Using Fisher's z-transform for comparing R2 values
    # For R2 comparison, we use the variance of R2 estimator:
    # Var(R2) ~ 4*R2*(1-R2)^2 * (n-k-1)^2 / ((n^2-1)*(n+3))
    # where k = number of predictors

    k = X_visc.shape[1] if prop_type == "viscosity" else X_st.shape[1]
    n = n_compounds

    # Minimum detectable R2 difference at p=0.05, power=0.80
    # Using Steiger's (1980) test for comparing dependent R2
    # Approximate: se(R2) ~ sqrt(4*R2*(1-R2)^2 / (n-k-1))

    r2_baseline = 0.85
    se_r2 = np.sqrt(4 * r2_baseline * (1 - r2_baseline)**2 / (n - k - 1)) if n > k + 1 else float('inf')

    # For two-sided test at alpha=0.05, power=0.80
    z_alpha = 1.96
    z_beta = 0.84  # power = 0.80

    min_detectable_diff = (z_alpha + z_beta) * se_r2 * np.sqrt(2)

    print(f"  SE(R2) at R2=0.85: {se_r2:.4f}")
    print(f"  Minimum detectable R2 difference (alpha=0.05, power=0.80): {min_detectable_diff:.4f}")

    # Can we distinguish R2=0.85 from R2=0.90?
    target_diff = 0.05
    # Required n for detecting diff=0.05
    # n_required ~ 4*R2*(1-R2)^2 * (z_alpha+z_beta)^2 * 2 / diff^2 + k + 1
    r2_mid = 0.875
    n_required = int(np.ceil(
        4 * r2_mid * (1 - r2_mid)**2 * (z_alpha + z_beta)**2 * 2 / target_diff**2
    )) + k + 1

    can_distinguish = min_detectable_diff <= target_diff

    print(f"  Can distinguish R2=0.85 vs R2=0.90? {'YES' if can_distinguish else 'NO'}")
    print(f"  Compounds needed to distinguish 0.85 vs 0.90: {n_required}")

    # Also compute via bootstrap-based approach (empirical)
    # Use the actual CV results to estimate R2 variance
    cv5_key = f"{prop_type}_5fold"
    if cv5_key in all_results and "recon_r2_all" in all_results[cv5_key]["summary"]:
        r2_vals = all_results[cv5_key]["summary"]["recon_r2_all"]["values"]
        empirical_std = np.std(r2_vals, ddof=1)
        empirical_min_detect = (z_alpha + z_beta) * empirical_std * np.sqrt(2)
        print(f"  Empirical R2 std from 5-fold: {empirical_std:.4f}")
        print(f"  Empirical min detectable diff: {empirical_min_detect:.4f}")
    else:
        empirical_std = None
        empirical_min_detect = None

    power_results[prop_type] = {
        "n_compounds": n_compounds,
        "n_features": int(k),
        "se_r2_at_085": round(se_r2, 6) if se_r2 != float('inf') else None,
        "min_detectable_r2_diff": round(min_detectable_diff, 6) if min_detectable_diff != float('inf') else None,
        "can_distinguish_085_vs_090": bool(can_distinguish),
        "n_required_for_005_diff": int(n_required),
        "empirical_r2_std_5fold": round(empirical_std, 6) if empirical_std is not None else None,
        "empirical_min_detectable_diff": round(empirical_min_detect, 6) if empirical_min_detect is not None else None,
    }

all_results["power_analysis"] = power_results

# ── 6. Stability assessment ────────────────────────────────────────────────

print(f"\n{'=' * 70}")
print("STEP 6: Stability Assessment")
print("=" * 70)

stability = {}
for key in all_results:
    if "fold" in key and "summary" in all_results[key]:
        summary = all_results[key]["summary"]
        stable = True
        notes = []
        for metric, vals in summary.items():
            if "r2" in metric and vals.get("std", 0) > 0.10:
                stable = False
                notes.append(f"{metric} has high variance (std={vals['std']:.4f})")
            if "mape" in metric and vals.get("std", 0) > 10.0:
                stable = False
                notes.append(f"{metric} has high variance (std={vals['std']:.4f})")

        stability[key] = {
            "stable": stable,
            "notes": notes if notes else ["Results appear stable across folds"],
        }
        status = "STABLE" if stable else "UNSTABLE"
        print(f"  {key}: {status}")
        for note in notes:
            print(f"    - {note}")

all_results["stability_assessment"] = stability

# ── 7. Save results ────────────────────────────────────────────────────────

output_path = Path("/Users/user/Downloads/untitled folder/src/phase1/outputs/cv_results.json")
output_path.parent.mkdir(parents=True, exist_ok=True)

# Convert numpy types for JSON serialization
def convert_for_json(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, dict):
        return {k: convert_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [convert_for_json(v) for v in obj]
    return obj

with open(output_path, "w") as f:
    json.dump(convert_for_json(all_results), f, indent=2)

print(f"\nResults saved to {output_path}")

# ── Final Summary ───────────────────────────────────────────────────────────

print(f"\n{'=' * 70}")
print("FINAL SUMMARY")
print("=" * 70)

for prop_type in ["viscosity", "surface_tension"]:
    print(f"\n{'─' * 40}")
    print(f"  {prop_type.upper()}")
    print(f"{'─' * 40}")

    for n_folds in [5, 10]:
        key = f"{prop_type}_{n_folds}fold"
        if key in all_results and "summary" in all_results[key]:
            s = all_results[key]["summary"]
            print(f"\n  {n_folds}-fold CV:")
            if "recon_r2_all" in s:
                v = s["recon_r2_all"]
                print(f"    Recon R2 (all):  {v['mean']:.4f} +/- {v['std']:.4f}  "
                      f"95% CI [{v['ci95_low']:.4f}, {v['ci95_high']:.4f}]")
            if "recon_r2_hc" in s:
                v = s["recon_r2_hc"]
                print(f"    Recon R2 (HC):   {v['mean']:.4f} +/- {v['std']:.4f}  "
                      f"95% CI [{v['ci95_low']:.4f}, {v['ci95_high']:.4f}]")
            if "recon_mape_all" in s:
                v = s["recon_mape_all"]
                print(f"    MAPE (all):      {v['mean']:.2f}% +/- {v['std']:.2f}%  "
                      f"95% CI [{v['ci95_low']:.2f}%, {v['ci95_high']:.2f}%]")
            if "recon_mape_hc" in s:
                v = s["recon_mape_hc"]
                print(f"    MAPE (HC):       {v['mean']:.2f}% +/- {v['std']:.2f}%  "
                      f"95% CI [{v['ci95_low']:.2f}%, {v['ci95_high']:.2f}%]")
            if "n_hc_test_compounds" in s:
                v = s["n_hc_test_compounds"]
                print(f"    HC per fold:     {v['mean']:.1f} +/- {v['std']:.1f}  "
                      f"(range: {min(v['values'])}-{max(v['values'])})")

    loo_key = f"{prop_type}_loocv_hc"
    if loo_key in all_results:
        loo = all_results[loo_key]
        print(f"\n  LOOCV (HC only, n={loo['n_compounds']}):")
        print(f"    Param R2(A):     {loo['param_r2_A']:.4f}")
        print(f"    Param R2(B):     {loo['param_r2_B']:.4f}")
        print(f"    Recon R2:        {loo['recon_r2']:.4f}")
        print(f"    MAPE:            {loo['recon_mape']:.2f}%")

print(f"\n{'─' * 40}")
print("  POWER ANALYSIS")
print(f"{'─' * 40}")
for prop_type, pa in power_results.items():
    print(f"\n  {prop_type} (n={pa['n_compounds']}):")
    print(f"    Min detectable R2 diff: {pa['min_detectable_r2_diff']}")
    print(f"    Can detect 0.85 vs 0.90: {pa['can_distinguish_085_vs_090']}")
    print(f"    Compounds needed for 0.05 diff: {pa['n_required_for_005_diff']}")
    if pa['empirical_r2_std_5fold'] is not None:
        print(f"    Empirical R2 std (5-fold): {pa['empirical_r2_std_5fold']:.4f}")
        print(f"    Empirical min detectable: {pa['empirical_min_detectable_diff']:.4f}")

print(f"\nDone. Full results at: {output_path}")
