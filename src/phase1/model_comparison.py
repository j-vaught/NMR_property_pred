"""
Model comparison: Tree-based vs Linear models for predicting
viscosity (Arrhenius) and surface tension (linear) coefficients.
5-fold cross-validation at compound level.
"""

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_percentage_error
from scipy.stats import uniform, loguniform, randint

import xgboost as xgb
import lightgbm as lgb

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shared.config import Paths as _Paths
_p = _Paths()
ROOT = _p.root
DATA = _p.data
OUTPUTS = ROOT / "src" / "phase1" / "outputs"

# ── Config ──────────────────────────────────────────────────────────────────
T_MIN, T_MAX = 200.0, 500.0
P_MAX = 110.0
MIN_PTS = 5
ARR_R2_THRESH = 0.95
N_FOLDS = 5
SEED = 42


def flush_print(*args, **kwargs):
    print(*args, **kwargs, flush=True)


# ── Data Loading (fast, no RDKit canonicalization) ──────────────────────────

def load_features():
    """Load Morgan FP + RDKit descriptors, keyed by SMILES."""
    flush_print("Loading fingerprints...")
    fp = pd.read_csv(DATA / "features" / "morgan_r2_2048.csv")
    smiles_col = "SMILES"
    bit_cols = [c for c in fp.columns if c not in ("CAS", "SMILES")]
    fp = fp.drop_duplicates(subset=[smiles_col], keep="first")
    fp = fp.set_index(smiles_col)[bit_cols].astype(np.float32)
    # Remove zero-variance
    fp = fp.loc[:, fp.std() > 0]
    n_fp = fp.shape[1]

    flush_print("Loading RDKit descriptors...")
    desc = pd.read_csv(DATA / "features" / "rdkit_descriptors.csv")
    desc_cols = [c for c in desc.columns if c not in ("CAS", "SMILES")]
    desc = desc.drop_duplicates(subset=["SMILES"], keep="first")
    desc = desc.set_index("SMILES")[desc_cols].astype(np.float32)
    desc = desc.replace([np.inf, -np.inf], np.nan).fillna(0)
    desc = desc.loc[:, desc.std() > 0]
    # Standardize descriptors
    desc = (desc - desc.mean()) / desc.std().replace(0, 1)

    common = fp.index.intersection(desc.index)
    features = pd.concat([fp.loc[common], desc.loc[common]], axis=1)
    flush_print(f"  {len(features)} compounds, {features.shape[1]} features "
                f"({n_fp} FP + {len(desc.columns)} desc)")
    return features


def load_gc_viscosity():
    """Load GC viscosity (already has SMILES)."""
    df = pd.read_csv(DATA / "thermo" / "group_contribution" / "chemicals_viscosity_expanded.csv")
    result = pd.DataFrame({
        "smiles": df["SMILES"],
        "T_K": pd.to_numeric(df["T_K"], errors="coerce"),
        "value": pd.to_numeric(df["viscosity_Pas"], errors="coerce"),
    }).dropna()
    result = result[(result["T_K"] >= T_MIN) & (result["T_K"] <= T_MAX) & (result["value"] > 0)]
    flush_print(f"  GC viscosity: {len(result)} rows, {result['smiles'].nunique()} compounds")
    return result


def load_gc_surface_tension():
    """Load GC surface tension."""
    df = pd.read_csv(DATA / "thermo" / "group_contribution" / "chemicals_surface_tension_expanded.csv")
    st_col = [c for c in df.columns if "surface" in c.lower()][0]
    result = pd.DataFrame({
        "smiles": df["SMILES"],
        "T_K": pd.to_numeric(df["T_K"], errors="coerce"),
        "value": pd.to_numeric(df[st_col], errors="coerce"),
    }).dropna()
    result = result[(result["T_K"] >= T_MIN) & (result["T_K"] <= T_MAX) & (result["value"] > 0) & (result["value"] < 1.0)]
    flush_print(f"  GC surface tension: {len(result)} rows, {result['smiles'].nunique()} compounds")
    return result


def load_thermoml_viscosity(smiles_lookup):
    """Load ThermoML viscosity, mapping InChI -> SMILES via prebuilt lookup."""
    df = pd.read_csv(DATA / "thermo" / "thermoml_parsed" / "pure_viscosity.csv")
    inchi_col = "InChI" if "InChI" in df.columns else [c for c in df.columns if "inchi" in c.lower()][0]
    df["smiles"] = df[inchi_col].map(smiles_lookup)
    df = df.dropna(subset=["smiles"])

    visc_col = [c for c in df.columns if "viscosity" in c.lower()][0]
    result = pd.DataFrame({
        "smiles": df["smiles"],
        "T_K": pd.to_numeric(df["T_K"], errors="coerce"),
        "value": pd.to_numeric(df[visc_col], errors="coerce"),
    }).dropna()

    # Pressure filter
    if "P_kPa" in df.columns:
        p = pd.to_numeric(df.loc[result.index, "P_kPa"], errors="coerce")
        result = result[p.isna() | (p <= P_MAX)]

    result = result[(result["T_K"] >= T_MIN) & (result["T_K"] <= T_MAX)]
    result = result[(result["value"] > 0) & (result["value"] < 100)]
    flush_print(f"  ThermoML viscosity: {len(result)} rows, {result['smiles'].nunique()} compounds")
    return result


def load_thermoml_surface_tension(smiles_lookup):
    """Load ThermoML surface tension."""
    df = pd.read_csv(DATA / "thermo" / "thermoml_parsed" / "pure_surface_tension.csv")
    inchi_col = "InChI" if "InChI" in df.columns else [c for c in df.columns if "inchi" in c.lower()][0]
    df["smiles"] = df[inchi_col].map(smiles_lookup)
    df = df.dropna(subset=["smiles"])

    st_col = [c for c in df.columns if "surface" in c.lower() or "tension" in c.lower()][0]
    result = pd.DataFrame({
        "smiles": df["smiles"],
        "T_K": pd.to_numeric(df["T_K"], errors="coerce"),
        "value": pd.to_numeric(df[st_col], errors="coerce"),
    }).dropna()
    result = result[(result["T_K"] >= T_MIN) & (result["T_K"] <= T_MAX)]
    result = result[(result["value"] > 0) & (result["value"] < 1.0)]
    flush_print(f"  ThermoML surface tension: {len(result)} rows, {result['smiles'].nunique()} compounds")
    return result


def build_inchi_smiles_lookup():
    """Build InChI -> SMILES mapping using RDKit, caching to disk."""
    cache_path = DATA / "features" / "_inchi_smiles_cache.csv"
    if cache_path.exists():
        flush_print("  Loading cached InChI->SMILES mapping...")
        df = pd.read_csv(cache_path)
        return dict(zip(df["InChI"], df["SMILES"]))

    flush_print("  Building InChI->SMILES mapping (one-time)...")
    from rdkit.Chem.inchi import MolFromInchi
    from rdkit import Chem

    # Gather all unique InChIs from ThermoML files
    all_inchis = set()
    for fname in ["pure_viscosity.csv", "pure_surface_tension.csv"]:
        p = DATA / "thermo" / "thermoml_parsed" / fname
        if p.exists():
            df = pd.read_csv(p)
            col = "InChI" if "InChI" in df.columns else [c for c in df.columns if "inchi" in c.lower()][0]
            all_inchis.update(df[col].dropna().unique())

    flush_print(f"    Converting {len(all_inchis)} unique InChIs...")
    lookup = {}
    for inchi in all_inchis:
        try:
            mol = MolFromInchi(str(inchi), sanitize=True, treatWarningAsError=False)
            if mol is not None:
                lookup[inchi] = Chem.MolToSmiles(mol)
        except Exception:
            pass

    # Cache
    pd.DataFrame({"InChI": list(lookup.keys()), "SMILES": list(lookup.values())}).to_csv(cache_path, index=False)
    flush_print(f"    Cached {len(lookup)} mappings to {cache_path}")
    return lookup


def load_all_data():
    """Load all data fast."""
    features = load_features()

    # Build InChI lookup (cached)
    smiles_lookup = build_inchi_smiles_lookup()

    flush_print("\nLoading labels...")
    gc_visc = load_gc_viscosity()
    gc_st = load_gc_surface_tension()
    tml_visc = load_thermoml_viscosity(smiles_lookup)
    tml_st = load_thermoml_surface_tension(smiles_lookup)

    # Merge: ThermoML takes priority over GC
    def merge(tml, gc):
        combined = pd.concat([tml, gc], ignore_index=True)
        combined["T_round"] = combined["T_K"].round(1)
        combined["src"] = [0]*len(tml) + [1]*len(gc)
        combined = combined.sort_values("src")
        combined = combined.drop_duplicates(subset=["smiles", "T_round"], keep="first")
        return combined.drop(columns=["T_round", "src"])

    visc = merge(tml_visc, gc_visc)
    st = merge(tml_st, gc_st)

    # Filter to compounds with features
    visc = visc[visc["smiles"].isin(features.index)]
    st = st[st["smiles"].isin(features.index)]

    flush_print(f"\n[Final] Viscosity: {len(visc)} rows, {visc['smiles'].nunique()} compounds")
    flush_print(f"[Final] Surface tension: {len(st)} rows, {st['smiles'].nunique()} compounds")

    return features, visc, st


# ── Arrhenius / Linear fitting ──────────────────────────────────────────────

def fit_arrhenius(T_K, eta_Pas):
    """ln(eta) = A + B/T. Returns A, B, r2."""
    ln_eta = np.log(eta_Pas)
    inv_T = 1.0 / T_K
    X = np.column_stack([np.ones_like(inv_T), inv_T])
    coeffs, _, _, _ = np.linalg.lstsq(X, ln_eta, rcond=None)
    A, B = coeffs
    ss_res = np.sum((ln_eta - X @ coeffs) ** 2)
    ss_tot = np.sum((ln_eta - ln_eta.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    return A, B, r2


def fit_linear_st(T_K, sigma):
    """sigma = A + B*T. Returns A, B, r2."""
    X = np.column_stack([np.ones_like(T_K), T_K])
    coeffs, _, _, _ = np.linalg.lstsq(X, sigma, rcond=None)
    A, B = coeffs
    ss_res = np.sum((sigma - X @ coeffs) ** 2)
    ss_tot = np.sum((sigma - sigma.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    return A, B, r2


def build_targets(label_df, fit_fn, property_name):
    """Fit per-compound coefficients."""
    records = []
    for smi, grp in label_df.groupby("smiles"):
        if len(grp) < MIN_PTS:
            continue
        T = grp["T_K"].values
        vals = grp["value"].values
        A, B, r2 = fit_fn(T, vals)
        if r2 >= ARR_R2_THRESH:
            records.append({"smiles": smi, "A": A, "B": B, "r2": r2, "n_pts": len(grp)})
    result = pd.DataFrame(records)
    flush_print(f"[{property_name}] {len(result)} compounds pass Arrhenius/linear fit "
                f"(from {label_df['smiles'].nunique()}, min_pts={MIN_PTS}, r2>={ARR_R2_THRESH})")
    return result


# ── Prepare X, y ────────────────────────────────────────────────────────────

def prepare_data(features, label_df, coeff_df, property_type):
    coeff_df = coeff_df[coeff_df["smiles"].isin(features.index)]
    smiles_arr = coeff_df["smiles"].values
    X = features.loc[smiles_arr].values.astype(np.float32)
    y = coeff_df[["A", "B"]].values.astype(np.float64)

    raw = {}
    for smi in smiles_arr:
        sub = label_df[label_df["smiles"] == smi]
        raw[smi] = {"T_K": sub["T_K"].values, "value": sub["value"].values}

    return X, y, smiles_arr, raw


# ── Evaluation ──────────────────────────────────────────────────────────────

def evaluate_reconstruction(y_test, y_pred, smiles_test, raw, prop_type):
    all_true, all_pred = [], []
    for i, smi in enumerate(smiles_test):
        if smi not in raw:
            continue
        T = raw[smi]["T_K"]
        true_vals = raw[smi]["value"]
        A_p, B_p = y_pred[i]
        if prop_type == "viscosity":
            pred_vals = np.exp(A_p + B_p / T)
        else:
            pred_vals = A_p + B_p * T
        valid = np.isfinite(pred_vals) & np.isfinite(true_vals) & (true_vals > 0)
        if valid.sum() > 0:
            all_true.extend(true_vals[valid])
            all_pred.extend(pred_vals[valid])

    all_true = np.array(all_true)
    all_pred = np.clip(np.array(all_pred), 1e-15, 1e10)
    if len(all_true) == 0:
        return {"r2": float("nan"), "mape": float("nan")}
    return {
        "r2": r2_score(all_true, all_pred),
        "mape": mean_absolute_percentage_error(all_true, all_pred) * 100,
    }


# ── Model definitions ──────────────────────────────────────────────────────

def get_models():
    return {
        "XGBoost": {
            "estimator": xgb.XGBRegressor(
                objective="reg:squarederror", tree_method="hist",
                n_jobs=-1, random_state=SEED, verbosity=0,
            ),
            "params": {
                "estimator__n_estimators": randint(100, 1000),
                "estimator__max_depth": randint(3, 12),
                "estimator__learning_rate": loguniform(0.01, 0.3),
                "estimator__subsample": uniform(0.6, 0.4),
                "estimator__colsample_bytree": uniform(0.3, 0.7),
                "estimator__reg_alpha": loguniform(1e-3, 10),
                "estimator__reg_lambda": loguniform(1e-3, 10),
                "estimator__min_child_weight": randint(1, 10),
            },
            "n_iter": 30,
        },
        "LightGBM": {
            "estimator": lgb.LGBMRegressor(
                objective="regression", n_jobs=-1, random_state=SEED, verbose=-1,
            ),
            "params": {
                "estimator__n_estimators": randint(100, 1000),
                "estimator__max_depth": randint(3, 12),
                "estimator__learning_rate": loguniform(0.01, 0.3),
                "estimator__subsample": uniform(0.6, 0.4),
                "estimator__colsample_bytree": uniform(0.3, 0.7),
                "estimator__reg_alpha": loguniform(1e-3, 10),
                "estimator__reg_lambda": loguniform(1e-3, 10),
                "estimator__num_leaves": randint(15, 127),
                "estimator__min_child_samples": randint(3, 30),
            },
            "n_iter": 30,
        },
        "RandomForest": {
            "estimator": RandomForestRegressor(n_jobs=-1, random_state=SEED),
            "params": {
                "estimator__n_estimators": randint(100, 1000),
                "estimator__max_depth": randint(5, 30),
                "estimator__min_samples_split": randint(2, 20),
                "estimator__min_samples_leaf": randint(1, 10),
                "estimator__max_features": uniform(0.1, 0.9),
            },
            "n_iter": 20,
        },
        "Ridge": {
            "estimator": Ridge(random_state=SEED),
            "params": {"estimator__alpha": loguniform(1e-3, 1e4)},
            "n_iter": 20,
        },
        "Lasso": {
            "estimator": Lasso(random_state=SEED, max_iter=10000),
            "params": {"estimator__alpha": loguniform(1e-5, 1e2)},
            "n_iter": 20,
        },
    }


# ── Cross-validation ───────────────────────────────────────────────────────

def run_cv(X, y, smiles_arr, raw, prop_type):
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    models = get_models()
    results = {}

    scaler_X = StandardScaler()
    X_s = scaler_X.fit_transform(X)

    for name, cfg in models.items():
        flush_print(f"\n{'='*60}")
        flush_print(f"  {name} -- {prop_type}")
        flush_print(f"{'='*60}")

        fold_metrics = {"r2_A": [], "r2_B": [], "r2_AB": [], "recon_r2": [], "recon_mape": []}
        t0 = time.time()
        best_params = None
        best_cv_score = -np.inf

        for fi, (tr_idx, te_idx) in enumerate(kf.split(X_s)):
            X_tr, X_te = X_s[tr_idx], X_s[te_idx]
            y_tr, y_te = y[tr_idx], y[te_idx]
            smi_te = smiles_arr[te_idx]

            sc_y = StandardScaler()
            y_tr_s = sc_y.fit_transform(y_tr)

            base = MultiOutputRegressor(
                cfg["estimator"].__class__(**cfg["estimator"].get_params())
            )

            search = RandomizedSearchCV(
                base, cfg["params"], n_iter=cfg["n_iter"], cv=3,
                scoring="r2", random_state=SEED + fi, n_jobs=1,
                refit=True, error_score=-999,
            )
            search.fit(X_tr, y_tr_s)

            if search.best_score_ > best_cv_score:
                best_cv_score = search.best_score_
                best_params = search.best_params_

            y_pred = sc_y.inverse_transform(search.predict(X_te))

            r2_A = r2_score(y_te[:, 0], y_pred[:, 0])
            r2_B = r2_score(y_te[:, 1], y_pred[:, 1])
            r2_AB = r2_score(y_te, y_pred, multioutput="uniform_average")

            fold_metrics["r2_A"].append(r2_A)
            fold_metrics["r2_B"].append(r2_B)
            fold_metrics["r2_AB"].append(r2_AB)

            test_raw = {s: raw[s] for s in smi_te if s in raw}
            recon = evaluate_reconstruction(y_te, y_pred, smi_te, test_raw, prop_type)
            fold_metrics["recon_r2"].append(recon["r2"])
            fold_metrics["recon_mape"].append(recon["mape"])

            flush_print(f"  Fold {fi+1}: R2(A,B)={r2_AB:.4f}  R2(A)={r2_A:.4f}  "
                        f"R2(B)={r2_B:.4f}  Recon R2={recon['r2']:.4f}  MAPE={recon['mape']:.1f}%")

        elapsed = time.time() - t0

        results[name] = {}
        for k, vals in fold_metrics.items():
            results[name][k] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
        results[name]["training_time_s"] = round(elapsed, 1)
        results[name]["best_params"] = {
            k: (int(v) if isinstance(v, (np.integer,)) else
                float(v) if isinstance(v, (np.floating, float)) else str(v))
            for k, v in (best_params or {}).items()
        }

        flush_print(f"\n  >>> {name} SUMMARY:")
        flush_print(f"      R2(A,B)    = {np.mean(fold_metrics['r2_AB']):.4f} +/- {np.std(fold_metrics['r2_AB']):.4f}")
        flush_print(f"      Recon R2   = {np.mean(fold_metrics['recon_r2']):.4f} +/- {np.std(fold_metrics['recon_r2']):.4f}")
        flush_print(f"      Recon MAPE = {np.mean(fold_metrics['recon_mape']):.1f}% +/- {np.std(fold_metrics['recon_mape']):.1f}%")
        flush_print(f"      Time: {elapsed:.1f}s")

    return results


def get_feature_importance(X, y, features_df, smiles_arr, prop_type):
    """Train LightGBM on full data, extract top-20 features."""
    sc_X = StandardScaler()
    X_s = sc_X.fit_transform(X)
    sc_y = StandardScaler()
    y_s = sc_y.fit_transform(y)

    model = MultiOutputRegressor(lgb.LGBMRegressor(
        n_estimators=500, max_depth=8, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.5,
        reg_alpha=0.1, reg_lambda=1.0, num_leaves=63,
        min_child_samples=5, n_jobs=-1, random_state=SEED, verbose=-1,
    ))
    model.fit(X_s, y_s)

    imp = np.zeros(X.shape[1])
    for est in model.estimators_:
        imp += est.feature_importances_
    imp /= len(model.estimators_)

    feat_names = list(features_df.columns)
    top_idx = np.argsort(imp)[::-1][:20]
    top = []
    for rank, idx in enumerate(top_idx, 1):
        top.append({"rank": rank, "feature": feat_names[idx], "importance": float(imp[idx])})

    flush_print(f"\n{'='*60}")
    flush_print(f"  Top 20 features -- {prop_type} (LightGBM)")
    flush_print(f"{'='*60}")
    for f in top:
        flush_print(f"  {f['rank']:2d}. {f['feature']:40s}  {f['importance']:.1f}")

    return top


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    flush_print("="*70)
    flush_print("  MODEL COMPARISON: Trees vs Linear for Arrhenius/Linear Coefficients")
    flush_print("="*70)

    features, visc_df, st_df = load_all_data()

    all_results = {}

    # ── VISCOSITY ──
    flush_print("\n" + "#"*70)
    flush_print("#  VISCOSITY (Arrhenius: ln(eta) = A + B/T)")
    flush_print("#"*70)

    visc_coeffs = build_targets(visc_df, fit_arrhenius, "viscosity")
    X_v, y_v, smi_v, raw_v = prepare_data(features, visc_df, visc_coeffs, "viscosity")
    flush_print(f"  Dataset: {len(smi_v)} compounds, {X_v.shape[1]} features")
    flush_print(f"  A range: [{y_v[:,0].min():.2f}, {y_v[:,0].max():.2f}]")
    flush_print(f"  B range: [{y_v[:,1].min():.1f}, {y_v[:,1].max():.1f}]")

    all_results["viscosity"] = run_cv(X_v, y_v, smi_v, raw_v, "viscosity")
    all_results["viscosity_top_features"] = get_feature_importance(
        X_v, y_v, features, smi_v, "viscosity"
    )

    # ── SURFACE TENSION ──
    flush_print("\n" + "#"*70)
    flush_print("#  SURFACE TENSION (Linear: sigma = A + B*T)")
    flush_print("#"*70)

    st_coeffs = build_targets(st_df, fit_linear_st, "surface_tension")
    X_s, y_s, smi_s, raw_s = prepare_data(features, st_df, st_coeffs, "surface_tension")
    flush_print(f"  Dataset: {len(smi_s)} compounds, {X_s.shape[1]} features")
    flush_print(f"  A range: [{y_s[:,0].min():.4f}, {y_s[:,0].max():.4f}]")
    flush_print(f"  B range: [{y_s[:,1].min():.6f}, {y_s[:,1].max():.6f}]")

    all_results["surface_tension"] = run_cv(X_s, y_s, smi_s, raw_s, "surface_tension")
    all_results["surface_tension_top_features"] = get_feature_importance(
        X_s, y_s, features, smi_s, "surface_tension"
    )

    # ── SUMMARY TABLE ──
    flush_print("\n\n" + "="*100)
    flush_print("  FINAL COMPARISON SUMMARY")
    flush_print("="*100)

    for prop in ["viscosity", "surface_tension"]:
        pr = all_results[prop]
        flush_print(f"\n  {prop.upper()}")
        flush_print(f"  {'Model':<15s} {'R2(A,B)':>14s} {'R2(A)':>14s} {'R2(B)':>14s} "
                     f"{'Recon R2':>14s} {'MAPE%':>12s} {'Time':>8s}")
        flush_print(f"  {'-'*15} {'-'*14} {'-'*14} {'-'*14} {'-'*14} {'-'*12} {'-'*8}")
        for m in ["XGBoost", "LightGBM", "RandomForest", "Ridge", "Lasso"]:
            r = pr[m]
            flush_print(
                f"  {m:<15s} "
                f"{r['r2_AB']['mean']:>6.4f}+/-{r['r2_AB']['std']:.4f} "
                f"{r['r2_A']['mean']:>6.4f}+/-{r['r2_A']['std']:.4f} "
                f"{r['r2_B']['mean']:>6.4f}+/-{r['r2_B']['std']:.4f} "
                f"{r['recon_r2']['mean']:>6.4f}+/-{r['recon_r2']['std']:.4f} "
                f"{r['recon_mape']['mean']:>5.1f}+/-{r['recon_mape']['std']:.1f} "
                f"{r['training_time_s']:>6.1f}s"
            )

    # ── NN reference ──
    flush_print("\n  Reference (current NN from train.py):")
    flush_print("  R2(A,B) ~ 0.89, Recon R2 ~ 0.88 (viscosity)")

    all_results["metadata"] = {
        "n_viscosity_compounds": int(len(smi_v)),
        "n_st_compounds": int(len(smi_s)),
        "n_features": int(X_v.shape[1]),
        "n_folds": N_FOLDS,
        "nn_reference": {
            "note": "Current NN baseline from train.py",
            "r2_AB_approx": 0.89,
            "recon_r2_approx": 0.88,
        },
    }

    OUTPUTS.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUTS / "model_comparison.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    flush_print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
