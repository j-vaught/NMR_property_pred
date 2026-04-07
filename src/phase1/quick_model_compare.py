"""Quick model comparison: XGBoost vs Ridge vs RF on Arrhenius coefficients.
No hyperparameter tuning — uses sensible defaults. 5-fold CV."""

import sys
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shared.config import Paths, DataConfig
from shared.data_utils import canonical_smiles, inchi_to_canonical_smiles, fit_arrhenius, fit_linear_st, is_hydrocarbon
from shared.metrics import compute_all_metrics

warnings.filterwarnings("ignore")

paths = Paths()


def load_data():
    # Features
    fp = pd.read_csv(paths.morgan_r2)
    fp["canonical_smiles"] = fp["SMILES"].apply(canonical_smiles)
    fp = fp.dropna(subset=["canonical_smiles"]).set_index("canonical_smiles")
    bit_cols = [c for c in fp.columns if c not in ("CAS", "SMILES")]
    fp = fp[bit_cols].astype(np.float32)
    fp = fp[~fp.index.duplicated(keep="first")]
    # Remove zero-variance
    fp = fp.loc[:, fp.std() > 0]

    # Add descriptors
    desc = pd.read_csv(paths.features / "rdkit_descriptors.csv")
    desc["canonical_smiles"] = desc["SMILES"].apply(canonical_smiles)
    desc = desc.dropna(subset=["canonical_smiles"]).set_index("canonical_smiles")
    desc_cols = [c for c in desc.columns if c not in ("CAS", "SMILES")]
    desc = desc[desc_cols].astype(np.float32).replace([np.inf, -np.inf], np.nan).fillna(0)
    desc = desc.loc[:, desc.std() > 0]
    desc = (desc - desc.mean()) / desc.std().replace(0, 1)

    common = fp.index.intersection(desc.index)
    features = pd.concat([fp.loc[common], desc.loc[common]], axis=1)
    print(f"Features: {features.shape[0]} compounds, {features.shape[1]} dims")

    # Labels - ThermoML viscosity
    visc = pd.read_csv(paths.thermoml_viscosity)
    visc = visc.dropna(subset=["InChI"])
    visc["canonical_smiles"] = visc["InChI"].apply(inchi_to_canonical_smiles)
    visc = visc.dropna(subset=["canonical_smiles"])
    v_col = [c for c in visc.columns if "viscosity" in c.lower()][0]
    t_col = "T_K" if "T_K" in visc.columns else [c for c in visc.columns if "temp" in c.lower()][0]
    visc = visc[["canonical_smiles", t_col, v_col]].rename(columns={t_col: "T_K", v_col: "value"})
    visc["T_K"] = pd.to_numeric(visc["T_K"], errors="coerce")
    visc["value"] = pd.to_numeric(visc["value"], errors="coerce")
    visc = visc.dropna()
    visc = visc[(visc["T_K"] >= 200) & (visc["T_K"] <= 500) & (visc["value"] > 0) & (visc["value"] < 100)]

    # Labels - ThermoML surface tension
    st = pd.read_csv(paths.thermoml_surface_tension)
    st = st.dropna(subset=["InChI"])
    st["canonical_smiles"] = st["InChI"].apply(inchi_to_canonical_smiles)
    st = st.dropna(subset=["canonical_smiles"])
    s_col = [c for c in st.columns if "surface" in c.lower() or "tension" in c.lower()][0]
    t_col2 = "T_K" if "T_K" in st.columns else [c for c in st.columns if "temp" in c.lower()][0]
    st = st[["canonical_smiles", t_col2, s_col]].rename(columns={t_col2: "T_K", s_col: "value"})
    st["T_K"] = pd.to_numeric(st["T_K"], errors="coerce")
    st["value"] = pd.to_numeric(st["value"], errors="coerce")
    st = st.dropna()
    st = st[(st["T_K"] >= 200) & (st["T_K"] <= 500) & (st["value"] > 0) & (st["value"] < 1)]

    return features, visc, st


def build_arrhenius_targets(features, prop_df, property_name, min_pts=5, r2_thresh=0.95):
    prop_in_features = prop_df[prop_df["canonical_smiles"].isin(features.index)]
    X_list, Y_list, smiles_list = [], [], []

    for smi, grp in prop_in_features.groupby("canonical_smiles"):
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
    print(f"  {property_name}: {len(X)} compounds ({hc_mask.sum()} HC), {X.shape[1]} features")
    return X, Y, smiles_list, hc_mask


def evaluate_fold(Y_true, Y_pred, smiles, hc_mask, prop_df, property_name):
    """Evaluate on [A,B] and reconstructed property."""
    results = {}
    results["ab_r2"] = float(1 - np.sum((Y_true - Y_pred)**2) / np.sum((Y_true - Y_true.mean(0))**2))

    # Reconstruct at actual temperatures
    all_true, all_pred, all_hc = [], [], []
    for i, smi in enumerate(smiles):
        A_p, B_p = Y_pred[i]
        cpd = prop_df[prop_df["canonical_smiles"] == smi]
        hc = hc_mask[i]
        for _, row in cpd.iterrows():
            T = row["T_K"]
            if property_name == "viscosity":
                pred = np.exp(A_p + B_p / T)
            else:
                pred = max(A_p + B_p * T, 1e-6)
            all_true.append(row["value"])
            all_pred.append(pred)
            all_hc.append(hc)

    all_true, all_pred, all_hc = np.array(all_true), np.array(all_pred), np.array(all_hc)
    m = compute_all_metrics(all_true, all_pred)
    results["recon_r2"] = m["r2"]
    results["recon_mape"] = m["mape"]

    if all_hc.any():
        mh = compute_all_metrics(all_true[all_hc], all_pred[all_hc])
        results["hc_recon_r2"] = mh["r2"]
        results["hc_recon_mape"] = mh["mape"]

    return results


def run_comparison(features, prop_df, property_name):
    print(f"\n{'#'*60}")
    print(f"  {property_name.upper()}")
    print(f"{'#'*60}")

    X, Y, smiles, hc_mask = build_arrhenius_targets(features, prop_df, property_name)
    if len(X) < 20:
        print("  Too few compounds, skipping")
        return {}

    models = {
        "Ridge(a=100)": Ridge(alpha=100),
        "Lasso(a=0.1)": Lasso(alpha=0.1, max_iter=5000),
        "RandomForest": RandomForestRegressor(n_estimators=200, max_depth=10, min_samples_leaf=3, random_state=42, n_jobs=-1),
        "XGBoost": xgb.XGBRegressor(n_estimators=200, max_depth=4, learning_rate=0.1, reg_alpha=1.0, reg_lambda=1.0, random_state=42, n_jobs=-1, verbosity=0),
        "LightGBM": lgb.LGBMRegressor(n_estimators=200, max_depth=4, learning_rate=0.1, reg_alpha=1.0, reg_lambda=1.0, random_state=42, n_jobs=-1, verbose=-1),
    }

    all_results = {}

    for name, base_model in models.items():
        print(f"\n  {name}...")
        fold_results = []
        kf = KFold(n_splits=5, shuffle=True, random_state=42)

        for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
            X_tr, X_te = X[train_idx], X[test_idx]
            Y_tr, Y_te = Y[train_idx], Y[test_idx]
            test_smiles = [smiles[i] for i in test_idx]
            test_hc = hc_mask[test_idx]

            # Scale
            sx = StandardScaler().fit(X_tr)
            sy = StandardScaler().fit(Y_tr)
            X_tr_s = sx.transform(X_tr)
            X_te_s = sx.transform(X_te)
            Y_tr_s = sy.transform(Y_tr)

            # Train
            if name in ("XGBoost", "LightGBM"):
                model = MultiOutputRegressor(base_model.__class__(**base_model.get_params()))
            else:
                model = MultiOutputRegressor(base_model.__class__(**base_model.get_params())) if hasattr(base_model, 'get_params') else base_model

            # For Ridge/Lasso, use directly (they support multi-output)
            if name.startswith("Ridge") or name.startswith("Lasso"):
                model = base_model.__class__(**base_model.get_params())
                model.fit(X_tr_s, Y_tr_s)
            else:
                model.fit(X_tr_s, Y_tr_s)

            Y_pred_s = model.predict(X_te_s)
            Y_pred = sy.inverse_transform(Y_pred_s.reshape(-1, 2))

            fr = evaluate_fold(Y_te, Y_pred, test_smiles, test_hc, prop_df, property_name)
            fold_results.append(fr)

        # Aggregate
        agg = {}
        for key in fold_results[0]:
            vals = [fr[key] for fr in fold_results if key in fr]
            agg[f"{key}_mean"] = float(np.mean(vals))
            agg[f"{key}_std"] = float(np.std(vals))

        all_results[name] = agg
        print(f"    [A,B] R²: {agg['ab_r2_mean']:.3f} ± {agg['ab_r2_std']:.3f}")
        print(f"    Recon R²: {agg['recon_r2_mean']:.3f} ± {agg['recon_r2_std']:.3f}")
        print(f"    Recon MAPE: {agg['recon_mape_mean']:.1f}% ± {agg['recon_mape_std']:.1f}%")
        if "hc_recon_r2_mean" in agg:
            print(f"    HC R²: {agg['hc_recon_r2_mean']:.3f} ± {agg['hc_recon_r2_std']:.3f}")

    return all_results


def main():
    print("=" * 60)
    print("  QUICK MODEL COMPARISON (default params, 5-fold CV)")
    print("=" * 60)

    features, visc, st = load_data()
    results = {}
    results["viscosity"] = run_comparison(features, visc, "viscosity")
    results["surface_tension"] = run_comparison(features, st, "surface_tension")

    out = paths.root / "src" / "phase1" / "outputs"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "quick_model_comparison.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {out / 'quick_model_comparison.json'}")


if __name__ == "__main__":
    main()
