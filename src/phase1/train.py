import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shared.config import Paths, DataConfig, SplitConfig, ModelConfig, TrainConfig
from shared.data_utils import fit_arrhenius, fit_linear_st, make_temperature_features
from shared.metrics import compute_all_metrics
from phase1.data_pipeline import build_dataset
from phase1.model import ArrheniusMultiTaskModel, DirectMultiTaskModel, count_parameters


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_direct_tensors(
    fp_df, visc_df, st_df, compound_list, scaler_visc=None, scaler_st=None
):
    fps, t_feats, y_visc, y_st, mask_visc, mask_st = [], [], [], [], [], []

    visc_sub = visc_df[visc_df["canonical_smiles"].isin(compound_list)]
    st_sub = st_df[st_df["canonical_smiles"].isin(compound_list)]

    for _, row in visc_sub.iterrows():
        smi = row["canonical_smiles"]
        if smi not in fp_df.index:
            continue
        fp = fp_df.loc[smi].values
        t = make_temperature_features(np.array([row["T_K"]]))[0]
        fps.append(fp)
        t_feats.append(t)
        y_visc.append(np.log(row["value"]))
        y_st.append(0.0)
        mask_visc.append(True)
        mask_st.append(False)

    for _, row in st_sub.iterrows():
        smi = row["canonical_smiles"]
        if smi not in fp_df.index:
            continue
        fp = fp_df.loc[smi].values
        t = make_temperature_features(np.array([row["T_K"]]))[0]
        fps.append(fp)
        t_feats.append(t)
        y_visc.append(0.0)
        y_st.append(row["value"])
        mask_visc.append(False)
        mask_st.append(True)

    fps = np.array(fps, dtype=np.float32)
    t_feats = np.array(t_feats, dtype=np.float32)
    y_visc = np.array(y_visc, dtype=np.float32).reshape(-1, 1)
    y_st = np.array(y_st, dtype=np.float32).reshape(-1, 1)
    mask_visc = np.array(mask_visc, dtype=bool)
    mask_st = np.array(mask_st, dtype=bool)

    if scaler_visc is None:
        scaler_visc = StandardScaler()
        y_visc[mask_visc] = scaler_visc.fit_transform(y_visc[mask_visc])
    else:
        y_visc[mask_visc] = scaler_visc.transform(y_visc[mask_visc])

    if scaler_st is None:
        scaler_st = StandardScaler()
        y_st[mask_st] = scaler_st.fit_transform(y_st[mask_st])
    else:
        y_st[mask_st] = scaler_st.transform(y_st[mask_st])

    return (
        torch.tensor(fps),
        torch.tensor(t_feats),
        torch.tensor(y_visc),
        torch.tensor(y_st),
        torch.tensor(mask_visc),
        torch.tensor(mask_st),
        scaler_visc,
        scaler_st,
    )


def build_arrhenius_tensors(
    fp_df, visc_df, st_df, compound_list, data_config,
    scaler_visc=None, scaler_st=None
):
    visc_sub = visc_df[visc_df["canonical_smiles"].isin(compound_list)]
    st_sub = st_df[st_df["canonical_smiles"].isin(compound_list)]

    visc_targets = {}
    for smi, group in visc_sub.groupby("canonical_smiles"):
        if len(group) < data_config.min_points_per_compound:
            continue
        A, B, r2 = fit_arrhenius(group["T_K"].values, group["value"].values)
        if r2 >= data_config.arrhenius_r2_threshold:
            visc_targets[smi] = (A, B)

    st_targets = {}
    for smi, group in st_sub.groupby("canonical_smiles"):
        if len(group) < data_config.min_points_per_compound:
            continue
        A, B, r2 = fit_linear_st(group["T_K"].values, group["value"].values)
        if r2 >= data_config.arrhenius_r2_threshold:
            st_targets[smi] = (A, B)

    all_smiles = sorted(set(visc_targets) | set(st_targets))
    all_smiles = [s for s in all_smiles if s in fp_df.index]

    fps, y_visc, y_st, mask_visc, mask_st = [], [], [], [], []
    for smi in all_smiles:
        fps.append(fp_df.loc[smi].values)
        if smi in visc_targets:
            y_visc.append(list(visc_targets[smi]))
            mask_visc.append(True)
        else:
            y_visc.append([0.0, 0.0])
            mask_visc.append(False)
        if smi in st_targets:
            y_st.append(list(st_targets[smi]))
            mask_st.append(True)
        else:
            y_st.append([0.0, 0.0])
            mask_st.append(False)

    fps = np.array(fps, dtype=np.float32)
    y_visc = np.array(y_visc, dtype=np.float32)
    y_st = np.array(y_st, dtype=np.float32)
    mask_visc = np.array(mask_visc, dtype=bool)
    mask_st = np.array(mask_st, dtype=bool)

    if scaler_visc is None:
        scaler_visc = StandardScaler()
        if mask_visc.any():
            y_visc[mask_visc] = scaler_visc.fit_transform(y_visc[mask_visc])
    else:
        if mask_visc.any():
            y_visc[mask_visc] = scaler_visc.transform(y_visc[mask_visc])

    if scaler_st is None:
        scaler_st = StandardScaler()
        if mask_st.any():
            y_st[mask_st] = scaler_st.fit_transform(y_st[mask_st])
    else:
        if mask_st.any():
            y_st[mask_st] = scaler_st.transform(y_st[mask_st])

    return (
        torch.tensor(fps),
        torch.tensor(y_visc),
        torch.tensor(y_st),
        torch.tensor(mask_visc),
        torch.tensor(mask_st),
        scaler_visc,
        scaler_st,
    )


def masked_mse(pred, target, mask):
    if not mask.any():
        return torch.tensor(0.0, device=pred.device)
    return nn.functional.mse_loss(pred[mask], target[mask])


def train_direct(dataset: dict, model_config: ModelConfig, train_config: TrainConfig):
    set_seed(train_config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    fp_df = dataset["fp_df"]
    visc_df = dataset["visc_df"]
    st_df = dataset["st_df"]
    splits = dataset["splits"]

    train_data = build_direct_tensors(fp_df, visc_df, st_df, splits["train"])
    fps_tr, t_tr, yv_tr, ys_tr, mv_tr, ms_tr, sc_v, sc_s = train_data

    val_data = build_direct_tensors(fp_df, visc_df, st_df, splits["val"], sc_v, sc_s)
    fps_val, t_val, yv_val, ys_val, mv_val, ms_val, _, _ = val_data

    train_ds = TensorDataset(fps_tr, t_tr, yv_tr, ys_tr, mv_tr, ms_tr)
    train_loader = DataLoader(train_ds, batch_size=train_config.batch_size, shuffle=True, drop_last=False)

    model = DirectMultiTaskModel(
        fp_dim=fps_tr.shape[1],
        t_feature_dim=t_tr.shape[1],
        encoder_layers=model_config.encoder_layers[1:],
        dropout=model_config.dropout,
    ).to(device)

    print(f"Model parameters: {count_parameters(model):,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=train_config.lr, weight_decay=train_config.weight_decay)

    if train_config.lr_scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=train_config.epochs, eta_min=train_config.lr_min)
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)

    best_val_loss = float("inf")
    patience_counter = 0
    history = {"train_loss": [], "val_loss": [], "lr": []}

    output_dir = Paths().outputs
    output_dir.mkdir(parents=True, exist_ok=True)

    fps_val_d = fps_val.to(device)
    t_val_d = t_val.to(device)
    yv_val_d = yv_val.to(device)
    ys_val_d = ys_val.to(device)

    t0 = time.time()
    for epoch in range(train_config.epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            fp_b, t_b, yv_b, ys_b, mv_b, ms_b = [x.to(device) for x in batch]
            optimizer.zero_grad()

            pred_v, pred_s = model(fp_b, t_b)
            loss_v = masked_mse(pred_v, yv_b, mv_b)
            loss_s = masked_mse(pred_s, ys_b, ms_b)
            loss = train_config.viscosity_weight * loss_v + train_config.surface_tension_weight * loss_s

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_train_loss = epoch_loss / max(n_batches, 1)

        model.eval()
        with torch.no_grad():
            pred_v_val, pred_s_val = model(fps_val_d, t_val_d)
            val_loss_v = masked_mse(pred_v_val, yv_val_d, mv_val.to(device))
            val_loss_s = masked_mse(pred_s_val, ys_val_d, ms_val.to(device))
            val_loss = (train_config.viscosity_weight * val_loss_v + train_config.surface_tension_weight * val_loss_s).item()

        current_lr = optimizer.param_groups[0]["lr"]
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(val_loss)
        history["lr"].append(current_lr)

        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_loss)
        else:
            scheduler.step()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "val_loss": val_loss,
                "scaler_visc_mean": sc_v.mean_.tolist(),
                "scaler_visc_scale": sc_v.scale_.tolist(),
                "scaler_st_mean": sc_s.mean_.tolist(),
                "scaler_st_scale": sc_s.scale_.tolist(),
                "model_config": {
                    "fp_dim": fps_tr.shape[1],
                    "t_feature_dim": t_tr.shape[1],
                    "encoder_layers": model_config.encoder_layers[1:],
                    "dropout": model_config.dropout,
                },
            }, output_dir / "best_direct_model.pt")
        else:
            patience_counter += 1

        if (epoch + 1) % 10 == 0 or epoch == 0:
            elapsed = time.time() - t0
            print(f"Epoch {epoch+1:3d}/{train_config.epochs} | "
                  f"train={avg_train_loss:.4f} val={val_loss:.4f} | "
                  f"lr={current_lr:.2e} | best={best_val_loss:.4f} | "
                  f"{elapsed:.0f}s")

        if patience_counter >= train_config.patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    with open(output_dir / "training_history.json", "w") as f:
        json.dump(history, f)

    checkpoint = torch.load(output_dir / "best_direct_model.pt", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    return model, sc_v, sc_s, history


def evaluate_direct(model, fp_df, visc_df, st_df, test_compounds, scaler_visc, scaler_st, device):
    test_data = build_direct_tensors(fp_df, visc_df, st_df, test_compounds, scaler_visc, scaler_st)
    fps, t_feats, yv, ys, mv, ms, _, _ = test_data

    model.eval()
    with torch.no_grad():
        pred_v, pred_s = model(fps.to(device), t_feats.to(device))

    results = {}

    if mv.any():
        y_true_v = scaler_visc.inverse_transform(yv[mv].numpy())
        y_pred_v = scaler_visc.inverse_transform(pred_v[mv].cpu().numpy())
        y_true_v_exp = np.exp(y_true_v)
        y_pred_v_exp = np.exp(y_pred_v)
        results["viscosity_log"] = compute_all_metrics(y_true_v.flatten(), y_pred_v.flatten())
        results["viscosity_exp"] = compute_all_metrics(y_true_v_exp.flatten(), y_pred_v_exp.flatten())

    if ms.any():
        y_true_s = scaler_st.inverse_transform(ys[ms].numpy())
        y_pred_s = scaler_st.inverse_transform(pred_s[ms].cpu().numpy())
        results["surface_tension"] = compute_all_metrics(y_true_s.flatten(), y_pred_s.flatten())

    return results


def main():
    paths = Paths()
    data_config = DataConfig()
    split_config = SplitConfig()
    model_config = ModelConfig()
    train_config = TrainConfig(approach="direct")

    dataset = build_dataset(paths, data_config, split_config)

    model, sc_v, sc_s, history = train_direct(dataset, model_config, train_config)

    device = next(model.parameters()).device
    results = evaluate_direct(
        model, dataset["fp_df"], dataset["visc_df"], dataset["st_df"],
        dataset["splits"]["test"], sc_v, sc_s, device
    )

    print("\n" + "=" * 60)
    print("TEST SET RESULTS")
    print("=" * 60)
    for prop, metrics in results.items():
        print(f"\n{prop}:")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")

    output_dir = Paths().outputs
    with open(output_dir / "test_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
