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
from shared.data_utils import make_temperature_features
from shared.metrics import compute_all_metrics
from phase1.data_pipeline import build_dataset
from phase1.model import DirectMultiTaskModel, SingleTaskModel, ArrheniusModel, count_parameters


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_single_task_tensors(fp_df, prop_df, compound_list, scaler=None):
    """Build tensors for a single property. One row per (compound, T) measurement."""
    sub = prop_df[prop_df["canonical_smiles"].isin(compound_list)]
    sub = sub[sub["canonical_smiles"].isin(fp_df.index)]

    fps = []
    t_feats = []
    targets = []

    for _, row in sub.iterrows():
        smi = row["canonical_smiles"]
        fp = fp_df.loc[smi].values
        t = make_temperature_features(np.array([row["T_K"]]))[0]
        fps.append(fp)
        t_feats.append(t)
        targets.append(row["value"])

    if len(fps) == 0:
        return None

    fps = np.array(fps, dtype=np.float32)
    t_feats = np.array(t_feats, dtype=np.float32)
    targets = np.array(targets, dtype=np.float32).reshape(-1, 1)

    if scaler is None:
        scaler = StandardScaler()
        targets = scaler.fit_transform(targets)
    else:
        targets = scaler.transform(targets)

    return (
        torch.tensor(fps),
        torch.tensor(t_feats),
        torch.tensor(targets, dtype=torch.float32),
        scaler,
    )


def build_multitask_tensors(fp_df, visc_df, st_df, compound_list,
                            scaler_visc=None, scaler_st=None):
    """Build tensors where compounds with both properties at same T share a row."""
    visc_sub = visc_df[visc_df["canonical_smiles"].isin(compound_list)]
    visc_sub = visc_sub[visc_sub["canonical_smiles"].isin(fp_df.index)]
    st_sub = st_df[st_df["canonical_smiles"].isin(compound_list)]
    st_sub = st_sub[st_sub["canonical_smiles"].isin(fp_df.index)]

    # Transform viscosity to log scale
    visc_sub = visc_sub.copy()
    visc_sub["log_value"] = np.log(visc_sub["value"])

    fps, t_feats = [], []
    y_visc, y_st = [], []
    mask_visc, mask_st = [], []

    # Viscosity rows
    for _, row in visc_sub.iterrows():
        fps.append(fp_df.loc[row["canonical_smiles"]].values)
        t_feats.append(make_temperature_features(np.array([row["T_K"]]))[0])
        y_visc.append(row["log_value"])
        y_st.append(0.0)
        mask_visc.append(True)
        mask_st.append(False)

    # Surface tension rows
    for _, row in st_sub.iterrows():
        fps.append(fp_df.loc[row["canonical_smiles"]].values)
        t_feats.append(make_temperature_features(np.array([row["T_K"]]))[0])
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
        torch.tensor(fps), torch.tensor(t_feats),
        torch.tensor(y_visc), torch.tensor(y_st),
        torch.tensor(mask_visc), torch.tensor(mask_st),
        scaler_visc, scaler_st,
    )


def masked_loss(pred, target, mask, delta=1.0):
    if not mask.any():
        return torch.tensor(0.0, device=pred.device, requires_grad=True)
    return nn.functional.huber_loss(pred[mask], target[mask], delta=delta)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train_single_task(dataset, property_name, train_config):
    """Train a single-task model for viscosity or surface tension."""
    set_seed(train_config.seed)
    device = get_device()
    print(f"\n{'='*60}")
    print(f"SINGLE TASK: {property_name}")
    print(f"Device: {device}")
    print(f"{'='*60}")

    fp_df = dataset["fp_df"]
    prop_df = dataset["visc_df"] if property_name == "viscosity" else dataset["st_df"]
    splits = dataset["splits"]

    # Transform viscosity to log scale
    if property_name == "viscosity":
        prop_df = prop_df.copy()
        prop_df["value"] = np.log(prop_df["value"])

    train_data = build_single_task_tensors(fp_df, prop_df, splits["train"])
    if train_data is None:
        print(f"No training data for {property_name}")
        return None
    fps_tr, t_tr, y_tr, scaler = train_data

    val_data = build_single_task_tensors(fp_df, prop_df, splits["val"], scaler)
    if val_data is None:
        print(f"No validation data for {property_name}")
        return None
    fps_val, t_val, y_val, _ = val_data

    print(f"Train: {len(fps_tr)} rows, Val: {len(fps_val)} rows")

    train_ds = TensorDataset(fps_tr, t_tr, y_tr)
    train_loader = DataLoader(train_ds, batch_size=train_config.batch_size, shuffle=True)

    model = SingleTaskModel(
        fp_dim=fps_tr.shape[1],
        t_feature_dim=t_tr.shape[1],
        dropout=0.3,
    ).to(device)
    print(f"Parameters: {count_parameters(model):,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=train_config.lr, weight_decay=train_config.weight_decay)
    warmup_epochs = 10
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(1, train_config.epochs - warmup_epochs)
        return max(train_config.lr_min / train_config.lr, 0.5 * (1 + np.cos(np.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    fps_val_d = fps_val.to(device)
    t_val_d = t_val.to(device)
    y_val_d = y_val.to(device)

    best_val_loss = float("inf")
    patience_counter = 0
    history = {"train_loss": [], "val_loss": [], "lr": []}

    output_dir = Paths().outputs
    output_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    for epoch in range(train_config.epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for fp_b, t_b, y_b in train_loader:
            fp_b, t_b, y_b = fp_b.to(device), t_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            pred = model(fp_b, t_b)
            loss = nn.functional.huber_loss(pred, y_b, delta=1.0)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        avg_train = epoch_loss / max(n_batches, 1)
        scheduler.step()

        model.eval()
        with torch.no_grad():
            pred_val = model(fps_val_d, t_val_d)
            val_loss = nn.functional.huber_loss(pred_val, y_val_d, delta=1.0).item()

        history["train_loss"].append(avg_train)
        history["val_loss"].append(val_loss)
        history["lr"].append(optimizer.param_groups[0]["lr"])

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "scaler_mean": scaler.mean_.tolist(),
                "scaler_scale": scaler.scale_.tolist(),
                "epoch": epoch,
                "val_loss": val_loss,
                "property": property_name,
            }, output_dir / f"best_{property_name}_model.pt")
        else:
            patience_counter += 1

        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}/{train_config.epochs} | "
                  f"train={avg_train:.4f} val={val_loss:.4f} | "
                  f"lr={optimizer.param_groups[0]['lr']:.2e} | best={best_val_loss:.4f} | "
                  f"{time.time()-t0:.0f}s")

        if patience_counter >= train_config.patience:
            print(f"  Early stopping at epoch {epoch+1}")
            break

    ckpt = torch.load(output_dir / f"best_{property_name}_model.pt", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])

    # Evaluate on test set
    test_data = build_single_task_tensors(fp_df, prop_df, splits["test"], scaler)
    if test_data is not None:
        fps_te, t_te, y_te, _ = test_data
        model.eval()
        with torch.no_grad():
            pred_te = model(fps_te.to(device), t_te.to(device)).cpu().numpy()

        y_true_scaled = y_te.numpy()
        y_pred_scaled = pred_te

        y_true = scaler.inverse_transform(y_true_scaled)
        y_pred = scaler.inverse_transform(y_pred_scaled)

        metrics_std = compute_all_metrics(y_true.flatten(), y_pred.flatten())
        print(f"\n  Test ({property_name}, standardized space):")
        for k, v in metrics_std.items():
            print(f"    {k}: {v:.4f}")

        if property_name == "viscosity":
            y_true_exp = np.exp(y_true)
            y_pred_exp = np.exp(y_pred)
            metrics_exp = compute_all_metrics(y_true_exp.flatten(), y_pred_exp.flatten())
            print(f"\n  Test (viscosity, Pa.s):")
            for k, v in metrics_exp.items():
                print(f"    {k}: {v:.4f}")

    return model, scaler, history


def train_multitask(dataset, train_config):
    """Train multi-task model for both properties simultaneously."""
    set_seed(train_config.seed)
    device = get_device()
    print(f"\n{'='*60}")
    print(f"MULTI-TASK TRAINING")
    print(f"Device: {device}")
    print(f"{'='*60}")

    fp_df = dataset["fp_df"]
    splits = dataset["splits"]

    train_data = build_multitask_tensors(fp_df, dataset["visc_df"], dataset["st_df"], splits["train"])
    fps_tr, t_tr, yv_tr, ys_tr, mv_tr, ms_tr, sc_v, sc_s = train_data

    val_data = build_multitask_tensors(fp_df, dataset["visc_df"], dataset["st_df"], splits["val"], sc_v, sc_s)
    fps_val, t_val, yv_val, ys_val, mv_val, ms_val, _, _ = val_data

    n_visc_tr = mv_tr.sum().item()
    n_st_tr = ms_tr.sum().item()
    print(f"Train: {len(fps_tr)} total rows ({n_visc_tr} visc, {n_st_tr} ST)")
    print(f"Val: {len(fps_val)} total rows ({mv_val.sum().item()} visc, {ms_val.sum().item()} ST)")

    train_ds = TensorDataset(fps_tr, t_tr, yv_tr, ys_tr, mv_tr, ms_tr)
    train_loader = DataLoader(train_ds, batch_size=train_config.batch_size, shuffle=True)

    model = DirectMultiTaskModel(
        fp_dim=fps_tr.shape[1],
        t_feature_dim=t_tr.shape[1],
        dropout=0.3,
    ).to(device)
    print(f"Parameters: {count_parameters(model):,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=train_config.lr, weight_decay=train_config.weight_decay)
    warmup_epochs = 10
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(1, train_config.epochs - warmup_epochs)
        return max(train_config.lr_min / train_config.lr, 0.5 * (1 + np.cos(np.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Balance loss weights inversely proportional to data size
    visc_weight = 1.0
    st_weight = n_visc_tr / max(n_st_tr, 1)
    print(f"Loss weights: visc={visc_weight:.2f}, ST={st_weight:.2f}")

    fps_val_d = fps_val.to(device)
    t_val_d = t_val.to(device)
    yv_val_d = yv_val.to(device)
    ys_val_d = ys_val.to(device)
    mv_val_d = mv_val.to(device)
    ms_val_d = ms_val.to(device)

    best_val_loss = float("inf")
    patience_counter = 0
    history = {"train_loss": [], "val_loss": [], "val_visc": [], "val_st": [], "lr": []}

    output_dir = Paths().outputs
    output_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    for epoch in range(train_config.epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for batch in train_loader:
            fp_b, t_b, yv_b, ys_b, mv_b, ms_b = [x.to(device) for x in batch]
            optimizer.zero_grad()
            pred_v, pred_s = model(fp_b, t_b)
            loss_v = masked_loss(pred_v, yv_b, mv_b)
            loss_s = masked_loss(pred_s, ys_b, ms_b)
            loss = visc_weight * loss_v + st_weight * loss_s
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_train = epoch_loss / max(n_batches, 1)

        model.eval()
        with torch.no_grad():
            pv, ps = model(fps_val_d, t_val_d)
            vl_v = masked_loss(pv, yv_val_d, mv_val_d).item()
            vl_s = masked_loss(ps, ys_val_d, ms_val_d).item()
            val_loss = visc_weight * vl_v + st_weight * vl_s

        history["train_loss"].append(avg_train)
        history["val_loss"].append(val_loss)
        history["val_visc"].append(vl_v)
        history["val_st"].append(vl_s)
        history["lr"].append(optimizer.param_groups[0]["lr"])

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "scaler_visc_mean": sc_v.mean_.tolist(),
                "scaler_visc_scale": sc_v.scale_.tolist(),
                "scaler_st_mean": sc_s.mean_.tolist(),
                "scaler_st_scale": sc_s.scale_.tolist(),
                "epoch": epoch,
                "val_loss": val_loss,
            }, output_dir / "best_multitask_model.pt")
        else:
            patience_counter += 1

        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}/{train_config.epochs} | "
                  f"train={avg_train:.4f} val={val_loss:.4f} (v={vl_v:.4f} s={vl_s:.4f}) | "
                  f"lr={optimizer.param_groups[0]['lr']:.2e} | best={best_val_loss:.4f} | "
                  f"{time.time()-t0:.0f}s")

        if patience_counter >= train_config.patience:
            print(f"  Early stopping at epoch {epoch+1}")
            break

    ckpt = torch.load(output_dir / "best_multitask_model.pt", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])

    # Test evaluation
    test_data = build_multitask_tensors(fp_df, dataset["visc_df"], dataset["st_df"], splits["test"], sc_v, sc_s)
    fps_te, t_te, yv_te, ys_te, mv_te, ms_te, _, _ = test_data

    model.eval()
    with torch.no_grad():
        pv, ps = model(fps_te.to(device), t_te.to(device))

    print(f"\n  TEST RESULTS (multi-task):")
    if mv_te.any():
        y_true_v = sc_v.inverse_transform(yv_te[mv_te].numpy())
        y_pred_v = sc_v.inverse_transform(pv[mv_te].cpu().numpy())
        m_log = compute_all_metrics(y_true_v.flatten(), y_pred_v.flatten())
        print(f"\n  Viscosity (log space):")
        for k, v in m_log.items():
            print(f"    {k}: {v:.4f}")

        y_true_exp = np.exp(y_true_v)
        y_pred_exp = np.exp(y_pred_v)
        m_exp = compute_all_metrics(y_true_exp.flatten(), y_pred_exp.flatten())
        print(f"\n  Viscosity (Pa.s):")
        for k, v in m_exp.items():
            print(f"    {k}: {v:.4f}")

    if ms_te.any():
        y_true_s = sc_s.inverse_transform(ys_te[ms_te].numpy())
        y_pred_s = sc_s.inverse_transform(ps[ms_te].cpu().numpy())
        m_st = compute_all_metrics(y_true_s.flatten(), y_pred_s.flatten())
        print(f"\n  Surface tension (N/m):")
        for k, v in m_st.items():
            print(f"    {k}: {v:.4f}")

    return model, sc_v, sc_s, history


def build_arrhenius_data(fp_df, prop_df, compound_list, property_name, data_config, scaler=None):
    """Fit [A, B] per compound, return (fps, targets, scaler)."""
    sub = prop_df[prop_df["canonical_smiles"].isin(compound_list)]
    sub = sub[sub["canonical_smiles"].isin(fp_df.index)]

    fps, targets, smiles_list = [], [], []
    for smi, group in sub.groupby("canonical_smiles"):
        if len(group) < data_config.min_points_per_compound:
            continue
        T = group["T_K"].values
        vals = group["value"].values

        if property_name == "viscosity":
            A, B, r2 = fit_arrhenius(T, vals)
        else:
            A, B, r2 = fit_linear_st(T, vals)

        if r2 >= data_config.arrhenius_r2_threshold:
            fps.append(fp_df.loc[smi].values)
            targets.append([A, B])
            smiles_list.append(smi)

    if len(fps) == 0:
        return None

    fps = np.array(fps, dtype=np.float32)
    targets = np.array(targets, dtype=np.float32)

    if scaler is None:
        scaler = StandardScaler()
        targets = scaler.fit_transform(targets)
    else:
        targets = scaler.transform(targets)

    return torch.tensor(fps), torch.tensor(targets, dtype=torch.float32), scaler, smiles_list


def train_arrhenius(dataset, property_name, train_config, data_config=None):
    """Train Arrhenius model: FP -> [A, B] coefficients."""
    set_seed(train_config.seed)
    device = get_device()

    if data_config is None:
        data_config = DataConfig()

    print(f"\n{'='*60}")
    print(f"ARRHENIUS: {property_name}")
    print(f"Device: {device}")
    print(f"{'='*60}")

    fp_df = dataset["fp_df"]
    prop_df = dataset["visc_df"] if property_name == "viscosity" else dataset["st_df"]
    splits = dataset["splits"]

    train_data = build_arrhenius_data(fp_df, prop_df, splits["train"], property_name, data_config)
    if train_data is None:
        print("No training data after Arrhenius fitting")
        return None
    fps_tr, y_tr, scaler, train_smiles = train_data

    val_data = build_arrhenius_data(fp_df, prop_df, splits["val"], property_name, data_config, scaler)
    if val_data is None:
        print("No validation data after Arrhenius fitting")
        return None
    fps_val, y_val, _, val_smiles = val_data

    print(f"Train: {len(fps_tr)} compounds, Val: {len(fps_val)} compounds")

    train_ds = TensorDataset(fps_tr, y_tr)
    train_loader = DataLoader(train_ds, batch_size=min(64, len(fps_tr)), shuffle=True)

    model = ArrheniusModel(fp_dim=fps_tr.shape[1], dropout=0.3).to(device)
    print(f"Parameters: {count_parameters(model):,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=train_config.lr, weight_decay=train_config.weight_decay)
    warmup_epochs = 10

    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(1, train_config.epochs - warmup_epochs)
        return max(train_config.lr_min / train_config.lr, 0.5 * (1 + np.cos(np.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    fps_val_d = fps_val.to(device)
    y_val_d = y_val.to(device)

    best_val_loss = float("inf")
    patience_counter = 0
    output_dir = Paths().outputs
    output_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    for epoch in range(train_config.epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for fp_b, y_b in train_loader:
            fp_b, y_b = fp_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            pred = model(fp_b)
            loss = nn.functional.huber_loss(pred, y_b, delta=1.0)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_train = epoch_loss / max(n_batches, 1)

        model.eval()
        with torch.no_grad():
            pred_val = model(fps_val_d)
            val_loss = nn.functional.huber_loss(pred_val, y_val_d, delta=1.0).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "scaler_mean": scaler.mean_.tolist(),
                "scaler_scale": scaler.scale_.tolist(),
                "epoch": epoch,
                "val_loss": val_loss,
                "property": property_name,
                "approach": "arrhenius",
            }, output_dir / f"best_arrhenius_{property_name}.pt")
        else:
            patience_counter += 1

        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}/{train_config.epochs} | "
                  f"train={avg_train:.4f} val={val_loss:.4f} | "
                  f"lr={optimizer.param_groups[0]['lr']:.2e} | best={best_val_loss:.4f} | "
                  f"{time.time()-t0:.0f}s")

        if patience_counter >= train_config.patience:
            print(f"  Early stopping at epoch {epoch+1}")
            break

    # Load best and evaluate on test set
    ckpt = torch.load(output_dir / f"best_arrhenius_{property_name}.pt", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])

    test_data = build_arrhenius_data(fp_df, prop_df, splits["test"], property_name, data_config, scaler)
    if test_data is not None:
        fps_te, y_te, _, test_smiles = test_data
        model.eval()
        with torch.no_grad():
            pred_te = model(fps_te.to(device)).cpu().numpy()

        y_true = scaler.inverse_transform(y_te.numpy())
        y_pred = scaler.inverse_transform(pred_te)

        # Metrics on [A, B] coefficients
        metrics_ab = compute_all_metrics(y_true.flatten(), y_pred.flatten())
        print(f"\n  Test ([A,B] coefficients):")
        for k, v in metrics_ab.items():
            print(f"    {k}: {v:.4f}")

        # Reconstruct property at actual temperatures and compute MAPE
        test_sub = prop_df[prop_df["canonical_smiles"].isin(test_smiles)]
        all_true, all_pred = [], []

        for i, smi in enumerate(test_smiles):
            A_pred, B_pred = y_pred[i]
            A_true, B_true = y_true[i]
            cpd_data = test_sub[test_sub["canonical_smiles"] == smi]

            for _, row in cpd_data.iterrows():
                T = row["T_K"]
                true_val = row["value"]

                if property_name == "viscosity":
                    pred_val = np.exp(A_pred + B_pred / T)
                else:
                    pred_val = A_pred + B_pred * T

                all_true.append(true_val)
                all_pred.append(pred_val)

        if all_true:
            all_true = np.array(all_true)
            all_pred = np.array(all_pred)
            # Clip negative predictions for surface tension
            if property_name == "surface_tension":
                all_pred = np.clip(all_pred, 1e-6, None)
            metrics_recon = compute_all_metrics(all_true, all_pred)
            print(f"\n  Test (reconstructed {property_name} at measured temperatures):")
            for k, v in metrics_recon.items():
                print(f"    {k}: {v:.4f}")

    return model, scaler


def train_ensemble(dataset, property_name, train_config, data_config, n_models=5):
    """Train an ensemble of Arrhenius models with different seeds."""
    print(f"\n{'='*60}")
    print(f"ENSEMBLE ({n_models} models): {property_name}")
    print(f"{'='*60}")

    models = []
    scalers = []
    for i in range(n_models):
        seed = train_config.seed + i * 111
        tc = TrainConfig(
            approach="arrhenius",
            epochs=train_config.epochs,
            batch_size=train_config.batch_size,
            lr=train_config.lr,
            weight_decay=train_config.weight_decay,
            patience=train_config.patience,
            lr_min=train_config.lr_min,
            seed=seed,
        )
        print(f"\n--- Ensemble member {i+1}/{n_models} (seed={seed}) ---")
        result = train_arrhenius(dataset, property_name, tc, data_config)
        if result is not None:
            models.append(result[0])
            scalers.append(result[1])

    if not models:
        print("No models trained successfully")
        return

    # Ensemble evaluation on test set
    device = get_device()
    fp_df = dataset["fp_df"]
    prop_df = dataset["visc_df"] if property_name == "viscosity" else dataset["st_df"]
    splits = dataset["splits"]

    test_data = build_arrhenius_data(fp_df, prop_df, splits["test"], property_name, data_config, scalers[0])
    if test_data is None:
        print("No test data")
        return

    fps_te, y_te, _, test_smiles = test_data

    # Average predictions from all models
    all_preds = []
    for model, scaler in zip(models, scalers):
        # Refit test data with this model's scaler
        td = build_arrhenius_data(fp_df, prop_df, splits["test"], property_name, data_config, scaler)
        if td is None:
            continue
        fps_t, y_t, _, _ = td
        model.eval()
        with torch.no_grad():
            pred = model(fps_t.to(device)).cpu().numpy()
        pred_unscaled = scaler.inverse_transform(pred)
        all_preds.append(pred_unscaled)

    ensemble_pred = np.mean(all_preds, axis=0)
    y_true = scalers[0].inverse_transform(y_te.numpy())

    metrics_ab = compute_all_metrics(y_true.flatten(), ensemble_pred.flatten())
    print(f"\n  ENSEMBLE Test ([A,B] coefficients):")
    for k, v in metrics_ab.items():
        print(f"    {k}: {v:.4f}")

    # Reconstruct at measured temperatures
    test_sub = prop_df[prop_df["canonical_smiles"].isin(test_smiles)]
    all_true, all_pred_vals = [], []

    for i, smi in enumerate(test_smiles):
        A_pred, B_pred = ensemble_pred[i]
        cpd_data = test_sub[test_sub["canonical_smiles"] == smi]

        for _, row in cpd_data.iterrows():
            T = row["T_K"]
            if property_name == "viscosity":
                pred_val = np.exp(A_pred + B_pred / T)
            else:
                pred_val = max(A_pred + B_pred * T, 1e-6)
            all_true.append(row["value"])
            all_pred_vals.append(pred_val)

    if all_true:
        metrics_recon = compute_all_metrics(np.array(all_true), np.array(all_pred_vals))
        print(f"\n  ENSEMBLE Test (reconstructed {property_name}):")
        for k, v in metrics_recon.items():
            print(f"    {k}: {v:.4f}")

    return models, scalers


def main():
    paths = Paths()
    data_config = DataConfig()
    split_config = SplitConfig()
    train_config = TrainConfig(
        approach="direct",
        epochs=500,
        batch_size=128,
        lr=3e-4,
        weight_decay=1e-4,
        patience=50,
        lr_min=1e-6,
        seed=42,
    )

    dataset = build_dataset(paths, data_config, split_config)

    # 1. Single-task direct baselines
    print("\n" + "#" * 60)
    print("# PHASE 1a: SINGLE-TASK DIRECT BASELINES")
    print("#" * 60)

    train_single_task(dataset, "viscosity", train_config)
    train_single_task(dataset, "surface_tension", train_config)

    # 2. Arrhenius approach (compound-level)
    print("\n" + "#" * 60)
    print("# PHASE 1b: ARRHENIUS MODELS")
    print("#" * 60)

    train_arrhenius(dataset, "viscosity", train_config, data_config)
    train_arrhenius(dataset, "surface_tension", train_config, data_config)

    # 3. Ensemble of Arrhenius models
    print("\n" + "#" * 60)
    print("# PHASE 1c: ENSEMBLE (5 models)")
    print("#" * 60)

    train_ensemble(dataset, "viscosity", train_config, data_config, n_models=5)
    train_ensemble(dataset, "surface_tension", train_config, data_config, n_models=5)

    output_dir = Paths().outputs
    print(f"\nAll results saved to {output_dir}")


if __name__ == "__main__":
    main()
