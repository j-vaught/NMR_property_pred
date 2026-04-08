"""
Phase 2 Training: NMR spectrum -> property prediction.

Trains 1D CNN, ResNet, or Transformer models on NMR spectra to predict
viscosity and surface tension. Uses 5-fold CV with 3-model ensemble.
"""

import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, TensorDataset, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shared.config import Paths
from shared.data_utils import (
    is_hydrocarbon,
    make_temperature_features,
    fit_arrhenius,
)
from shared.metrics import compute_all_metrics
from phase1.physics_loss import HybridArrheniusLoss
from phase2.models import NMR1DCNN, NMR1DResNet, NMRTransformer, count_parameters
from phase2.nmr_dataset import (
    build_nmr_property_dataset,
    NMRDirectDataset,
    NMRArrheniusDataset,
)
from phase2.augmentation import NMRAugmentor, SpectrumAugmentation


paths = Paths()
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"


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


# ---------------------------------------------------------------------------
# Transformer peak-list dataset
# ---------------------------------------------------------------------------

# Multiplicity one-hot encoding (8 classes)
_MULT_TO_IDX = {
    "s": 0, "d": 1, "t": 2, "q": 3, "quint": 4,
    "dd": 5, "m": 6, "other": 7,
}


def _encode_peak_features(peak: dict, max_j=20.0) -> np.ndarray:
    """Encode a single peak into a 12-dim feature vector.

    Features: [shift_center, shift_range, mult_onehot(8), J1_norm, J2_norm]
    """
    features = np.zeros(12, dtype=np.float32)

    # Chemical shift center (normalized to [0, 1] over 0-12 ppm range)
    features[0] = peak.get("center_ppm", 0.0) / 12.0

    # Shift range (high - low, normalized)
    shift_range = abs(peak.get("shift_high", 0.0) - peak.get("shift_low", 0.0))
    features[1] = min(shift_range / 2.0, 1.0)

    # Multiplicity one-hot (indices 2-9)
    mult = peak.get("multiplicity", "s").lower().strip()
    mult_idx = _MULT_TO_IDX.get(mult, _MULT_TO_IDX["other"])
    features[2 + mult_idx] = 1.0

    # J-couplings (normalized by max_j)
    j_list = peak.get("j_couplings_hz", [])
    if len(j_list) >= 1:
        features[10] = min(j_list[0] / max_j, 1.0)
    if len(j_list) >= 2:
        features[11] = min(j_list[1] / max_j, 1.0)

    return features


class NMRTransformerDataset(Dataset):
    """Dataset for Transformer model: peak lists as (N, 12) feature tensors.

    Each sample is (peak_features [max_peaks, 12], padding_mask [max_peaks],
                     t_features [3], target [1]) for direct,
    or (peak_features, padding_mask, target_AB [2]) for Arrhenius.
    """

    def __init__(self, peak_lists, labels_rows, property_name="viscosity",
                 approach="direct", max_peaks=50):
        """
        Parameters
        ----------
        peak_lists : list of list[dict]
            One parsed peak list per compound.
        labels_rows : list of dict
            Each dict has 'compound_idx', 'T_K', 'value'.
        property_name : str
        approach : 'direct' or 'arrhenius'
        max_peaks : int
        """
        self.max_peaks = max_peaks
        self.approach = approach
        self.property_name = property_name

        # Encode all peak lists to fixed-size arrays
        self.peak_features = []
        self.padding_masks = []
        for peaks in peak_lists:
            n = min(len(peaks), max_peaks)
            feat = np.zeros((max_peaks, 12), dtype=np.float32)
            mask = np.ones(max_peaks, dtype=bool)  # True = padded
            for i in range(n):
                feat[i] = _encode_peak_features(peaks[i])
                mask[i] = False
            self.peak_features.append(feat)
            self.padding_masks.append(mask)

        self.peak_features = np.array(self.peak_features)
        self.padding_masks = np.array(self.padding_masks)

        # Build per-row data
        self.compound_indices = []
        self.t_features = []
        self.targets = []

        for row in labels_rows:
            cidx = row["compound_idx"]
            self.compound_indices.append(cidx)
            if approach == "direct":
                t_feat = make_temperature_features(np.array([row["T_K"]]))[0]
                self.t_features.append(t_feat)
                val = np.log(row["value"]) if property_name == "viscosity" else row["value"]
                self.targets.append([val])
            else:
                # Arrhenius: target is [A, B]
                self.targets.append(row["target_AB"])

        self.compound_indices = np.array(self.compound_indices)
        if approach == "direct":
            self.t_features = np.array(self.t_features, dtype=np.float32)
        self.targets = np.array(self.targets, dtype=np.float32)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        cidx = self.compound_indices[idx]
        peaks = torch.from_numpy(self.peak_features[cidx]).float()
        mask = torch.from_numpy(self.padding_masks[cidx]).bool()
        target = torch.from_numpy(self.targets[idx]).float()

        if self.approach == "direct":
            t_feat = torch.from_numpy(self.t_features[idx]).float()
            return peaks, mask, t_feat, target
        else:
            return peaks, mask, target


# ---------------------------------------------------------------------------
# Training loop for a single model
# ---------------------------------------------------------------------------

def train_one_model_spectrum(
    model, train_loader, val_loader, device,
    approach="direct", loss_fn=None,
    epochs=500, lr=3e-4, patience=50, print_every=20,
):
    """Train a single spectrum-based model (CNN or ResNet).

    Returns the best model (loaded with best validation weights).
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    warmup = 10

    def lr_lambda(ep):
        if ep < warmup:
            return (ep + 1) / warmup
        progress = (ep - warmup) / max(1, epochs - warmup)
        return max(1e-6 / lr, 0.5 * (1 + np.cos(np.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    if loss_fn is None:
        loss_fn = nn.functional.huber_loss

    best_val = float("inf")
    best_state = None
    wait = 0

    for epoch in range(epochs):
        model.train()
        train_loss_sum = 0.0
        n_batches = 0
        for batch in train_loader:
            if approach == "direct":
                spec, t_feat, target = batch
                spec, t_feat, target = spec.to(device), t_feat.to(device), target.to(device)
                pred = model(spec, t_feat)
            else:
                spec, target = batch
                spec, target = spec.to(device), target.to(device)
                pred = model(spec)

            optimizer.zero_grad()
            if callable(loss_fn) and isinstance(loss_fn, nn.Module):
                loss = loss_fn(pred, target)
            else:
                loss = nn.functional.huber_loss(pred, target, delta=1.0)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss_sum += loss.item()
            n_batches += 1
        scheduler.step()

        # Validation
        model.eval()
        val_loss_sum = 0.0
        val_batches = 0
        with torch.no_grad():
            for batch in val_loader:
                if approach == "direct":
                    spec, t_feat, target = batch
                    spec, t_feat, target = spec.to(device), t_feat.to(device), target.to(device)
                    pred = model(spec, t_feat)
                else:
                    spec, target = batch
                    spec, target = spec.to(device), target.to(device)
                    pred = model(spec)

                if callable(loss_fn) and isinstance(loss_fn, nn.Module):
                    vloss = loss_fn(pred, target)
                else:
                    vloss = nn.functional.huber_loss(pred, target, delta=1.0)
                val_loss_sum += vloss.item()
                val_batches += 1

        val_loss = val_loss_sum / max(val_batches, 1)

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1

        if (epoch + 1) % print_every == 0:
            train_loss = train_loss_sum / max(n_batches, 1)
            print(f"    Epoch {epoch+1:4d}: train={train_loss:.4f}, val={val_loss:.4f}, best={best_val:.4f}, wait={wait}")

        if wait >= patience:
            print(f"    Early stop at epoch {epoch+1}")
            break

    model.load_state_dict(best_state)
    return model


def train_one_model_transformer(
    model, train_loader, val_loader, device,
    approach="direct", loss_fn=None,
    epochs=500, lr=3e-4, patience=50, print_every=20,
):
    """Train a single Transformer model (peak-list input)."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    warmup = 10

    def lr_lambda(ep):
        if ep < warmup:
            return (ep + 1) / warmup
        progress = (ep - warmup) / max(1, epochs - warmup)
        return max(1e-6 / lr, 0.5 * (1 + np.cos(np.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    if loss_fn is None:
        loss_fn = nn.functional.huber_loss

    best_val = float("inf")
    best_state = None
    wait = 0

    for epoch in range(epochs):
        model.train()
        train_loss_sum = 0.0
        n_batches = 0
        for batch in train_loader:
            if approach == "direct":
                peaks, mask, t_feat, target = batch
                peaks = peaks.to(device)
                mask = mask.to(device)
                t_feat = t_feat.to(device)
                target = target.to(device)
                pred = model(peaks, t_feat, padding_mask=mask)
            else:
                peaks, mask, target = batch
                peaks = peaks.to(device)
                mask = mask.to(device)
                target = target.to(device)
                pred = model(peaks, padding_mask=mask)

            optimizer.zero_grad()
            if callable(loss_fn) and isinstance(loss_fn, nn.Module):
                loss = loss_fn(pred, target)
            else:
                loss = nn.functional.huber_loss(pred, target, delta=1.0)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss_sum += loss.item()
            n_batches += 1
        scheduler.step()

        # Validation
        model.eval()
        val_loss_sum = 0.0
        val_batches = 0
        with torch.no_grad():
            for batch in val_loader:
                if approach == "direct":
                    peaks, mask, t_feat, target = batch
                    peaks, mask, t_feat, target = (
                        peaks.to(device), mask.to(device),
                        t_feat.to(device), target.to(device),
                    )
                    pred = model(peaks, t_feat, padding_mask=mask)
                else:
                    peaks, mask, target = batch
                    peaks, mask, target = (
                        peaks.to(device), mask.to(device), target.to(device),
                    )
                    pred = model(peaks, padding_mask=mask)

                if callable(loss_fn) and isinstance(loss_fn, nn.Module):
                    vloss = loss_fn(pred, target)
                else:
                    vloss = nn.functional.huber_loss(pred, target, delta=1.0)
                val_loss_sum += vloss.item()
                val_batches += 1

        val_loss = val_loss_sum / max(val_batches, 1)

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1

        if (epoch + 1) % print_every == 0:
            train_loss = train_loss_sum / max(n_batches, 1)
            print(f"    Epoch {epoch+1:4d}: train={train_loss:.4f}, val={val_loss:.4f}, best={best_val:.4f}, wait={wait}")

        if wait >= patience:
            print(f"    Early stop at epoch {epoch+1}")
            break

    model.load_state_dict(best_state)
    return model


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train_nmr_model(
    model_type="cnn",
    property_name="viscosity",
    approach="direct",
    n_folds=5,
    n_ensemble=3,
    epochs=500,
    lr=3e-4,
    patience=50,
):
    """Train NMR-based property prediction model with CV and ensemble.

    Parameters
    ----------
    model_type : str
        'cnn', 'resnet', or 'transformer'
    property_name : str
        'viscosity' or 'surface_tension'
    approach : str
        'direct' (T as input -> 1 output) or 'arrhenius' ([A, B] prediction)
    n_folds : int
        Number of CV folds.
    n_ensemble : int
        Number of ensemble members per fold.
    epochs : int
        Maximum training epochs.
    lr : float
        Learning rate.
    patience : int
        Early stopping patience.

    Returns
    -------
    list of dict : per-fold results
    """
    print(f"\n{'#'*60}")
    print(f"  {property_name.upper()} - {model_type.upper()} - {approach.upper()}")
    print(f"  {n_folds}-fold CV, {n_ensemble} ensemble")
    print(f"{'#'*60}")

    device = get_device()
    print(f"  Device: {device}")

    # Load data
    data = build_nmr_property_dataset(property_name)
    spectra = data["spectra"]
    labels_df = data["labels_df"]
    smiles_list = data["smiles_list"]
    hc_mask = data["hc_mask"]
    smi_to_idx = data["smi_to_idx"]

    # Compound-level split
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    all_compounds = smiles_list
    hc_set = {s for i, s in enumerate(smiles_list) if hc_mask[i]}

    fold_results = []
    t0 = time.time()

    for fold, (train_cpd_idx, test_cpd_idx) in enumerate(kf.split(all_compounds)):
        train_cpds = set(all_compounds[i] for i in train_cpd_idx)
        test_cpds = set(all_compounds[i] for i in test_cpd_idx)

        print(f"\n--- Fold {fold+1}/{n_folds} (train={len(train_cpds)} cpds, test={len(test_cpds)} cpds) ---")

        # Split labels
        train_labels = labels_df[labels_df["canonical_smiles"].isin(train_cpds)]
        test_labels = labels_df[labels_df["canonical_smiles"].isin(test_cpds)]

        if model_type == "transformer":
            fold_result = _train_fold_transformer(
                fold, train_labels, test_labels, data, train_cpds, test_cpds,
                hc_set, property_name, approach, n_ensemble, device,
                epochs, lr, patience,
            )
        else:
            fold_result = _train_fold_spectrum(
                fold, train_labels, test_labels, data, train_cpds, test_cpds,
                hc_set, model_type, property_name, approach, n_ensemble, device,
                epochs, lr, patience,
            )

        fold_results.append(fold_result)

    # Summary
    elapsed = time.time() - t0
    _print_summary(property_name, model_type, approach, fold_results, elapsed)

    return fold_results


def _train_fold_spectrum(
    fold, train_labels, test_labels, data, train_cpds, test_cpds,
    hc_set, model_type, property_name, approach, n_ensemble, device,
    epochs, lr, patience,
):
    """Train one fold for spectrum-based models (CNN/ResNet)."""
    spectra = data["spectra"]
    smi_to_idx = data["smi_to_idx"]

    if approach == "direct":
        # Create direct datasets
        train_ds = NMRDirectDataset(spectra, train_labels, smi_to_idx, property_name, training=True)
        test_ds = NMRDirectDataset(spectra, test_labels, smi_to_idx, property_name, training=False)

        # Standardize targets
        scaler_y = StandardScaler().fit(train_ds.targets)
        train_ds.targets = scaler_y.transform(train_ds.targets).astype(np.float32)
        test_ds.targets = scaler_y.transform(test_ds.targets).astype(np.float32)

        # Train/val split (90/10)
        n_val = max(1, int(0.1 * len(train_ds)))
        perm = np.random.RandomState(42 + fold).permutation(len(train_ds))
        val_idx = perm[:n_val]
        tr_idx = perm[n_val:]

        train_loader = DataLoader(
            torch.utils.data.Subset(train_ds, tr_idx.tolist()),
            batch_size=min(128, len(tr_idx)), shuffle=True,
        )
        val_loader = DataLoader(
            torch.utils.data.Subset(train_ds, val_idx.tolist()),
            batch_size=min(128, len(val_idx)),
        )

        # Train ensemble
        models = []
        for e in range(n_ensemble):
            set_seed(42 + fold * 1000 + e * 111)
            if model_type == "cnn":
                model = NMR1DCNN(approach="direct").to(device)
            else:
                model = NMR1DResNet(approach="direct").to(device)

            if e == 0:
                print(f"  Model: {model_type.upper()}, {count_parameters(model):,} params")

            model = train_one_model_spectrum(
                model, train_loader, val_loader, device,
                approach="direct", epochs=epochs, lr=lr, patience=patience,
            )
            models.append(model)

        # Evaluate ensemble on test set
        test_loader = DataLoader(test_ds, batch_size=256)
        all_preds = []
        for model in models:
            model.eval()
            preds = []
            with torch.no_grad():
                for batch in test_loader:
                    spec, t_feat, target = batch
                    pred = model(spec.to(device), t_feat.to(device)).cpu().numpy()
                    preds.append(pred)
            preds = np.concatenate(preds, axis=0)
            all_preds.append(scaler_y.inverse_transform(preds))

        ensemble_pred = np.mean(all_preds, axis=0).flatten()
        y_true = scaler_y.inverse_transform(test_ds.targets).flatten()

    else:
        # Arrhenius approach
        train_arr_ds = NMRArrheniusDataset(spectra, train_labels, smi_to_idx)
        test_arr_ds = NMRArrheniusDataset(spectra, test_labels, smi_to_idx)

        if len(train_arr_ds) == 0 or len(test_arr_ds) == 0:
            print("  Skipping fold: insufficient Arrhenius-fit compounds")
            return None

        scaler_y = StandardScaler().fit(train_arr_ds.targets)
        train_arr_ds.targets = scaler_y.transform(train_arr_ds.targets).astype(np.float32)
        test_arr_ds.targets = scaler_y.transform(test_arr_ds.targets).astype(np.float32)

        n_val = max(1, int(0.1 * len(train_arr_ds)))
        perm = np.random.RandomState(42 + fold).permutation(len(train_arr_ds))
        val_idx = perm[:n_val]
        tr_idx = perm[n_val:]

        train_loader = DataLoader(
            torch.utils.data.Subset(train_arr_ds, tr_idx.tolist()),
            batch_size=min(64, len(tr_idx)), shuffle=True,
        )
        val_loader = DataLoader(
            torch.utils.data.Subset(train_arr_ds, val_idx.tolist()),
            batch_size=min(64, len(val_idx)),
        )

        # Loss function
        loss_fn = HybridArrheniusLoss(
            lam=0.5,
            scaler_mean=scaler_y.mean_.tolist(),
            scaler_scale=scaler_y.scale_.tolist(),
        )
        loss_fn = loss_fn.to(device)

        models = []
        for e in range(n_ensemble):
            set_seed(42 + fold * 1000 + e * 111)
            if model_type == "cnn":
                model = NMR1DCNN(approach="arrhenius").to(device)
            else:
                model = NMR1DResNet(approach="arrhenius").to(device)

            if e == 0:
                print(f"  Model: {model_type.upper()}, {count_parameters(model):,} params")

            model = train_one_model_spectrum(
                model, train_loader, val_loader, device,
                approach="arrhenius", loss_fn=loss_fn,
                epochs=epochs, lr=lr, patience=patience,
            )
            models.append(model)

        # Evaluate
        test_loader = DataLoader(test_arr_ds, batch_size=256)
        all_preds = []
        for model in models:
            model.eval()
            preds = []
            with torch.no_grad():
                for batch in test_loader:
                    spec, target = batch
                    pred = model(spec.to(device)).cpu().numpy()
                    preds.append(pred)
            preds = np.concatenate(preds, axis=0)
            all_preds.append(scaler_y.inverse_transform(preds))

        ensemble_pred = np.mean(all_preds, axis=0)
        y_true = scaler_y.inverse_transform(test_arr_ds.targets)

    return _compute_fold_metrics(
        y_true, ensemble_pred, test_labels, smi_to_idx, hc_set,
        property_name, approach, fold,
    )


def _train_fold_transformer(
    fold, train_labels, test_labels, data, train_cpds, test_cpds,
    hc_set, property_name, approach, n_ensemble, device,
    epochs, lr, patience,
):
    """Train one fold for the Transformer model (peak-list input)."""
    from phase2.spectrum_converter import parse_1h_peaks

    smi_to_idx = data["smi_to_idx"]

    # Parse peak lists for all compounds
    import pandas as pd
    nmr_df = pd.read_parquet(paths.nmrexp)
    nmr_df = nmr_df[nmr_df["NMR_type"] == "1H NMR"].copy()
    nmr_df = nmr_df.dropna(subset=["NMR_processed", "SMILES"])
    from shared.data_utils import canonical_smiles
    nmr_df["canonical_smiles"] = nmr_df["SMILES"].apply(canonical_smiles)
    nmr_df = nmr_df.dropna(subset=["canonical_smiles"])
    nmr_df = nmr_df.drop_duplicates(subset=["canonical_smiles"], keep="first")
    nmr_dict = dict(zip(nmr_df["canonical_smiles"], nmr_df["NMR_processed"]))

    smiles_list = data["smiles_list"]

    # Build peak lists aligned with smi_to_idx
    peak_lists = []
    for smi in smiles_list:
        raw = nmr_dict.get(smi, None)
        if raw is not None:
            peaks = parse_1h_peaks(raw)
            if peaks is not None:
                peak_lists.append(peaks)
            else:
                peak_lists.append([{"center_ppm": 0, "shift_high": 0, "shift_low": 0,
                                    "multiplicity": "s", "j_couplings_hz": [], "integral": 1}])
        else:
            peak_lists.append([{"center_ppm": 0, "shift_high": 0, "shift_low": 0,
                                "multiplicity": "s", "j_couplings_hz": [], "integral": 1}])

    if approach == "direct":
        # Build label rows
        train_rows = []
        for _, row in train_labels.iterrows():
            smi = row["canonical_smiles"]
            if smi not in smi_to_idx:
                continue
            train_rows.append({
                "compound_idx": smi_to_idx[smi],
                "T_K": row["T_K"],
                "value": row["value"],
            })

        test_rows = []
        for _, row in test_labels.iterrows():
            smi = row["canonical_smiles"]
            if smi not in smi_to_idx:
                continue
            test_rows.append({
                "compound_idx": smi_to_idx[smi],
                "T_K": row["T_K"],
                "value": row["value"],
            })

        train_ds = NMRTransformerDataset(peak_lists, train_rows, property_name, "direct")
        test_ds = NMRTransformerDataset(peak_lists, test_rows, property_name, "direct")

        scaler_y = StandardScaler().fit(train_ds.targets)
        train_ds.targets = scaler_y.transform(train_ds.targets).astype(np.float32)
        test_ds.targets = scaler_y.transform(test_ds.targets).astype(np.float32)

        n_val = max(1, int(0.1 * len(train_ds)))
        perm = np.random.RandomState(42 + fold).permutation(len(train_ds))
        val_idx = perm[:n_val]
        tr_idx = perm[n_val:]

        train_loader = DataLoader(
            torch.utils.data.Subset(train_ds, tr_idx.tolist()),
            batch_size=min(128, len(tr_idx)), shuffle=True,
        )
        val_loader = DataLoader(
            torch.utils.data.Subset(train_ds, val_idx.tolist()),
            batch_size=min(128, len(val_idx)),
        )

        models = []
        for e in range(n_ensemble):
            set_seed(42 + fold * 1000 + e * 111)
            model = NMRTransformer(approach="direct").to(device)
            if e == 0:
                print(f"  Model: Transformer, {count_parameters(model):,} params")

            model = train_one_model_transformer(
                model, train_loader, val_loader, device,
                approach="direct", epochs=epochs, lr=lr, patience=patience,
            )
            models.append(model)

        # Evaluate
        test_loader = DataLoader(test_ds, batch_size=256)
        all_preds = []
        for model in models:
            model.eval()
            preds = []
            with torch.no_grad():
                for batch in test_loader:
                    peaks, mask, t_feat, target = batch
                    pred = model(
                        peaks.to(device), t_feat.to(device),
                        padding_mask=mask.to(device),
                    ).cpu().numpy()
                    preds.append(pred)
            preds = np.concatenate(preds, axis=0)
            all_preds.append(scaler_y.inverse_transform(preds))

        ensemble_pred = np.mean(all_preds, axis=0).flatten()
        y_true = scaler_y.inverse_transform(test_ds.targets).flatten()

    else:
        # Arrhenius for transformer
        train_rows = []
        for smi, group in train_labels.groupby("canonical_smiles"):
            if smi not in smi_to_idx:
                continue
            if len(group) < 5:
                continue
            T_K = group["T_K"].values
            eta = group["value"].values
            if np.any(eta <= 0):
                continue
            A, B, r2 = fit_arrhenius(T_K, eta)
            if r2 < 0.95:
                continue
            train_rows.append({
                "compound_idx": smi_to_idx[smi],
                "T_K": 300.0,  # placeholder
                "value": 0.0,  # placeholder
                "target_AB": [A, B],
            })

        test_rows = []
        for smi, group in test_labels.groupby("canonical_smiles"):
            if smi not in smi_to_idx:
                continue
            if len(group) < 5:
                continue
            T_K = group["T_K"].values
            eta = group["value"].values
            if np.any(eta <= 0):
                continue
            A, B, r2 = fit_arrhenius(T_K, eta)
            if r2 < 0.95:
                continue
            test_rows.append({
                "compound_idx": smi_to_idx[smi],
                "T_K": 300.0,
                "value": 0.0,
                "target_AB": [A, B],
            })

        if len(train_rows) == 0 or len(test_rows) == 0:
            print("  Skipping fold: insufficient Arrhenius compounds for Transformer")
            return None

        train_ds = NMRTransformerDataset(peak_lists, train_rows, property_name, "arrhenius")
        test_ds = NMRTransformerDataset(peak_lists, test_rows, property_name, "arrhenius")

        scaler_y = StandardScaler().fit(train_ds.targets)
        train_ds.targets = scaler_y.transform(train_ds.targets).astype(np.float32)
        test_ds.targets = scaler_y.transform(test_ds.targets).astype(np.float32)

        n_val = max(1, int(0.1 * len(train_ds)))
        perm = np.random.RandomState(42 + fold).permutation(len(train_ds))
        val_idx = perm[:n_val]
        tr_idx = perm[n_val:]

        train_loader = DataLoader(
            torch.utils.data.Subset(train_ds, tr_idx.tolist()),
            batch_size=min(64, len(tr_idx)), shuffle=True,
        )
        val_loader = DataLoader(
            torch.utils.data.Subset(train_ds, val_idx.tolist()),
            batch_size=min(64, len(val_idx)),
        )

        loss_fn = HybridArrheniusLoss(
            lam=0.5,
            scaler_mean=scaler_y.mean_.tolist(),
            scaler_scale=scaler_y.scale_.tolist(),
        ).to(device)

        models = []
        for e in range(n_ensemble):
            set_seed(42 + fold * 1000 + e * 111)
            model = NMRTransformer(approach="arrhenius").to(device)
            if e == 0:
                print(f"  Model: Transformer, {count_parameters(model):,} params")

            model = train_one_model_transformer(
                model, train_loader, val_loader, device,
                approach="arrhenius", loss_fn=loss_fn,
                epochs=epochs, lr=lr, patience=patience,
            )
            models.append(model)

        test_loader = DataLoader(test_ds, batch_size=256)
        all_preds = []
        for model in models:
            model.eval()
            preds = []
            with torch.no_grad():
                for batch in test_loader:
                    peaks, mask, target = batch
                    pred = model(
                        peaks.to(device),
                        padding_mask=mask.to(device),
                    ).cpu().numpy()
                    preds.append(pred)
            preds = np.concatenate(preds, axis=0)
            all_preds.append(scaler_y.inverse_transform(preds))

        ensemble_pred = np.mean(all_preds, axis=0)
        y_true = scaler_y.inverse_transform(test_ds.targets)

    return _compute_fold_metrics(
        y_true, ensemble_pred, test_labels, smi_to_idx, hc_set,
        property_name, approach, fold,
    )


# ---------------------------------------------------------------------------
# Metrics computation
# ---------------------------------------------------------------------------

def _compute_fold_metrics(
    y_true, ensemble_pred, test_labels, smi_to_idx, hc_set,
    property_name, approach, fold,
):
    """Compute and print metrics for one fold."""
    if approach == "direct":
        # y_true and ensemble_pred are in log-space (viscosity) or real (ST)
        m_std = compute_all_metrics(y_true, ensemble_pred)

        if property_name == "viscosity":
            y_true_real = np.exp(y_true)
            y_pred_real = np.exp(ensemble_pred)
        else:
            y_true_real = y_true
            y_pred_real = np.clip(ensemble_pred, 1e-6, None)
        m_real = compute_all_metrics(y_true_real, y_pred_real)

        # HC-only metrics
        from shared.data_utils import is_hydrocarbon as _is_hc
        hc_indices = []
        idx = 0
        for _, row in test_labels.iterrows():
            smi = row["canonical_smiles"]
            if smi in smi_to_idx:
                if smi in hc_set:
                    hc_indices.append(idx)
                idx += 1

        m_hc = None
        if hc_indices and len(hc_indices) > 0:
            hc_idx = np.array(hc_indices)
            if max(hc_idx) < len(y_true):
                if property_name == "viscosity":
                    m_hc = compute_all_metrics(np.exp(y_true[hc_idx]), np.exp(ensemble_pred[hc_idx]))
                else:
                    m_hc = compute_all_metrics(y_true[hc_idx], np.clip(ensemble_pred[hc_idx], 1e-6, None))

        unit = "log" if property_name == "viscosity" else "N/m"
        print(f"  {unit} R2={m_std['r2']:.3f} | Real R2={m_real['r2']:.3f} MAPE={m_real['mape']:.1f}%", end="")
        if m_hc:
            print(f" | HC R2={m_hc['r2']:.3f} MAPE={m_hc['mape']:.1f}% ({len(hc_indices)} pts)")
        else:
            print(f" | HC: 0 pts")

        return {
            "fold": fold,
            "std_metrics": m_std,
            "real_metrics": m_real,
            "hc_metrics": m_hc,
            "n_hc_test_pts": len(hc_indices),
        }

    else:
        # Arrhenius: y_true and ensemble_pred are [A, B] arrays
        m_coeff = compute_all_metrics(y_true.flatten(), ensemble_pred.flatten())
        print(f"  Coeff R2={m_coeff['r2']:.3f} MAPE={m_coeff['mape']:.1f}%")

        return {
            "fold": fold,
            "coeff_metrics": m_coeff,
        }


def _print_summary(property_name, model_type, approach, fold_results, elapsed):
    """Print summary statistics across folds."""
    # Filter None results
    fold_results = [fr for fr in fold_results if fr is not None]
    if not fold_results:
        print("  No valid fold results.")
        return

    print(f"\n{'='*60}")
    print(f"  {property_name.upper()} - {model_type.upper()} - {approach.upper()} SUMMARY ({elapsed:.0f}s)")
    print(f"{'='*60}")

    if approach == "direct":
        for key, label in [("std_metrics", "Standardized"), ("real_metrics", "Real units"), ("hc_metrics", "HC only")]:
            vals = {}
            for metric in ["r2", "mape", "rmse"]:
                fold_vals = [fr[key][metric] for fr in fold_results if fr.get(key) is not None and metric in fr[key]]
                if fold_vals:
                    vals[metric] = (np.mean(fold_vals), np.std(fold_vals))
            if vals:
                print(f"\n  {label}:")
                for metric, (mean, std) in vals.items():
                    print(f"    {metric}: {mean:.4f} +/- {std:.4f}")
    else:
        vals = {}
        for metric in ["r2", "mape", "rmse"]:
            fold_vals = [fr["coeff_metrics"][metric] for fr in fold_results if "coeff_metrics" in fr]
            if fold_vals:
                vals[metric] = (np.mean(fold_vals), np.std(fold_vals))
        if vals:
            print(f"\n  Coefficient space:")
            for metric, (mean, std) in vals.items():
                print(f"    {metric}: {mean:.4f} +/- {std:.4f}")


# ---------------------------------------------------------------------------
# Phase 1 comparison
# ---------------------------------------------------------------------------

def _load_phase1_results():
    """Load Phase 1 direct optimal results for comparison."""
    p1_path = paths.root / "src" / "phase1" / "outputs" / "direct_optimal_results.json"
    if not p1_path.exists():
        return None
    with open(p1_path) as f:
        return json.load(f)


def _print_comparison(phase2_results, phase1_results):
    """Print comparison table between Phase 1 and Phase 2."""
    print(f"\n{'='*60}")
    print("  PHASE 1 vs PHASE 2 COMPARISON")
    print(f"{'='*60}")

    for prop in ["viscosity", "surface_tension"]:
        if prop not in phase2_results or prop not in phase1_results:
            continue

        print(f"\n  {prop.upper()}:")
        print(f"  {'Method':<30s} {'log/std R2':>10s} {'Real R2':>10s} {'HC R2':>10s} {'Real MAPE':>10s}")
        print(f"  {'-'*70}")

        # Phase 1
        p1_folds = phase1_results[prop]
        p1_std_r2 = np.mean([f["std_metrics"]["r2"] for f in p1_folds])
        p1_real_r2 = np.mean([f["real_metrics"]["r2"] for f in p1_folds])
        p1_hc_r2 = np.mean([f["hc_metrics"]["r2"] for f in p1_folds if f["hc_metrics"]])
        p1_real_mape = np.mean([f["real_metrics"]["mape"] for f in p1_folds])
        print(f"  {'Phase1 FP+desc (direct)':<30s} {p1_std_r2:>10.3f} {p1_real_r2:>10.3f} {p1_hc_r2:>10.3f} {p1_real_mape:>10.1f}%")

        # Phase 2
        for model_key, label in phase2_results[prop].items():
            folds = [fr for fr in label if fr is not None]
            if not folds:
                continue
            if "std_metrics" in folds[0]:
                p2_std_r2 = np.mean([f["std_metrics"]["r2"] for f in folds])
                p2_real_r2 = np.mean([f["real_metrics"]["r2"] for f in folds])
                p2_hc_vals = [f["hc_metrics"]["r2"] for f in folds if f.get("hc_metrics")]
                p2_hc_r2 = np.mean(p2_hc_vals) if p2_hc_vals else float("nan")
                p2_real_mape = np.mean([f["real_metrics"]["mape"] for f in folds])
                print(f"  {label[0].get('model_label', model_key):<30s} {p2_std_r2:>10.3f} {p2_real_r2:>10.3f} {p2_hc_r2:>10.3f} {p2_real_mape:>10.1f}%")


# ---------------------------------------------------------------------------
# JSON serialization helper
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("  PHASE 2: NMR SPECTRUM -> PROPERTY PREDICTION")
    print("  CNN direct for viscosity and surface tension")
    print("=" * 60)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results = {}

    # CNN direct for viscosity
    visc_cnn = train_nmr_model(
        model_type="cnn", property_name="viscosity", approach="direct",
        n_folds=5, n_ensemble=3,
    )
    results.setdefault("viscosity", {})["cnn_direct"] = visc_cnn

    # CNN direct for surface tension
    st_cnn = train_nmr_model(
        model_type="cnn", property_name="surface_tension", approach="direct",
        n_folds=5, n_ensemble=3,
    )
    results.setdefault("surface_tension", {})["cnn_direct"] = st_cnn

    # Save results
    with open(OUTPUT_DIR / "phase2_results.json", "w") as f:
        json.dump(_to_serializable(results), f, indent=2)
    print(f"\nResults saved to {OUTPUT_DIR / 'phase2_results.json'}")

    # Print comparison with Phase 1
    phase1_results = _load_phase1_results()
    if phase1_results:
        _print_comparison(results, phase1_results)


if __name__ == "__main__":
    main()
