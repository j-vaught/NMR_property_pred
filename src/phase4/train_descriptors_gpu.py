"""
Train NMR → descriptors MLP on GPU (PyTorch CUDA).

Loads data/nmr/phase4_pretraining_set.parquet (~1.5M compounds).
Trains a multi-task MLP to predict 8 RDKit descriptors from NMR features.

Architecture:
  Input:  120 NMR features (95 spectrum + 25 peaklist)
  Hidden: 256 → 256 → 128 (with dropout + BN)
  Output: 8 descriptors (one head per target, shared backbone)

Training:
  - Adam optimizer, lr=1e-3, cosine schedule
  - Batch size: 4096
  - ~20 epochs
  - 90/10 train/val split
  - Saves best model by val loss to phase4_descriptors_model.pt
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shared.config import Paths

paths = Paths()
PARQUET = paths.data / "nmr" / "phase4_pretraining_set.parquet"
OUT_DIR = Path(__file__).parent / "outputs"
OUT_DIR.mkdir(exist_ok=True)
MODEL_PATH = OUT_DIR / "phase4_descriptors_model.pt"
SCALERS_PATH = OUT_DIR / "phase4_descriptors_scalers.npz"

DESCRIPTOR_COLS = [
    "mw", "logp", "tpsa", "n_heavy_atoms", "h_count_molecular",
    "n_rotatable", "n_aromatic_rings", "n_peaks",
]


class NMRDescriptorsMLP:
    """PyTorch MLP wrapper kept simple — builds the model on the fly."""

    def __init__(self, input_dim=120, n_outputs=8, hidden=(256, 256, 128), dropout=0.2):
        import torch
        import torch.nn as nn

        layers = []
        in_d = input_dim
        for h in hidden:
            layers.extend([
                nn.Linear(in_d, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            in_d = h
        layers.append(nn.Linear(in_d, n_outputs))
        self.net = nn.Sequential(*layers)

    def __call__(self, x):
        return self.net(x)


def load_data():
    print(f"Loading {PARQUET}")
    df = pd.read_parquet(PARQUET)
    print(f"  {len(df)} compounds")

    # Filter out any rows with NaN descriptors
    df = df.dropna(subset=DESCRIPTOR_COLS).reset_index(drop=True)
    print(f"  {len(df)} after NaN filter")

    # Stack features
    spec = np.stack([np.array(s, dtype=np.float32) for s in df["spectrum_features"]])
    peak = np.stack([np.array(s, dtype=np.float32) for s in df["peaklist_features"]])
    X = np.hstack([spec, peak])  # (N, 120)

    # Stack targets
    Y = df[DESCRIPTOR_COLS].values.astype(np.float32)  # (N, 8)

    print(f"  X: {X.shape}, Y: {Y.shape}")
    return X, Y


def main():
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    print("=" * 70)
    print("  Phase 4 GPU: Train NMR → descriptors MLP")
    print("=" * 70)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Load data
    X, Y = load_data()

    # Standardize features and targets
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0) + 1e-8
    X_scaled = (X - X_mean) / X_std

    Y_mean = Y.mean(axis=0)
    Y_std = Y.std(axis=0) + 1e-8
    Y_scaled = (Y - Y_mean) / Y_std

    print(f"\nTarget stats (original):")
    for j, name in enumerate(DESCRIPTOR_COLS):
        print(f"  {name:<22s} mean={Y_mean[j]:8.2f}  std={Y_std[j]:8.2f}")

    # Train/val split (90/10, random)
    rng = np.random.RandomState(42)
    idx = rng.permutation(len(X_scaled))
    n_val = max(10000, int(0.1 * len(X_scaled)))
    val_idx = idx[:n_val]
    tr_idx = idx[n_val:]

    X_tr = torch.from_numpy(X_scaled[tr_idx]).float()
    Y_tr = torch.from_numpy(Y_scaled[tr_idx]).float()
    X_va = torch.from_numpy(X_scaled[val_idx]).float()
    Y_va = torch.from_numpy(Y_scaled[val_idx]).float()

    print(f"\nTrain: {len(X_tr)}, Val: {len(X_va)}")

    # Build model
    model = NMRDescriptorsMLP(
        input_dim=X.shape[1],
        n_outputs=Y.shape[1],
        hidden=(256, 256, 128),
        dropout=0.2,
    ).net.to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters: {n_params:,}")
    print(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    n_epochs = 30
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=1e-5)

    batch_size = 4096
    train_ds = TensorDataset(X_tr, Y_tr)
    val_ds = TensorDataset(X_va, Y_va)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=2, pin_memory=True)

    # Training loop
    print(f"\nTraining for {n_epochs} epochs...")
    best_val_loss = float("inf")
    best_state = None
    history = []

    for epoch in range(n_epochs):
        # Train
        model.train()
        t0 = time.time()
        train_losses = []
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            optimizer.zero_grad()
            pred = model(xb)
            loss = ((pred - yb) ** 2).mean()
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        train_loss = np.mean(train_losses)

        # Validate
        model.eval()
        val_losses = []
        per_target_sse = np.zeros(Y.shape[1])
        per_target_n = np.zeros(Y.shape[1])
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                pred = model(xb)
                loss = ((pred - yb) ** 2).mean()
                val_losses.append(loss.item())
                # Per-target SSE (on scaled space)
                sq_err = ((pred - yb) ** 2).cpu().numpy()
                per_target_sse += sq_err.sum(axis=0)
                per_target_n += sq_err.shape[0]

        val_loss = np.mean(val_losses)
        scheduler.step()

        # Per-target R² (on scaled space)
        per_target_mse = per_target_sse / per_target_n
        # Since target was standardized (unit variance), R² = 1 - MSE
        per_target_r2 = 1 - per_target_mse

        elapsed = time.time() - t0
        print(f"  Epoch {epoch+1:2d}/{n_epochs}: "
              f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
              f"avg_R²={per_target_r2.mean():.3f}  ({elapsed:.0f}s)")

        history.append({
            "epoch": epoch + 1,
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "per_target_r2": per_target_r2.tolist(),
            "lr": optimizer.param_groups[0]["lr"],
        })

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # Load best model
    model.load_state_dict(best_state)
    model.eval()

    # Final per-target metrics
    print(f"\n{'='*70}")
    print(f"Best val loss: {best_val_loss:.4f}")
    print(f"\nPer-target val R² (final model):")
    last = history[-1]
    for j, (name, r2) in enumerate(zip(DESCRIPTOR_COLS, last["per_target_r2"])):
        print(f"  {name:<22s} R² = {r2:.4f}")

    # Save model + scalers
    torch.save({
        "state_dict": model.state_dict(),
        "input_dim": X.shape[1],
        "hidden": (256, 256, 128),
        "n_outputs": Y.shape[1],
        "descriptor_cols": DESCRIPTOR_COLS,
    }, MODEL_PATH)
    np.savez(SCALERS_PATH, X_mean=X_mean, X_std=X_std, Y_mean=Y_mean, Y_std=Y_std)
    print(f"\nModel saved to {MODEL_PATH}")
    print(f"Scalers saved to {SCALERS_PATH}")

    with open(OUT_DIR / "phase4_descriptors_history.json", "w") as f:
        json.dump(history, f, indent=2)


if __name__ == "__main__":
    main()
