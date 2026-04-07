"""
Physics-informed loss functions for Arrhenius viscosity prediction.

Problem: The model predicts [A, B] where ln(eta) = A + B/T. Training with
MSE on [A, B] ignores the exponential mapping eta = exp(A + B/T), so small
coefficient errors amplify into large viscosity errors.

Analysis summary (from actual dataset):
  - A ~ N(-12.2, 2.9), B ~ N(1513, 1482)
  - A 5% relative error in both A and B produces a MEDIAN 133% error in eta
  - Error is asymmetric (overestimates are larger than underestimates)
  - Error amplification is worst at low temperatures (|B|/T is larger)
  - A errors dominate B errors on a per-percent-relative basis at T > 300K

Solution: Compute loss in the RECONSTRUCTED property space. The model still
outputs [A, B], but the loss evaluates ln(eta_pred(T)) vs ln(eta_true(T))
at a grid of reference temperatures. This makes the gradient aware of how
coefficient errors map to actual viscosity errors.
"""

import torch
import torch.nn as nn
import numpy as np


# Reference temperature grid (K) for reconstruction loss
DEFAULT_TEMPS_K = [250.0, 300.0, 350.0, 400.0, 450.0]


class ArrheniusReconstructionLoss(nn.Module):
    """
    Physics-informed loss that evaluates Arrhenius predictions in the
    reconstructed ln(viscosity) space rather than coefficient space.

    The model outputs [A, B] (possibly standardized). This loss:
      1. Unstandardizes the predictions to physical [A, B]
      2. Computes ln(eta) = A + B/T at each reference temperature
      3. Returns MSE of ln(eta_pred) vs ln(eta_true) averaged over temperatures

    Working in ln(eta) space (not eta space) avoids numerical overflow and
    ensures the loss gradient is proportional to RELATIVE viscosity error,
    which is the physically meaningful metric.

    Parameters
    ----------
    temps_K : list of float
        Reference temperatures for reconstruction. Default: [250, 300, 350, 400, 450].
    scaler_mean : array-like of shape (2,)
        Mean used to standardize [A, B] targets. None if targets are unscaled.
    scaler_scale : array-like of shape (2,)
        Std used to standardize [A, B] targets. None if targets are unscaled.
    delta : float
        Huber delta for the reconstruction loss. Use float('inf') for pure MSE.
    """

    def __init__(self, temps_K=None, scaler_mean=None, scaler_scale=None, delta=1.0):
        super().__init__()
        if temps_K is None:
            temps_K = DEFAULT_TEMPS_K

        # Store 1/T as a buffer (moves to device with model)
        inv_T = torch.tensor([1.0 / T for T in temps_K], dtype=torch.float32)
        self.register_buffer("inv_T", inv_T)  # shape (n_temps,)

        if scaler_mean is not None:
            self.register_buffer("scaler_mean",
                                 torch.tensor(scaler_mean, dtype=torch.float32))
            self.register_buffer("scaler_scale",
                                 torch.tensor(scaler_scale, dtype=torch.float32))
        else:
            self.scaler_mean = None
            self.scaler_scale = None

        self.delta = delta

    def _unstandardize(self, ab):
        """Convert standardized [A, B] back to physical units."""
        if self.scaler_mean is not None:
            return ab * self.scaler_scale + self.scaler_mean
        return ab

    def _compute_ln_eta(self, ab_physical):
        """
        Compute ln(eta) = A + B/T at all reference temperatures.

        Parameters
        ----------
        ab_physical : tensor of shape (batch, 2)
            Physical [A, B] values.

        Returns
        -------
        ln_eta : tensor of shape (batch, n_temps)
        """
        A = ab_physical[:, 0:1]  # (batch, 1)
        B = ab_physical[:, 1:2]  # (batch, 1)
        inv_T = self.inv_T.unsqueeze(0)  # (1, n_temps)
        return A + B * inv_T  # (batch, n_temps)

    def forward(self, pred_ab, true_ab):
        """
        Parameters
        ----------
        pred_ab : tensor of shape (batch, 2), standardized [A, B]
        true_ab : tensor of shape (batch, 2), standardized [A, B]

        Returns
        -------
        loss : scalar tensor
        """
        pred_phys = self._unstandardize(pred_ab)
        true_phys = self._unstandardize(true_ab)

        ln_eta_pred = self._compute_ln_eta(pred_phys)  # (batch, n_temps)
        ln_eta_true = self._compute_ln_eta(true_phys)  # (batch, n_temps)

        if self.delta == float("inf"):
            return nn.functional.mse_loss(ln_eta_pred, ln_eta_true)
        else:
            return nn.functional.huber_loss(ln_eta_pred, ln_eta_true, delta=self.delta)


class HybridArrheniusLoss(nn.Module):
    """
    Weighted combination of coefficient-space and reconstruction-space losses.

    L = (1 - lambda) * L_coeff + lambda * L_recon

    This preserves some direct supervision on [A, B] (which stabilizes early
    training) while steering the model toward accurate viscosity predictions.

    Parameters
    ----------
    lam : float
        Weight on the reconstruction loss. 0 = pure coefficient loss,
        1 = pure reconstruction loss. Default: 0.5.
    temps_K : list of float
        Reference temperatures for reconstruction.
    scaler_mean, scaler_scale : array-like or None
        Standardization parameters for [A, B].
    delta_coeff : float
        Huber delta for coefficient loss.
    delta_recon : float
        Huber delta for reconstruction loss.
    """

    def __init__(self, lam=0.5, temps_K=None, scaler_mean=None, scaler_scale=None,
                 delta_coeff=1.0, delta_recon=1.0):
        super().__init__()
        self.lam = lam
        self.recon_loss = ArrheniusReconstructionLoss(
            temps_K=temps_K,
            scaler_mean=scaler_mean,
            scaler_scale=scaler_scale,
            delta=delta_recon,
        )
        self.delta_coeff = delta_coeff

    def forward(self, pred_ab, true_ab):
        loss_coeff = nn.functional.huber_loss(pred_ab, true_ab, delta=self.delta_coeff)
        loss_recon = self.recon_loss(pred_ab, true_ab)
        return (1.0 - self.lam) * loss_coeff + self.lam * loss_recon


class TemperatureWeightedReconstructionLoss(nn.Module):
    """
    Reconstruction loss with per-temperature weights.

    By default, weights are proportional to 1/T, placing more emphasis on
    low-temperature accuracy where error amplification is worst.

    Parameters
    ----------
    temps_K : list of float
        Reference temperatures.
    temp_weights : list of float or None
        Per-temperature weights. If None, uses 1/T weighting (normalized).
    scaler_mean, scaler_scale : array-like or None
        Standardization parameters.
    delta : float
        Huber delta.
    """

    def __init__(self, temps_K=None, temp_weights=None, scaler_mean=None,
                 scaler_scale=None, delta=1.0):
        super().__init__()
        if temps_K is None:
            temps_K = DEFAULT_TEMPS_K

        inv_T = torch.tensor([1.0 / T for T in temps_K], dtype=torch.float32)
        self.register_buffer("inv_T", inv_T)

        if temp_weights is None:
            # Weight proportional to 1/T (low T gets more weight)
            w = torch.tensor([1.0 / T for T in temps_K], dtype=torch.float32)
        else:
            w = torch.tensor(temp_weights, dtype=torch.float32)
        w = w / w.sum()
        self.register_buffer("weights", w)

        if scaler_mean is not None:
            self.register_buffer("scaler_mean",
                                 torch.tensor(scaler_mean, dtype=torch.float32))
            self.register_buffer("scaler_scale",
                                 torch.tensor(scaler_scale, dtype=torch.float32))
        else:
            self.scaler_mean = None
            self.scaler_scale = None

        self.delta = delta

    def _unstandardize(self, ab):
        if self.scaler_mean is not None:
            return ab * self.scaler_scale + self.scaler_mean
        return ab

    def forward(self, pred_ab, true_ab):
        pred_phys = self._unstandardize(pred_ab)
        true_phys = self._unstandardize(true_ab)

        A_p, B_p = pred_phys[:, 0:1], pred_phys[:, 1:2]
        A_t, B_t = true_phys[:, 0:1], true_phys[:, 1:2]
        inv_T = self.inv_T.unsqueeze(0)

        ln_eta_pred = A_p + B_p * inv_T
        ln_eta_true = A_t + B_t * inv_T

        # Per-temperature Huber loss, then weighted sum
        per_T_loss = torch.zeros(len(self.weights), device=pred_ab.device)
        for i in range(len(self.weights)):
            per_T_loss[i] = nn.functional.huber_loss(
                ln_eta_pred[:, i], ln_eta_true[:, i], delta=self.delta
            )
        return (self.weights * per_T_loss).sum()


class MultiOutputReferenceLoss(nn.Module):
    """
    Alternative approach: predict ln(eta) at reference temperatures directly.

    Instead of predicting [A, B] (2 outputs), the model predicts
    [ln(eta_T1), ln(eta_T2), ..., ln(eta_Tn)] (n outputs).

    Post-hoc, [A, B] can be recovered via linear regression on 1/T.

    This loss is a simple MSE on the predicted vs true ln(eta) values.
    It avoids the error amplification problem entirely because the loss
    operates directly on the quantity of interest.

    Parameters
    ----------
    temps_K : list of float
        The reference temperatures. The model's output dimension must match len(temps_K).
    """

    def __init__(self, temps_K=None, delta=1.0):
        super().__init__()
        if temps_K is None:
            temps_K = DEFAULT_TEMPS_K
        self.temps_K = temps_K
        self.delta = delta

    @staticmethod
    def compute_targets(A, B, temps_K):
        """
        Convert [A, B] Arrhenius parameters to ln(eta) at reference temperatures.

        Parameters
        ----------
        A, B : array-like
            Arrhenius coefficients (can be numpy or torch).
        temps_K : list of float

        Returns
        -------
        ln_eta : array of shape (n_compounds, n_temps)
        """
        inv_T = np.array([1.0 / T for T in temps_K])
        A = np.atleast_1d(np.asarray(A))
        B = np.atleast_1d(np.asarray(B))
        return A[:, None] + B[:, None] * inv_T[None, :]

    @staticmethod
    def recover_arrhenius(ln_eta_predictions, temps_K):
        """
        Recover [A, B] from predicted ln(eta) at reference temperatures
        via least-squares regression: ln(eta) = A + B * (1/T).

        Parameters
        ----------
        ln_eta_predictions : array of shape (n_compounds, n_temps)
        temps_K : list of float

        Returns
        -------
        A, B : arrays of shape (n_compounds,)
        """
        inv_T = np.array([1.0 / T for T in temps_K])
        X = np.column_stack([np.ones_like(inv_T), inv_T])
        # Solve for each compound
        # ln_eta = X @ [A, B]^T  -> [A, B] = (X^T X)^{-1} X^T ln_eta^T
        coeffs = np.linalg.lstsq(X, ln_eta_predictions.T, rcond=None)[0]
        return coeffs[0], coeffs[1]  # A_array, B_array

    def forward(self, pred_ln_eta, true_ln_eta):
        """
        Parameters
        ----------
        pred_ln_eta : tensor of shape (batch, n_temps)
        true_ln_eta : tensor of shape (batch, n_temps)
        """
        return nn.functional.huber_loss(pred_ln_eta, true_ln_eta, delta=self.delta)


def make_loss(loss_type, scaler=None, temps_K=None, lam=0.5, delta=1.0):
    """
    Factory function to create a loss for the Arrhenius training loop.

    Drop-in replacement for nn.functional.huber_loss in train.py.

    Parameters
    ----------
    loss_type : str
        One of: "huber" (baseline), "reconstruction", "hybrid",
        "weighted_reconstruction".
    scaler : sklearn StandardScaler or None
        The scaler used for [A, B] targets.
    temps_K : list of float or None
    lam : float
        Lambda for hybrid loss.
    delta : float
        Huber delta.

    Returns
    -------
    loss_fn : nn.Module with forward(pred, target) -> scalar
    """
    scaler_mean = scaler.mean_.tolist() if scaler is not None else None
    scaler_scale = scaler.scale_.tolist() if scaler is not None else None

    if loss_type == "huber":
        return _HuberWrapper(delta=delta)
    elif loss_type == "reconstruction":
        return ArrheniusReconstructionLoss(
            temps_K=temps_K, scaler_mean=scaler_mean,
            scaler_scale=scaler_scale, delta=delta,
        )
    elif loss_type == "hybrid":
        return HybridArrheniusLoss(
            lam=lam, temps_K=temps_K,
            scaler_mean=scaler_mean, scaler_scale=scaler_scale,
            delta_coeff=delta, delta_recon=delta,
        )
    elif loss_type == "weighted_reconstruction":
        return TemperatureWeightedReconstructionLoss(
            temps_K=temps_K, scaler_mean=scaler_mean,
            scaler_scale=scaler_scale, delta=delta,
        )
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


class _HuberWrapper(nn.Module):
    """Thin wrapper so all losses have the same .forward(pred, target) API."""
    def __init__(self, delta=1.0):
        super().__init__()
        self.delta = delta

    def forward(self, pred, target):
        return nn.functional.huber_loss(pred, target, delta=self.delta)
