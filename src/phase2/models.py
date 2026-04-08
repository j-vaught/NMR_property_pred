"""
Phase 2: 1D neural network architectures for NMR spectrum -> property prediction.

Three architectures:
  A. NMR1DCNN      - 1D convolutional network (~145K params)
  B. NMR1DResNet   - 1D residual network (~190K params)
  C. NMRTransformer - Transformer encoder for peak lists (~130K params)

Each model produces an embedding vector, with a configurable prediction head
for either direct (T as input -> 1 output) or Arrhenius (no T -> 2 outputs).
"""

import math
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# ---------------------------------------------------------------------------
# Prediction heads
# ---------------------------------------------------------------------------

class DirectHead(nn.Module):
    """Direct prediction: concat embedding with T features -> 1 output."""

    def __init__(self, embed_dim, t_feature_dim=3, dropout=0.3):
        super().__init__()
        hidden = min(64, embed_dim)
        self.net = nn.Sequential(
            nn.Linear(embed_dim + t_feature_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, embedding, t_features):
        x = torch.cat([embedding, t_features], dim=1)
        return self.net(x)


class ArrheniusHead(nn.Module):
    """Arrhenius prediction: embedding -> 2 outputs [A, B] (no temperature)."""

    def __init__(self, embed_dim, dropout=0.3):
        super().__init__()
        hidden = min(64, embed_dim)
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 2),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, embedding):
        return self.net(embedding)


# ---------------------------------------------------------------------------
# A. NMR1DCNN (~145K parameters)
# ---------------------------------------------------------------------------

class NMR1DCNN(nn.Module):
    """1D CNN for NMR spectrum -> property prediction.

    Two size variants controlled by `width`:
      - 'small' (~12K params): designed for <1000 compound datasets
        Conv1d(1, 16, k=7) -> BN -> ReLU -> MaxPool(4)     # (16, 300)
        Conv1d(16, 32, k=5) -> BN -> ReLU -> MaxPool(4)    # (32, 75)
        Conv1d(32, 64, k=3) -> BN -> ReLU -> MaxPool(3)    # (64, 25)
        GlobalAvgPool1d -> Dropout(0.5)                     # (64,)

      - 'large' (~145K params): original architecture
        Conv1d(1, 32, k=7) -> BN -> ReLU -> MaxPool(4)     # (32, 300)
        Conv1d(32, 64, k=5) -> BN -> ReLU -> MaxPool(4)    # (64, 75)
        Conv1d(64, 128, k=3) -> BN -> ReLU -> MaxPool(3)   # (128, 25)
        Conv1d(128, 256, k=3) -> BN -> ReLU               # (256, 25)
        GlobalAvgPool1d -> Dropout(0.3)                     # (256,)

    Parameters
    ----------
    in_channels : int
        Number of input channels (default 1 for single spectrum).
    approach : str
        'direct' or 'arrhenius'. Determines the prediction head.
    t_feature_dim : int
        Number of temperature features (only used for direct approach).
    width : str
        'small' or 'large'. Controls model capacity.
    """

    def __init__(self, in_channels=1, approach="direct", t_feature_dim=3, width="small"):
        super().__init__()
        self.approach = approach

        if width == "small":
            self.embed_dim = 64
            drop = 0.5
            self.features = nn.Sequential(
                nn.Conv1d(in_channels, 16, kernel_size=7, padding=3),
                nn.BatchNorm1d(16),
                nn.ReLU(),
                nn.MaxPool1d(4),

                nn.Conv1d(16, 32, kernel_size=5, padding=2),
                nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.MaxPool1d(4),

                nn.Conv1d(32, 64, kernel_size=3, padding=1),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.MaxPool1d(3),
            )
        else:
            self.embed_dim = 256
            drop = 0.3
            self.features = nn.Sequential(
                nn.Conv1d(in_channels, 32, kernel_size=7, padding=3),
                nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.MaxPool1d(4),

                nn.Conv1d(32, 64, kernel_size=5, padding=2),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.MaxPool1d(4),

                nn.Conv1d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.MaxPool1d(3),

                nn.Conv1d(128, 256, kernel_size=3, padding=1),
                nn.BatchNorm1d(256),
                nn.ReLU(),
            )

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(drop)

        if approach == "arrhenius":
            self.head = ArrheniusHead(self.embed_dim, dropout=drop)
        else:
            self.head = DirectHead(self.embed_dim, t_feature_dim, dropout=drop)

        self._init_weights()

    def _init_weights(self):
        for m in self.features.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward_embed(self, x):
        """Extract embedding from spectrum input.

        Parameters
        ----------
        x : tensor, shape (batch, C_in, 1200)

        Returns
        -------
        embedding : tensor, shape (batch, embed_dim)
        """
        h = self.features(x)
        h = self.pool(h).squeeze(-1)
        h = self.dropout(h)
        return h

    def forward(self, x, t_features=None):
        """Forward pass through embedding + prediction head.

        Parameters
        ----------
        x : tensor, shape (batch, C_in, 1200)
        t_features : tensor, shape (batch, t_feature_dim) or None
            Required for direct approach, ignored for arrhenius.

        Returns
        -------
        prediction : tensor, shape (batch, 1) for direct, (batch, 2) for arrhenius
        """
        embed = self.forward_embed(x)
        if self.approach == "arrhenius":
            return self.head(embed)
        else:
            return self.head(embed, t_features)


# ---------------------------------------------------------------------------
# B. NMR1DResNet (~190K parameters)
# ---------------------------------------------------------------------------

class ResBlock1d(nn.Module):
    """1D Residual block: Conv(k=3)->BN->ReLU->Conv(k=3)->BN + shortcut."""

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.shortcut = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")

    def forward(self, x):
        identity = self.shortcut(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = F.relu(out + identity)
        return out


class NMR1DResNet(nn.Module):
    """1D ResNet for NMR spectrum -> property prediction.

    Two size variants controlled by `width`:
      - 'small' (~18K params): designed for <1000 compound datasets
      - 'large' (~190K params): original architecture

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    approach : str
        'direct' or 'arrhenius'.
    t_feature_dim : int
        Number of temperature features (direct approach only).
    width : str
        'small' or 'large'. Controls model capacity.
    """

    def __init__(self, in_channels=1, approach="direct", t_feature_dim=3, width="small"):
        super().__init__()
        self.approach = approach

        if width == "small":
            drop = 0.5
            self.stem = nn.Sequential(
                nn.Conv1d(in_channels, 12, kernel_size=7, stride=2, padding=3, bias=False),
                nn.BatchNorm1d(12),
                nn.ReLU(),
                nn.MaxPool1d(2),
            )
            self.layer1 = nn.Sequential(
                ResBlock1d(12, 24, stride=2),
                ResBlock1d(24, 24, stride=1),
            )
            self.layer2 = nn.Sequential(
                ResBlock1d(24, 48, stride=2),
            )
            self.layer3 = ResBlock1d(48, 64, stride=2)
            self.embed_dim = 64
        else:
            drop = 0.3
            self.stem = nn.Sequential(
                nn.Conv1d(in_channels, 18, kernel_size=7, stride=2, padding=3, bias=False),
                nn.BatchNorm1d(18),
                nn.ReLU(),
                nn.MaxPool1d(2),
            )
            self.layer1 = nn.Sequential(
                ResBlock1d(18, 36, stride=2),
                ResBlock1d(36, 36, stride=1),
            )
            self.layer2 = nn.Sequential(
                ResBlock1d(36, 72, stride=2),
                ResBlock1d(72, 72, stride=1),
            )
            self.layer3 = ResBlock1d(72, 144, stride=2)
            self.embed_dim = 144

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(drop)

        if approach == "arrhenius":
            self.head = ArrheniusHead(self.embed_dim, dropout=drop)
        else:
            self.head = DirectHead(self.embed_dim, t_feature_dim, dropout=drop)

        # Init stem conv
        for m in self.stem.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")

    def forward_embed(self, x):
        h = self.stem(x)
        h = self.layer1(h)
        h = self.layer2(h)
        h = self.layer3(h)
        h = self.pool(h).squeeze(-1)
        h = self.dropout(h)
        return h

    def forward(self, x, t_features=None):
        embed = self.forward_embed(x)
        if self.approach == "arrhenius":
            return self.head(embed)
        else:
            return self.head(embed, t_features)


# ---------------------------------------------------------------------------
# C. NMRTransformer (~130K parameters)
# ---------------------------------------------------------------------------

class NMRTransformer(nn.Module):
    """Transformer encoder for peak-list NMR -> property prediction.

    Input: peak list as (N_peaks, 12) where 12 features are:
        [shift_center, shift_range, mult_onehot(8), J1_norm, J2_norm]

    Architecture:
        Embed: Linear(12, 64) -> ReLU -> Linear(64, 64)
        Positional encoding (learned, max 50 peaks)
        TransformerEncoder: 4 layers, 4 heads, d_model=64, d_ff=128, dropout=0.1
        Mean pool over non-padded peaks -> (64,)
        Dropout(0.3) -> (64,)

    Parameters
    ----------
    peak_dim : int
        Input feature dimension per peak (default 12).
    d_model : int
        Transformer model dimension (default 64).
    n_heads : int
        Number of attention heads (default 4).
    n_layers : int
        Number of transformer encoder layers (default 4).
    d_ff : int
        Feed-forward hidden dimension (default 128).
    max_peaks : int
        Maximum number of peaks (default 50).
    approach : str
        'direct' or 'arrhenius'.
    t_feature_dim : int
        Temperature feature dimension (direct only).
    """

    def __init__(self, peak_dim=12, d_model=64, n_heads=4, n_layers=4,
                 d_ff=128, max_peaks=50, approach="direct", t_feature_dim=3,
                 width="small"):
        super().__init__()
        self.approach = approach
        self.max_peaks = max_peaks

        if width == "small":
            d_model = 32
            n_heads = 4
            n_layers = 2
            d_ff = 64
            drop = 0.5
            enc_drop = 0.2
        else:
            drop = 0.3
            enc_drop = 0.1

        self.embed_dim = d_model

        # Peak embedding
        self.embed = nn.Sequential(
            nn.Linear(peak_dim, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )

        # Learned positional encoding
        self.pos_embed = nn.Parameter(torch.randn(1, max_peaks, d_model) * 0.02)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=enc_drop,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.dropout = nn.Dropout(drop)

        if approach == "arrhenius":
            self.head = ArrheniusHead(self.embed_dim, dropout=drop)
        else:
            self.head = DirectHead(self.embed_dim, t_feature_dim, dropout=drop)

        self._init_weights()

    def _init_weights(self):
        for m in self.embed.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward_embed(self, x, padding_mask=None):
        """Extract embedding from peak list input.

        Parameters
        ----------
        x : tensor, shape (batch, N_peaks, 12)
            Peak features, zero-padded to max length.
        padding_mask : tensor, shape (batch, N_peaks), bool
            True for padded positions (to be ignored).

        Returns
        -------
        embedding : tensor, shape (batch, 64)
        """
        batch_size, seq_len, _ = x.shape

        # Embed peaks
        h = self.embed(x)  # (batch, seq_len, d_model)

        # Add positional encoding (truncate or pad as needed)
        h = h + self.pos_embed[:, :seq_len, :]

        # Transformer encoder
        h = self.transformer(h, src_key_padding_mask=padding_mask)  # (batch, seq_len, d_model)

        # Mean pool over non-padded positions
        if padding_mask is not None:
            # Invert mask: True -> padded, so we want False positions
            valid_mask = ~padding_mask  # (batch, seq_len)
            valid_mask_expanded = valid_mask.unsqueeze(-1).float()  # (batch, seq_len, 1)
            h = (h * valid_mask_expanded).sum(dim=1) / valid_mask_expanded.sum(dim=1).clamp(min=1)
        else:
            h = h.mean(dim=1)

        h = self.dropout(h)
        return h

    def forward(self, x, t_features=None, padding_mask=None):
        """Forward pass.

        Parameters
        ----------
        x : tensor, shape (batch, N_peaks, 12)
        t_features : tensor or None
        padding_mask : tensor or None, shape (batch, N_peaks)

        Returns
        -------
        prediction : tensor
        """
        embed = self.forward_embed(x, padding_mask)
        if self.approach == "arrhenius":
            return self.head(embed)
        else:
            return self.head(embed, t_features)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def count_parameters(model):
    """Count total trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    print("=" * 60)
    print("  Phase 2 Model Architecture Tests")
    print("=" * 60)

    batch = 4
    spec_len = 1200
    x_spec = torch.randn(batch, 1, spec_len)
    t_feat = torch.randn(batch, 3)

    for width in ["small", "large"]:
        print(f"\n{'='*40} WIDTH={width} {'='*40}")

        cnn_d = NMR1DCNN(approach="direct", width=width)
        out = cnn_d(x_spec, t_feat)
        print(f"  CNN direct:     {count_parameters(cnn_d):>8,} params, embed={cnn_d.embed_dim}, out={out.shape}")

        resnet_d = NMR1DResNet(approach="direct", width=width)
        out = resnet_d(x_spec, t_feat)
        print(f"  ResNet direct:  {count_parameters(resnet_d):>8,} params, embed={resnet_d.embed_dim}, out={out.shape}")

        transformer_d = NMRTransformer(approach="direct", width=width)
        n_peaks = 15
        x_peaks = torch.randn(batch, n_peaks, 12)
        mask = torch.zeros(batch, n_peaks, dtype=torch.bool)
        mask[0, 10:] = True
        out = transformer_d(x_peaks, t_feat, padding_mask=mask)
        print(f"  Transformer direct: {count_parameters(transformer_d):>8,} params, embed={transformer_d.embed_dim}, out={out.shape}")
