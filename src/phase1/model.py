import torch
import torch.nn as nn


class SharedEncoder(nn.Module):
    def __init__(self, layer_dims: list[int], dropout: float = 0.3, use_batch_norm: bool = True):
        super().__init__()
        layers = []
        for i in range(len(layer_dims) - 1):
            layers.append(nn.Linear(layer_dims[i], layer_dims[i + 1]))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(layer_dims[i + 1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)


class PropertyHead(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: int = 64, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)


class ArrheniusMultiTaskModel(nn.Module):
    """Predicts Arrhenius [A, B] coefficients for viscosity and [A, B] for surface tension."""

    def __init__(self, fp_dim: int = 2048, encoder_layers: list[int] = None,
                 dropout: float = 0.3):
        super().__init__()
        if encoder_layers is None:
            encoder_layers = [fp_dim, 512, 256, 128]
        else:
            encoder_layers = [fp_dim] + encoder_layers

        self.encoder = SharedEncoder(encoder_layers, dropout=dropout)
        embed_dim = encoder_layers[-1]
        self.visc_head = PropertyHead(embed_dim, 2, hidden=64, dropout=0.2)
        self.st_head = PropertyHead(embed_dim, 2, hidden=64, dropout=0.2)

    def forward(self, fp):
        z = self.encoder(fp)
        return self.visc_head(z), self.st_head(z)


class DirectMultiTaskModel(nn.Module):
    """Predicts property values directly, with temperature as input."""

    def __init__(self, fp_dim: int = 2048, t_feature_dim: int = 3,
                 encoder_layers: list[int] = None, dropout: float = 0.3):
        super().__init__()
        if encoder_layers is None:
            encoder_layers = [fp_dim, 512, 256, 128]
        else:
            encoder_layers = [fp_dim] + encoder_layers

        self.encoder = SharedEncoder(encoder_layers, dropout=dropout)
        embed_dim = encoder_layers[-1]
        head_in = embed_dim + t_feature_dim
        self.visc_head = PropertyHead(head_in, 1, hidden=64, dropout=0.2)
        self.st_head = PropertyHead(head_in, 1, hidden=64, dropout=0.2)

    def forward(self, fp, t_features):
        z = self.encoder(fp)
        z_t = torch.cat([z, t_features], dim=1)
        return self.visc_head(z_t), self.st_head(z_t)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    arr_model = ArrheniusMultiTaskModel()
    direct_model = DirectMultiTaskModel()

    print(f"Arrhenius model: {count_parameters(arr_model):,} parameters")
    print(f"Direct model:    {count_parameters(direct_model):,} parameters")

    fp = torch.randn(4, 2048)
    t = torch.randn(4, 3)

    v_arr, s_arr = arr_model(fp)
    print(f"Arrhenius output: visc {v_arr.shape}, ST {s_arr.shape}")

    v_dir, s_dir = direct_model(fp, t)
    print(f"Direct output: visc {v_dir.shape}, ST {s_dir.shape}")
