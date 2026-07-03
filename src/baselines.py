"""
Non-spatial recurrent baselines (LSTM / GRU).

Each node's time series is encoded independently — no graph, no message passing.
Two heads:
  - forecast head: mu [B, N] point forecast (softplus, count domain), trained with MSE.
  - reconstruction head: [B, N, T, D] per-timestep reconstruction, trained with MSE.
    This makes the baseline's imputation and kriging attempts *genuine*: when the
    harness blanks a held-out node's inputs, the reconstruction there is honestly
    terrible — which is exactly the ablation evidence we want, backed by real
    numbers instead of fabricated distribution parameters.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class RecurrentBaseline(nn.Module):
    def __init__(
        self,
        time_series_dim: int,
        hidden_dim: int = 32,
        num_layers: int = 1,
        dropout: float = 0.1,
        rnn_type: str = "lstm",
    ):
        super().__init__()
        torch.manual_seed(42)

        assert rnn_type in ("lstm", "gru"), f"Unknown rnn_type: {rnn_type}"
        rnn_cls = nn.LSTM if rnn_type == "lstm" else nn.GRU
        self.rnn = rnn_cls(
            input_size=time_series_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.forecast_head = nn.Linear(hidden_dim, 1)
        self.recon_head = nn.Linear(hidden_dim, time_series_dim)
        self.time_series_dim = time_series_dim
        self.hidden_dim = hidden_dim

    def forward(self, dynamic_features: torch.Tensor):
        """
        Args:
            dynamic_features: [B, N, T, D] (NaNs already replaced with 0.0)
        Returns:
            mu:    [B, N] non-negative point forecast
            recon: [B, N, T, D] per-timestep reconstruction
        """
        assert dynamic_features.dim() == 4, \
            f"Expected [B, N, T, D], got {tuple(dynamic_features.shape)}"
        b, n, t, d = dynamic_features.shape
        assert d == self.time_series_dim, \
            f"time_series_dim mismatch: {d} != {self.time_series_dim}"

        x = dynamic_features.reshape(b * n, t, d)
        hidden, _ = self.rnn(x)                              # [B*N, T, H]

        recon = self.recon_head(hidden).reshape(b, n, t, d)  # [B, N, T, D]
        mu = F.softplus(self.forecast_head(hidden[:, -1, :])).squeeze(-1).reshape(b, n)
        return mu, recon


class LSTMForecaster(RecurrentBaseline):
    def __init__(self, time_series_dim: int, hidden_dim: int = 32,
                 num_layers: int = 1, dropout: float = 0.1):
        super().__init__(time_series_dim, hidden_dim, num_layers, dropout, rnn_type="lstm")


class GRUForecaster(RecurrentBaseline):
    def __init__(self, time_series_dim: int, hidden_dim: int = 32,
                 num_layers: int = 1, dropout: float = 0.1):
        super().__init__(time_series_dim, hidden_dim, num_layers, dropout, rnn_type="gru")
