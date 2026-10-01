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
        input_dim: int = None,
        head: str = "mse",
    ):
        """v5: input_dim > time_series_dim when the baseline also reads masks,
        covariates and static features (model.baseline_inputs); head = nb adds a
        dispersion output so the forecast is trained by NB NLL (model.baseline_head).
        Defaults reproduce the v4 module."""
        super().__init__()
        assert rnn_type in ("lstm", "gru"), f"Unknown rnn_type: {rnn_type}"
        assert head in ("mse", "nb"), f"Unknown baseline head: {head}"
        input_dim = time_series_dim if input_dim is None else input_dim
        rnn_cls = nn.LSTM if rnn_type == "lstm" else nn.GRU
        self.rnn = rnn_cls(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.forecast_head = nn.Linear(hidden_dim, 1 if head == "mse" else 2)
        self.recon_head = nn.Linear(hidden_dim, time_series_dim)
        self.time_series_dim = time_series_dim
        self.input_dim = input_dim
        self.head = head
        self.hidden_dim = hidden_dim

    def forward(self, dynamic_features: torch.Tensor):
        mu, _, recon = self.forward_heads(dynamic_features)
        return mu, recon

    def forward_heads(self, dynamic_features: torch.Tensor):
        """
        Args:
            dynamic_features: [B, N, T, input_dim] (NaNs already replaced with 0.0)
        Returns:
            mu:    [B, N] non-negative point forecast
            theta: [B, N] NB dispersion (head = nb) or None
            recon: [B, N, T, D] per-timestep reconstruction
        """
        assert dynamic_features.dim() == 4, \
            f"Expected [B, N, T, D], got {tuple(dynamic_features.shape)}"
        b, n, t, d = dynamic_features.shape
        assert d == self.input_dim, \
            f"input_dim mismatch: {d} != {self.input_dim}"

        x = dynamic_features.reshape(b * n, t, d)
        hidden, _ = self.rnn(x)                              # [B*N, T, H]

        recon = self.recon_head(hidden).reshape(b, n, t, self.time_series_dim)  # [B, N, T, D]
        out = self.forecast_head(hidden[:, -1, :]).reshape(b, n, -1)
        if self.head == "mse":
            return F.softplus(out[..., 0]), None, recon
        return F.softplus(out[..., 0].float()) + 1e-6, F.softplus(out[..., 1].float()) + 1e-6, recon


class LSTMForecaster(RecurrentBaseline):
    def __init__(self, time_series_dim: int, hidden_dim: int = 32,
                 num_layers: int = 1, dropout: float = 0.1, input_dim: int = None, head: str = "mse"):
        super().__init__(time_series_dim, hidden_dim, num_layers, dropout, rnn_type="lstm",
                         input_dim=input_dim, head=head)


class GRUForecaster(RecurrentBaseline):
    def __init__(self, time_series_dim: int, hidden_dim: int = 32,
                 num_layers: int = 1, dropout: float = 0.1, input_dim: int = None, head: str = "mse"):
        super().__init__(time_series_dim, hidden_dim, num_layers, dropout, rnn_type="gru",
                         input_dim=input_dim, head=head)


class NaiveBaseline(nn.Module):
    """Non-learned floor (zero-collapse report E0.3 / Phase 7 baseline_profile_median).

    kind = profile:  forecast = train-slice seasonal slot-of-week median of the node
                     (the inductive substitute for blanked nodes — never their own data).
    kind = last_obs: last value the model could see in the input window; the profile
                     when the whole window is hidden or the node is blanked.
    Holds no parameters; NaiveAdapter does the lookups through the CovariateBuilder.
    """

    def __init__(self, kind: str):
        super().__init__()
        assert kind in ("profile", "last_obs"), f"Unknown naive baseline kind: {kind}"
        self.kind = kind
        # A registered buffer keeps state_dict() non-empty and device moves meaningful.
        self.register_buffer("is_naive", torch.ones(()))
