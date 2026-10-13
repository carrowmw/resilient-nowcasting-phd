"""
Shared temporal encoders: [B*, T, D] → [B*, F].

The SAME classes serve as (a) the per-node temporal baselines' backbone and
(b) the GAT models' temporal stage. This is deliberate: it guarantees that
"GAT + temporal encoder vs per-node temporal baseline" differs only in the
spatial message-passing block, not in how time is encoded.
"""
import torch
import torch.nn as nn

TEMPORAL_ENCODERS = ("none", "lstm", "transformer")


def build_temporal_encoder(kind: str, input_dim: int, window: int,
                           hidden_dim: int, spec: dict) -> nn.Module:
    assert kind in TEMPORAL_ENCODERS, f"CRITICAL: unknown temporal_encoder {kind!r}"
    if kind == "none":
        return FlattenMLPEncoder(input_dim, window, hidden_dim,
                                 dropout=float(spec.get("dropout", 0.1)))
    if kind == "lstm":
        return LSTMEncoder(input_dim, hidden_dim,
                           num_layers=int(spec["num_layers"]),
                           dropout=float(spec.get("dropout", 0.1)))
    return TransformerEncoder(
        input_dim, hidden_dim, window,
        nhead=int(spec["nhead"]),
        num_encoder_layers=int(spec["num_encoder_layers"]),
        dim_feedforward=int(spec["dim_feedforward"]),
        dropout=float(spec.get("dropout", 0.1)),
    )


class FlattenMLPEncoder(nn.Module):
    """Window flattened through a 2-layer MLP — the 'no temporal model' stage."""

    def __init__(self, input_dim: int, window: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim * window, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [B*, T, D] → [B*, F]
        return self.net(x.flatten(1))


class LSTMEncoder(nn.Module):
    """Unidirectional LSTM; last hidden state. (The window is short — 12 steps
    at the reference config — so bidirectionality buys little and would break
    parameter parity with the causal reading of the window.)"""

    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, dropout: float):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers,
                            batch_first=True,
                            dropout=dropout if num_layers > 1 else 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (h, _) = self.lstm(x)
        return h[-1]                                      # [B*, F]


class TransformerEncoder(nn.Module):
    """Linear projection + learned positional embedding + encoder + mean pool."""

    def __init__(self, input_dim: int, hidden_dim: int, window: int, nhead: int,
                 num_encoder_layers: int, dim_feedforward: int, dropout: float):
        super().__init__()
        self.project = nn.Linear(input_dim, hidden_dim)
        self.pos = nn.Embedding(window, hidden_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_encoder_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t = x.shape[1]
        pos = self.pos(torch.arange(t, device=x.device))[None]
        return self.encoder(self.project(x) + pos).mean(dim=1)   # [B*, F]
