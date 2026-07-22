import torch
import pytest
from decoders import DecoupledDecoders, compute_loss

def test_decoupled_decoders_and_loss():
    torch.manual_seed(42)

    batch_size = 4
    num_nodes = 50
    seq_len = 12
    hidden_dim = 64
    time_series_dim = 3

    model = DecoupledDecoders(hidden_dim=hidden_dim, time_series_dim=time_series_dim)

    latent_embeddings = torch.randn(batch_size, num_nodes, seq_len, hidden_dim)
    target_forecast = torch.randint(0, 100, (batch_size, num_nodes)).float()
    target_impute = torch.randn(batch_size, num_nodes, seq_len, time_series_dim)

    imputed_features, mu, theta, pi, _ = model(latent_embeddings)

    # PHASE 1/2: compute_loss requires scaler buffers and returns 4 values
    # (total, impute, forecast, kriging). With w_kriging=0 the kriging term is 0.
    loss, loss_impute, loss_forecast, loss_kriging = compute_loss(
        imputed_features=imputed_features,
        target_impute=target_impute,
        mu=mu,
        theta=theta,
        pi=pi,
        target_forecast=target_forecast,
        traffic_mean=torch.tensor(0.0),
        traffic_std=torch.tensor(1.0),
    )

    assert not torch.isnan(loss).any()
    # Kriging path is bypassed when w_kriging defaults to 0.0
    assert loss_kriging.item() == 0.0

    loss.backward()

    # impute_layer and forecast_layer are now nn.Sequential MLPs.
    # Verify gradient flow through the first Linear in each head.
    assert model.impute_layer[0].weight.grad is not None
    assert model.forecast_layer[0].weight.grad is not None
