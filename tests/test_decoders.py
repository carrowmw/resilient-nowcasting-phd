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

    imputed_features, mu, theta, pi = model(latent_embeddings)

    loss, loss_impute, loss_forecast = compute_loss(
        imputed_features=imputed_features,
        target_impute=target_impute,
        mu=mu,
        theta=theta,
        pi=pi,
        target_forecast=target_forecast
    )

    assert not torch.isnan(loss).any()

    loss.backward()

    # impute_layer and forecast_layer are now nn.Sequential MLPs.
    # Verify gradient flow through the first Linear in each head.
    assert model.impute_layer[0].weight.grad is not None
    assert model.forecast_layer[0].weight.grad is not None
