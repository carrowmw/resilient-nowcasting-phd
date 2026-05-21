import torch
import pytest
from metrics import calculate_metrics


def _make_base_tensors(batch_size: int = 2, num_nodes: int = 4, seq_len: int = 3, dim: int = 2):
    """Helper: creates zero-filled tensors matching the expected shapes."""
    target_forecast = torch.zeros(batch_size, num_nodes)
    mu = torch.zeros(batch_size, num_nodes)
    pi = torch.zeros(batch_size, num_nodes)
    imputed_features = torch.zeros(batch_size, num_nodes, seq_len, dim)
    target_impute = torch.zeros(batch_size, num_nodes, seq_len, dim)
    node_mask = torch.zeros(batch_size, num_nodes, seq_len, dim, dtype=torch.bool)
    return target_forecast, mu, pi, imputed_features, target_impute, node_mask


class TestZeroCounts:
    def test_all_zeros_in_target__all_pi_high(self):
        """All targets are 0 and pi > 0.5 everywhere → full counts."""
        target_forecast, mu, pi, imputed, target_imp, mask = _make_base_tensors()
        # All targets are 0 (already), pi set above threshold
        pi = torch.full_like(pi, 0.9)

        metrics = calculate_metrics(target_forecast, mu, pi, imputed, target_imp, mask)

        expected_true = target_forecast.numel()  # 2 * 4 = 8
        assert metrics["zero_true_count"] == expected_true
        assert metrics["zero_correct_count"] == expected_true

    def test_all_zeros_in_target__all_pi_low(self):
        """All targets are 0 but pi ≤ 0.5 everywhere → zero correct count."""
        target_forecast, mu, pi, imputed, target_imp, mask = _make_base_tensors()
        pi = torch.full_like(pi, 0.3)

        metrics = calculate_metrics(target_forecast, mu, pi, imputed, target_imp, mask)

        assert metrics["zero_true_count"] == target_forecast.numel()
        assert metrics["zero_correct_count"] == 0

    def test_no_zeros_in_target__returns_zero_counts(self):
        """No target zeros → both counts must be 0 (no fallback 1.0 bias)."""
        target_forecast, mu, pi, imputed, target_imp, mask = _make_base_tensors()
        target_forecast = torch.ones_like(target_forecast) * 5  # all non-zero
        pi = torch.full_like(pi, 0.9)

        metrics = calculate_metrics(target_forecast, mu, pi, imputed, target_imp, mask)

        assert metrics["zero_true_count"] == 0
        assert metrics["zero_correct_count"] == 0

    def test_mixed_targets__counts_are_exact(self):
        """Only some positions are zero; verify exact integer counts."""
        # Shape [1, 4]: target = [0, 5, 0, 10], pi = [0.9, 0.9, 0.3, 0.9]
        target_forecast = torch.tensor([[0.0, 5.0, 0.0, 10.0]])
        mu = torch.zeros(1, 4)
        pi = torch.tensor([[0.9, 0.9, 0.3, 0.9]])
        imputed = torch.zeros(1, 4, 3, 2)
        target_imp = torch.zeros(1, 4, 3, 2)
        mask = torch.zeros(1, 4, 3, 2, dtype=torch.bool)

        metrics = calculate_metrics(target_forecast, mu, pi, imputed, target_imp, mask)

        # Zeros at positions 0 and 2; pi[0]=0.9>0.5 ✓, pi[2]=0.3≤0.5 ✗
        assert metrics["zero_true_count"] == 2
        assert metrics["zero_correct_count"] == 1

    def test_return_types_are_int(self):
        """zero_true_count and zero_correct_count must be Python ints."""
        target_forecast, mu, pi, imputed, target_imp, mask = _make_base_tensors()
        pi = torch.full_like(pi, 0.9)

        metrics = calculate_metrics(target_forecast, mu, pi, imputed, target_imp, mask)

        assert isinstance(metrics["zero_true_count"], int)
        assert isinstance(metrics["zero_correct_count"], int)

    def test_zero_acc_key_absent(self):
        """Legacy 'zero_acc' key must no longer be present in the output."""
        target_forecast, mu, pi, imputed, target_imp, mask = _make_base_tensors()

        metrics = calculate_metrics(target_forecast, mu, pi, imputed, target_imp, mask)

        assert "zero_acc" not in metrics


class TestOtherMetrics:
    def test_rmse_and_mae_non_negative(self):
        target_forecast = torch.tensor([[3.0, 5.0]])
        mu = torch.tensor([[1.0, 7.0]])
        pi = torch.zeros(1, 2)
        imputed = torch.zeros(1, 2, 3, 2)
        target_imp = torch.zeros(1, 2, 3, 2)
        mask = torch.zeros(1, 2, 3, 2, dtype=torch.bool)

        metrics = calculate_metrics(target_forecast, mu, pi, imputed, target_imp, mask)

        assert metrics["rmse"] >= 0.0
        assert metrics["mae"] >= 0.0

    def test_impute_mse_with_active_mask(self):
        """Imputation MSE should be non-zero when mask has True entries."""
        target_forecast = torch.zeros(1, 2)
        mu = torch.zeros(1, 2)
        pi = torch.zeros(1, 2)
        imputed = torch.ones(1, 2, 3, 2)   # predicted = 1
        target_imp = torch.zeros(1, 2, 3, 2)  # truth = 0  → MSE = 1
        mask = torch.ones(1, 2, 3, 2, dtype=torch.bool)

        metrics = calculate_metrics(target_forecast, mu, pi, imputed, target_imp, mask)

        assert pytest.approx(metrics["impute_mse"], abs=1e-5) == 1.0

    def test_impute_mse_zero_without_mask(self):
        """With no masked positions, impute_mse should be 0.0."""
        target_forecast, mu, pi, imputed, target_imp, mask = _make_base_tensors()
        # mask is all-False by default
        metrics = calculate_metrics(target_forecast, mu, pi, imputed, target_imp, mask)
        assert metrics["impute_mse"] == 0.0

    def test_shannon_entropy_clamp(self):
        """Test entropy calculation with epsilon clamp for log(0) safety."""
        # Represents per_edge_attn where one attention is exactly 0.0
        p = torch.tensor([1.0, 0.0])
        # H = -sum p log(p + eps)
        eps = 1e-6
        entropy = -float(torch.sum(p * torch.log(p + eps)).item())
        # log(1.0 + 1e-6) is approx 1e-6, so 1.0 * 1e-6 ~ 1e-6
        # 0.0 * log(1e-6) = 0.0
        # - (1e-6 + 0) = -1e-6 ~ 0.0
        assert entropy > -1e-4 and entropy < 1e-4
