# Implementation Plan - DecoupledDecoders and Loss

## Objective
Implement `DecoupledDecoders` and `compute_loss` in `src/decoders.py` to handle joint imputation and uncertainty-aware forecasting using a ZINB distribution.

## Proposed Changes

### 1. Module Implementation (`src/decoders.py`)
- **DecoupledDecoders Class**:
  - `__init__`:
    - `impute_layer`: `nn.Linear(hidden_dim, time_series_dim)`
    - `forecast_layer`: `nn.Linear(hidden_dim, 3)` (for $\mu, \theta, \pi$)
  - `forward(latent_embeddings)`:
    - Input: `[batch_size, num_nodes, seq_len, hidden_dim]`
    - **Imputation**: Apply `impute_layer` to all timesteps.
    - **Forecasting**: Extract last timestep `[:, :, -1, :]`. Apply `forecast_layer`.
    - **ZINB Constraints**:
      - $\pi = \text{sigmoid}(\text{logits}_{\pi})$
      - $\mu = \text{softplus}(\text{logits}_{\mu}) + 1e-6$
      - $\theta = \text{softplus}(\text{logits}_{\theta}) + 1e-6$

### 2. Loss Implementation
- **compute_zinb_nll(y_true, mu, theta, pi)**:
  - Implement the Negative Log-Likelihood for ZINB.
  - Use `torch.lgamma` for factorial terms.
  - Clamp `torch.log` to `1e-6`.
  - Handle $y=0$ and $y>0$ cases for numerical stability.
- **compute_loss(...)**:
  - Imputation Loss: `MSE(imputed, target_impute)` (Note: ARCHITECTURE.md says "strictly on artificially masked nodes", but for the module level, we'll implement the basic MSE and can add masking logic if the mask is provided).
  - Forecasting Loss: `compute_zinb_nll`.
  - Total Loss: $w_{forecast} \cdot \mathcal{L}_{forecast} + w_{impute} \cdot \mathcal{L}_{impute}$.

### 3. Offensive Unit Test
- Instantiate `DecoupledDecoders`.
- Mock inputs and targets.
- Verify:
  - Output shapes.
  - NaN tripwire on loss.
  - Gradient flow to both decoder heads.

## Verification Plan
- Run `python src/decoders.py` and verify all assertions pass.
