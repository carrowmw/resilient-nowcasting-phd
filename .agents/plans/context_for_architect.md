# Investigation Context: Fixing Zero-Inflation Representation Collapse

Based on the investigation into the ML pipeline's representation collapse (where the zero-inflation probability `pi` spikes for masked/simulated nodes), we have evaluated the downstream effects of the two proposed fixes.

## 1. Effects of Preserving NaNs in Targets
Removing `val_df.fillna(0.0)` in `src/dataset.py` will ripple through the data loading process to fix the target leakage problem.

- **Scaling Implications**: `StandardScaler` naturally computes mean/std ignoring NaNs, but it outputs NaNs. Therefore, `dataset.dynamic_data` will contain true NaNs instead of the scaled mean.
- **Safety Assertions**: The safeguard `assert not torch.isnan(self.dynamic_data).any()` (`dataset.py:190`) will instantly crash the pipeline and must be removed.
- **Encoder Inputs**: No architectural changes are needed to handle the input side. The existing `torch.nan_to_num(..., nan=0.0)` in `pipeline.py:147` will intercept these true NaNs and map them to `0.0` for the encoder safely.
- **Loss Masking (The Fix)**: `target_impute` and `target_forecast` in `train.py` will now correctly retain NaNs for structurally missing data. The loss logic (`valid_mask = ~torch.isnan(target_impute) & ~torch.isnan(target_forecast)`) will correctly evaluate to `False` for these points, entirely dropping them from MSE and ZINB NLL calculations. 
- **Outcome**: The model will no longer associate `structural_mask=1` with an artificial target of `0`. This will cure the zero-inflation spike for simulated nodes.

## 2. Effects of Explicit Indicator Channels
Passing the temporal mask (`node_mask`) and spatial holdout mask to the encoder will teach the model to distinguish between "artificially masked data" and "data that naturally equals the mean".

- **Mask Expansions**: In `pipeline.py`, the 2D masks from the batch dict (`node_mask` and `spatial_holdout_mask`) will need to be reshaped/broadcasted to `[Batch, Num_Nodes, Seq_Len, 1]` and concatenated to `encoder_input`.
- **Dimension Shifts**: `MaskedSpatiotemporalEncoder` (`encoder.py`) hardcodes `mask_dim=2`. This must be parameterized and updated to `mask_dim=4` in `pipeline.py`. Consequently, `GATv2Conv` `in_channels` will increase from 5 to 7.
- **Breaking Checkpoints**: Expanding the input dimension of the first layer is a breaking architectural change. **All previous checkpoints will become incompatible, requiring a full retraining run from scratch.**
- **Inference Exporter Updates**: The inference scripts (`export_paginated_payload.py` and `export_ui_payload.py`) do not generate `node_mask` or structure holdout masks in the same way `train.py` does. They will need to be updated to manually inject a zeroed temporal mask and an expanded spatial holdout mask into the inference batch so the model receives the expected 7 dimensions.
- **Outcome**: The encoder will have explicit context for why an input sequence is a flatline of `0.0`s. For spatial holdout nodes, the model will learn to route its predictions through neighbor graph embeddings rather than collapsing to the blind zero-inflated spatial prior.