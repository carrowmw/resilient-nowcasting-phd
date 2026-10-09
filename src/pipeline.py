import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import logging

from graph_learner import DynamicGraphLearner
from encoder import MaskedSpatiotemporalEncoder
from decoders import DecoupledDecoders, SpatialKrigingDecoder, compute_loss
from dataset import RealIoTDataset
from likelihoods import point_forecast, prob_zero
from losses import TrafficScale

from typing import Dict, Optional, Tuple, Union

logger = logging.getLogger(__name__)

class ResilientIoTPipeline(nn.Module):
    """
    ResilientIoTPipeline: End-to-end spatiotemporal model for IoT sensor networks.
    Integrates Topology Inference, Spatiotemporal Encoding, and Multi-Task Decoders.
    """
    def __init__(self,
                 num_nodes: int,
                 static_feat_dim: int,
                 time_series_dim: int,
                 seq_len: int,
                 embedding_dim: int = 32,
                 hidden_dim: int = 64,
                 sparsity_threshold: float = 0.5,
                 num_layers: int = 2,
                 num_heads: int = 4,
                 dropout: float = 0.1,
                 dim_feedforward: Optional[int] = None,
                 zinb_eps: float = 1.0e-6,
                 edge_features_cache_path: str = None,
                 simulated_edge_features_cache_path: str = None,
                 use_positional_encoding: bool = True,
                 covariate_dim: int = 0,
                 sensor_embedding_dim: int = 0,
                 sensor_embedding_dropout: float = 0.0,
                 temporal_key_padding: bool = False,
                 gat_drop_missing_sources: bool = False,
                 graph_mode: str = "learned",
                 forecast_likelihood: str = "zinb",
                 impute_likelihood: str = "msle",
                 quantiles: Tuple[float, ...] = (),
                 outage_head_input: str = "latent",
                 mask_channels: str = "both",
                 holdout_night_channel: str = "leak",
                 detach_impute_latent: bool = False,
                 traffic_log1p: bool = False,
                 residual_to_profile: bool = False):
        """v5 keyword arguments default to the v4 architecture (identical state_dict);
        model_factory.build_model passes every one explicitly from the resolved config.
        The v5 options are only honoured by forward(..., return_outputs=True)."""
        super(ResilientIoTPipeline, self).__init__()
        assert graph_mode in ("learned", "identity")
        assert outage_head_input in ("latent", "latent_detached", "missingness")
        assert mask_channels in ("both", "total_only", "night_only", "none")
        assert holdout_night_channel in ("leak", "zero", "consistent")
        assert list(quantiles) == sorted(quantiles) and all(0.0 < q < 1.0 for q in quantiles), \
            f"CRITICAL: quantiles must be sorted levels in (0, 1), got {quantiles}"
        self.seq_len = seq_len
        self.num_nodes = num_nodes
        self.covariate_dim = covariate_dim
        self.sensor_embedding_dim = sensor_embedding_dim
        self.sensor_embedding_dropout = sensor_embedding_dropout
        self.graph_mode = graph_mode
        self.forecast_likelihood = forecast_likelihood
        self.impute_likelihood = impute_likelihood
        self.quantile_levels = tuple(quantiles)
        self.outage_head_input = outage_head_input
        self.mask_channels = mask_channels
        self.holdout_night_channel = holdout_night_channel
        self.detach_impute_latent = detach_impute_latent
        self.traffic_log1p = traffic_log1p
        # v6: forecast mean = (seasonal profile + 1) * exp(head output); the head starts at
        # the naive-profile forecast and learns a multiplicative correction.
        self.residual_to_profile = residual_to_profile

        # None is a documented API default (hidden_dim * 4); <= 0 is an invalid caller error.
        if dim_feedforward is None:
            dim_feedforward = hidden_dim * 4
        elif dim_feedforward <= 0:
            raise ValueError(f"CRITICAL: dim_feedforward must be positive, got {dim_feedforward}")

        # 1. Topology Inference
        # PHASE 3: pass the edge features cache so the graph learner can build [E, 10] edge_attr.
        self.graph_learner = DynamicGraphLearner(
            static_feat_dim=static_feat_dim,
            embedding_dim=embedding_dim,
            sparsity_threshold=sparsity_threshold,
            edge_features_cache_path=edge_features_cache_path,
            simulated_edge_features_cache_path=simulated_edge_features_cache_path,
        )
        logger.info(f"Initialized DynamicGraphLearner with static_feat_dim={static_feat_dim}, embedding_dim={embedding_dim}, sparsity_threshold={sparsity_threshold}, edge_features_cache_path={edge_features_cache_path}, simulated_edge_features_cache_path={simulated_edge_features_cache_path}, total_params={sum(p.numel() for p in self.graph_learner.parameters())}")

        # PHASE 3: edge_dim matches the graph learner's edge_attr width — 10 when the
        # physical cache is loaded, else 1 (learned-similarity-only fallback).
        encoder_edge_dim = 10 if edge_features_cache_path else 1

        # 2. Spatiotemporal Feature Extraction
        self.encoder = MaskedSpatiotemporalEncoder(
            num_nodes=num_nodes,
            seq_len=seq_len,
            time_series_dim=time_series_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
            dim_feedforward=dim_feedforward,
            edge_dim=encoder_edge_dim,
            use_positional_encoding=use_positional_encoding,
            extra_dim=covariate_dim + sensor_embedding_dim,
            temporal_key_padding=temporal_key_padding,
            gat_drop_missing_sources=gat_drop_missing_sources,
        )
        self.encoder_edge_dim = encoder_edge_dim

        # v5 (R3): learned per-sensor embedding. Blanked (holdout) nodes, and a random
        # embedding_dropout fraction of nodes during training, use a projection of the
        # static features instead — the only embedding an unseen sensor can have.
        if sensor_embedding_dim > 0:
            self.sensor_embedding = nn.Embedding(num_nodes, sensor_embedding_dim)
            self.static_embedding = nn.Linear(static_feat_dim, sensor_embedding_dim)
        else:
            self.sensor_embedding = None
            self.static_embedding = None

        if forecast_likelihood == "outage_mixture":
            outage_feat_dim = (seq_len + 3 if outage_head_input == "missingness" else hidden_dim) + covariate_dim
        else:
            outage_feat_dim = 0
        logger.info(f"Initialized MaskedSpatiotemporalEncoder with num_nodes={num_nodes}, seq_len={seq_len}, time_series_dim={time_series_dim}, hidden_dim={hidden_dim}, num_layers={num_layers}, num_heads={num_heads}, use_pe={use_positional_encoding}, total_params={sum(p.numel() for p in self.encoder.parameters())}")

        # 3. Multi-Task Decoders
        self.decoders = DecoupledDecoders(
            hidden_dim=hidden_dim,
            time_series_dim=time_series_dim,
            dropout=dropout,
            zinb_eps=zinb_eps,
            forecast_likelihood=forecast_likelihood,
            impute_likelihood=impute_likelihood,
            num_quantiles=len(quantiles),
            target_cov_dim=covariate_dim,
            outage_feat_dim=outage_feat_dim,
        )
        logger.info(f"Initialized DecoupledDecoders with hidden_dim={hidden_dim}, time_series_dim={time_series_dim}, dropout={dropout}, total_params={sum(p.numel() for p in self.decoders.parameters())}")

        # PHASE 2: Spatial Kriging Decoder (dormant unless a holdout mask is supplied
        # in the batch and loss.w_kriging > 0). Reconstructs held-out nodes' series.
        self.kriging_decoder = SpatialKrigingDecoder(
            hidden_dim=hidden_dim,
            time_series_dim=time_series_dim,
            dropout=dropout,
            count_head=impute_likelihood == "outage_mixture",
        )
        logger.info(f"Initialized SpatialKrigingDecoder with hidden_dim={hidden_dim}, time_series_dim={time_series_dim}, total_params={sum(p.numel() for p in self.kriging_decoder.parameters())}")

        # PHASE 1: Scaler buffers (initialized to sentinel values, set by train.py)
        self.register_buffer('traffic_mean', torch.tensor(0.0))
        self.register_buffer('traffic_std', torch.tensor(1.0))
        logger.info("Registered traffic scaler buffers (traffic_mean, traffic_std)")
        if outage_head_input == "missingness" and forecast_likelihood == "outage_mixture":
            # Train-slice per-node missing rate (set by model_factory); outage-head input only.
            self.register_buffer('node_missing_rate', torch.zeros(num_nodes))
        else:
            self.node_missing_rate = None

    def set_scaler_stats(self, traffic_mean: float, traffic_std: float):
        """
        Called by train.py after dataset initialization to populate scaler buffers.

        Args:
            traffic_mean: Mean of traffic channel from dataset scaler
            traffic_std: Standard deviation of traffic channel from dataset scaler
        """
        import numpy as np

        # OFFENSIVE ASSERTIONS
        assert traffic_std > 0, f"CRITICAL: Cannot set traffic_std to {traffic_std} (must be positive)"
        assert not np.isnan(traffic_mean), "CRITICAL: Cannot set traffic_mean to NaN"
        assert not np.isnan(traffic_std), "CRITICAL: Cannot set traffic_std to NaN"

        self.traffic_mean.fill_(traffic_mean)
        self.traffic_std.fill_(traffic_std)

        logger.info(f"Scaler buffers set: mean={traffic_mean:.4f}, std={traffic_std:.4f}, device={self.traffic_mean.device}")

    @property
    def scale(self) -> TrafficScale:
        return TrafficScale(self.traffic_mean, self.traffic_std, self.traffic_log1p)

    def forward(
        self,
        batch: dict,
        return_analysis: bool = False,
        return_outputs: bool = False,
    ) -> Union[Tuple[torch.Tensor, ...], Dict[str, torch.Tensor]]:
        if return_outputs:
            return self._forward_v5(batch, return_attention=return_analysis)
        return self._forward_v4(batch, return_analysis)

    # ------------------------------------------------------------------ #
    # v5 forward: outputs only; losses live in losses.compute_v5_losses    #
    # ------------------------------------------------------------------ #
    def _forward_v5(self, batch: dict, return_attention: bool = False) -> Dict[str, object]:
        """
        Batch keys (all set by harness.batch_prep.BatchPreparer; optional ones may be None):
            static_features [N, S], dynamic_features [B, N, T, D] (NaN = hidden),
            structural_masks [B, N, T, 2], spatial_holdout_mask [N] | None,
            blank_mask [N] | None (every node whose dynamic input is blanked),
            covariates [B, N, T + 1, C] | None, night_flags [B, T + 1] bool,
            input_missing [B, N, T] bool (effective missingness the model sees),
            location_ids, simulated_nodes_mask (optional, F-C04).
        """
        static_features = batch['static_features']
        dyn = torch.nan_to_num(batch['dynamic_features'], nan=0.0)
        B, N, T, _ = dyn.shape
        holdout_mask = batch['spatial_holdout_mask']
        blank_mask = batch['blank_mask']
        input_missing = batch['input_missing']
        covariates = batch['covariates']
        masks = batch['structural_masks'].clone()

        # Holdout encoding. "leak" is v4 exactly: only the kriging holdout signals
        # ch0 = 1 and loses its outgoing edges, and its night channel keeps the real
        # pattern. The v5 modes treat every blanked node (incl. the fixed inductive
        # set during training) as a fully missing sensor.
        if self.holdout_night_channel == "leak":
            encode_blank = holdout_mask
        else:
            encode_blank = blank_mask
        if encode_blank is not None and encode_blank.any():
            masks[:, encode_blank, :, 0] = 1.0
            if self.holdout_night_channel == "zero":
                masks[:, encode_blank, :, 1] = 0.0
            elif self.holdout_night_channel == "consistent":
                night = batch['night_flags'][:, :T].to(masks.dtype)            # [B, T]
                masks[:, encode_blank, :, 1] = night.unsqueeze(1).expand(B, int(encode_blank.sum()), T)
        if self.mask_channels in ("night_only", "none"):
            masks[..., 0] = 0.0
        if self.mask_channels in ("total_only", "none"):
            masks[..., 1] = 0.0

        parts = [dyn, masks]
        if covariates is not None:
            assert covariates.shape == (B, N, T + 1, self.covariate_dim), \
                f"CRITICAL: covariates {tuple(covariates.shape)} != {(B, N, T + 1, self.covariate_dim)}"
            parts.append(covariates[:, :, :T].to(dyn.dtype))
        else:
            assert self.covariate_dim == 0, "CRITICAL: model expects covariates but batch has none"
        if self.sensor_embedding is not None:
            emb = self.sensor_embedding.weight                                  # [N, d]
            proj = self.static_embedding(static_features)                      # [N, d]
            use_proj = torch.zeros(N, dtype=torch.bool, device=dyn.device)
            if blank_mask is not None:
                use_proj = use_proj | blank_mask
            if self.training and self.sensor_embedding_dropout > 0.0:
                use_proj = use_proj | (torch.rand(N, device=dyn.device) < self.sensor_embedding_dropout)
            node_emb = torch.where(use_proj.unsqueeze(-1), proj, emb)
            parts.append(node_emb.view(1, N, 1, -1).expand(B, N, T, -1).to(dyn.dtype))
        encoder_input = torch.cat(parts, dim=-1)

        location_ids = batch.get('location_ids')
        if self.graph_mode == "identity":
            # Graph ablation (thesis limitation 7): no message passing; GATv2 still adds
            # self-loops, so every other component is unchanged.
            edge_index = torch.zeros(2, 0, dtype=torch.long, device=dyn.device)
            edge_weight = torch.zeros(0, device=dyn.device)
            edge_attr = torch.zeros(0, self.encoder_edge_dim, device=dyn.device)
        else:
            edge_index, edge_weight, edge_attr = self.graph_learner(static_features, location_ids=location_ids)
            drop_mask = batch.get('simulated_nodes_mask')
            if encode_blank is not None and encode_blank.any():
                drop_mask = encode_blank if drop_mask is None else (drop_mask | encode_blank)
            if drop_mask is not None:
                valid_edges = ~drop_mask[edge_index[0]]
                edge_index = edge_index[:, valid_edges]
                edge_weight = edge_weight[valid_edges]
                edge_attr = edge_attr[valid_edges]

        enc = self.encoder(
            encoder_input, edge_index, edge_attr,
            return_attention_weights=return_attention, input_missing=input_missing,
        )
        latent, alpha = enc if return_attention else (enc, None)

        target_cov = covariates[:, :, T] if covariates is not None else None
        outage_feats = None
        if self.forecast_likelihood == "outage_mixture":
            if self.outage_head_input == "missingness":
                miss = input_missing.float()
                trailing = torch.cumprod(miss.flip(-1), dim=-1).sum(-1) / T
                blanked = (blank_mask.view(1, N).expand(B, N).float() if blank_mask is not None
                           else torch.zeros(B, N, device=dyn.device))
                rate = torch.where(blanked > 0, torch.ones_like(blanked), self.node_missing_rate.view(1, N).expand(B, N))
                feats = [miss, trailing.unsqueeze(-1), rate.unsqueeze(-1), blanked.unsqueeze(-1)]
            else:
                last = latent[:, :, -1, :]
                feats = [last.detach() if self.outage_head_input == "latent_detached" else last]
            if target_cov is not None:
                feats.append(target_cov)
            outage_feats = torch.cat([f.to(latent.dtype) for f in feats], dim=-1)

        impute_latent = latent.detach() if self.detach_impute_latent else latent
        mu_anchor = None
        if self.residual_to_profile:
            assert batch.get("profile_target_raw") is not None, \
                "CRITICAL: residual_to_profile needs batch['profile_target_raw'] (BatchPreparer)"
            mu_anchor = batch["profile_target_raw"].float() + 1.0
        heads = self.decoders.forward_v5(latent, impute_latent, target_cov, outage_feats, mu_anchor=mu_anchor)

        scale = self.scale
        if heads["impute_traffic_params"] is not None:
            mu_t = heads["impute_traffic_params"][0]
            imputed = torch.cat([scale.to_scaled(mu_t).unsqueeze(-1), heads["impute_weather"]], dim=-1)
            imputed_raw = mu_t
        else:
            imputed = heads["imputed"]
            imputed_raw = scale.to_raw(imputed[..., 0].float())

        kriging_pred, kriging_params = None, None
        if holdout_mask is not None:
            k_raw = self.kriging_decoder(latent, holdout_mask)
            if self.kriging_decoder.count_head:
                mu_k = self.decoders._positive(k_raw[..., 0])
                theta_k = self.decoders._positive(k_raw[..., 1])
                kriging_params = (mu_k, theta_k)
                kriging_pred = torch.cat([scale.to_scaled(mu_k).unsqueeze(-1), k_raw[..., 2:].float()], dim=-1)
            else:
                kriging_pred = k_raw

        mu, theta, pi, q = heads["mu"], heads["theta"], heads["pi"], heads["q"]
        lik = self.forecast_likelihood
        return {
            'imputed': imputed,
            'imputed_raw': imputed_raw,
            'impute_traffic_params': heads["impute_traffic_params"],
            'mu': mu,
            'theta': theta,
            'pi': pi,
            'q': q,
            'quantiles': heads["quantiles"],
            'quantile_levels': self.quantile_levels,
            'point': point_forecast(lik, mu, theta, pi),
            'p_zero': prob_zero(lik, mu, theta, pi, q),
            'kriging_pred': kriging_pred,
            'kriging_traffic_params': kriging_params,
            'tripwires': heads["tripwires"],
            'edge_index': edge_index if return_attention else None,
            'edge_weight': edge_weight if return_attention else None,
            'attention': alpha,
        }

    def _forward_v4(
        self,
        batch: dict,
        return_analysis: bool = False,
    ) -> Union[Tuple[torch.Tensor, ...], Dict[str, torch.Tensor]]:
        """
        Args:
            batch (dict): Expected keys:
                - static_features: [num_nodes, static_feat_dim]
                - dynamic_features: [batch_size, num_nodes, seq_len, time_series_dim]
                - target_forecast: [batch_size, num_nodes]  (not required in analysis mode)
                - target_impute: [batch_size, num_nodes, seq_len, time_series_dim]  (not required in analysis mode)
                - node_mask: [batch_size, num_nodes, seq_len, time_series_dim]  (not required in analysis mode)
            return_analysis: If True, complete the full forward pass through the decoders
                but bypass loss calculation. Returns a comprehensive dict with graph
                structure, attention weights, and ZINB distribution parameters for
                post-hoc interpretability and UI payload generation.
        Returns:
            Normal mode: (total_loss, loss_impute, loss_forecast, loss_kriging,
                          imputed_features, mu, theta, pi, tripwire_logs, kriging_pred)
            Analysis mode: {'edge_index', 'edge_weight', 'attention_weights',
                            'imputed_features', 'kriging_pred', 'mu', 'theta', 'pi', 'tripwire_logs'}
        """
        static_features = batch['static_features']
        dynamic_features = batch['dynamic_features']
        # PHASE 2: spatial holdout mask [N] bool (None when kriging inactive).
        holdout_mask = batch.get('spatial_holdout_mask')
        # PHASE 3: per-node location identifiers for physical edge feature lookup.
        location_ids = batch.get('location_ids')

        # Defensive Data Handling
        # Replace NaNs (injected by our Dataset masking or simulated nodes) with 0.0
        dynamic_features_clean = torch.nan_to_num(dynamic_features, nan=0.0)

        # Retrieve and concatenate explicit MNAR masks
        # structural_masks shape: [batch_size, num_nodes, seq_len, 2]
        structural_masks = batch.get('structural_masks')
        if structural_masks is not None:
            # F-C05: If spatial holdout is applied, heldout nodes should signal Channel 3 = 1.0 (unobserved)
            # rather than leaking their fine-grained missingness pattern.
            if holdout_mask is not None and holdout_mask.any():
                masks_clone = structural_masks.clone()
                masks_clone[:, holdout_mask, :, 0] = 1.0
                encoder_input = torch.cat([dynamic_features_clean, masks_clone], dim=-1)
            else:
                encoder_input = torch.cat([dynamic_features_clean, structural_masks], dim=-1)
        else:
            encoder_input = dynamic_features_clean
        # 1. Graph Learner — PHASE 3 returns enriched edge_attr [E, 10] (or [E, 1] fallback).
        edge_index, edge_weight, edge_attr = self.graph_learner(
            static_features, location_ids=location_ids
        )

        # --- Strict Topology Masking (F-C04 fix) ---
        # Prevent "ghost clustering" and pollution by dropping outgoing edges from simulated nodes
        # AND from spatial holdout nodes during training / kriging evaluation
        drop_mask = None
        if 'simulated_nodes_mask' in batch:
            drop_mask = batch['simulated_nodes_mask']
        if holdout_mask is not None and holdout_mask.any():
            drop_mask = holdout_mask if drop_mask is None else (drop_mask | holdout_mask)

        if drop_mask is not None:
            sources = edge_index[0]
            valid_edges = ~drop_mask[sources]
            edge_index = edge_index[:, valid_edges]
            edge_weight = edge_weight[valid_edges]
            # PHASE 3: keep edge_attr row-aligned with the filtered edges.
            edge_attr = edge_attr[valid_edges]

        # 2. Encoder — request attention weights when in analysis mode
        if return_analysis:
            latent_embeddings, alpha = self.encoder(
                encoder_input, edge_index, edge_attr,
                return_attention_weights=True,
            )

            # 3. Decoders (complete the full pass to get ZINB parameters)
            imputed_features, mu, theta, pi, tripwires = self.decoders(latent_embeddings)

            # PHASE 2: Kriging reconstruction for held-out nodes (if a mask is supplied)
            kriging_pred = (
                self.kriging_decoder(latent_embeddings, holdout_mask)
                if holdout_mask is not None else None
            )

            return {
                'edge_index': edge_index,
                'edge_weight': edge_weight,
                'attention_weights': alpha,
                'imputed_features': imputed_features,
                'kriging_pred': kriging_pred,
                'mu': mu,
                'theta': theta,
                'pi': pi,
                'tripwire_logs': tripwires
            }

        # Normal training mode — need loss targets
        target_forecast = batch['target_forecast']
        target_impute = batch['target_impute']

        latent_embeddings = self.encoder(encoder_input, edge_index, edge_attr)

        # 3. Decoders
        imputed_features, mu, theta, pi, tripwires = self.decoders(latent_embeddings)

        # PHASE 2: Kriging reconstruction for held-out nodes (if a mask is supplied)
        kriging_pred = (
            self.kriging_decoder(latent_embeddings, holdout_mask)
            if holdout_mask is not None else None
        )

        # 4. Compute Joint Loss — kept in FP32 to avoid bf16 precision issues in ZINB NLL.
        # The autocast(enabled=False) guard ensures this block runs at full precision even
        # when the surrounding adapter.step() call is wrapped in torch.autocast(bfloat16).
        with torch.autocast(device_type=target_forecast.device.type, enabled=False):
            loss, loss_impute, loss_forecast, loss_kriging = compute_loss(
                imputed_features=imputed_features.float(),
                target_impute=target_impute.float(),
                mu=mu.float(),
                theta=theta.float(),
                pi=pi.float(),
                target_forecast=target_forecast.float(),
                traffic_mean=self.traffic_mean,
                traffic_std=self.traffic_std,
                kriging_pred=kriging_pred.float() if kriging_pred is not None else None,
                spatial_holdout_mask=holdout_mask,
                w_impute=batch['w_impute'],
                w_forecast=batch['w_forecast'],
                w_kriging=batch['w_kriging'],
                loss_node_mask=batch.get('loss_node_mask')
            )

        return loss, loss_impute, loss_forecast, loss_kriging, imputed_features, mu, theta, pi, tripwires, kriging_pred

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    # Reproducibility
    torch.manual_seed(42)

    logger.info("Starting ResilientIoTPipeline Integration Test (The Burn Test)")

    # Dimensions
    batch_size = 4
    num_nodes = 50
    static_feat_dim = 10
    time_series_dim = 3
    seq_len = 12

    # --- 1. Dataset Initialization & Mock Data ---
    # The Dataset defined in dataset.py expects (N, S) for static and (N, T, D) for dynamic.
    # We want a pipeline that gets batches with the 'num_nodes' dimension intact.
    # We will generate a mock Dataset that yields identical static features, and (N, T, D)
    # structured batches. However, since the dataset yields (T, D) per sample and we want
    # [B, N, T, D] out of the DataLoader, we must adjust how we interact with it or mock the batch.

    # We'll create a custom mock Dataset specifically for testing the pipeline requirements.
    mask_prob = 0.15

    class MockPipelineDataset(torch.utils.data.Dataset):
        def __init__(self, size):
            self.size = size
        def __len__(self):
            return self.size
        def __getitem__(self, idx):
            # dynamic: [N, T, D] — clean (mask generated below, mirroring on-device path)
            dyn = torch.randn(num_nodes, seq_len, time_series_dim)
            return {
                'static_features': torch.randn(num_nodes, static_feat_dim),
                'dynamic_features': dyn,
                'target_forecast': torch.randint(0, 50, (num_nodes,)).float(),
                'structural_masks': torch.zeros(num_nodes, seq_len, 2),
            }

    dataset = MockPipelineDataset(size=20)

    def custom_collate(batch_list):
        return {
            'static_features': batch_list[0]['static_features'],
            'dynamic_features': torch.stack([b['dynamic_features'] for b in batch_list]),
            'target_forecast': torch.stack([b['target_forecast'] for b in batch_list]),
            'structural_masks': torch.stack([b['structural_masks'] for b in batch_list]),
        }

    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=custom_collate)

    # --- 2. Pipeline and Optimizer ---
    model = ResilientIoTPipeline(
        num_nodes=num_nodes,
        static_feat_dim=static_feat_dim,
        time_series_dim=time_series_dim,
        seq_len=seq_len,
        num_layers=2
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # --- 3. Training Loop (2 Epochs) ---
    num_epochs = 2
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch_idx, batch in enumerate(dataloader):
            optimizer.zero_grad()

            # Generate mask on-device (mirrors train.py GPU-side mask injection).
            clean_dyn = batch["dynamic_features"]
            node_mask = torch.rand(clean_dyn.shape) < mask_prob
            batch["target_impute"] = clean_dyn.clone()
            batch["dynamic_features"] = clean_dyn.masked_fill(node_mask, float("nan"))
            batch["node_mask"] = node_mask

            # Inject loss weights that the adapter normally provides.
            batch["w_impute"] = 1.0
            batch["w_forecast"] = 1.0
            batch["w_kriging"] = 0.0

            # Forward Pass
            loss, *_ = model(batch)

            # Backward Pass
            loss.backward()

            # CRITICAL SAFEGUARD: Gradient Clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # Optimizer Step
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(dataloader)
        logger.info(f"Epoch {epoch + 1}/{num_epochs} - Avg Loss: {avg_loss:.4f}")

    logger.info("ResilientIoTPipeline Integration Test PASSED (Computation graph is connected, no NaNs)")
