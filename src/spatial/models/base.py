"""
Shared training machinery for all trainable spatial forecasters.

One fit loop for every torch model (the quality track duplicated ~120 lines
across its three wrappers and flags this in its own docstrings — fixed here):
Adam + StepLR (per epoch) + masked MSE + gradient clipping + early stopping
with mandatory best-state restore, reseeded at every fit() so results are
order-independent.

Subclasses implement only `_build_module`. Module forward contract:

    module(x [B,N,T,D], edge_index [2,E] long, edge_attr [E,dim]|None,
           static [N,S]) -> pred [B,N,H]
"""
import logging
from abc import abstractmethod
from typing import Dict, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

from spatial.contract import FitReport, GraphForecaster, GraphSpec, GraphSplit
from spatial.dataset import GraphWindowDataset, collate_graph, input_dim, masked_mse

logger = logging.getLogger(__name__)

# Hyperparameters every trainable model spec must provide (no Python defaults
# for anything scientific — NSTAR rule).
TRAINING_KEYS = ("lr", "batch_size", "epochs", "patience",
                 "scheduler_step_size", "scheduler_gamma", "grad_clip")


class TorchGraphForecaster(GraphForecaster):

    def __init__(self, spec: Dict, seed: int, device: Optional[str],
                 inputs_cfg: Dict):
        for key in TRAINING_KEYS:
            assert key in spec, f"CRITICAL: model spec missing training key {key!r}"
        self.spec = dict(spec)
        self.lr = float(spec["lr"])
        self.batch_size = int(spec["batch_size"])
        self.epochs = int(spec["epochs"])
        self.patience = int(spec["patience"])
        self.scheduler_step_size = int(spec["scheduler_step_size"])
        self.scheduler_gamma = float(spec["scheduler_gamma"])
        self.grad_clip = float(spec["grad_clip"])
        self.seed = int(seed)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.mask_channel = bool(inputs_cfg["mask_channel"])
        self.time_features = tuple(inputs_cfg["time_features"])
        self.module: Optional[torch.nn.Module] = None
        self._graph_cache: Dict = {}
        self.diagnostics: Dict = {}

    # ------------------------------------------------------------ subclass API

    @abstractmethod
    def _build_module(self, in_dim: int, n_nodes: int, window: int,
                      n_horizons: int, graph: GraphSpec) -> torch.nn.Module:
        """Return the nn.Module implementing the forward contract."""

    # ---------------------------------------------------------------- fit

    def _graph_tensors(self, graph: GraphSpec):
        key = id(graph)
        if key not in self._graph_cache:
            ei = torch.from_numpy(graph.edge_index).long().to(self.device)
            ea = None
            if graph.edge_attr is not None:
                ea = torch.from_numpy(graph.edge_attr).float().to(self.device)
            self._graph_cache = {key: (ei, ea)}   # single-entry: one graph per job
        return self._graph_cache[key]

    def _loader(self, split: GraphSplit, shuffle: bool) -> DataLoader:
        ds = GraphWindowDataset(split, mask_channel=self.mask_channel,
                                time_features=self.time_features)
        return DataLoader(ds, batch_size=self.batch_size, shuffle=shuffle,
                          num_workers=0, drop_last=False, collate_fn=collate_graph)

    def fit(self, train: GraphSplit, val: GraphSplit, graph: GraphSpec) -> FitReport:
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        ctx = train.context
        in_dim = input_dim(self.mask_channel, self.time_features)
        self.module = self._build_module(
            in_dim, ctx.n_nodes, ctx.window, len(ctx.horizons), graph
        ).to(self.device)
        static = torch.from_numpy(ctx.static_features).float().to(self.device)
        ei, ea = self._graph_tensors(graph)

        optimiser = torch.optim.Adam(self.module.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimiser, step_size=self.scheduler_step_size, gamma=self.scheduler_gamma
        )
        train_loader = self._loader(train, shuffle=True)
        val_loader = self._loader(val, shuffle=False)

        best_val = float("inf")
        best_state = None
        epochs_no_improve = 0
        epochs_run = 0

        for epoch in range(self.epochs):
            epochs_run = epoch + 1
            self.module.train()
            for batch in train_loader:
                x = batch["x"].to(self.device)
                y = batch["y"].to(self.device)
                y_mask = batch["y_mask"].to(self.device)
                optimiser.zero_grad()
                pred = self.module(x, ei, ea, static)
                loss = masked_mse(pred, y, y_mask)
                assert torch.isfinite(loss), "CRITICAL: non-finite train loss"
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.module.parameters(), self.grad_clip)
                optimiser.step()
            scheduler.step()

            # Exact epoch-level masked MSE on val (sums, not batch averages).
            self.module.eval()
            sq_sum, n_obs = 0.0, 0
            with torch.no_grad():
                for batch in val_loader:
                    x = batch["x"].to(self.device)
                    y = batch["y"].to(self.device)
                    y_mask = batch["y_mask"].to(self.device)
                    pred = self.module(x, ei, ea, static)
                    sq_sum += float(((pred - y) ** 2 * y_mask).sum())
                    n_obs += int(y_mask.sum())
            assert n_obs > 0, "CRITICAL: val split has zero observed targets"
            val_loss = sq_sum / n_obs

            if val_loss < best_val - 1e-6:
                best_val = val_loss
                best_state = {k: v.detach().cpu().clone()
                              for k, v in self.module.state_dict().items()}
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= self.patience:
                    break

        assert best_state is not None, \
            f"CRITICAL: {self.name} training produced no best state"
        self.module.load_state_dict(best_state)
        return FitReport(fitted=True, epochs_run=epochs_run, best_val_loss=best_val)

    # -------------------------------------------------------------- predict

    def predict(self, split: GraphSplit, graph: GraphSpec) -> np.ndarray:
        assert self.module is not None, \
            f"CRITICAL: {self.name}.predict() before fit()"
        ctx = split.context
        static = torch.from_numpy(ctx.static_features).float().to(self.device)
        ei, ea = self._graph_tensors(graph)
        self.module.eval()
        outs = []
        with torch.no_grad():
            for batch in self._loader(split, shuffle=False):
                pred = self.module(batch["x"].to(self.device), ei, ea, static)
                outs.append(pred.cpu().numpy())
        preds = np.concatenate(outs, axis=0)
        assert preds.shape == (len(split), ctx.n_nodes, len(ctx.horizons)), \
            f"CRITICAL: {self.name} prediction shape {preds.shape}"
        assert np.isfinite(preds).all(), \
            f"CRITICAL: {self.name} produced non-finite predictions"
        return preds

    def get_params(self) -> Dict:
        return {**self.spec, "seed": self.seed, "device": str(self.device),
                "mask_channel": self.mask_channel,
                "time_features": list(self.time_features)}
