"""
Artificial-gap generators for self-supervised imputation (v5 R2).

Every generator returns a boolean tensor [B, N, T + 1] on the batch device,
True = artificially hidden. Columns [0, T) cover the input window; column T is
the forecast target step ("target outage" — the forecast is made from inside a
simulated outage). Only the `tail` strategy (and `mixed` through it) ever sets
column T.

Strategies (docs/design_reports/zero_collapse_investigation.md §R2):
  iid        — Bernoulli(mask_prob) per step (the v4 behaviour, traffic only).
  block      — contiguous runs per (b, n); lengths drawn from the empirical
               training gap-length bank (count-weighted) or a geometric law.
  transplant — a real (seq_len + 1)-step missingness pattern copied from the
               training slice onto this window.
  tail       — the last k input steps AND the target, k ~ U{1..T}.
  mixed      — categorical choice of the above per (b, n).

Block / transplant / tail are applied to a (b, n) row with probability
p_apply, calibrated so the expected masked fraction of the input window equals
mask_prob (tail: of the input window too; its target column is extra).

All randomness flows through an optional torch.Generator so validation / test
masks can be seeded and identical across epochs (deterministic selection).
"""
import math
from typing import Dict, Optional

import torch

STRATEGIES = ("iid", "block", "transplant", "tail")


class GapMasker:
    def __init__(
        self,
        strategy: str,
        mask_prob: float,
        seq_len: int,
        gap_length_bank: torch.Tensor,
        transplant_bank: torch.Tensor,
        block_len_dist: str,
        block_mean_len: float,
        mask_mix: Dict[str, float],
        device: torch.device,
    ):
        assert strategy in STRATEGIES + ("mixed",), f"CRITICAL: unknown mask strategy {strategy!r}"
        assert 0.0 <= mask_prob < 1.0, f"CRITICAL: mask_prob must be in [0, 1), got {mask_prob}"
        assert block_len_dist in ("empirical", "geometric")
        assert transplant_bank.dim() == 2 and transplant_bank.shape[1] == seq_len + 1, \
            f"CRITICAL: transplant bank must be [P, seq_len + 1], got {tuple(transplant_bank.shape)}"
        self.strategy = strategy
        self.mask_prob = mask_prob
        self.seq_len = seq_len
        self.device = device
        self.block_len_dist = block_len_dist
        self.block_mean_len = float(block_mean_len)

        # Count-weighted empirical gap lengths: a slot is in a run of length L with
        # probability ∝ L · freq(L), and "a gap covering this region" should follow
        # that size-biased law — long outages dominate the missing mass.
        lengths = gap_length_bank.to(torch.float32)
        self.gap_lengths = lengths.to(device)
        self.gap_weights = (lengths / lengths.sum()).to(device)

        # Transplant bank: only the input columns contribute to coverage calibration.
        self.transplant_bank = transplant_bank.to(device)
        transplant_cov = float(transplant_bank[:, :seq_len].float().mean())
        assert transplant_cov > 0.0, "CRITICAL: transplant bank has no input-window missingness"

        self.mix_names = [k for k in STRATEGIES if mask_mix.get(k, 0.0) > 0.0] if strategy == "mixed" else []
        self.mix_probs = (
            torch.tensor([mask_mix[k] for k in self.mix_names], dtype=torch.float32, device=device)
            if strategy == "mixed" else None
        )

        # p_apply per structured strategy, calibrated once (Monte Carlo for block).
        self.p_apply = {
            "block": self._calibrate_block(),
            "transplant": min(1.0, mask_prob / transplant_cov),
            "tail": min(1.0, mask_prob / ((seq_len + 1) / 2.0 / seq_len)),
        }

    # ------------------------------------------------------------------ #
    def _block_lengths(self, n: int, gen: Optional[torch.Generator]) -> torch.Tensor:
        if self.block_len_dist == "empirical":
            idx = torch.multinomial(self.gap_weights, n, replacement=True, generator=gen)
            return self.gap_lengths[idx].long()
        p = 1.0 / self.block_mean_len
        u = torch.rand(n, device=self.device, generator=gen).clamp_min(1e-12)
        return (torch.floor(torch.log(u) / math.log(1.0 - p)) + 1).long()

    def _block_rows(self, n: int, gen: Optional[torch.Generator]) -> torch.Tensor:
        """[n, T] block masks. The block start is uniform over positions that let
        the block overlap the window (so long blocks often cover all of it)."""
        T = self.seq_len
        lengths = self._block_lengths(n, gen)
        span = (lengths + T - 1).clamp_min(1)
        start = (torch.rand(n, device=self.device, generator=gen) * span).floor().long() - (lengths - 1)
        t = torch.arange(T, device=self.device).view(1, -1)
        return (t >= start.view(-1, 1)) & (t < (start + lengths).view(-1, 1))

    def _calibrate_block(self) -> float:
        gen = torch.Generator(device=self.device)
        gen.manual_seed(0)
        cov = float(self._block_rows(20000, gen).float().mean())
        assert cov > 0.0, "CRITICAL: block masks cover nothing — check the gap-length bank"
        return min(1.0, self.mask_prob / cov)

    # ------------------------------------------------------------------ #
    def _strategy_rows(self, name: str, n: int, gen: Optional[torch.Generator]) -> torch.Tensor:
        """[n, T + 1] masks for one strategy (before p_apply for structured ones)."""
        T = self.seq_len
        if name == "iid":
            rows = torch.rand(n, T, device=self.device, generator=gen) < self.mask_prob
            return torch.cat([rows, torch.zeros(n, 1, dtype=torch.bool, device=self.device)], dim=1)
        if name == "block":
            rows = self._block_rows(n, gen)
            return torch.cat([rows, torch.zeros(n, 1, dtype=torch.bool, device=self.device)], dim=1)
        if name == "transplant":
            idx = torch.randint(0, self.transplant_bank.shape[0], (n,), device=self.device, generator=gen)
            rows = self.transplant_bank[idx].clone()
            rows[:, T] = False  # target column is never hidden by a transplant (truth stays visible to loss routing)
            return rows
        assert name == "tail"
        k = torch.randint(1, T + 1, (n,), device=self.device, generator=gen)
        t = torch.arange(T + 1, device=self.device).view(1, -1)
        return t >= (T - k).view(-1, 1)

    def sample(self, batch_size: int, num_nodes: int, gen: Optional[torch.Generator] = None) -> torch.Tensor:
        n = batch_size * num_nodes
        if self.strategy == "mixed":
            choice = torch.multinomial(self.mix_probs, n, replacement=True, generator=gen)
            out = torch.zeros(n, self.seq_len + 1, dtype=torch.bool, device=self.device)
            for i, name in enumerate(self.mix_names):
                rows = self._strategy_rows(name, n, gen)
                if name != "iid":
                    rows &= (torch.rand(n, 1, device=self.device, generator=gen) < self.p_apply[name])
                out = torch.where((choice == i).view(-1, 1), rows, out)
        else:
            out = self._strategy_rows(self.strategy, n, gen)
            if self.strategy != "iid":
                out &= (torch.rand(n, 1, device=self.device, generator=gen) < self.p_apply[self.strategy])
        return out.view(batch_size, num_nodes, self.seq_len + 1)


def synthetic_outage_mask(
    batch_size: int,
    num_nodes: int,
    seq_len: int,
    length: int,
    node_frac: float,
    eligible: torch.Tensor,
    gen: Optional[torch.Generator],
    device: torch.device,
) -> torch.Tensor:
    """Evaluation-only synthetic outage of `length` steps ending at the last input
    step (the forecast target is then predicted from inside the outage).

    Returns [B, N] bool `selected` rows; the window slice is the last
    min(length, seq_len) steps. `eligible` [B, N] restricts selection (target
    observed, node not blanked) so the forecast truth is real.
    """
    assert length >= 1
    selected = torch.rand(batch_size, num_nodes, device=device, generator=gen) < node_frac
    return selected & eligible
