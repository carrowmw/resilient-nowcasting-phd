# Comprehensive Thesis Audit Report

**Thesis:** *Resilient Spatio-Temporal Forecasting for Urban Sensor Networks: A Deep Learning Framework for Operational Situational Awareness Under Data Imperfection* (Newcastle University, School of Engineering — Geospatial AI)
**Audit date:** 2026-09-25
**Auditor role:** External examiner / forensic AI-ML auditor (read-only; no source files were modified)

---

## 0. Audit Scope, Sources, and Limitations (read first)

The thesis LaTeX tree (`symlinks/symlinks → /Users/administrator/Code/latex-documents/thesis`) is a host-side symlink that **is not mounted** in the audit environment. The only thesis material recoverable from the repository (git `HEAD`, deleted from the working tree) was:

| Source | Content | Used as |
|---|---|---|
| `docs/context/chapter4_methodology.tex` (732 lines) | Chapter 4 — Methodology | **Primary audit object.** All `L###` line numbers below refer to this file. |
| `docs/context/appendix_design_rationale.tex` (105 lines) | Appendix — Design Rationale | Primary audit object (`App:L###`). |
| `docs/context/thesis.pdf` (29 pp.) | Full ToC (Ch. 1–7) + compiled Ch. 4 + references page | ToC cross-checks for Ch. 1–7 headings, rendered Figure 4.1, staleness checks. |
| `docs/mdpi_sensors/mdpi_sensors.tex` | MDPI *Sensors* paper (source of Experiment 1 / Ch. 5) | Reconciling Exp. 1 numbers. |
| `latex-documents/thesis/figures/**` | 3 sensor-availability figures (×3 duplicated folders) | Figure grounding. |
| `docs/poster/riot_architecture.{png,svg,pdf}`, `docs/diagrams/*.mmd` | Architecture schematics | Diagram fidelity. |
| `src/**`, `configs/**`, `artifacts/**`, `data/processed/master_dataset_15T.parquet` | Implementation, configs, manifests, data | Grounding every implementation claim in Ch. 4. |

**Consequence:** Chapters 1, 2, 3, 5, 6, and 7 **could not be audited at body-text level**. Their numbers, tables, and figures (MAE/RMSE/CVRMSE/MASE/R²/p-values) cannot be verified against the narrative. Where the ToC or the linked artefacts (manifests, figures, MDPI paper) show what those chapters must contain, the cross-chapter checks (Dimension E) were done against those artefacts. **A second audit pass on the full LaTeX tree is required before submission.**

Verdict labels: **CONFIRMED** means the evidence was reproduced directly (code read, data queried, figure rendered). **PLAUSIBLE** means it is strongly indicated but depends on an artefact I could not inspect.

---

## 1. Executive Summary & Integrity Scorecard

| Metric | Value |
|---|---|
| **Overall empirical grounding score** | **3 / 10** |
| Critical findings | **12** |
| Major findings | **29** |
| Minor findings | **18** |
| **Total** | **59** |

### High-level verdict

**Not ready for submission.** Chapter 4 is well written and its core equations mostly match the code: the graph learner, 10-dimensional edge attributes, ZINB likelihood, parameter constraints, MSLE on inverse-transformed counts, 2-layer GELU decoders, and night-mask window all check out. However, several **load-bearing methodological claims are contradicted by the implementation and data** that would produce the Chapter 7 results:

1. **Evaluation protocol.** The chapter asserts a chronological train/val/test split with a train-only scaler. The RIoT training code uses `torch.utils.data.random_split` over overlapping 12-step windows (11/12 timesteps shared between neighbours). It has **no test set**, and fits scalers on train+val. The project's own experiment ledger records this scaler leakage as "legacy leakage" fixed only in the Exp. 1 track.
2. **Missingness encoding.** The central thesis device, the explicit MNAR mask channels, is largely neutralised by the direction-aggregation step. In the default `target_dir: all` configuration, `groupby().sum()` converts all-null station slots to **count = 0 with mask = 0**. Measured on the pedestrian data: **36.3%** of station-slots are fully null, but only **5.5%** survive as NaN. A further **25.5%** are silent partial sums.
3. **Kriging protocol.** "Topology masking" of held-out nodes is described in the text and both diagrams but **is not implemented**: only simulated nodes lose outgoing edges. Held-out nodes also keep their own MNAR mask channels and self-loops. This contradicts the claim that their embeddings are "shaped entirely by message passing from observable neighbours" and creates a target-information leak.
4. **Metrics.** Point RMSE/MAE are computed against the NB mean `μ`, not the ZINB mean `μ(1−π)` the chapter specifies. Zero-accuracy and the zero-inflation reliability diagram are defined inconsistently with the ZINB model.
5. **Architecture description.** The Transformer temporal encoder has **no positional encoding**. The text nonetheless attributes diurnal-cycle and long-term-trend modelling to it, within a 3-hour (T = 12) window. Figure 4.1 omits the GATv2 spatial encoder entirely. The poster/architecture diagram swaps the imputation and forecasting decoder labels. The per-head edge projection dimension `D_in/K` is mathematically invalid (5/4).
6. **Cross-experiment comparability.** Chapter 4 promises "controlled comparisons" between successive experiments. But the three experiments use different units (110 directional streams / 54 stations / 81 stations), horizons (h = 8, 24 / 1, 2, 4 / 1), missing-data treatments (gap-preserving / — / zero-fill), and split regimes (per-stream chronological / global calendar / random).

**Key vulnerability areas:** evaluation leakage (split and scaler), the null→zero pipeline contradiction, kriging leakage, diagram fidelity, notation overload, and empty skeleton chapters (the ToC shows Ch. 5 and Ch. 6 occupying one page each, and Ch. 7 ablation sections without content).

---

## 2. Cross-Chapter Traceability Matrix

| Element Label | Type | File & Line | Referenced in Text? | Data/Result Grounded? | Status / Flags |
|---|---|---|---|---|---|
| `fig:conceptual_framework` | Figure | `ch4:L212–218` | Yes (L210, L425) | **No.** Asset `figures/methodology/conceptual_pipeline_framework.pdf` is not in repo; rendered PDF version omits GATv2 | **[CRITICAL]** Misrepresents architecture (F-C09); draft comment L214 |
| `fig:ambiguity_of_missing_data` | Figure | `ch4:L586–591` | Yes (L584) | **No.** Asset `ambiguity_of_missing_data.png` not in repo; absent from compiled PDF | **[MAJOR]** Missing asset; no generating script (F-M23) |
| `tab:encoder-channels` | Table | `ch4:L353–369` | **No** explicit `\ref` (only implied by L351) | **Partially.** Channels exist in code, but Ch. 3/4 mask rates contradicted by aggregation | **[CRITICAL]** Mask semantics broken in default config (F-C03); orphan table |
| `tab:capability_contrast` | Table | `ch4:L681–697` | Yes (L679) | **Partially.** Baseline "kriging ×" contradicts harness `kriging=True` | **[MAJOR]** (F-M28); caption acronym "RIOT" |
| `eq:gat-static`…`eq:gat-multihead` | Equations | `ch4:L33–81` | Yes | Theory (cited) | **[MAJOR]** W dimension inconsistent (F-M14); `\Bigr\|` typo |
| `eq:scaled-dot-product`…`eq:transformer-ffn` | Equations | `ch4:L96–121` | Yes | Code matches post-LN, ReLU FFN | **[CRITICAL]** No positional encoding in text or code (F-C08) |
| `eq:encoder-latent` | Equation | `ch4:L136–139` | Yes | Yes (`encoder.py` output `[B,N,T,H]`) | OK |
| `eq:joint-loss` | Equation | `ch4:L155–158` **and** `L633–636` | Yes (ambiguous) | Yes (`compute_loss`) | **[MAJOR]** Duplicate label (F-M03) |
| `eq:graph-projection/similarity/sparsify` | Equations | `ch4:L438–455` | Yes | Yes (`graph_learner.py`) | OK. Undisclosed self-loops (F-m04) |
| `eq:edge-attr` | Equation | `ch4:L464–467` | Yes (L487) | Yes (`graph_learner.py` 9+1) | **[MINOR]** "min-max" vs ÷max (F-m04) |
| `eq:gatv2-edge-conditioned` | Equation | `ch4:L482–485` | Yes | **No.** `Θ_e ∈ ℝ^{(D_in/K)×10}` wrong | **[CRITICAL]** (F-C07) |
| `eq:spatial-output` | Equation | `ch4:L489–492` | Yes | Yes (single GATv2Conv, concat) | **[MINOR]** σ omitted vs eq:gat-multihead |
| `eq:imputation-decoder`, `eq:forecast-decoder`, `eq:kriging-decoder` | Equations | `ch4:L530–555` | Yes | Yes (`decoders.py`) | **[MAJOR]** Output dim D vs D_ts (F-M16) |
| `eq:zinb_likelihood` | Equation | `ch4:L599–606` | Yes | Yes (`compute_zinb_nll`) | **[MAJOR]** NB pmf undefined, no citation (F-M21) |
| `eq:loss-fore` | Equation | `ch4:L608–611` | Yes | **No.** Random holdout nodes are *not* excluded | **[MAJOR]** (F-M09) |
| MSLE equation (unlabelled) | Equation | `ch4:L622–624` | No label | Yes (`scale_invariant_mse`) | **[MINOR]** Unlabelled; not over O only (F-M09) |
| `sec:meth-evolution` | Section label | `ch4:L228` **and** `L266` | — | — | **[MAJOR]** Duplicate label, duplicate section (F-M02) |
| `sec:meth-prelims` | Section | `ch4:L14` + `L16` | Yes (L17) | — | **[MAJOR]** Duplicated heading → empty §4.1 (F-M01) |
| `subsec:meth-graph` | Subsection | `ch4:L415–418` | Not referenced | — | **[MAJOR]** Empty placeholder (F-M04) |
| `ch:exp1`, `ch:exp2`, `ch:exp3`, `ch:data` | Chapter refs | `ch4:L12, 186, 237–275, 282, 413, 517` | Yes | **Unverifiable** (chapters absent) | Ch. 5/6 appear to be empty skeletons in ToC (F-M23) |
| `experiment_sensor_availability_comparison.{pdf,png}` | Figure asset | `figures/{methodology,experiment2,experiment3}/` | **Not in Ch. 4** (may be in Ch. 5–7) | Yes (`scripts/plot_experiment_sensor_availability.py`, manifests) | **[CRITICAL]** Exp. 3 panel claims calendar train/val/test split not used by RIoT code (F-C12) |
| `sensor_availability_exp1_vs_exp2.{pdf,png}` | Figure asset | same | Not in Ch. 4 | Yes | [MINOR] Duplicated in 2 folders; caption wording |
| `sensor_availability_exp2_vs_exp3.{pdf,png}` | Figure asset | same | Not in Ch. 4 | Yes | **[CRITICAL]** Same split misrepresentation (F-C12) |
| `riot_architecture.{png,svg,pdf}` | Diagram (poster; likely reused Ch. 7) | `docs/poster/` | Unknown | Partially | **[CRITICAL]** Decoder labels swapped; holdout edge-drop not implemented (F-C11) |
| `architecture_current.mmd` | Diagram | `docs/diagrams/` | Unknown | Partially | [MAJOR/MINOR] Holdout edge-drop, p = 0.15, order of masking vs scaling (F-C04, F-m14) |
| Appendix `sec:appendix-graph-interface` | Appendix section | `App:L13` | Unknown (no back-ref from Ch. 4) | Descriptive | [MINOR] Example N = 29 matches no experiment (F-M26); not cited from Ch. 4 |

**Positive verifications (claims confirmed against code/data):** d_e = 32 (`graph_learner.py`, `train.yaml`); ReLU → subtract τ → ReLU sparsification; 10-dim edge attribute with 6 collapsed road categories summing to 1/0; single-layer GATv2Conv with `out = H/K` and concatenation; post-LN Transformer with ReLU FFN; `[B,T,N,H]→[T,B·N,H]` reshape; 2-layer MLP + GELU + dropout decoders; `softplus+ε` for μ, θ and clamped sigmoid for π with ε = 1e-6; ZINB NLL formula; MSLE on inverse-transformed raw counts and MSE on scaled weather; night window [00:00, 05:00); 15 static feature names; fail-fast NaN assertions on the weather join; T = 12 at 15-min resolution (grid verified regular: 96,576 steps, 0 gaps); city-centre coordinates (54.9738, −1.6131); **raw pedestrian data contains zero explicit zero counts (0 of 15.6M rows) and ~7.9M nulls**, which grounds the null-zero conflation premise (L582).

---

## 3. Detailed Forensic Findings (Ordered by Chapter)

### Chapter 4 — Methodology

---

### [F-C01] [CRITICAL] Claimed chronological train/val/test split is not what the RIoT pipeline does; no test set exists
- **File & Location:** `chapter4_methodology.tex` (L192–194, §4.3.2 "Train, Validation, and Test Protocol")
- **Category:** Hallucination / Cross-Chapter Consistency
- **Verdict:** CONFIRMED
- **Quoted Text / LaTeX:**
  > "All experiments use a chronological train/validation/test split of the time series data to prevent information leakage from future observations into the training process. [...] The test set is used only for final evaluation and is not examined during model development."
- **Discrepancy / Evidence:** `src/train.py` L264 and L343: `train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])` with `train_split: 0.8` (`configs/train/train.yaml`). Samples are overlapping sliding windows (`dataset.py __getitem__`: `dynamic_data[idx: idx+seq_len]`), so a random split puts windows that share 11 of 12 timesteps (and each other's forecast targets) into both train and validation. Early stopping and Optuna model selection run on this leaked validation loss. No test split exists in `train.py`. The Exp. 2 context manifest (`artifacts/spatial_full/context_manifest.json`: train/val/test = 83,787/17,939/17,941) and the Exp. 1 track (per-run chronological 70/15/15, per `EXPERIMENTS.md`) *do* use chronological splits. The statement is therefore false for Experiment 3, the thesis's headline model.
- **Recommended Action:** Either re-run all RIoT experiments with a chronological, purged split and a held-out test period, or disclose the protocol. Suggested replacement text (valid only after re-running):
  > "All experiments use a chronological split of the 15-minute grid into training (70%), validation (15%) and test (15%) periods, with a purge gap of *T + h* steps at each boundary so that no input window or forecast target straddles two partitions. Scalers are fitted on the training period only. Hyperparameters and early stopping use the validation period; the test period is evaluated once per final configuration."

---

### [F-C02] [CRITICAL] Scalers are fitted on the full dataset (train + validation), not the training set
- **File & Location:** `chapter4_methodology.tex` (L344; L389; Table `tab:encoder-channels` L361–363)
- **Category:** Hallucination / Math Rigor
- **Verdict:** CONFIRMED
- **Quoted Text / LaTeX:**
  > "All 15 features are standardised (zero mean, unit variance) via a \texttt{StandardScaler} fitted on the training set before being passed to the model."
- **Discrepancy / Evidence:** `src/dataset.py` fits `traffic_scaler`, `weather_scaler` and `static_scaler` with `fit_transform` on the entire `[start_date, end_date]` range *before* `random_split` in `train.py`. `train.py` L436–451 even asserts train/val scaler equality "because both subsets share the same full_dataset". `EXPERIMENTS.md` (2026-07-22 entry) explicitly describes a "TRAIN-only scaler (legacy leakage fixed)" in the Exp. 1 track only. Also, nulls are zero-filled *before* fitting (`val_df.fillna(0.0)` precedes `fit_transform`), so the traffic mean and σ are biased by fill values. That makes the registered buffers `μ_traffic` and `σ_traffic` (L389) statistics of a mixture of real counts and artificial zeros.
- **Recommended Action:** Fit all scalers on training-period rows only, and exclude null positions from the fit (`StandardScaler().fit(x[~mask])`). Replace L344 with:
  > "Static features are standardised using statistics computed over training nodes only; dynamic channels are standardised using statistics computed over *observed* (non-null) training-period values only."

---

### [F-C03] [CRITICAL] MNAR mask channels are largely suppressed in the default configuration by direction aggregation (null → 0, mask = 0)
- **File & Location:** `chapter4_methodology.tex` (L351–377, Table `tab:encoder-channels`; L582–584)
- **Category:** Hallucination / Cross-Chapter Consistency (missing-data treatment)
- **Verdict:** CONFIRMED on `data/processed/master_dataset_15T.parquet` (the documented mirror of the TimescaleDB source; `configs/infra.yaml: data_source: "database"`)
- **Quoted Text / LaTeX:**
  > "Channel~3 (total MNAR mask) is set to 1 wherever the raw traffic count is \texttt{null}, regardless of time of day."
- **Discrepancy / Evidence:** With `target_dir: "all"` (production config), `dataset.py` performs `df.groupby(['dt','location']).agg({'value':'sum', ...})` **before** the mask is computed (`val_df.isna()`). Pandas `sum` uses `min_count=0`, so an all-null group becomes **0.0, not NaN**. Measured on pedestrian data (Jan–Feb 2025 sample, 81 stations):
  - Station-slots with **all** directions null: **36.3%**
  - Station-slots that remain NaN after aggregation (i.e., mask = 1): **5.5%**
  - Station-slots equal to 0 after aggregation: **34.3%** (raw data contains **0** explicit zeros)
  - Station-slots with *some* directions null (silent under-count, mask = 0): **25.5%**

  So roughly 85% of genuinely null station-slots enter the model as observed zero traffic with Channel 3 = 0 and Channel 4 = 0. The forecast target (`unscaled_traffic`) inherits the same zeros. The mechanism the chapter presents as its answer to null-zero conflation is therefore inactive for most of the data, and the ZINB zero-component is fitted to aggregation artefacts.
- **Recommended Action:** Aggregate with `min_count=1` (NaN if all directions null) and define a policy for partial-null slots (e.g., mark as missing, or add a "partial" channel). Then re-run Exp. 3. Add a sentence to §4.7.3:
  > "Directional streams are summed to station level only when every direction reports; station-slots with any missing direction are treated as missing (Channel 3 = 1). Under this rule, X% of station-slots are masked (Y% at night)."

---

### [F-C04] [CRITICAL] "Topology masking" of spatially held-out nodes is described but not implemented
- **File & Location:** `chapter4_methodology.tex` (L568, §4.8.6 Spatial Holdout, item 2)
- **Category:** Hallucination / Diagram Fidelity
- **Verdict:** CONFIRMED
- **Quoted Text / LaTeX:**
  > "\textbf{Topology Masking}: Outgoing edges from held-out nodes are dropped from the graph to prevent information leakage in the reverse direction."
- **Discrepancy / Evidence:** `src/pipeline.py` L160–168 drops outgoing edges **only** for `batch['simulated_nodes_mask']`. `spatial_holdout_mask` is never used to filter `edge_index`, in either `pipeline.py` or `harness/adapters.py`. Observable nodes therefore aggregate messages from zeroed held-out nodes, which contradicts the text. `docs/diagrams/architecture_current.mmd` ("Drop Outgoing Edges from Simulated & Held-out Nodes") and `docs/poster/riot_architecture.png` repeat the unimplemented claim.
- **Recommended Action:** Implement the filter (`valid = ~(sim_mask | holdout_mask)[edge_index[0]]`) and re-run, or delete item 2 and describe the actual behaviour:
  > "Held-out nodes remain in the graph as (zeroed) message sources; the effect of this on observable nodes is quantified in Section X."

---

### [F-C05] [CRITICAL] Held-out nodes retain their own MNAR mask channels and self-loops, leaking target information to the kriging head
- **File & Location:** `chapter4_methodology.tex` (L557; L567)
- **Category:** Hallucination / Math Rigor
- **Verdict:** CONFIRMED (code path); leakage magnitude unmeasured
- **Quoted Text / LaTeX:**
  > "The dynamic features of held-out nodes are set to zero across all time steps and channels [...] This forces their latent embeddings to be determined entirely by spatial message passing from observable neighbours."
  > "Critically, the kriging decoder's latent input $\mathbf{z}_{j,t}$ for held-out nodes is shaped \emph{entirely} by message passing from observable neighbours"
- **Discrepancy / Evidence:**
  1. `apply_spatial_holdout` zeroes the 3 observation channels in `train.py`. The 2 structural-mask channels are concatenated **afterwards** in `pipeline.py` L149–153 and are not blanked. Held-out nodes therefore expose their own null pattern. Given F-C03, that pattern marks exactly where the kriging target equals the fill value.
  2. `GATv2Conv` defaults to `add_self_loops=True`. In addition, the graph learner's `A_raw = EEᵀ` has diagonal entries ‖e_i‖² that generally survive the threshold. The comment in `graph_learner.py` asserting that "self-pairs never survive sparsity" is not guaranteed. Each held-out node therefore attends to its own (mask-bearing) features.
  3. The per-node Transformer then mixes these across time.
- **Recommended Action:** Blank mask channels for held-out nodes (or set Channel 3 = 1 for all their timesteps to signal "unobserved"). Disable self-loops for held-out nodes, and document self-loop handling. Reword L557 to:
  > "...shaped by message passing from observable neighbours and by the node's static features via the learned topology; the node's own dynamic inputs, including missingness indicators, are withheld."

---

### [F-C06] [CRITICAL] Point-forecast metrics are computed on μ, not on the ZINB expectation μ(1−π) stated in the text
- **File & Location:** `chapter4_methodology.tex` (L655, §4.9.1)
- **Category:** Hallucination / Math Rigor
- **Verdict:** CONFIRMED
- **Quoted Text / LaTeX:**
  > "\textbf{Forecasting (Point) Metrics:} The expected value $\hat{\mu} \cdot (1 - \hat{\pi})$ is compared against the unscaled target using \gls{rmse} and \gls{mae-metric}."
- **Discrepancy / Evidence:** `src/harness/evaluate.py` L61–62: `rmse = sqrt(mean((target - preds.mu)**2))`, `mae = mean(|target - preds.mu|)`. `GNNAdapter` sets `Predictions.mu = mu` (the NB mean) with no `(1−π)` factor. `src/metrics.py` does the same, and `compute_two_extremes` uses `mu_predictions`. Every RMSE/MAE reported for RIoT in Ch. 7 therefore evaluates a different estimator from the one described. With π non-negligible (≈31–34% zero targets, F-C03), the bias is material.
- **Recommended Action:** Change the harness to `mu * (1 - pi)` and regenerate all Ch. 7 tables. If the thesis intends to keep μ, change L655 to:
  > "The Negative Binomial mean $\hat{\mu}$ (i.e., the conditional mean given a non-structural-zero state) is compared..."

  and justify that choice explicitly.

---

### [F-C07] [CRITICAL] Edge-projection dimension $(D_{\text{in}}/K)$ is mathematically invalid
- **File & Location:** `chapter4_methodology.tex` (L487, below `eq:gatv2-edge-conditioned`)
- **Category:** Math Rigor / Diagram Fidelity
- **Verdict:** CONFIRMED
- **Quoted Text / LaTeX:**
  > "$\boldsymbol{\Theta}_e^{(k)} \in \mathbb{R}^{(D_{\text{in}}/K) \times 10}$ is a learnable edge projection, and $K$ is the number of attention heads."
- **Discrepancy / Evidence:** D_in = 5 and K = 4 (`train.yaml num_heads: 4`), so D_in/K = 1.25, which is not an integer dimension. In Eq. `gatv2-edge-conditioned`, Θ_e a_ij is added to W^(k)[x_i‖x_j], which lives in the per-head output space ℝ^{H/K}. PyG `GATv2Conv(out_channels=hidden_dim//num_heads, heads=num_heads, edge_dim=10)` implements `lin_edge: ℝ^{10} → ℝ^{K·H/K}`, i.e. Θ_e^(k) ∈ ℝ^{(H/K)×10} = ℝ^{16×10}. The same equation also leaves W^(k) undefined; it should be W^(k) ∈ ℝ^{(H/K)×2D_in}.
- **Recommended Action:** Replace with:
  > "where $\mathbf{W}^{(k)} \in \mathbb{R}^{(H/K) \times 2D_{\text{in}}}$, $\boldsymbol{\Theta}_e^{(k)} \in \mathbb{R}^{(H/K) \times 10}$ and $\mathbf{a}^{(k)} \in \mathbb{R}^{H/K}$ are the per-head parameters (with $H = 64$, $K = 4$ in all reported experiments)."

---

### [F-C08] [CRITICAL] Transformer temporal encoder has no positional encoding; text attributes diurnal/trend modelling to a 3-hour order-agnostic encoder
- **File & Location:** `chapter4_methodology.tex` (L90–123 §4.2.2; L112; L396; L501–503 §4.8.3)
- **Category:** Hallucination / Diagram Fidelity / Math Rigor
- **Verdict:** CONFIRMED
- **Quoted Text / LaTeX:**
  > "Each head can attend to different temporal patterns (e.g., short-range diurnal cycles vs.\ longer-term trends)."
  > "...while the Transformer captures intra-node temporal dynamics across the context window."
- **Discrepancy / Evidence:**
  1. `src/encoder.py` passes the GATv2 output straight into `nn.TransformerEncoder`, with no positional or time encoding, and no time-of-day or day-of-week features anywhere in the 5-channel input (Table `tab:encoder-channels`). Self-attention without positional information is permutation-equivariant over the T positions. The forecasting head reads position −1 only, so the model cannot distinguish temporal order except through the last token's own content.
  2. With T = 12 at 15 min (L396), the window spans **3 hours**, so a 24-hour "diurnal cycle" and "longer-term trends" cannot be attended to at all.
  3. ToC §7.2.4 is titled "Temporal Encoding and Permutation Equivariance Limitations", so the author appears aware of the problem. Chapter 4 nonetheless presents the encoder without caveat, and Eq. `transformer-attn` lacks the standard `X + PE` term.
- **Recommended Action:** Add positional (and ideally time-of-day) encodings and re-run. Alternatively, rewrite L112 and L503 as:
  > "In the configuration evaluated here, no positional encoding is applied, so the temporal encoder is permutation-equivariant over the $T = 12$ (3-hour) context; the consequences of this design are analysed in Section~\ref{sec:exp3-temporal-encoding}."

  Delete the "diurnal cycles vs. longer-term trends" example.

---

### [F-C09] [CRITICAL] Figure 4.1 (conceptual framework) misrepresents the architecture and its data flow
- **File & Location:** `chapter4_methodology.tex` (L210–218; referenced again at L425 as "illustrates the end-to-end data flow")
- **Category:** Diagram Fidelity / Figure-Table Grounding
- **Verdict:** CONFIRMED (rendered from `thesis.pdf` p. 59)
- **Quoted Text / LaTeX:**
  > "\caption{Conceptual framework [...] illustrating the flow from raw multi-modal inputs through dynamic topology inference to decoupled forecasting, imputation, and spatial kriging heads.}"
  > "% Ensure you export from draw.io as a PDF with 'Crop to Content' selected"
- **Discrepancy / Evidence:** The rendered figure is a serial four-box chain: *Data Ingestion → Dynamic Graph Learning → "Spatio-Temporal Forecasting" (MHSA + FFN block) → Decoupled Multi-Task Decoders*.
  - The **GATv2 spatial encoder (Component 2) is absent**, even though L425 says the figure illustrates all four components.
  - The serial arrows imply that dynamic data passes *through* the graph learner. In the code the graph learner consumes static features only, and dynamic features bypass it into the encoder (`pipeline.py`).
  - There are no MNAR mask channels, edge-feature cache, holdout masking, or loss terms. The decoder heads (imputation, forecasting, kriging, ZINB) are not individually shown, despite the caption.
  - The encoder block is labelled "Spatio-Temporal Forecasting", conflating the encoder with one decoder task.
  - A draft authoring comment remains (L214), and the source asset `conceptual_pipeline_framework.pdf` is not in the repository.
- **Recommended Action:** Redraw with two input streams: static features → Graph Learner → (edge_index, edge_attr[10]); dynamic 3 channels ⊕ 2 mask channels → GATv2 (per time step) → Transformer → **Z** → {Imputation, ZINB Forecast, Kriging}. Remove the draft comment. Suggested caption:
  > "Conceptual framework of the RIoT pipeline. Static urban-morphology features parameterise a learned graph whose edges are enriched with nine OSM attributes; dynamic observations and structural missingness masks are encoded by a per-time-step GATv2 layer followed by a per-node Transformer; the shared latent tensor $\mathbf{Z}$ feeds imputation, ZINB forecasting and spatial-kriging heads."

---

### [F-C10] [CRITICAL] Claim of "controlled comparison" across experiments is contradicted by incompatible node sets, horizons, splits, and missing-data treatments
- **File & Location:** `chapter4_methodology.tex` (L182, L186, L231, L241)
- **Category:** Cross-Chapter Consistency / Hallucination
- **Verdict:** CONFIRMED (via manifests, MDPI paper, configs)
- **Quoted Text / LaTeX:**
  > "...enabling a controlled comparison between successive architectures." (L182)
  > "Experiment~2 [...] enabling a direct measurement of the gain attributable to incorporating inter-sensor relationships." (L186)
- **Discrepancy / Evidence:**

  | | Exp. 1 (Ch. 5 / MDPI) | Exp. 2 (Ch. 6) | Exp. 3 (Ch. 7, RIoT) |
  |---|---|---|---|
  | Unit | **110 directional streams** (53 stations) (`artifacts/quality_full_v2/sensor_manifest.json`) | **54 station nodes** (`artifacts/spatial_full/context_manifest.json`) | **81 station nodes** (`train.yaml num_nodes: 81`) |
  | Directions | Per direction | Summed | Summed (with null→0 artefact, F-C03) |
  | Horizon | h = 8, 24 (2 h, 6 h) | h ∈ {1, 2, 4} | h = 1 (one-step, L401) |
  | Missing data | Gaps preserved; paper warns *against* zero-fill | (not audited) | Zero-filled (L377) |
  | Split | Per-stream chronological 70/15/15 | Global calendar 70/15/15 | Random 80/20, no test (F-C01) |

  Because the error metrics are not computed on the same targets, horizons, or periods, differences between experiments cannot be attributed to "inter-sensor relationships" or to "reconstruction".
- **Recommended Action:** Define a common evaluation core (same stations, same test period, same horizons, same missing-value policy on the target) that all three chapters report. Otherwise, soften L182/L186 to:
  > "Each experiment is evaluated under its own protocol (Table X summarises node sets, horizons and splits); cross-experiment comparisons are therefore qualitative, and quantitative claims are restricted to within-experiment comparisons."

---

### [F-C11] [CRITICAL] RIoT architecture schematic swaps the imputation and forecasting decoder labels and depicts unimplemented behaviour
- **File & Location:** `docs/poster/riot_architecture.{png,svg,pdf}` (a candidate asset for Ch. 7 §7.2.1; also `docs/diagrams/architecture_current.mmd`)
- **Category:** Diagram Fidelity
- **Verdict:** CONFIRMED (image inspected); inclusion in Ch. 7 PLAUSIBLE
- **Quoted Text / LaTeX:** Diagram labels: "Imputation Decoder — *MLP_fore(Z)*"; "Forecasting Decoder — *MLP_imp(Z)*"; "Drop Outgoing Edges from Simulated and Held-out nodes"; "Spatial: GATv2Conv layers"; "Edge Index & Edge Weight (N,N,1)"; "MSE Loss" (imputation).
- **Discrepancy / Evidence:**
  - The MLP subscripts are **swapped** between the Imputation and Forecasting decoders.
  - The held-out edge-drop is not implemented (F-C04).
  - The code uses a **single** GATv2Conv layer.
  - Edge weights are `[E]` and edge_index is `[2,E]`, not `(N,N,1)`.
  - The imputation loss is traffic-MSLE + weather-MSE (L625), not "MSE".
  - Editing artefacts remain: "Horizontal Container" and the typo "simluated".
  - The forecasting-decoder input is shown as the full Z rather than Z[:,:,−1,:].
- **Recommended Action:** Correct the subscripts (Imputation → MLP_imp, Forecasting → MLP_fore(Z_{:,:,T})). Label the imputation loss "MSLE (traffic) + MSE (weather)". Change "layers" to "layer (K = 4 heads)" and fix the tensor shapes. Remove the draw.io artefacts. Keep the held-out edge-drop only if it is implemented.

---

### [F-C12] [CRITICAL] Sensor-availability figures assign a global calendar train/val/test split to RIoT that the RIoT code does not use
- **File & Location:** `latex-documents/thesis/figures/{methodology,experiment2,experiment3}/experiment_sensor_availability_comparison.*` and `sensor_availability_exp2_vs_exp3.*` (panel "Experiment 3: Resilient Architecture (RIoT) — N = 81 Resilient Graph Nodes (Global Calendar Split)")
- **Category:** Figure/Table Grounding / Cross-Chapter Consistency
- **Verdict:** CONFIRMED
- **Quoted Text / LaTeX:** Panel subtitle "N = 81 Resilient Graph Nodes (Global Calendar Split)"; annotation "27 Sparse Locations Reclaimed by RIoT (100% of 81 physical locations active)".
- **Discrepancy / Evidence:**
  - The RIoT trainer uses `random_split` (80/20, no test; F-C01), so the red/blue/green calendar blocks do not describe how RIoT was trained or evaluated.
  - "100% of 81 physical locations active" is overstated. The figure itself shows most of the 27 reclaimed stations with **no observations** in the validation or test periods (rows ≈ 60–81 end in 2024), so they cannot contribute to evaluation there.
  - The figure omits the **24/81 inductive holdout** (`artifacts/riot-full-inductive:v4/holdout_manifest.json`), which is excluded from all training losses.
  - The generating script's docstring (`scripts/plot_experiment_sensor_availability.py` L9) says "Experiment 3 [...] N=110", contradicting the rendered N = 81.
- **Recommended Action:** Regenerate after fixing F-C01, or relabel the panel "(random window split — see §7.x)". Replace the annotation with:
  > "27 low-completeness stations included as graph nodes; k of 27 have observations in the evaluation period"

  Overlay the inductive holdout nodes and fix the script docstring.

---

### [F-M01] [MAJOR] Duplicated `\section{Theoretical Preliminaries}` produces an empty §4.1
- **File & Location:** L14 and L16
- **Category:** Figure/Table Grounding (artefacts)
- **Verdict:** CONFIRMED (ToC shows §4.1 and §4.2 with the same title, both on p. 53)
- **Quoted Text / LaTeX:** `\section{Theoretical Preliminaries}` followed immediately by a second `\section{Theoretical Preliminaries}\label{sec:meth-prelims}`
- **Discrepancy / Evidence:** A leftover draft line renumbers every subsequent section (+1).
- **Recommended Action:** Delete L14.

### [F-M02] [MAJOR] Two sections titled "Methodological Evolution Across Experiments" with the same label
- **File & Location:** L227–228 and L265–266
- **Category:** Cross-reference integrity / Narrative consistency
- **Verdict:** CONFIRMED (ToC §4.5 and §4.6)
- **Quoted Text / LaTeX:** `\section{Methodological Evolution Across Experiments}\label{sec:meth-evolution}` (twice)
- **Discrepancy / Evidence:** Duplicate `\label` (LaTeX "multiply defined" warning; refs resolve to the last one). The second copy (L269–275) is an older, shorter version of the first. Its description of Exp. 3 ("resilience without requiring a separate imputation stage") differs in scope from L243–244 (dynamic graph learning, ZINB, kriging).
- **Recommended Action:** Delete L264–275.

### [F-M03] [MAJOR] Duplicate equation label `eq:joint-loss`
- **File & Location:** L157 and L635
- **Category:** Math Rigor / Cross-reference integrity
- **Verdict:** CONFIRMED
- **Quoted Text / LaTeX:** `\label{eq:joint-loss}` on both $\mathcal{L}_{\text{joint}}$ (L156) and $\mathcal{L}_{\text{total}}$ (L634)
- **Discrepancy / Evidence:** The same objective is defined twice under two symbols ($\mathcal{L}_{\text{joint}}$ vs $\mathcal{L}_{\text{total}}$). References resolve ambiguously.
- **Recommended Action:** Keep one definition (L634), rename the symbol consistently, and replace L155–158 with a forward reference: "combined as in Eq.~\eqref{eq:joint-loss} (Section~\ref{subsubsec:meth-joint-loss})".

### [F-M04] [MAJOR] Empty placeholder subsection "Graph Construction"
- **File & Location:** L415–418
- **Category:** Placeholders & Artefacts
- **Verdict:** CONFIRMED (ToC §4.7.6 present)
- **Quoted Text / LaTeX:** `\subsection{Graph Construction}\label{subsec:meth-graph}` followed only by `% How the sensor network is represented as a graph: nodes, edges, adjacency`
- **Discrepancy / Evidence:** Node definition (station vs directional stream), the direction-aggregation rule, and the Exp. 2 static-graph construction (54 nodes, 30% completeness threshold per the availability figure) are nowhere formalised.
- **Recommended Action:** Populate it. Suggested content:
  > "Each node $i \in V$ is a physical counting station; directional streams at a station are summed when all directions report (otherwise the slot is missing). Experiment~2 uses a static graph over the 54 stations meeting a ≥30\% completeness threshold, with edges defined by [...]; Experiment~3 uses the 81-station learned graph of Section~\ref{subsec:meth-graph-learner}."

### [F-M05] [MAJOR] Promised "masked auto-encoding training strategy" is never described (although it exists in code)
- **File & Location:** L10 (chapter introduction)
- **Category:** Concept Validation
- **Verdict:** CONFIRMED
- **Quoted Text / LaTeX:**
  > "...followed by detailed descriptions of the proposed spatio-temporal forecasting architecture, the masked auto-encoding training strategy, and the uncertainty quantification framework."
- **Discrepancy / Evidence:** No section defines it. The code (`train.py` L590–593; `mask_prob: 0.15`) injects element-wise NaNs into the observation channels, which are then zero-filled (`pipeline.py nan_to_num`) **without** an indicator channel. Imputation MSE is scored on these entries (`harness/evaluate.py`). The mechanism behind the headline imputation metric is therefore undocumented.
- **Recommended Action:** Add §4.8.x "Masked Auto-Encoding": "At each iteration a Bernoulli mask with $p_{\text{mask}} = 0.15$ is sampled independently over $(b, i, t, c)$; masked entries are set to 0 before encoding (no indicator is provided); the imputation metric is computed on masked entries whose ground truth was observed."

### [F-M06] [MAJOR] Zero-inflation reliability diagram compares π with the wrong empirical quantity
- **File & Location:** L666
- **Category:** Math Rigor
- **Verdict:** CONFIRMED (theory; code `compute_zero_inflation_reliability` implements the text)
- **Quoted Text / LaTeX:**
  > "Plots the predicted inflation parameter $\hat{\pi}$ against the empirically observed zero fraction in bins. Points lying on the diagonal $y=x$ indicate well-calibrated zero-inflation estimates."
- **Discrepancy / Evidence:** Under the ZINB model, $P(y=0) = \pi + (1-\pi)\,\text{NB}(0\mid\mu,\theta) \ge \pi$. A perfectly calibrated model lies **above** the diagonal whenever NB(0) > 0, so "on the diagonal" is not the calibration criterion.
- **Recommended Action:** Bin the predicted $\hat{P}(y=0) = \hat\pi + (1-\hat\pi)(\hat\theta/(\hat\theta+\hat\mu))^{\hat\theta}$ against the observed zero fraction. Revise the text: "Plots the predicted total zero probability $\hat{P}(y=0)$ ...".

### [F-M07] [MAJOR] Zero-accuracy definition differs from implementation
- **File & Location:** L656
- **Category:** Math Rigor / Hallucination
- **Verdict:** CONFIRMED
- **Quoted Text / LaTeX:**
  > "...(i.e., where $y=0$ and the model's most probable prediction is zero)."
- **Discrepancy / Evidence:** `harness/evaluate.py` scores a hit when `pi > 0.5`. The ZINB mode is zero whenever $P(0) \ge \max_{y>0}P(y)$, which can hold with π < 0.5. The metric is also recall-only: false positives are ignored. Moreover, all y = 0 targets are null-derived (F-C03), so the metric measures prediction of *missingness*, not of genuine inactivity.
- **Recommended Action:** Either implement the mode or rewrite: "Zero-recall: the fraction of targets with $y=0$ for which $\hat\pi > 0.5$. Because the raw data contains no explicit zeros, $y=0$ arises only from missing observations; this metric therefore measures the model's ability to anticipate non-reporting."

### [F-M08] [MAJOR] PIT for a discrete distribution requires randomisation; not stated
- **File & Location:** L665
- **Category:** Math Rigor / Concept Validation
- **Verdict:** CONFIRMED (code `export_ui_payload._zinb_randomized_pit` does randomise; text does not)
- **Quoted Text / LaTeX:** "A perfectly calibrated model produces a uniform (flat) PIT histogram."
- **Discrepancy / Evidence:** For count data the non-randomised PIT is not uniform even under perfect calibration. There is no citation and no definition.
- **Recommended Action:** Add: "For the discrete ZINB we use the randomised PIT $u = F(y-1) + v\,[F(y) - F(y-1)]$, $v \sim U(0,1)$ \citep{czadoPredictiveModelAssessment2009}."

### [F-M09] [MAJOR] Loss masking claim is false for the random (transductive) holdout
- **File & Location:** L569; L607–612; L625
- **Category:** Hallucination / Math Rigor
- **Verdict:** CONFIRMED
- **Quoted Text / LaTeX:**
  > "The imputation and forecasting losses are computed only over \emph{observable} nodes, while the kriging loss is computed only over held-out nodes."
- **Discrepancy / Evidence:** `compute_loss` slices imputation and forecast terms by `loss_node_mask` (the fixed *inductive* holdout) only. The per-batch random holdout (`spatial_holdout_mask`) is used solely for the kriging term, so zeroed nodes still contribute to $\mathcal{L}_{\text{imp}}$ and $\mathcal{L}_{\text{fore}}$. The normalisation $1/(|\mathcal{O}|\cdot T)$ in the MSLE equation is therefore also incorrect.
- **Recommended Action:** Implement `loss_node_mask & ~spatial_holdout_mask` for the imputation and forecast terms, or revise L569: "Imputation and forecasting losses exclude the fixed inductive holdout set; nodes in the per-batch random holdout contribute to all three terms."

### [F-M10] [MAJOR] Imputation loss and metric are computed against zero-fill values at null positions
- **File & Location:** L377; L535; L653
- **Category:** Math Rigor / Cross-Chapter Consistency (missing-data treatment)
- **Verdict:** CONFIRMED
- **Quoted Text / LaTeX:**
  > "This fill is intentionally uninformative: the model is expected to rely on the mask channels, not on the fill value" (L377)
  > "This head reconstructs the \emph{entire} input sequence [...] providing a dense training signal." (L535)
  > "Mean Squared Error (MSE) is used to evaluate the reconstruction of the observable context window." (L653)
- **Discrepancy / Evidence:**
  - `target_impute = clean_dyn.clone()` contains zero-filled nulls (raw 0 → scaled −μ/σ, **not** 0.0 as L377 implies). The imputation MSLE is unmasked, so the model is trained to *reproduce* fill zeros, which is the opposite of "uninformative".
  - The imputation MSE metric is computed on the artificially masked entries (`node_mask`), not on "the observable context window". It includes masked positions whose ground truth was itself a fill value.
- **Recommended Action:** Mask both loss and metric with `(~structural_mask_total)`. Correct L377 ("nulls are filled with raw 0 prior to scaling") and L653 ("MSE on artificially masked entries whose ground truth was observed").

### [F-M11] [MAJOR] Holdout sampling and sizes differ from text
- **File & Location:** L564; L572
- **Category:** Hallucination (numerical grounding)
- **Verdict:** CONFIRMED
- **Quoted Text / LaTeX:** "a random subset of $p_{\text{holdout}} \cdot N$ nodes (typically $p_{\text{holdout}} = 0.15$)"
- **Discrepancy / Evidence:**
  - `make_spatial_holdout_mask` draws an independent Bernoulli(p) per node, so the count is random and can be 0 (the code guards against this).
  - The production config uses `spatial_holdout_prob: 0.2`, not 0.15. The 0.15 value appears only in `architecture_current.mmd`.
  - The inductive holdout (`inductive_holdout_frac: 0.2`, seed 42) selected **24 of 81 nodes (29.6%)** in all four `riot-full-inductive` manifests, not 16.
- **Recommended Action:** Sample exactly $\lfloor pN \rceil$ nodes without replacement, or state: "each node is held out independently with probability $p = 0.2$ (expected 16.2 of 81); the fixed inductive set comprises 24 stations (29.6\%, seed 42)".

### [F-M12] [MAJOR] Target modality inconsistent: "vehicle counts" vs pedestrian data
- **File & Location:** L614; L292; L361; `architecture_current.mmd` ("Raw Unscaled Vehicle Counts")
- **Category:** Cross-Chapter Consistency
- **Verdict:** CONFIRMED
- **Quoted Text / LaTeX:** "...the forecasting loss is computed strictly against the raw, unscaled vehicle counts"
- **Discrepancy / Evidence:** Configs use `target_veh_class: "person"`, Ch. 3 §3.3.3 is "Pedestrian Counting Sensors", and the MDPI paper concerns pedestrians. L171 says "pedestrian and vehicle flow".
- **Recommended Action:** Replace "vehicle counts" with "pedestrian counts" (L614), and state the class explicitly in §4.7.1: "All reported experiments use the \texttt{person} class".

### [F-M13] [MAJOR] Pervasive notation collisions
- **File & Location:** Throughout Ch. 4 and the Appendix
- **Category:** Math Rigor / Notation
- **Verdict:** CONFIRMED
- **Evidence (non-exhaustive):**

  | Symbol | Meaning 1 | Meaning 2 | Meaning 3 |
  |---|---|---|---|
  | $H$ | hidden dim (L83, L137) | forecast horizon $\{t+1..t+H\}$ (L222) | $\mathcal{H}$ held-out set (L150) |
  | $K$ | attention heads (L76, L487) | context length $\{t-K+1..t\}$ (L222) | softmax index $k$ (L65) collides with head index $k$ |
  | $\mathbf{a}$ | attention vector $\mathbf{a}^{(k)}$ | edge attribute $\mathbf{a}_{ij}$ (**same equation**, L483) | — |
  | $e$ | attention logit $e_{ij}$ | edge feature $\mathbf{e}_{ij}$ (L51) | node embedding $\mathbf{e}_i$ (L439) |
  | $E$ / $\mathbf{E}$ | edge set | embedding matrix (L446) | — |
  | $D$ | input features (L134) | $D_e$ edge dim, $D_{\text{ts}}$ | index of dispersion "$D \gg 1$" (L596, undefined) |
  | Tensor layout | $B{\times}N{\times}T{\times}D$ (L134) | $B{\times}T{\times}N{\times}H$ (L494) | $T{\times}N{\times}F$ (App:L36) |
- **Recommended Action:** Add a notation table at the start of §4.2. Rename: horizon → $h$ (max $h_{\max}$); context length → $T$ (drop $K$ at L222); heads → $K$; edge attribute → $\mathbf{r}_{ij}$ (keep $\mathbf{a}$ for attention); node embedding → $\mathbf{q}_i$ / $\mathbf{Q}$; dispersion index → $\text{ID}$ with a definition. Fix App:L36 to $B{\times}N{\times}T{\times}D$.

### [F-M14] [MAJOR] GAT/GATv2 weight dimensions inconsistent in preliminaries
- **File & Location:** L38; L43; L54; L72
- **Category:** Math Rigor
- **Verdict:** CONFIRMED
- **Quoted Text / LaTeX:** "$\mathbf{W} \in \mathbb{R}^{F' \times F}$" (L38), then "$\mathbf{W}[\mathbf{h}_i \| \mathbf{h}_j]$" (L43, L54)
- **Discrepancy / Evidence:** $[\mathbf{h}_i\|\mathbf{h}_j] \in \mathbb{R}^{2F}$ cannot be multiplied by an $F'\times F$ matrix. In GATv2, $\mathbf{W} = [\mathbf{W}_l\,\|\,\mathbf{W}_r] \in \mathbb{R}^{F'\times 2F}$, and aggregation uses $\mathbf{W}_r\mathbf{h}_j$ (L72).
- **Recommended Action:** After L43 insert: "where, for GATv2, $\mathbf{W} = [\mathbf{W}_l \,\|\, \mathbf{W}_r] \in \mathbb{R}^{F' \times 2F}$ and the aggregated message in Eq.~\eqref{eq:gat-aggregate} uses $\mathbf{W}_r\mathbf{h}_j$."

### [F-M15] [MAJOR] Forecast horizon defined inconsistently
- **File & Location:** L222 vs L401, L513; cross-chapter
- **Category:** Cross-Chapter Consistency
- **Verdict:** CONFIRMED
- **Quoted Text / LaTeX:** "...to predict future traffic states $\{t+1, \dots, t+H\}$" (L222) vs "raw unscaled traffic counts at the next time step" (L401)
- **Discrepancy / Evidence:** RIoT is strictly one-step (h = 1, 15 min). Exp. 1 uses h = 8 and 24; Exp. 2 uses h ∈ {1, 2, 4}. The operational scenarios in Ch. 1 (drone fleet, crowd management) imply multi-step lead times.
- **Recommended Action:** State: "Experiment~3 forecasts one step ($h=1$, 15 minutes) ahead; multi-step forecasting ($h \in \{1,2,4\}$, $h \in \{8,24\}$) is studied in Experiments~2 and~1 respectively."

### [F-M16] [MAJOR] Decoder output dimensionality inconsistent
- **File & Location:** L148, L150 vs L535, L553
- **Category:** Math Rigor
- **Verdict:** CONFIRMED
- **Quoted Text / LaTeX:** "$g_{\text{imp}}(\mathbf{Z}) \in \mathbb{R}^{B \times N \times T \times D}$ reconstructs the full input sequence" vs "$D_{\text{ts}} = 3$ [...] excluding the mask channels"
- **Discrepancy / Evidence:** D = 5 (input) but the decoders output D_ts = 3 (`time_series_dim: 3`).
- **Recommended Action:** Replace $D$ with $D_{\text{ts}}$ at L148 and L150, and "full input sequence" with "the observation channels of the input sequence".

### [F-M17] [MAJOR] Static-feature provenance, radii and definitions misreported
- **File & Location:** L304–344
- **Category:** Hallucination (numerical grounding) / Concept Validation
- **Verdict:** CONFIRMED (`src/features.py`)
- **Evidence:**
  - L331–341: the "Network Centrality Metrics (5 features)" from "the OpenStreetMap pedestrian walking network" includes `nearby_bus_stops` (Google Places `bus_station`, 300 m) and `nearby_parking` (Google Places `parking`, 500 m). Only 3 features are OSM centralities; the code docstring says "Google Places API results (12 features) with OSMnx centrality (3 features)".
  - L327: `is_school_zone` uses a **250 m** radius for type `school` only (text: 200 m, "educational facility").
  - L328: distance is **haversine** (great-circle), not "Euclidean".
  - L326: entropy is **normalised** to [0, 1] (÷ log₂ 6).
  - L337: betweenness is **approximated** from k = 500 sampled sources.
  - "fixed radius" is unspecified; the code uses 500 m for POI categories, with Text Search pagination for transit, retail, and restaurants.
- **Recommended Action:** Regroup as "POI (7) / land-use & context (5: entropy, school zone, distance, bus stops, parking) / OSM centrality (3)". State all radii (500 m POIs; 250 m schools; 300 m bus; 500 m parking), "haversine distance", "normalised Shannon entropy $H/\log_2 6$", and "betweenness approximated with $k=500$ sampled sources".

### [F-M18] [MAJOR] "Lessons learned" present empirical findings without evidence
- **File & Location:** L254; L257
- **Category:** Hallucination (unsubstantiated claims)
- **Verdict:** CONFIRMED (no table, figure, or ablation cited; Ch. 7 §7.5.1–7.5.2 have no content in the ToC)
- **Quoted Text / LaTeX:**
  > "Early investigations in this research revealed that deterministic pre-imputation propagates synthetic errors directly into the forecasting model."
  > "Relying strictly on fixed Euclidean distance matrices fails to capture non-linear urban functional similarities..."
- **Discrepancy / Evidence:** No experiment in the recoverable material compares two-stage pre-imputation vs joint training, or static vs learned adjacency, for RIoT. `configs/train/ablation/` and `train_riot_no_impute_inductive.yaml` exist, but no results are referenced.
- **Recommended Action:** Cite the specific ablation (table/figure) or rephrase as motivation: "Prior work suggests \citep{...} that deterministic pre-imputation can propagate [...]; this hypothesis is tested in Section~\ref{sec:exp3-ablation}."

### [F-M19] [MAJOR] Experiment 1 summary misdescribes the MDPI study
- **File & Location:** L237–238
- **Category:** Cross-Chapter Consistency / Hallucination
- **Verdict:** CONFIRMED (`docs/mdpi_sensors/mdpi_sensors.tex`)
- **Quoted Text / LaTeX:**
  > "...non-spatial recurrent deep learning models (LSTM, GRU), demonstrating that naive preprocessing heuristics (such as global mean imputation or deletion) introduce severe distributional distortion under structural missingness."
- **Discrepancy / Evidence:**
  - The MDPI paper evaluates persistence, seasonal naïve (1 d, 7 d), ARIMA, and UV/MV LSTM across **110 sensors**; GRU appears only as a hyperparameter-search option.
  - Interpolation variants are none / night_zero / linear / spline / kNN, not "global mean imputation".
  - The paper's headline finding is that **sequence-length structure, not completeness**, drives error (breakpoint ≈ 3 h; RF OOB R² = 0.81), not "distributional distortion".
  - GRU and Transformer forecasters were added to the codebase later (commits 7b4e925, d738644), so Ch. 5 must state which version is reported.
- **Recommended Action:** Replace with:
  > "Experiment~1 benchmarks persistence, seasonal-naïve, ARIMA and univariate/multivariate LSTM forecasters on 110 directional pedestrian streams (53 stations) at 2-hour and 6-hour horizons, preserving gaps as missing, and shows that the temporal structure of missingness (mean contiguous-sequence length) rather than overall completeness governs forecast error."

### [F-M20] [MAJOR] ZINB motivation internally contradictory (overdispersion "induced by true zeros"; π semantics)
- **File & Location:** L582; L596; L598; L375
- **Category:** Concept Validation / Math Rigor
- **Verdict:** CONFIRMED
- **Quoted Text / LaTeX:**
  > "...capturing the severe overdispersion ($D \gg 1$) induced by the presence of true zeros." (L596)
  > "...the complete absence of explicit zero-count reporting in the raw data." (L582)
- **Discrepancy / Evidence:**
  - The data has 0 explicit zeros (verified), so "true zeros" cannot induce overdispersion. D is undefined, and no dispersion statistic is reported.
  - L596 assigns π to "structural zero (sensor failure)". Yet all y = 0 targets are null-derived, both night (L375 "sampling") and day ("structural"), so the NB(0) component has no observed genuine zeros to learn from and the π/NB split is **not identifiable** from the data.
  - L598, "missing values are mapped to 0 during evaluation", also applies during *training*.
- **Recommended Action:** Report the empirical variance-to-mean ratio of observed counts per station. Remove "induced by the presence of true zeros". Add an identifiability caveat: "Because no explicit zeros are observed, the decomposition of $P(y=0)$ into $\pi$ and $(1-\pi)\text{NB}(0)$ is not identifiable from data alone; $\pi$ is interpreted as the probability of non-reporting."

### [F-M21] [MAJOR] Undefined constructs and missing citations
- **File & Location:** L564 (IGNNK), L596–606 (ZINB, NB pmf), L621 (MSLE), L665–667 (PIT, reliability diagram), L719 (Optuna), L723 (W&B)
- **Category:** Concept Validation
- **Verdict:** CONFIRMED (the PDF references page lists only a handful of entries; none for these)
- **Quoted Text / LaTeX:** "this thesis adopts an IGNNK-style node masking protocol"
- **Discrepancy / Evidence:**
  - IGNNK (Wu et al., AAAI 2021) is not cited.
  - The NB pmf in the (μ, θ) parameterisation is used but never written (the code docstring has it).
  - ZINB (Lambert 1992) is not cited.
  - MSLE is labelled "scale-invariant" without qualification. log(1+y) is only approximately scale-invariant for y ≫ 1 and behaves like MSE near 0.
- **Recommended Action:** Add $\text{NB}(y\mid\mu,\theta)=\frac{\Gamma(y+\theta)}{\Gamma(\theta)\,y!}\left(\frac{\theta}{\theta+\mu}\right)^{\theta}\left(\frac{\mu}{\theta+\mu}\right)^{y}$ with $\mathbb{E}[y]=\mu$, $\text{Var}[y]=\mu+\mu^2/\theta$. Add citations for IGNNK, ZINB, PIT, and Optuna. Qualify MSLE as "approximately scale-invariant for counts ≫ 1".

### [F-M22] [MAJOR] Chapter summary is a stub and points to the wrong next chapter
- **File & Location:** L732
- **Category:** Placeholders & Artefacts / Cross-Chapter Consistency
- **Verdict:** CONFIRMED
- **Quoted Text / LaTeX:** "This chapter has presented the research methodology employed in this thesis. The following chapter describes the data and study areas used for experimental evaluation."
- **Discrepancy / Evidence:** Data is Chapter 3 (before Ch. 4); Chapter 5 follows (Experiment 1). The header comment (L3) also says "Chapter 3: Methodology", which suggests a reordering was never propagated.
- **Recommended Action:** Replace with a substantive summary. Suggested ending: "...The following chapter (Chapter~\ref{ch:exp1}) applies the single-sensor components of this methodology in Experiment~1."

### [F-M23] [MAJOR] Compiled PDF is stale vs source; figure assets missing; experiment chapters appear to be skeletons
- **File & Location:** `thesis.pdf` ToC; L579–591; `figures/methodology/`
- **Category:** Placeholders & Artefacts / Grounded Origins
- **Verdict:** CONFIRMED
- **Discrepancy / Evidence:**
  - The PDF lacks the "Loss Formulation and Uncertainty Quantification" subsection, the "Null-Zero Conflation" subsubsection, and Figure `fig:ambiguity_of_missing_data`. In the PDF, the ZINB/MSLE subsubsections nest under §4.8.6 "Spatial Holdout", which is a structural error in the compiled document.
  - Neither `ambiguity_of_missing_data.png` nor `conceptual_pipeline_framework.pdf` exists anywhere in the repository, and no script generates them.
  - Page numbering regresses (Ch. 4 ends p. 73; Ch. 5 starts p. 67).
  - Every Ch. 5 subsection is on p. 67 and every Ch. 6 subsection on p. 69, which indicates empty skeleton chapters.
- **Recommended Action:** Commit all figure sources and their generating scripts under version control. Recompile the full document with a clean `.aux`. Complete Chapters 5–6 before submission.

### [F-M24] [MAJOR] "FP32 mixed-precision autocast" is a contradiction and misreports the implementation
- **File & Location:** L724
- **Category:** Hallucination
- **Verdict:** CONFIRMED
- **Quoted Text / LaTeX:** "...combined with gradient clipping and FP32 mixed-precision autocast for numerical stability."
- **Discrepancy / Evidence:** `train.py` uses `torch.autocast(dtype=torch.bfloat16)` gated by `amp`, and `train.yaml` sets `amp: false` by default. The loss is kept in FP32.
- **Recommended Action:** "...gradient clipping (max norm 1.0). Optional bfloat16 autocast is available but was [enabled/disabled] for all reported runs; the loss is always computed in FP32."

### [F-M25] [MAJOR] "Reconstruction is a spatial problem" is not what the architecture implements
- **File & Location:** L223; L514–517
- **Category:** Diagram Fidelity / Unsubstantiated generalisation
- **Verdict:** CONFIRMED
- **Quoted Text / LaTeX:**
  > "...reconstruction performance measures how well the spatial encoder's message passing recovers information at nodes with missing or masked inputs"
- **Discrepancy / Evidence:** The imputation and kriging heads consume **Z**, the output of the bidirectional per-node Transformer, which mixes all T time steps of the node. Masked entries can therefore be recovered from the node's own adjacent timesteps (temporal interpolation), not only from spatial message passing. L223 ("at time slice t") conflicts with the heads operating on the full window.
- **Recommended Action:** Rephrase: "Reconstruction draws on both spatial message passing and within-node temporal context; the spatial contribution is isolated by the ablation in Section~X (e.g., Transformer removed / GATv2 removed)."

### [F-M26] [MAJOR] Node count N is never stated in the Methodology; Appendix example matches no experiment
- **File & Location:** Ch. 4 (entire); App:L45
- **Category:** Cross-Chapter Consistency
- **Verdict:** CONFIRMED
- **Quoted Text / LaTeX:** "Pre-train on a source graph (e.g., $N=29$ sensors)"
- **Discrepancy / Evidence:** The actual node sets are 110 streams / 53 stations (Exp. 1), 54 (Exp. 2), and 81 (Exp. 3). N = 29 appears in no manifest or config.
- **Recommended Action:** Add a table in §4.7 listing N, unit, horizon, split, and missing-data policy per experiment (see F-C10). Change App:L45 to "(e.g., the $N=54$-node graph of Experiment~2)".

### [F-M27] [MAJOR] Model acronym and role inconsistent
- **File & Location:** L244, L683 ("RIOT"); ToC §7.2.1 "Baseline Architecture: The Resilient Graph Neural Network (RIoT)"
- **Category:** Cross-Chapter Consistency
- **Verdict:** CONFIRMED (ToC) / PLAUSIBLE (glossary entry not available)
- **Discrepancy / Evidence:** The model appears as "RIOT", `\gls{riot}`, and "RIoT". The Ch. 7 expansion "Resilient Graph Neural Network" does not produce the acronym. The model is the *proposed* method in Ch. 4 but is called the "Baseline Architecture" in Ch. 7.
- **Recommended Action:** Use `\gls{riot}` everywhere with a single glossary expansion (e.g., "Resilient IoT spatio-temporal network (RIoT)"). Rename §7.2.1 "Proposed Architecture: RIoT".

### [F-M28] [MAJOR] Capability table contradicts evaluation harness
- **File & Location:** L676; Table `tab:capability_contrast` (L681–697)
- **Category:** Figure/Table Grounding
- **Verdict:** CONFIRMED
- **Quoted Text / LaTeX:** "Zero-Shot Spatial Kriging & $\times$ & \checkmark"
- **Discrepancy / Evidence:** `harness/adapters.py RecurrentBaselineAdapter.capabilities = Capabilities(..., imputation=True, kriging=True, graph=False)`. Baselines therefore report kriging metrics (from their reconstruction head on zeroed nodes). The table also omits Exp. 1 comparators (persistence, seasonal naïve, ARIMA) and the newer Transformer forecaster, and "Temporal Imputation ✓ (MSE)/(MSLE)" mixes loss choice with capability.
- **Recommended Action:** Change the baseline kriging cell to "naïve (per-node reconstruction of zeroed input)", add rows for loss functions, and list all comparators used in Ch. 5–7.

### [F-M29] [MAJOR] Mask-channel "structured prior" is asserted, not validated
- **File & Location:** L375
- **Category:** Unsubstantiated claim
- **Verdict:** CONFIRMED
- **Quoted Text / LaTeX:**
  > "...if both channels are active, the missingness is plausibly \emph{sampling} (genuine inactivity); if only Channel~3 is active (daytime \texttt{null}), the missingness is more likely \emph{structural} (sensor failure)."
- **Discrepancy / Evidence:** Channel 4 is a strict subset of Channel 3 by construction (`dataset.py`), so it encodes only "hour ∈ [0, 5)" given null, i.e. a time-of-day indicator. No evidence (e.g., gap-length distributions by hour, Ch. 3 §3.4.1) is cross-referenced. The 00:00–05:00 cut-off is unjustified, and the "MNAR" label is asserted without a test.
- **Recommended Action:** Cite the Ch. 3 analysis supporting the cut-off, e.g.: "Chapter~\ref{ch:data} (Figure~X) shows that Y\% of nulls between 00:00 and 05:00 belong to gaps shorter than 5 h recurring daily, consistent with non-reporting under low activity." Rename the channels "null indicator" and "night-null indicator".

---

### [F-m01] [MINOR] Header comment mislabels the chapter — L3: `% Chapter 3: Methodology`. Change to "Chapter 4".
### [F-m02] [MINOR] Delimiter typo in Eq. `gat-multihead` — L79 `\Bigr\|` should be `\Big\|` (as in L490).
### [F-m03] [MINOR] Section ordering — "Theoretical Preliminaries" (§4.2) precedes "Research Philosophy" (§4.3) and "Conceptual Framework" (§4.4). Conventional order is philosophy → design → framework → theory → pipeline.
### [F-m04] [MINOR] Edge-feature normalisation and self-loops misdescribed — L469 says "min-max normalised"; the code divides by the global max shortest-path distance (no min subtraction), with the Euclidean distance also divided by the *shortest-path* max. Self-loops (diagonal of $\mathbf{E}\mathbf{E}^\top$, plus PyG `add_self_loops=True` with mean-filled edge_attr) are undisclosed. Suggested: "normalised by the maximum shortest-path distance over all pairs; self-loops are added by the GATv2 layer with mean-imputed edge attributes."
### [F-m05] [MINOR] Clamp rationale — L404 attributes the `y ≥ 0` clamp to "floating-point artefacts introduced by the zero-fill step". A raw 0.0 fill cannot create negative artefacts; the clamp is a defensive guard (`dataset.py` comment). Reword accordingly.
### [F-m06] [MINOR] Redundant sample component — L400 lists $\mathbf{M}$ separately although it is already Channels 3–4 of $\mathbf{X}$ (L399). State that $\mathbf{X}_{:,:,3:5} \equiv \mathbf{M}$ (in code the masks are concatenated inside the model).
### [F-m07] [MINOR] Scaler wording in Table `tab:encoder-channels` — "Dedicated StandardScaler" for humidity and "(shared with Ch. 1)" for solar radiation reads as contradictory. Use "Weather scaler (joint, Ch. 1–2)" for both rows.
### [F-m08] [MINOR] Unpinned versions — L711 "PyTorch (v2.0)" but `requirements.txt` pins neither `torch` nor `torch-geometric`. Report exact versions (`pip freeze` of the training image). Next.js is 16.2.4 and deck.gl 9.3.1 (`frontend/package.json`).
### [F-m09] [MINOR] SQLite for multi-worker parallel sweeps (L719) — SQLite locks under concurrent writers. Confirm this is how sweeps were run, or state that workers were serialised.
### [F-m10] [MINOR] Eq. `spatial-output` omits σ whereas Eq. `gat-multihead` includes it. Consistent with code (no activation after GATv2Conv) but inconsistent with §4.2; note it explicitly.
### [F-m11] [MINOR] Unlabelled MSLE equation (L622–624) cannot be referenced; add `\label{eq:msle}`.
### [F-m12] [MINOR] Orphaned table — `tab:encoder-channels` is never `\ref`'d. Add "(Table~\ref{tab:encoder-channels})" at L351.
### [F-m13] [MINOR] Duplicate figure assets — identical sensor-availability figures are stored in `figures/methodology/`, `figures/experiment2/`, and `figures/experiment3/`. Keep one canonical copy to avoid divergence.
### [F-m14] [MINOR] `architecture_current.mmd` inaccuracies — it shows "Artificial NaN Masking → StandardScalers" (code scales in the dataset and masks later in the training loop), `p=0.15` (config 0.2), and "Min-Max Normalised" (÷max).
### [F-m15] [MINOR] Availability-figure caption grammar — "27 Locations Excluded from Graph (Failed ≥30% completeness threshold)" should read "(below the 30% completeness threshold)".
### [F-m16] [MINOR] Ablation example conflates factors — L190: "substituting GATv2 spatial encoding with an independent per-node recurrent model" changes both the spatial and temporal encoder. Suggest "removing the GATv2 layer (identity spatial mapping) while retaining the Transformer".
### [F-m17] [MINOR] ToC typesetting — "GA Tv2" and "V alidation" kerning artefacts (font/ligature issue). Use `\mbox{GATv2}` or check the font setup.
### [F-m18] [MINOR] Appendix not linked from Chapter 4 — `ch:appendix-design-rationale` is never referenced from the methodology, although its App:L105 choice (graph passed to `forward`) underpins §4.8. Add a reference at L457.

---

## 4. Diagram & Schematic Verification Notes

### 4.1 Figure 4.1 — `conceptual_pipeline_framework.pdf` (rendered from `thesis.pdf` p. 59)
- **Depicted:** Data Ingestion (database icon) → Dynamic Graph Learning (graph icon) → "Spatio-Temporal Forecasting" (stacked MHSA/FFN block) → Decoupled Multi-Task Decoders (clip-art target icon). Single serial arrows.
- **Against equations (§4.8):**
  - Missing Component 2 (GATv2, Eq. `gatv2-edge-conditioned`).
  - Missing the static/dynamic input split (Eq. `graph-projection` takes $\mathbf{S}$ only).
  - Missing the mask concatenation (Table `tab:encoder-channels`) and edge enrichment (Eq. `edge-attr`).
  - Missing the three heads and their outputs (Eqs. `imputation-decoder`, `forecast-decoder`, `kriging-decoder`).
- **Against implementation:** The serial arrow from the graph learner into the Transformer block is spurious. The graph learner outputs topology into the GATv2 layer, and dynamic data never passes through it.
- **Verdict:** Does not meet the caption's claims. **[CRITICAL] F-C09.**

### 4.2 Figure `ambiguity_of_missing_data.png`
- The asset is not present in the repository and is absent from the compiled PDF. It cannot be verified. There is no generating script, and its claims (two regimes: diurnal vs aperiodic multi-day) are not quantified in Ch. 4. **[MAJOR] F-M23.**

### 4.3 `docs/poster/riot_architecture.{png,svg,pdf}` (likely Ch. 7 §7.2.1 asset)
- **Correct:**
  - Graph-learner pipeline (static features → $W_{\text{static}}$ projection → pairwise similarity → ReLU/threshold → edge list).
  - Edge Feature Table ⊕ learned weight → 10-dim edge attributes.
  - Clean dynamic ⊕ MNAR masks → GATv2 → flatten → TransformerEncoder → $\mathbf{Z}\in[B,N,T,H]$.
  - Three MLP heads, ZINB (μ, θ, π) with ε-clamping, joint loss, and inductive loss-node mask gating.
  - Sweep → Training → W&B → Model Store.
- **Incorrect:**
  - **MLP_fore / MLP_imp labels swapped.**
  - Held-out edge-drop (not implemented).
  - "GATv2Conv layers" (single layer in code).
  - "(N,N,1)" edge-weight shape.
  - Imputation loss shown as "MSE" (it is MSLE + MSE).
  - Forecasting head drawn from full Z (it uses the last step only).
  - No positional-encoding block. This is consistent with the code but should be flagged as a limitation.
  - Editing artefacts ("Horizontal Container", "simluated").
- **Verdict:** **[CRITICAL] F-C11.**

### 4.4 `docs/diagrams/architecture_current.mmd`
- Broadly faithful to the code structure (batch dict, 5-channel encoder input, `[E,10]` edge attributes, three decoders).
- Incorrect: "Drop Outgoing Edges from Simulated & Held-out Nodes" (F-C04), `p=0.15` (config 0.2), masking-before-scaling order, "Min-Max Normalised", and "Raw Unscaled Vehicle Counts" (F-M12).

### 4.5 Sensor-availability figures (`scripts/plot_experiment_sensor_availability.py`, `plot_experiment1_split_comparison.py`)
- **Grounded:** Node counts (110 streams / 53 locations; 54 graph nodes; 81 stations) match `artifacts/quality_full_v2/sensor_manifest.json`, `artifacts/spatial_full/context_manifest.json`, and `train.yaml`. The Exp. 2 calendar split matches the context manifest (≈70/15/15 of 119,712 steps).
- **Not grounded:** The Exp. 3 calendar split (RIoT uses `random_split`) and "100% of 81 physical locations active". Script docstring N = 110 vs figure N = 81. **[CRITICAL] F-C12.**

### 4.6 Tensor-shape and index-convention audit (Ch. 4 vs code)

| Stage | Thesis | Code | Match |
|---|---|---|---|
| Encoder input | $B{\times}N{\times}T{\times}5$ (L351) | `[B,N,T,3+2]` (`encoder.py` assert) | ✔ |
| GATv2 working layout | $B{\times}T{\times}N{\times}H$ (L494) | `permute(0,2,1,3)` → `[B·T·N, D]` → `[B,T,N,H]` | ✔ |
| Transformer layout | $[T, B\cdot N, H]$ (L501) | `batch_first=False`, `[T, B·N, H]` | ✔ |
| Latent | $B{\times}N{\times}T{\times}H$ | `[B,N,T,H]` | ✔ |
| Per-head edge projection | $(D_{\text{in}}/K){\times}10$ | $(H/K){\times}10$ | ✘ F-C07 |
| Imputation output | $D$ (L148) / $D_{\text{ts}}=3$ (L535) | `time_series_dim=3` | ✘ F-M16 |
| Forecast head input | $\mathbf{Z}_{:,:,-1,:}$ | `latent[:, :, -1, :]` | ✔ |
| Edge attribute | 9 physical + 1 learned | `[E,10]` | ✔ |
| Positional encoding | not mentioned | absent | ✘ F-C08 (omission) |

---

## 5. Prioritized Action Checklist

**P0: Blocks submission (invalidates reported results; requires re-running Exp. 3)**
- [ ] **F-C01 / F-C02:** Replace `random_split` with a chronological, purged train/val/test split. Fit all scalers on observed training-period values only. Re-run RIoT and the baselines; regenerate all Ch. 7 tables and figures.
- [ ] **F-C03:** Fix direction aggregation (`min_count=1` plus a partial-null policy). Report the resulting mask rates in Ch. 3/4. Re-run.
- [ ] **F-C04 / F-C05:** Implement held-out edge dropping, blank or flag mask channels for held-out nodes, and control self-loops. Re-run the kriging evaluation.
- [ ] **F-C06:** Evaluate point forecasts with $\hat\mu(1-\hat\pi)$ (or change the text and justify μ). Regenerate the metrics.
- [ ] **F-M09 / F-M10:** Exclude random-holdout nodes and null positions from $\mathcal{L}_{\text{imp}}$, $\mathcal{L}_{\text{fore}}$, and the imputation metric.
- [ ] **F-C08:** Add positional/time encodings (or document the permutation-equivariant design and remove the diurnal/trend claims).

**P1: Must fix in text and figures before submission**
- [ ] **F-C07, F-M14, F-M16:** Correct the GATv2 dimensions and decoder output dimensions.
- [ ] **F-C09, F-C11, F-C12:** Redraw Figure 4.1, fix the RIoT schematic labels, and regenerate or relabel the availability figures.
- [ ] **F-C10, F-M15, F-M26:** Add a per-experiment protocol table (N, unit, horizon, split, missing-data policy). Remove or soften "controlled comparison" claims.
- [ ] **F-M06, F-M07, F-M08:** Correct the reliability-diagram reference quantity, the zero-accuracy definition, and the randomised PIT.
- [ ] **F-M18, F-M19, F-M20:** Ground or soften the "lessons learned". Correct the Exp. 1 summary to match the MDPI paper. Fix the ZINB motivation and add the identifiability caveat.
- [ ] **F-M17, F-M11, F-M12, F-M24:** Correct static-feature provenance and radii, holdout sampling and sizes, "vehicle" → "pedestrian", and the AMP statement.
- [ ] **F-M05:** Document the masked auto-encoding strategy.
- [ ] **F-M21:** Add the NB pmf and the citations (IGNNK, ZINB, PIT, Optuna).

**P2: Structural and editorial**
- [ ] **F-M01, F-M02, F-M03, F-M04:** Remove the duplicate sections and labels; populate "Graph Construction".
- [ ] **F-M22, F-M23:** Rewrite the chapter summary; commit the missing figure assets and scripts; recompile from a clean build; complete the Ch. 5/6 skeletons.
- [ ] **F-M13:** Add a notation table and resolve the symbol collisions across chapters and the appendix.
- [ ] **F-M25, F-M27, F-M28, F-M29:** Qualify the "reconstruction is spatial" claim; unify the RIoT acronym and role; fix the capability table; justify the night-mask prior.

**P3: Minor polish**
- [ ] **F-m01 – F-m18:** Header comment, `\Bigr` typo, section order, normalisation and self-loop wording, clamp rationale, redundant $\mathbf{M}$, scaler wording, version pinning, SQLite note, σ note, equation labels, table `\ref`, duplicate assets, `.mmd` fixes, caption grammar, ablation wording, ToC kerning, appendix cross-reference.

**P4: Re-audit**
- [ ] Mount the full LaTeX tree and re-run this audit on Chapters 1–3 and 5–7, including every reported metric (MAE, RMSE, CVRMSE, MASE, R², DM p-values, the "≈3 h breakpoint", "OOB R² = 0.81", and "92–98% of sensors significant") against `artifacts/**` outputs.
