# Final pre-submission review: Chapters 4, 5 and 6 (2026-10-13)

Read-only review of `docs/thesis/full/chapters/chapter{4,5,6}_*.tex` after the changes from `implementation_ch5_ch6_tier2_figures_2026-10-13.md` were applied. Line numbers refer to the current files (Ch4: 935 lines, Ch5: 580, Ch6: 714). Each OLD string is quoted so it can be found even if the line numbers shift.

Severity:
- **BLOCKER**: a factual statement that contradicts the thesis's own results or the code. Fix before sending.
- **SHOULD-FIX**: overclaims or internal inconsistencies a supervisor is likely to notice.
- **MINOR**: style, hyperbole or hygiene.

## Verdict

| Chapter | State | Blockers | Comment |
|---|---|---|---|
| 4 Methodology | **Not ready** | 8 | The Ch5/Ch6 corrections have not reached Ch4: it still states the old Experiment 1 headline ("sequence continuity governs", R² > 0.80, Runge oscillations) and the old Experiment 2 framing ("representation collapse", walking-corridor graphs). All fixes are text-only. |
| 5 Experiment 1 | **Ready after 1 fix** | 1 | 26/28 brief items applied correctly; Limitations verbatim; no v1 numbers left in the text. The remaining blocker is a figure (Fig. 5.5) built from the mixed v1/v2 data. |
| 6 Experiment 2 | **Ready after 2 fixes** | 2 | New figures, Limitations, references and numbers all check out. Two sentences carry pre-correction claims. |

LaTeX hygiene in all three chapters is clean:
- every `\ref` resolves;
- no duplicate labels;
- all `\gls` and `\cite` keys exist;
- all figure files exist;
- environments are balanced.

Estimated effort: about 1 hour of text edits, plus regenerating one figure.

---

## Chapter 4: Methodology

### BLOCKERS

**4-B1. L262: Experiment 1 summary states the superseded headline.** Replace the passage from `The core empirical finding demonstrates that the temporal structure of missingness (mean contiguous sequence length), rather than aggregate data completeness, governs forecast performance` up to and including `This demonstrates that` with:

```latex
Relative forecast error is associated with both mean contiguous sequence length and aggregate completeness, which are strongly correlated ($\rho = 0.78$); segmented regression gives a breakpoint at $2.95~\text{h}$ (95\% CI 1.90--3.42~h) at $h=8$ only. A random forest using four quality features explains almost none of the cross-sensor variance in CV-RMSE (out-of-bag $R^2 = 0.04$ at $h=8$, $-0.01$ at $h=24$). This indicates that
```

Chapter 5 no longer reports R² > 0.80 or 0.25–0.37 (Ch5 L14, L510).

**4-B2. L39, L323, L393: "proving / governs".**
- L39 and L323: `proving that consecutive sequence continuity governs predictive utility` → `indicating that sequence continuity and completeness are both associated with forecast error`
- L393: `established that consecutive contiguous sequence continuity, rather than aggregate completeness, governs predictive utility` → `found that forecast error is associated with both sequence continuity and completeness`

**4-B3. L389 and L346: spline Runge oscillations cannot occur.** Every quality config sets `max_gap_steps: 8`, so linear and spline fill only gaps ≤ 2 h. Chapter 5 shows spline performing like linear interpolation.
- L389, replace the whole `\item \textbf{Cubic spline interpolation} ...` with:
  ```latex
  \item \textbf{Cubic spline interpolation} was limited to gaps of at most 8 steps ($2~\text{h}$) and performed similarly to linear interpolation (Table~\ref{tab:exp1-interpolation}); its apparent RMSE gains reflect additional low-count evaluation windows rather than better forecasts.
  ```
- L346: `cubic splines exhibited runaway polynomial oscillations during prolonged sensor outages` → `interpolation gains reflected sample composition, and $k$-NN raised matched-timestamp CV-RMSE by $4.34~\text{pp}$`

**4-B4. L390: 4.34 is in percentage points, applies to KNN only, and comes from matched pairs.** `Linear and $k$-NN interpolation proved superficially stable on benign, short dropouts, but introduced a 4.34\% noise penalty even across clean verification sequences,` →

```latex
On the 50,051 timestamps shared with the gap-preserving model, $k$-NN imputation raised mean sensor CV-RMSE by $4.34~\text{pp}$ ($\pm 8.49~\text{pp}$; Section~\ref{subsec:exp1-matched-pairs}),
```

(Also change the item label to `\textbf{$k$-NN interpolation}`.)

**4-B5. "Representation collapse" contradicts Ch6 L99** ("precautionary ... no threshold sweep"). Experiment 2 never observed a collapse.
- L40: `forcing the exclusion of peripheral monitoring locations to prevent representation collapse during spatial message passing` → `leading to the precautionary exclusion of 27 sparse locations`
- L324: `without risking representation collapse` → `and so excludes them as a precaution`
- L358: `brittle under sensor outages, forcing the exclusion of 27 candidate stations ($<30\%$ completeness) to prevent representation collapse.` → `27 sparse stations ($<30\%$ completeness) excluded as a precaution.`
- L403: `maintaining training stability under static graph message passing required discarding` → `27 candidate installations were excluded as a precaution, without a threshold sweep, because`. Re-read the sentence afterwards for grammar.

**4-B6. L51: split and tuning description is wrong for Experiments 1 and 2.** `All experiments use a chronological train/validation/test split ... Hyperparameters are tuned exclusively on the validation set.` →

```latex
Experiments~2 and~3 use a global chronological split; Experiment~1 splits each contiguous run chronologically and assigns short runs whole (Section~\ref{subsec:meth-protocols}). Experiments~1 and~2 use fixed hyperparameters set a priori; any model selection uses the validation split only.
```

**4-B7. L571: refers to analyses Chapter 5 does not contain** (no state-space decomposition, no artificial-masking validation). `it is justified using \textbf{state-space decomposition} ... (described in Experiment~1, Chapter~\ref{ch:exp1}).` →

```latex
its effect is assessed by evaluating only on originally observed steps and by a matched-timestamp comparison (Chapter~\ref{ch:exp1}, Section~\ref{subsec:exp1-matched-pairs}).
```

**4-B8. L313, L578: Experiment 2 graphs are Euclidean, not walking graphs.**
- L313 (table): `Static geographic / OSM walking graph` → `Static Euclidean (Gabriel, $k$NN-8, FC) + 9 OSM edge attributes`
- L578: `planar Gabriel proximity graphs representing natural pedestrian walking corridors` → `planar Gabriel graphs built from straight-line (haversine) distances`

### SHOULD-FIX

| # | Line | OLD | NEW |
|---|---|---|---|
| 4-S1 | 578, 359 | `cosine similarity over projected` / `learnable threshold $\tau$` | `dot-product similarity of projected` / `tunable threshold $\tau$` (matches Eq. graph-similarity and L610/617) |
| 4-S2 | 78 | `(with $T=12$)` | `(with $T=12$ in Experiments~2--3 and $T=24$ in Experiment~1)` |
| 4-S3 | 310 | `54 stations ($\ge 30\%$ completeness)` | `54 stations ($\ge 30\%$ completeness; 48 with test targets)` |
| 4-S4 | 346, 388 | `by $>20\%$` / `by more than 20\%` | `by 20--24\% (median RMSE)` / `by 20--24\% in median RMSE` (GRU-MV is 19.5%) |
| 4-S5 | 265 | `isolates how dynamic inter-sensor correlation improves temporal interpolation and robustness during localised sensor dropouts` | `measures the forecasting gain from spatial message passing at 15--60~min horizons` (Exp 2 tested neither dropouts nor interpolation) |
| 4-S6 | 366 | `Exp~2 (\S\ref{sec:exp2-results})` | `theoretical argument (no GAT-v1 ablation was run)` |
| 4-S6 | 376 | `pure recurrent networks (LSTM/GRU) for long-horizon multi-sensor contexts` | `pure recurrent temporal encoders (retained in Exp~2 as GAT-LSTM)` |
| 4-S7 | 647 | `(with $H = 64$, $K = 4 \implies 16 \times 10$)` | `(with $H = 64$ and $K = 4$, so $\boldsymbol{\Theta}_e^{(k)} \in \mathbb{R}^{16 \times 10}$)` |
| 4-S8 | 43 | `Crucially, this` / `rigorously enforced` | `This` / `made` |
| 4-S8 | 403 | `is unacceptable` | `is a substantial loss` |
| 4-S8 | 540 | `A critical design decision` | `A key design decision` |
| 4-S8 | 824 | `To robustly evaluate` | `To evaluate` |
| 4-S8 | 922 | `To ensure rigorous reproducibility` | `To support reproducibility` |
| 4-S8 | 395 | `are expected to significantly improve` | `may improve` |
| 4-S8 | 731, 740 | `Critically,` / `Crucially,` | delete |

### MINOR

- **Spelling:**
  - L792 `standardizing` → `standardising`
  - L852 `modeling` → `modelling`
  - L531, L799 `nighttime` → `night-time`
- **L912:** `"zero host execution"` → `` `zero host execution' `` (LaTeX straight quotes).
- **L347:** `masked auto-encoding (MAE)` → `masked auto-encoding` with no acronym. Bare "MAE" clashes with the error metric `\gls{mae-metric}`.
- **L765:** `To explicitly model this bimodal missingness` → `To model the excess zeros arising from inactivity and non-reporting` ("this" has no antecedent).

---

## Chapter 5: Experiment 1

Status: 26 of 28 brief items applied correctly. E.1 Limitations were inserted verbatim and no v1 statistics remain in the text.

### BLOCKER

**5-B1. Fig. 5.5 `figures/experiment1/ext_correlation_heatmap.png` (L277; cited at L255, L282, L328) still uses the mixed v1/v2 data.**
- Its CV-RMSE column shows: sequence −0.52, completeness −0.59, diurnal −0.33, volume −0.40.
- These contradict Table 5.4 (v2). Under v2, sequence length correlates slightly more strongly than completeness, so the figure reverses the ordering the text now reports.
- Fix: regenerate from `artifacts/quality_full_v2` with your heatmap script in `scripts/eda/`. That directory was left untouched because you are editing it.
- Interim alternative: add to the caption at L278: `Correlations with CV-RMSE in this figure are from a preliminary run and differ slightly from Table~\ref{tab:exp1-correlations}, which is authoritative.`

### SHOULD-FIX

| # | Line | Issue | Fix |
|---|---|---|---|
| 5-S1 | 450 | The 5-17b sentence was inserted twice | Delete the second occurrence |
| 5-S2 | 464 | "100% of weekends" is not supported by any computed statistic | Delete the percentage, or replace it with the count of weekends examined |
| 5-S3 | 363 | `as demonstrated by matched-pairs` | `as indicated by the matched-pairs comparison` |
| 5-S4 | 577 | `revealed` | `indicated` |
| 5-S5 | 302 | Causal wording ("drives", "causes") about the breakpoint | `is associated with`; the analysis is cross-sectional |
| 5-S6 | 464 | `provides a critical conceptual insight` | `illustrates` |

### MINOR

- **Hyperbole** ("critical", "crucially", "clearly", "dramatically", "robust") at L32, 34, 53, 133, 136, 199, 282, 300, 325, 347, 363, 439, 496, 529, 575. Delete the intensifier or replace it with the number.
- **L347:** double brackets `((...))`.
- **L160:** `0.1\% relative` → `0.1 percentage points`.
- **L162:** duplicated phrase.
- **L267–270:** negative correlations typed with a hyphen. Use `$-0.54$`.
- **First-use acronyms:** LSTM/GRU (L12), MAPE (L110), ΔAIC (L290), LOWESS (L471).
- **L263:** unit spacing, `3~h` rather than `3h`.

### Open author TODOs (deliberately left for you)

- **L327:** ρ(distance) still needs recomputing on v2.
- **L452:** Heber de-duplication. The processed data contain no duplicate, so state the source of the original duplicate or drop the claim.
- **L459:** April fixture dates need a source.
- **L508–509:** confirm that no R² 0.81 remnant is left.
- **Log-volume regression:** the `corr2.py` script that produced it is not in the repository. Commit it under `scripts/experiment1/` or cite the v2 re-run in `ch5_v2_figures/exp1_verification_scripts/`.

---

## Chapter 6: Experiment 2

All new figures, the E.2 Limitations, references, citations and table numbers check out against `src/spatial` outputs.

### BLOCKERS

**6-B1. L646 still uses the old Experiment 1 headline and ρ = −0.62.** Replace the sentence beginning `In Experiment~1, univariate forecasting accuracy was shown to be strongly bounded by contiguous sequence length` with:

```latex
In Experiment~1, relative forecast error was associated with both mean contiguous sequence length (breakpoint $\approx 3.0$~h at $h=8$; $W=24$) and observation completeness ($\rho = -0.54$ at $h=8$, Table~\ref{tab:exp1-correlations}), two measures too strongly correlated to separate.
```

The −0.622 at L380 is a different quantity (the Spearman correlation in the table) and is correct.

**6-B2. L286: "pre-registered" is not accurate.** There was no pre-registration. `Following the pre-registered protocol selection criterion (minimum seed-mean of node-median validation MAE at $h=1$` →

```latex
Following the selection criterion of Section~\ref{subsec:exp2-training} (lowest seed-mean node-median validation MAE at $h=1$
```

### SHOULD-FIX

| # | Line | Issue | Fix |
|---|---|---|---|
| 6-S1 | 341 | `establishes a clear empirical frontier across all horizons` is contradicted: GAT-plain at h=1 (4.740) is worse than the temporal LSTM (4.597) | `contains the best model at every horizon, although the plain GAT is not better than the temporal LSTM at $h=1$` |
| 6-S2 | 359 | Same claim in the table caption | Same correction |
| 6-S3 | 144 | A sentence was inserted between the colon and the list it introduces | Move the inserted sentence after the list |
| 6-S4 | 675 | The sign convention is reversed relative to Table 6.x (negative = improvement) | Align the text with the table convention |
| 6-S5 | 136, 514, 546 | Overclaims ("demonstrates", "confirms" for single-seed attention) | `suggests`, `is consistent with` |
| 6-S6 | 10, 23, 25, 28, 29, 155, 479, 529, 628, 631, 637, 646 | Hyperbole ("illuminating", "striking", "fundamental", "crucially") | Delete or quantify |

### MINOR

- **L688:** `parameter-matched control` → `architecturally matched control`. Parameter counts differ (`param_counts.csv`).
- **L689:** the data contain no summer period; say so explicitly in the seasonal limitation.
- **L381:** missing ± SD in one table cell.
- **L493–494:** the bold marks the wrong value in a row (not the best).
- **L546–548:** use `$\bar{\rho}$` for the seed-averaged correlation.
- **L568, L712:** minor wording.
- **Unused figure:** `fig_ch6_node_map_h4.pdf` is in the figures folder but never included. Either include it or leave it out; this does not affect the build.

### Open author TODO

- **L227–234:** grid-search paragraph, currently commented out. No grid-search evidence exists in the repository (hyperparameters were fixed). Keep it out, or state that hyperparameters were fixed a priori. This is already in E.2 Limitations.

---

## Cross-chapter consistency checklist (after fixes)

| Fact | Ch4 | Ch5 | Ch6 |
|---|---|---|---|
| Exp1 split = run-level micro-split | ✔ (L290/294/315) | ✔ | ✔ (L111/148) |
| Exp1 headline = sequence length and completeness both associated, not separable | ✘ → 4-B1, 4-B2 | ✔ | ✘ → 6-B1 |
| Linear/spline fill ≤ 2 h; KNN fills all; KNN +4.34 pp on matched timestamps | ✘ → 4-B3, 4-B4 | ✔ | n/a |
| 30% threshold is precautionary, no sweep | ✘ → 4-B5 | n/a | ✔ |
| Exp2 graphs Euclidean (haversine) + OSM edge attributes | ✘ → 4-B8 | n/a | ✔ |
| Fixed hyperparameters, no grid search | ✘ → 4-B6 | ✔ | ✔ (TODO L227) |
| 54 nodes / 48 evaluated | partial → 4-S3 | n/a | ✔ |
| Figure images carry "alternating-day" labels | Fig. in Ch4 ✘ | n/a | Fig. 6.1 ✘ |

The figure-label issue in the last row is in `scripts/eda/plot_experiment_sensor_availability.py` and `plot_experiment1_split_comparison.py` (brief item 5-28). Regenerate both figures after updating the labels.
