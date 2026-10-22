# Implementation brief: Chapters 5 and 6, remaining Tier 2 items, new Chapter 6 figures and validation

Date: 2026-10-13. Applies to:
- `full/chapters/chapter5_experiment1_single_node.tex` (558 lines, as of 16:38)
- `full/chapters/chapter6_experiment2_multi_node.tex` (678 lines, as of 16:37)
- `full/chapters/chapter4_methodology.tex`: only the three rows of 5-28

The chapters were being edited while this brief was written. Every OLD string was checked against the files as of the times above; line numbers may be off by one or two.

Follows on from: `audit_report_ch5_ch6_2026-10-13.md`.

---

## 0. Instructions for the thesis agent

1. **Matching.** Line numbers refer to the files as they stand today.
   - Always locate an edit by its quoted `OLD` text, not by line number alone.
   - Within each file, apply edits from the bottom up so that earlier line numbers stay valid.
2. **Tags.** Every item carries one tag:
   - **[CORRECTION]**: the text is factually wrong, or contradicted by the code or artefacts. It must change.
   - **[LIMITATION]**: the text may stay (sometimes with softer wording), but the weakness must be disclosed
     in the chapter's Limitations subsection. Part E gives that subsection ready to paste.
   - **[BOTH]**: correct the local text *and* list the issue in the Limitations subsection.
   - **[STYLE]**: tone or notation only.
3. **[AUTHOR] flag.** Some items need the author's knowledge, e.g. whether an unlogged grid search took place.
   - Apply the conservative wording given.
   - Insert the comment `% TODO(author): <question>` on the line above.
   - Do not delete the original text. Comment it out with `%` so that it can be restored (AGENTS.md: no silent
     destructive deletions).
4. **Style.** Keep British English, `\citep`/`\citet`, non-breaking spaces before units, en-dashes in number ranges, and "Diebold--Mariano".
5. **New figures.** Chapter 5: replace one file in `full/figures/experiment1/` (5-10 / D.0; keep a backup). Chapter 6: copy these files from `docs/thesis/ch6_new_figures/` into `full/figures/experiment2/`:
   - `fig_ch6_node_map_h1.pdf`
   - `fig_ch6_node_map_h4.pdf`
   - `fig_ch6_completeness.pdf`
   - `fig_ch6_distance_decay.pdf`
   - `fig_ch6_paired_gain.pdf`

   Part D gives the LaTeX.
6. **New bibliography entries** (Part D.6): add them to `shared/bibliography/references.bib` before compiling.
7. **Compile.** Compile once with the Smart Build recipe and check for undefined references.

### Evidence base

Every number in this brief comes from one of these:

| Source | Content |
|---|---|
| `artifacts/spatial_full/runs/*/{results.csv,predictions.parquet,manifest}` | Experiment 2 runs: 46 runs, seeds 42–44 |
| `docs/thesis/ch6_new_figures/figure_stats.md` | Statistics behind the new figures |
| `docs/thesis/ch6_new_figures/_exp2_verification.md` | Experiment 2 code facts with file:line references |
| `docs/thesis/ch6_new_figures/dm_champion_seedavg.csv` (script `dm_champion_seedavg.py`, per-node results `dm_champion_seedavg_pernode.csv`) | New Diebold–Mariano tests on the champion |
| `docs/thesis/ch6_new_figures/param_counts.csv` | Trainable parameter counts |
| `docs/thesis/ch6_new_figures/_exp1_verification.md` | Experiment 1 code and artefact facts |

The Experiment 2 source code is not in the working tree. It is on branch `feature/ch6-spatial-gat-rebuild`
(commit `2fb322a`). Two checks confirm that this code produced the artefacts:
- the compiled `.pyc` files in the working tree match it;
- the MASE scale recomputed from it agrees with the artefacts to a relative error of 4×10⁻⁸.

## A. Validation of the changes made since the audit

### A.0 New finding: Chapter 5 draws on two different Experiment 1 run sets [CORRECTION]

`_exp1_verification.md` §0 shows that Chapter 5 mixes two run sets.

| Set | Path | Viable streams | Models | Thesis items that reproduce from it |
|---|---|---|---|---|
| **v2 (canonical)** | `artifacts/quality_full_v2` | 110 | all 12 | Tables 5.1–5.3 (leaderboard, interpolation, DM), Table 5.6 (outliers), matched pairs, extreme events, cluster medians, Heber |
| **v1** | `artifacts/quality_full` | 114 | LSTM-MV/UV + baselines | Table 5.4 (correlations), RF $R^2$ 0.25/0.37 |
| **mixed** (v1 errors + v2 quality features) | — | 110 | — | Fig. 5.6, the 2.90 h breakpoint, the line 294 outlier values |

**Decision applied in this brief:** move every Chapter 5 statistic to **v2**, the run behind the headline tables.
- The v2 values below were recomputed by the verification agent (scripts preserved in `ch5_v2_figures/exp1_verification_scripts/`).
- A v2 version of Fig. 5.6 is in `ch5_v2_figures/` (script `make_fig5_6_threshold_v2.py`, statistics in `fig5_6_threshold_v2_stats.txt`).

**Consequences of the move:**
- **Threshold:** largely unchanged (2.90 h → 2.95 h).
- **Line 294 contradiction resolved:** the CV-RMSE values at line 294 no longer contradict Table 5.6.
- **Headline claim changes:** sequence length now correlates slightly more strongly than completeness, which is the opposite of v1. Even so, completeness, not sequence length, keeps the independent partial association. The headline is therefore reframed (5-01).
- **RF meta-model:** under v2 it explains almost no variance.

### A.1 Chapter 5: validation of the author's changes

| Audit item | Status | Note |
|---|---|---|
| C5-1 to C5-12, C5-14, C5-15 | Fixed | Fig. 5.1 (framework PDF) not inspected visually |
| T2-A: partial correlations in Table 5.4 | Added, but **from the v1 run**, and the caption's "all significant" is false under v2 | 5-07 |
| T2-B: log-volume regression (line 284) | Added, but $t = 0.31$ is not produced by any repository script | v2 re-run gives $t = 0.24$ ($h=8$) and $1.85$, $p = 0.07$ ($h=24$); see 5-08 |
| T2-C: sample-composition caveats (lines 57, 104, 133–137) | Fixed | Line 57 describes a split mechanism that the code does not implement; see 5-02 |
| T2-D: breakpoint vs window (lines 298–301) | Fixed | Good addition |
| Line 294: outlier list | **Wrong source** | The CV-RMSE values are v1; Table 5.6 is v2. See 5-11 |
| Line 449: Heber de-duplication | Added, **unverifiable** | The processed data contain no duplicate. Other matchday readings reach 1,810–1,966, and 1,154 is outside the test split. See 5-17 |
| Line 461: omitted covariate | Fixed | Last sentence overclaims what Chapter 6 tests; see 5-18 |
| Headline vs Table 5.4 (lines 14, 553) | **Not fixed** | 5-01, 5-26 |
| Matched-pairs units and "3.7x" | **Not fixed** | 5-23, 5-24 |
| Extreme events (lines 482–497) | **Not fixed** | 5-20, 5-21 |
| Clusters "significantly lower" | **Not fixed, and wrong**: $p = 0.067$ | 5-14 |

### A.2 Chapter 6: validation of the author's changes

| Audit item | Status | Note |
|---|---|---|
| C6-1 (seed-stability figure vs caption) | Fixed | New `attention_stability.png` (Fig. at line 531) plus a recaptioned `seed_stability.png` (line 613) |
| C6-2 (SDs) | Fixed | Tables 6.3 and 6.4 now carry ±1 SD |
| C6-3 to C6-11, C6-14 | Fixed | Checked against the artefacts: the leaderboard, baselines and spatial-gain values all reproduce to 3 d.p. |
| X2 (cross-chapter breakpoint/completeness wording, line 633) | Fixed in substance | The sentence that follows still overclaims; see 6-23 |
| Completeness correlation (line 629) | Fixed | r = −0.33, p = 0.021; ρ = −0.37, p = 0.0097 reproduce exactly |
| Falconar/Barras sign (lines 651–654) | **Partly fixed, new error** | The direction and percentages are now right. The absolute MAEs (4.81→5.12, 11.23→11.77) are wrong; see 6-25 |
| Line 657 mechanism | **Contradicted by data** | Both nodes have the *highest* Gabriel in-degree (5), and their neighbours are *not* sparse; see 6-25 |
| Line 150 (the 6 excluded nodes) | **New error introduced** | The MASE-denominator explanation is wrong. The six nodes simply have no test targets; see 6-06 |
| Line 101 (45 steps) | **Partly wrong** | The count of 45 is right, but the val/test dates ignore the purge gaps; see 6-03 |
| Fig. 6.5 caption (line 501) | **New inconsistency** | The caption says "degree-normalised", but the images and r values are raw α. Replace with the new figure; see 6-18 |
| Line 356 caption (MASE) | **Wrong (newly detected)** | The code uses a *daily* seasonal-naive scale (m = 96), not weekly; see 6-15 |
| Table 6.7 (capacity) | **Wrong (newly detected)** | The ablation rows are GAT-Transformer runs, but the "Champion" row is GAT-LSTM; see 6-22 |

---

## B. Chapter 5: edits, in file order

File: `chapters/chapter5_experiment1_single_node.tex`. Apply from the bottom up.

### 5-01 [CORRECTION] Line 14: headline claim, threshold and RF

Under v2:
- **Bivariate correlations:** sequence length correlates slightly more strongly with CV-RMSE than completeness ($\rho = -0.58$ vs $-0.54$ at $h=8$).
- **Partial correlations:** sequence length controlling for completeness is marginal ($\rho = -0.18$, $p = 0.058$); completeness controlling for sequence length is significant ($\rho = -0.29$, $p = 0.002$).
- **Threshold:** exists at $h=8$, but not at $h=24$ (ΔAIC 0.39).
- **RF meta-model:** out-of-bag $R^2$ is 0.04 ($h=8$) and −0.01 ($h=24$).
- **"$R^2 = 0.81$":** not reproduced by the stated pipeline. Values near 0.81 appear only when volume is added as a predictor.

Replace the passage from `The core empirical finding of this chapter is that` up to and including `motivating the integration of spatial network context in subsequent chapters.` with:

```latex
The core empirical finding of this chapter is that relative forecast error is associated with both the temporal structure of missingness (mean contiguous sequence length) and aggregate data completeness, and that the effect of sequence length is non-linear. The two quality measures are strongly correlated ($\rho = 0.78$), so their separate contributions cannot be cleanly isolated. Segmented regression at the 2-hour horizon identifies a breakpoint at $\approx 3.0$~hours of continuous data (2.95~h, 95\% bootstrap CI 1.90--3.42~h): below it, relative forecast error (Coefficient of Variation of the Root Mean Square Error, CV-RMSE) rises steeply as sequences shorten, whereas above it, error changes little. No breakpoint is supported at the 6-hour horizon. A Random Forest meta-model using four quality features explains almost none of the cross-sensor variance in relative error (out-of-bag $R^2 = 0.04$ at $h=8$ and $-0.01$ at $h=24$). This indicates that temporal data quality alone is insufficient to explain forecast variability, motivating the integration of spatial network context in subsequent chapters.
```

The rest of line 14, from `Finally, while linear interpolation`, is unchanged.

### 5-02 [CORRECTION] Line 57: split mechanism

`src/quality/quality_metrics.py:134-205` (`assign_split_windows`) has **no day-by-day logic**. It works on contiguous runs:
- A run long enough to give every split at least one window after purging is split chronologically 70/15/15, with a purge of $W+H-1$ windows.
- Shorter runs are assigned **whole** to one split by a running deficit counter.

The same function is used for every interpolation arm.

Replace the text from `To evaluate model behaviour across these fragmented series` up to and including `permitting a continuous 70/15/15\% chronological block split.` with:

```latex
To evaluate model behaviour across these fragmented series without fabricating synthetic observations, every arm uses the same chronological split procedure, applied separately to each contiguous run of observations. A run long enough to supply at least one window to every split is divided in time order into training (70\%), validation (15\%) and test (15\%) windows, and the first $W+H-1$ windows of the validation and test portions are discarded so that no input or target step is shared across splits. Shorter runs are assigned whole to a single split, chosen by a running deficit counter that keeps the overall window counts close to the 70/15/15 targets; whole-run assignment cannot leak across splits and keeps low-quality sensors in the evaluation. Under the primary \textbf{Run-Level Micro-Split protocol} (no interpolation), dropouts fragment each stream into thousands of short runs, many of which are assigned whole. Under the secondary \textbf{Per-Stream Macro-Split protocol} evaluated in Section~\ref{sec:exp1-interpolation}, single-node interpolation bridges gaps (all gaps for $k$-NN; gaps of up to 2~hours for linear and spline interpolation), the runs become much longer, and the same procedure approaches a 70/15/15 chronological block split over each stream's active lifespan.
```

The sentence starting `Crucially, because extracting valid un-imputed sequences` is unchanged.

**Follow-on edits:**
- Line 148: replace `Conversely, under the primary \textbf{Alternating Daily Micro-Split protocol} (Panel~B), raw uninterpolated sequences are preserved, where temporal voids fragment the data into thousands of short runs per stream. To prevent temporal leakage under this extreme fragmentation while preserving chronological balance, an alternating day-by-day split assignment mechanism is employed (Panel~C), dynamically allocating daily sequences to training, validation, or test sets based on running deficit counters.` with `Conversely, under the primary \textbf{Run-Level Micro-Split protocol} (Panel~B), raw sequences are preserved and missingness fragments each stream into thousands of short runs. Runs too short to be split internally are assigned whole to training, validation or test by a running deficit counter (Panel~C), which prevents leakage while keeping the overall split proportions close to 70/15/15.`
- Line 158: `relies strictly on the primary Alternating Daily Micro-Split protocol` → `uses the primary Run-Level Micro-Split protocol`.
- Fig. 5.3 caption (line 144): `illustrating the alternating daily partition assignment algorithm that balances train, validation, and test quotas dynamically.` → `illustrating how whole runs are assigned to the training, validation and test splits by a running deficit counter.`
- **[AUTHOR]** Fig. 5.3 is produced by `scripts/eda/plot_experiment1_split_comparison.py`, which you are currently editing. Check that the Panel C title and legend do not say "alternating daily".

### 5-03 [BOTH] Line 48: single seed (T2-K)

Append to line 48: `All neural models were trained once, with a single random seed (42); seed-to-seed variability was therefore not measured in this experiment.`

Limitation: E.1, item 1.

### 5-04 [BOTH] Line 87: the masked Transformer is functionally inert (T2-K)

The windows are drawn from contiguous runs only, so the mask and time-delta channels are always zero. The masked and standard Transformers produce the same 333,917 test predictions.

- Append to the item at line 87: `In this experiment, however, input windows are drawn only from contiguous runs without missing steps, so the mask and time-since-last-observation channels are always zero: the masked variants never encounter a gap and receive no information beyond that available to the standard Transformers. Their results therefore do not test mask-aware learning, which is evaluated in Chapter~\ref{ch:exp3}.`
- Caption of `tab:exp1-baseline` (line 168): append, just before the closing `}` of `\caption{...}`, `The masked Transformer variants never receive missing input steps (Section~\ref{sec:exp1-baselines}), so their masking is inactive.`

Limitation: E.1, item 2.

### 5-05 [CORRECTION] Line 100: describe the KNN imputer (T2-H)

The KNN imputer is implemented in `src/quality/interpolation.py` (`_knn_impute`). Append after `To keep comparisons fair, error metrics were reported on the originally observed (real) time steps only.`:

```latex
The KNN imputer estimates each missing step from the $k = 5$ observed steps of the same stream that are nearest in a feature space of time-of-day and day-of-week (both sine--cosine encoded) and normalised calendar time, weighting donors by inverse distance; it uses no information from other sensors, so in practice the donors are the same 15-minute slot on the same weekday in nearby weeks. The imputer is fitted on all observed steps of a stream, including those in the validation and test periods, and fills gaps of any length; linear and spline interpolation are limited to gaps of at most 8 steps ($2~\text{h}$).
```

Limitation: E.1, item 3 (KNN fitted on test-period observations).

**[AUTHOR]** The 8-step limit for linear and spline interpolation comes from `max_gap_steps: 8` in the v2 config. Check that it matches your intended description.

### 5-06 [CORRECTION] Line 259: Table 5.4 caption

- OLD: `Volume corresponds to mean raw pedestrian count ($\rho(\text{avg\_seq\_len}, \text{volume}) = 0.88$). All reported bivariate and partial correlations are statistically significant at $p < 0.05$.`
- NEW: `Volume corresponds to mean raw pedestrian count ($\rho(\text{avg\_seq\_len}, \text{volume}) = 0.86$). Unless marked $^{\dagger}$ (not significant, $p \ge 0.05$), all correlations are significant at $p < 0.05$.`

### 5-07 [CORRECTION] Lines 267–270: Table 5.4 body, v2 values (T2-A)

Replace the four data rows with the rows below. Sequence length is now listed first, because it is the strongest bivariate correlate under v2.

```latex
        Average Sequence Length (\texttt{avg\_seq\_len\_hours})    & -0.58 & -0.44 & -0.41 & -0.45 \\
        Observation Completeness (\texttt{completeness\_active}) & -0.54 & -0.39 & -0.28 & -0.42 \\
        Diurnal Capture Ratio (\texttt{diurnal\_capture\_ratio})  & -0.34 & -0.11$^{\dagger}$ & -0.15$^{\dagger}$ & -0.11$^{\dagger}$ \\
        Mean Pedestrian Count (\texttt{mean\_count\_raw})         & -0.42 & ---   & -0.10$^{\dagger}$ & ---   \\
```

### 5-08 [CORRECTION] Lines 282–286: confound paragraphs (T2-A, T2-B)

**(a) Line 282.** `($\rho = 0.88$, Figure~\ref{fig:correlation_heatmap})` → `($\rho = 0.86$, Figure~\ref{fig:correlation_heatmap})`.

**[AUTHOR]** The heatmap (Fig. 5.5, `ext_correlation_heatmap.png`) has unknown provenance. If it was built from v1 or mixed data, its cells will differ slightly from the v2 Table 5.4. Regenerate it from v2, or add "(computed on an earlier processing run)" to its caption.

**(b) Line 284: replace the whole paragraph with:**

```latex
The partial correlation analysis in Table~\ref{tab:exp1-spearman} does not support this confound. When controlling for mean pedestrian volume, both average sequence length ($\rho = -0.44$ at $h=8$; $\rho = -0.45$ at $h=24$) and observation completeness ($\rho = -0.39$ at $h=8$; $\rho = -0.42$ at $h=24$) retain significant negative associations with relative forecast error ($p < 0.001$). In a multiple regression of CV-RMSE on sequence length, active completeness and log-transformed volume, the volume coefficient is not significant at $h=8$ ($t = 0.24$, $p = 0.81$) and is marginal at $h=24$ ($t = 1.85$, $p = 0.07$). Volume therefore adds little explanatory power once data quality is included, particularly at the 2-hour horizon. The diurnal capture ratio, by contrast, is not associated with error once volume is controlled ($\rho = -0.11$), suggesting that its bivariate correlation largely reflects volume.
```

**[AUTHOR]** No repository script produces this regression. The values come from the verification re-run (`ch5_v2_figures/exp1_verification_scripts/corr2.py`). Add a script, or cite it, before submission.

**(c) Line 286: replace the paragraph with:**

```latex
However, because sequence length and completeness are strongly correlated ($\rho = 0.78$), their separate contributions are difficult to isolate. Controlling for completeness, the partial correlation of sequence length with CV-RMSE is weak and only marginally significant ($\rho = -0.18$, $p = 0.058$ at $h=8$), whereas completeness controlling for sequence length remains significant ($\rho = -0.29$, $p = 0.002$). A linear partial correlation cannot, however, capture a threshold effect. The segmented regression below indicates that sequence length matters mainly below about three hours, beyond which further continuity brings little benefit.
```

### 5-09 [CORRECTION] Line 290: threshold statistics (v2)

| Item | OLD | NEW |
|---|---|---|
| Breakpoint | `occurs at $2.90~\text{h}$` | `occurs at $2.95~\text{h}$` |
| Confidence interval | `$[1.87, 3.23]~\text{h}$ (approximately $1.9$--$3.2~\text{h}$; $\Delta\text{AIC} = 9.94$` | `$[1.90, 3.42]~\text{h}$ ($\Delta\text{AIC} = 7.17$` |
| Slope below | `at a slope of $-27.37~\text{pp}$ CV-RMSE per hour of average sequence length ($p < 0.001$)` | `at a slope of $-22.26~\text{pp}$ CV-RMSE per hour of average sequence length` |
| Slope above | `a negligible slope of $-0.44~\text{pp}$ per hour` | `a negligible slope of $-0.48~\text{pp}$ per hour` |

The $p$-value for the below-breakpoint slope was not recomputed, so it is removed.

### 5-10 [CORRECTION] Fig. 5.6: replace the image and caption

- Copy `docs/thesis/ch5_v2_figures/threshold_cvrmse_lstm_mv_h8.png` (and `.pdf`) into `full/figures/experiment1/`.
- **First** rename the existing file to `threshold_cvrmse_lstm_mv_h8_mixed_v1v2.png` so that nothing is lost.

New caption (line 306):

```latex
\caption[Sequence length threshold]{Segmented regression of CV-RMSE against average sequence length for the LSTM-MV at $h=8$ (2~hours), $N = 110$ directional streams. The broken-stick regression identifies a breakpoint at $2.95~\text{h}$ (95\% bootstrap CI: $[1.90, 3.42]~\text{h}$; $\Delta\text{AIC} = 7.17$), separating a steep error decline ($-22.26~\text{pp}/\text{h}$) from a plateau ($-0.48~\text{pp}/\text{h}$).}
```

### 5-11 [CORRECTION] Lines 294–295: outlier leverage (v2 values) and the h=24 result

**(a) Line 294.** Replace `(mean sequence length $1.67~\text{h}$, CV-RMSE $225.4\%$), \nolinkurl{NclUrbanScienceBuildingSquareMid__west_to_east} ($2.26~\text{h}$, $219.7\%$), and \nolinkurl{NclQueenVictoriaRdPercyBuilding__from_south_to_north} ($1.79~\text{h}$, $191.4\%$)` with `(mean sequence length $1.67~\text{h}$, CV-RMSE $163.3\%$), \nolinkurl{NclUrbanScienceBuildingSquareMid__west_to_east} ($2.26~\text{h}$, $224.6\%$), and \nolinkurl{NclQueenVictoriaRdPercyBuilding__from_south_to_north} ($1.79~\text{h}$, $157.1\%$)`.

These now match Table 5.6.

**(b) Line 294, later in the paragraph.** Replace `the empirical breakpoint remains consistent at $2.96~\text{h}$ (95\% bootstrap CI: $[1.94, 3.39]~\text{h}$; $\Delta\text{AIC} = 7.78$), while the below-breakpoint slope attenuates to $-14.69~\text{pp}/\text{h}$ (with an above-breakpoint slope of $-0.50~\text{pp}/\text{h}$).` with:

`the breakpoint estimate is almost unchanged at $2.94~\text{h}$, but its 95\% bootstrap CI widens to $[1.94, 4.91]~\text{h}$ and the support for a breakpoint weakens ($\Delta\text{AIC} = 4.50$); the below-breakpoint slope attenuates to $-14.12~\text{pp}/\text{h}$ (above: $-0.53~\text{pp}/\text{h}$). The location of the threshold therefore depends partly on a few severely fragmented streams.`

**(c) Add a third item** after the `Sparse Support Beyond 6 Hours` item (line 295):

```latex
    \item \textbf{No Threshold at the 6-Hour Horizon:} Repeating the fit at $h=24$ gives a breakpoint at $2.57~\text{h}$ with a wide confidence interval ($[1.93, 4.92]~\text{h}$) and no improvement over a single-slope model ($\Delta\text{AIC} = 0.39$). The threshold is therefore specific to the 2-hour horizon.
```

**(d) Line 299.** `Therefore, the 2.9-hour breakpoint cannot be interpreted` → `Therefore, the 3-hour breakpoint cannot be interpreted`.

Limitation: E.1, item 4.

### 5-12 [STYLE] Line 288: subsection title

Optional: `The Three-Hour Threshold and Sequence Length Dynamics` → `A Three-Hour Threshold at the 2-Hour Horizon`.

### 5-13 [STYLE] Line 312

No change needed. The confounding caveat is already present.

### 5-14 [CORRECTION] Lines 324–326: clusters (T2-J)

**(a) Line 324: replace the first sentence with:**

```latex
A $k$-means cluster analysis ($k = 2$; seven standardised features: average sequence length, completeness, diurnal capture ratio, mean count, point-of-interest count, transit-stop count and distance to the city centre) segments the sensors into two groups: an Urban Core cluster (28 streams from 14 locations; higher volume, longer sequences, closer to the city centre) and a Suburban cluster (82 streams from 39 locations). The number of clusters was chosen by silhouette score, although the margin over $k = 6$ is small (0.460 versus 0.456).
```

**(b) Line 326, first two sentences.** Replace `Interestingly, despite the Urban Core having much higher absolute RMSE due to higher pedestrian volumes, they actually have significantly lower relative error (CV-RMSE), as visualised in Figure~\ref{fig:cluster_boxplot}. The Urban Core median CV-RMSE is 47.7\%, compared to 59.3\% for Suburban sensors.` with:

`Despite their higher absolute RMSE, which follows from higher pedestrian volumes, Urban Core sensors have lower relative error (Figure~\ref{fig:cluster_boxplot}): the median CV-RMSE is 47.7\% for the Urban Core and 59.3\% for Suburban sensors. This difference is not statistically significant (Mann--Whitney $U = 881$, $p = 0.067$ at $h=8$; $p = 0.51$ at $h=24$).`

**(c) Line 326, end of the paragraph.**
- `This indicates that model predictability is closely linked` → `This suggests that predictability is linked more`.
- `shows negligible correlation with CV-RMSE ($\rho = -0.01$).` → `shows negligible correlation with CV-RMSE ($|\rho| < 0.05$).`
- Insert `% TODO(author): rho(distance, CV) = -0.037 on the mixed v1/v2 data; recompute on v2 for the heatmap.`

**(d) Fig. 5.8 caption (line 338).** Append: `The difference is not statistically significant (Mann--Whitney $p = 0.067$).`

### 5-15 [CORRECTION] Line 345: KNN rationale and the 3.7-fold figure (T2-G)

Of the KNN windows, 349,673 (28.6%) have an imputed target. Counting only windows with observed targets gives 871,897 windows, a 2.6-fold increase.

- OLD: `and enables a 3.7-fold expansion in nominal test window availability`
- NEW: `and increases the number of nominal forecast windows 3.7-fold (2.6-fold if only windows with an observed target are counted)`

### 5-16 [CORRECTION] Line 361: matched hourly claims rest on n ≤ 11 (T2-G)

The matched hourly bins at 07:00, 08:00, 09:00 and 10:00 contain 8, 8, 8 and 11 steps respectively.

Replace the text from `When evaluating strictly on identical matched timestamps (Figure~\ref{fig:exp1-matched} Panel~B), the un-imputed 08:00 UTC spike` to the end of the paragraph with:

```latex
On identical matched timestamps (Figure~\ref{fig:exp1-matched} Panel~B), the early-morning comparison cannot be made reliably: only 4.2\% of matched steps fall between 00:00 and 08:59~UTC, and each hourly bin from 07:00 to 10:00 contains only 8--11 matched steps. The matched steps are concentrated between 13:00 and 22:59~UTC (90.6\%), and over this set KNN has the higher error (pooled RMSE 17.62 versus 14.67 counts; Table~\ref{tab:exp1-matched-summary}).
```

### 5-17 [CORRECTION + AUTHOR] Lines 446–449: failure regimes, UrbanScience (T2-E) and Heber

**(a) Line 446.** `fall into two distinct failure regimes:` → `fall into three failure regimes:`.

**(b) Line 448.** `causing CV-RMSE to exceed 100\%--160\%.` → `with CV-RMSE between 107\% and 163\%.`

**(c) Insert a new item between the Queen Victoria Road item (line 448) and the Heber item (line 449):**

```latex
    \item \textbf{Isolated Event Spikes on a Low-Volume Stream:} The highest-error stream, \nolinkurl{NclUrbanScienceBuildingSquareMid__west_to_east} (CV-RMSE 224.62\%, RMSE 42.12 counts), combines short sequences (2.26~h) with a low typical count (median 12 pedestrians per 15~min) and a few very large spikes, the largest being 1,018 pedestrians on Saturday 27 April 2024 at 16:00 (predicted: 28). The five largest residuals account for 94.1\% of the squared error; removing the single largest reduces CV-RMSE to 122.85\%, and removing the three largest to 67.21\%. The other high readings at this location also fall on Saturday afternoons, suggesting campus events rather than sensor faults. The opposite direction at the same location has a CV-RMSE of 87.6\%.
```

**(d) Line 449, Heber Street.**
- **[AUTHOR]** "Duplicate records" cannot be verified:
  - the processed data contain no duplicate timestamps;
  - other Saturday matchday readings at this sensor reach 1,810–1,966.

  Insert the line `% TODO(author): confirm in the raw database whether the 2,242 reading is a duplicate; if not, keep the wording below.`
- Replace the text from `A data audit reveals that this headline metric is partly elevated by an ingestion artefact:` up to and including `(or 138.98\% if omitted entirely).` with:

  `A large part of this error comes from a single reading of 2,242 pedestrians at 16:00~UTC on Sunday 23 February 2025, a Newcastle United home matchday. Halving this reading reduces RMSE to 41.96 counts and CV-RMSE to 151.23\%, and omitting it gives a CV-RMSE of 138.98\%.`
- `Even after accounting for this ingestion duplicate,` → `Even without this reading,`.
- `(peaking between 700 and 1,154 pedestrians in 15 minutes during Newcastle United home fixtures)` → `(up to 952 pedestrians per 15~minutes in the test split, excluding the 2,242 reading)`.
- `Restricting evaluation strictly to non-surge hours ($\le 260$ counts) yields a routine CV-RMSE of 66.51\%.` → `Restricting evaluation to steps with at most 260 counts, which removes 10 surge steps, gives a CV-RMSE of 66.51\%.`

### 5-18 [AUTHOR + CORRECTION] Fig. 5.12 caption (line 455), and lines 459 and 461

**(a) Fig. 5.12 caption.**
- Tuesday 21 and 28 April 2026 peak at only 150 and 206 counts. The matchday peaks that month were on Saturday 18 April (1,880) and Saturday 2 May (1,810). Labelling the Tuesdays as matchday surges is therefore doubtful.
- Insert `% TODO(author): check whether 21 and 28 April 2026 were fixture days; if not, use the caption below.`
- Replace `While routine weekdays (14 April) show good alignment, episodic matchday crowd surges adjacent to St James' Park stadium (21 and 28 April, peaking at 150 and 206 pedestrians) cannot be anticipated from local temporal history alone without stadium fixture schedules.` with `14 April shows a routine weekday profile, whereas 21 and 28 April contain higher afternoon peaks (150 and 206 pedestrians). The largest surges at this sensor occur on Saturday matchdays (e.g.\ 1,880 pedestrians on 18 April 2026) and are not shown.`

**(b) Line 459, [AUTHOR].** "active on all 141 weekend days … averaging 93\% completeness" is not traced to any artefact, and Table 5.6 gives 91.38\%. Verify it, or change it to `(91\% completeness)`.

**(c) Line 461, last sentence.** Chapter 6 does not test surge anticipation.
- OLD: `where spatial graph models are introduced to borrow predictive signals from surrounding urban corridors before surges arrive at individual bottleneck nodes, and for future operational deployments incorporating calendar schedule covariates.`
- NEW: `where spatial graph models are introduced to test whether neighbouring sensors provide additional predictive information, and for future deployments incorporating event-calendar covariates.`

### 5-19 [CORRECTION] Line 477: residual volume dependence (v2)

- `($\rho = -0.40$ at $h=8$; $\rho = -0.19$ at $h=24$, Table~\ref{tab:exp1-spearman})` → `($\rho = -0.42$ at $h=8$; $\rho = -0.10$, not significant, at $h=24$; Table~\ref{tab:exp1-spearman})`
- `were necessary to confirm that sequence continuity and completeness govern predictability independently of footfall volume.` → `were used to check that the associations of sequence continuity and completeness with error are not explained by footfall volume alone.`

### 5-20 [CORRECTION] Lines 482 and 491: extreme-event definition and numbers (T2-I)

**(a) Line 482.** σ and μ are computed per sensor over the **test-split targets**, not history.
- Replace `defined as time steps where actual pedestrian counts exceed $3\sigma$ above the sensor's historical mean ($y_{\text{true}} > \mu + 3\sigma$). Figure~\ref{fig:exp1-extreme} compares performance during normal operational conditions ($\mu \pm 1\sigma$) against extreme shock events.` with:

```latex
defined as time steps where the observed count exceeds the sensor's mean by more than three standard deviations ($y_{\text{true}} > \mu + 3\sigma$, with $\mu$ and $\sigma$ computed per sensor over its test-split targets; sensors with $\sigma < 1$ are omitted). Figure~\ref{fig:exp1-extreme} compares these extreme steps (1.4\% of steps for the median sensor) with normal conditions ($|y_{\text{true}} - \mu| \le \sigma$, 77.8\%); the intermediate steps (20.7\%) belong to neither group. Because the extreme group contains, by construction, the largest counts, its absolute errors are larger partly because of scale alone.
```

**(b) Line 491.**
- `$5.0\times$` formatting: already fixed by the author.
- `surging from 4.87 counts during normal conditions to 24.2 counts during shocks` → `rising from 4.87 counts under normal conditions to 24.2 counts during extreme events`.
- `with peak residuals reaching 690 counts` → `with the largest per-sensor extreme-event RMSE, at Heber Street, reaching 690 counts`.

**(c) Line 493.** `dropping from 58.9\% down to 45.1\%.` → `the mean across sensors dropping from 58.9\% to 45.1\% (median: 50.4\% to 43.7\%).`

**(d) Fig. 5.10 caption (line 487).** `While absolute RMSE surges five-fold` → `While median absolute RMSE increases five-fold`.

### 5-21 [CORRECTION] Lines 494–497: MAPE equation and the size of the denominator

**(a) Replace the equation** (lines 494–496). This avoids reusing $N$, which denotes the number of sensors elsewhere:

```latex
\begin{equation}
    \text{MAPE} = \frac{100\%}{T} \sum_{t=1}^{T} \frac{|y_t - \hat{y}_t|}{\max(|y_t|, 1)}
\end{equation}
where $T$ is the number of evaluated steps; the denominator $\max(|y_t|, 1)$ avoids division by zero for zero counts.
```

This matches `src/quality/evaluate.py:37-43`.

**(b) Line 497, first sentence.**
- OLD: `During extreme surges, the denominator $y_t$ is in the hundreds or thousands, which compresses the ratio even when the absolute prediction misses the surge by a substantial margin (e.g., an error of 100 on a true count of 500 yields a deceptively low 20\% error).`
- NEW: `During extreme surges the denominator $y_t$ is several times the sensor's typical count (the median of the per-sensor mean count in the extreme band is 58 pedestrians per 15~min, interquartile range 33--130), which compresses the ratio even when the absolute miss is large (e.g.\ an error of 20 on a true count of 100 yields a 20\% error).`

**(c) Line 497, last sentence [STYLE].** `This demonstrates that MAPE is fundamentally pathological for zero-inflated, intermittent urban sensor streams, reinforcing the utility of adopting scale-normalised CV-RMSE as a reliable metric for municipal forecasting.` → `MAPE is therefore poorly suited to intermittent, low-count urban sensor streams, which supports the use of scale-normalised metrics such as CV-RMSE.`

### 5-22 [CORRECTION] Line 502: RF meta-model (T2-F)

The code (`src/quality/stats.py:326`):
- 300 trees, out-of-bag $R^2$ with **row-level** bootstrap;
- predictors: `completeness_active`, `avg_seq_len_hours`, `std_seq_len_hours`, `seq_count`, with **no volume**.

Results:

| Data | Validation | $h=8$ | $h=24$ |
|---|---|---|---|
| v1, 114 sensors | OOB (the thesis's 0.25 / 0.37) | 0.25 | 0.37 |
| v1, 114 sensors | location-grouped | −0.08 | −0.74 |
| v2 | OOB | 0.04 | −0.01 |
| v2 | grouped | 0.04 | −0.14 |
| v2 | OOB with volume | 0.08 | 0.11 |

"0.81 for RMSE" is not reproduced by this specification.

- Insert `% TODO(author): the R^2 = 0.81 'preliminary study' value is not reproducible from the current pipeline (values near 0.81 arise only when volume is a predictor). Original sentence commented out below.`
- Comment out the current paragraph with `%`.
- Insert:

```latex
A random forest regressor (300 trees) was trained to predict relative error (CV-RMSE) from four quality features: active completeness, the mean and standard deviation of sequence length, and the number of sequences (volume is not included). It explains almost none of the cross-sensor variance (out-of-bag $R^2 = 0.04$ at $h=8$ and $-0.01$ at $h=24$). Holding out whole locations, so that both directions of a sensor fall in the same fold, gives similar values ($0.04$ and $-0.14$); this matters because out-of-bag estimates on individual streams can be optimistic when the two directions of one location are split between the bootstrap sample and the out-of-bag set. Adding mean volume as a fifth feature raises out-of-bag $R^2$ only to $0.08$ and $0.11$. Although the quality features are correlated with relative error in rank terms (Table~\ref{tab:exp1-spearman}), they therefore do not predict it at the level of individual sensors, which points to the need for further context, such as the spatial information explored in Chapter~\ref{ch:exp2}.
```

Limitation: E.1, item 5.

### 5-23 [CORRECTION] Lines 509–522: matched pairs (T2-G)

**(a) Line 509.**
- First sentence: already softened by the author; no change.
- `This paired benchmark isolates exactly 50,051 matched prediction steps across 108 viable sensors spanning January 2024 to June 2026.` → `This gives 50,051 matched prediction steps across 108 sensors (51--1,147 per sensor, median 377), spanning January 2024 to June 2026. The remaining two of the 110 streams, both directions of \nolinkurl{NclQuaysideWesleySquare}, have no test timestamps in common between the two arms.`

**(b) Fig. 5.13 caption (line 514).** Already fixed by the author; no change.

**(c) Line 520.** After `($p < 0.001$, paired $t$-test)`, insert `; median paired difference $+2.11~\text{pp}$`.

**(d) Line 521.**
- `Crucially, over 90\% of all matched prediction steps occur between 13:00 and 22:00 UTC. The un-imputed model produces almost no matched test predictions between 00:00 and 08:00 UTC` → `Of the matched prediction steps, 90.6\% occur between 13:00 and 22:59~UTC and only 4.2\% between 00:00 and 08:59~UTC,`
- `rendering the un-imputed model entirely blind during early mornings.` → `so the un-imputed model rarely produces forecasts in the early morning.`

**(e) Line 522, append:** `Of the KNN windows, 349,673 (28.6\%) have an imputed target value; counting only windows with an observed target (871,897), the expansion is 2.6-fold.`

### 5-24 [CORRECTION] Table 5.7 (lines 535–538): units

- Line 535: already `pp`; no change.
- Line 537: `$+4.11~\text{pp}$` → `$+4.11~\text{pp}$ (difference of medians)`.
- Line 538: `$\mathbf{+887{,}653\text{ windows (3.7}\times\text{)}}$` → `$\mathbf{+887{,}653}$ ($3.7\times$; $2.6\times$ with observed targets)`.

### 5-25 [STYLE] Lines 546 and 557

- **Line 546.** `the apparent $+4.34~\text{pp}$ reconstruction noise penalty on matched data` → `the $+4.34~\text{pp}$ CV-RMSE penalty (and 20\% higher pooled RMSE, 17.62 versus 14.67 counts) on matched data`.
- **Line 557.**
  - `exposed the profound operational limitation` → `exposed an operational limitation`.
  - `to deliver a 3.7-fold expansion in nominal forecast windows` → `to deliver a 3.7-fold increase in nominal forecast windows (2.6-fold with observed targets)`.
  - `risks fabricating a dangerous illusion of situational awareness` → `risks giving a misleading impression of situational awareness`.

### 5-26 [CORRECTION] Lines 553 and 555: chapter summary

**(a) Line 553, second sentence.**
- OLD: `Crucially, by preserving genuine data absences rather than resorting to naive zero-filling, the experiments demonstrated that consecutive sequence length—rather than marginal data completeness—governs forecast accuracy, establishing a concrete operational breakpoint at approximately 3.0 hours.`
- NEW: `By preserving genuine data absences rather than zero-filling them, the experiments showed that relative forecast error is associated with both consecutive sequence length and data completeness, which are themselves strongly correlated. Sequence length shows a threshold relationship at the 2-hour horizon: error falls steeply as average run length increases up to about 3.0 hours (95\% CI 1.9--3.4~h) and changes little beyond it. No such threshold was found at the 6-hour horizon.`

**(b) Line 555.** `outlier analysis on Shields Road confirmed the stability of the 3-hour threshold for routine commuting` → `the best-performing Shields Road sensors were consistent with the 3-hour threshold`.

**(c) Add a Limitations section.** Insert before `\section{Chapter Summary}` at line 549 (text in Part E.1).

### 5-27 [BOTH] Line 158: a metric shared with Chapter 6 (T2-V)

Chapter 5 reports RMSE, MAPE and CV-RMSE; Chapter 6 reports MAE and MASE. Append to the end of the paragraph at line 158:

```latex
For reference against Chapter~\ref{ch:exp2}, which reports mean absolute error (MAE), the LSTM-MV has a median per-sensor MAE of 4.32 counts at $h=8$ (interquartile range 3.21--8.24) and 3.35 counts at $h=24$ (2.49--7.08). These values are per directional stream rather than per station, and at longer horizons than those of Chapter~\ref{ch:exp2}, so they indicate scale only.
```

- Source: `artifacts/quality_full_v2/runs/lstm_mv_h{8,24}_none/predictions.parquet`, mean |error| per sensor, median over 110 sensors.
- The lower value at $h=24$ is the same sample-composition effect already explained at lines 133–137.

Limitation: E.2, item 11.

### 5-28 [CORRECTION] Cross-chapter: the "alternating daily" split description in Chapters 4 and 6

The split wording corrected in 5-02 also appears outside Chapter 5. Use the same terminology ("run-level micro-split") everywhere.

| File | Line | OLD | NEW |
|---|---|---|---|
| `chapter6_experiment2_multi_node.tex` | 111 (Fig. 6.1 caption) | `un-imputed observations are partitioned via alternating-day micro-splits across each device's idiosyncratic lifespan` | `un-imputed observations are partitioned run by run (run-level micro-splits) across each device's lifespan` |
| `chapter6_experiment2_multi_node.tex` | 148 | `(both uninterpolated alternating micro-splits and interpolated chronological macro-splits)` | `(both uninterpolated run-level micro-splits and interpolated chronological macro-splits)` |
| `chapter4_methodology.tex` | 290 (caption) | `using primary uninterpolated alternating daily micro-splits` | `using primary uninterpolated run-level micro-splits` |
| `chapter4_methodology.tex` | 294 | `employs an alternating daily micro-split protocol` | `employs a run-level micro-split protocol` |
| `chapter4_methodology.tex` | 315 (table) | `Primary: Alternating daily micro-splits (un-imputed, 70/15/15\% deficit-balanced)` | `Primary: Run-level micro-splits (un-imputed, 70/15/15\% deficit-balanced)` |

**[AUTHOR]** The images themselves carry the old wording:
- `scripts/eda/plot_experiment_sensor_availability.py` (lines 132, 319, 354) prints "Un-imputed Alternating-Day Splits" on the panel title used in Fig. 6.1 and in the Chapter 4 figure.
- `scripts/eda/plot_experiment1_split_comparison.py` (Panel C) describes "Daily assignment".

Both scripts are under your active edit, so they were not changed here. Regenerate the figures after updating the labels.

Context: runs are cut only at missing steps (`src/quality/quality_metrics.py:57-79`), never at midnight. Because nightly dropouts usually end a run overnight, many runs span roughly one day of activity, which probably explains the original "daily" description. The assignment unit is still the run, not the day.

---

## C. Chapter 6: edits, in file order

The six nodes with no test targets are referred to below as **the six nodes**:
- `NclJohnDobsonStNorthumberlandRdNorth`
- `NclSideStCrownPosadaNorth`
- `NclSideStCrownPosadaSouth`
- `NclNinthAveSecondAve`
- `NclJesmondDeneRdBentonBank`
- `NclHeatonParkViewHeatonParkRd`

### 6-01 [STYLE] Line 12: "hyperparameter sweeps"

The only sweeps are the four capacity ablations, and they were run on GAT-Transformer (see 6-22).

- OLD: `structural edge conditioning, and hyperparameter sweeps, this chapter isolates`
- NEW: `structural edge conditioning, and capacity ablations, this chapter isolates`

The count of 46 runs is correct: 14 trainable configurations × 3 seeds, plus 4 deterministic baselines.

### 6-02 [BOTH] Lines 90–99: the 30% threshold is asserted, not tested (T2-L)

No threshold sweep and no N = 81 run of the Experiment 2 models exists. The model also *does* receive a
mask channel (line 95), so the claim that zeros are indistinguishable needs softening.

**(a) Line 90, paragraph heading.**
- OLD: `\paragraph{Vulnerability of Standard GNNs to Missingness Collapse and the 30\% Threshold Rationale.}`
- NEW: `\paragraph{Rationale for the 30\% Completeness Threshold.}`

**(b) Line 91.**
- OLD: `The operational necessity of pruning sensors with $< 30\%$ availability is rooted in the mathematical mechanics of spatial message passing in standard Graph Neural Networks.`
- NEW: `The decision to prune sensors with $< 30\%$ availability was motivated by how spatial message passing propagates inputs between nodes.`

**(c) Line 95: append to the end of the paragraph.**
`The mask channel allows the network, in principle, to distinguish a zero-filled input from an observed zero, but nothing in the architecture forces it to do so.`

**(d) Line 97: replace the last two sentences.** The OLD text starts at `In spatial graph convolutions, this distortion does not remain localised:` and ends at `inducing representation collapse across entire subgraphs.`
- NEW: `In a graph model this distortion need not remain local: during message passing, the zero-filled inputs of a sparse node are aggregated into the representations of its neighbours $j \in \mathcal{N}(i)$, which could bias their attention weights $\alpha_{ij}$ and their gradients. We did not measure this effect directly in this chapter.`

**(e) Line 99: replace the first sentence.**
- OLD: `Enforcing the $\ge 30\%$ historical completeness threshold therefore serves as a defensive constraint, isolating a robust core subgraph of 54 stations where observation density is sufficient to stabilise spatial attention weights without triggering numerical degeneration.`
- NEW: `The $\ge 30\%$ threshold is therefore a precautionary design choice rather than an empirically optimised value: no threshold sweep was run, and the sensitivity of the Experiment~2 models to including sparse nodes is left to Chapter~\ref{ch:exp3}, which trains on all 81 candidate stations.`

**(f) Line 99: also soften the end of the paragraph.**
- OLD: `which reclaims all 81 candidate stations natively through joint self-supervised masked reconstruction and inductive spatial kriging without any data deletion.`
- NEW: `which retains all 81 candidate stations by combining forecasting with self-supervised masked reconstruction and inductive spatial kriging.`

**Limitation:** Part E.2, item 1.

### 6-03 [CORRECTION] Lines 101–106: split dates and the 45 steps

The splits are assigned by **window start** (W + H_max = 16 steps). The first 15 windows of validation and of test are
**purged** so that no step is shared across splits. Those 2 × 15 purged windows, plus the 15 trailing grid positions that cannot
start a full window, make up the 45. The data run to 7 June 2026 23:45.

The dates in the current text are computed as if the splits were contiguous, so they are wrong. The values below
are from `_exp2_verification.md` §4; the dates are target timestamps.

Replace lines 101–106 with:

```latex
The dataset spans 3.5 years, from 8 January 2023 00:00 to 7 June 2026 23:45, aggregated at a 15-minute interval ($\Delta t = 15~\text{min}$), giving $T = 119{,}712$ time steps. Sliding windows of $W + H_{\max} = 16$ steps are assigned chronologically to the training, validation and test splits (70/15/15\%) by their start time. To prevent leakage, the first 15 windows of the validation and test splits are discarded, so no input or target step of a validation or test window is used by an earlier split. Together with the 15 trailing positions that cannot start a complete window, this removes 45 window positions and leaves:
\begin{itemize}
    \item \textbf{Training split:} 83,787 windows (targets from 8 January 2023 03:00 to 29 May 2025 22:15);
    \item \textbf{Validation split:} 17,939 windows (targets from 30 May 2025 01:30 to 2 December 2025 22:45);
    \item \textbf{Test split:} 17,941 windows (targets from 3 December 2025 02:00 to 7 June 2026 23:45).
\end{itemize}
```

**Follow-on edit:** line 283, `on the held-out test split (17,941 time steps)` → `on the held-out test split (17,941 windows)`.

**Optional edit:** the Fig. 6.1 caption (line 111) gives approximate dates such as "29 May–2 December 2025". These are acceptable as they stand.

### 6-04 [CORRECTION] Line 120: D_in = 6, not 3

The code builds the input as the count, the mask, sin/cos of time-of-day and sin/cos of time-of-week (`spatial/dataset.py:5-13`; config
`time_features: [day, week]`).

- OLD: `where $D_{\text{in}} = 3$ channels correspond to the $z$-score standardised pedestrian count (zero-filled during missing intervals), an explicit binary missingness indicator mask $m_{i,t} \in \{0, 1\}$ (where $1$ denotes an observed count and $0$ an unobserved step), and temporal cyclical time-of-day/day-of-week embeddings.`
- NEW: `where the $D_{\text{in}} = 6$ channels are: the pedestrian count, $z$-score standardised per node using training-split statistics and zero-filled during missing intervals; a binary observation mask $m_{i,t} \in \{0, 1\}$ (1 = observed, 0 = unobserved); and the sine and cosine of time-of-day (period $24~\text{h}$) and of time-of-week (period $168~\text{h}$).`

### 6-05 [CORRECTION] Lines 128–133: training loss is masked MSE, not MAE

Two facts from the code:
- **Training** minimises masked **MSE on $z$-scored targets**, pooled over nodes, horizons and the batch (`dataset.py:109-122`, `base.py:113`).
- **Evaluation** uses MAE on raw counts over observed targets only, as described.

Replace lines 128–133 with:

```latex
To ensure that zero-filled inputs do not distort gradient updates or benchmark metrics, both the training loss and all evaluation metrics are computed \textbf{only over observed target points}. The networks are trained with a masked mean squared error on $z$-scored targets $z_{i,t+h}$:
\begin{equation}
    \mathcal{L}_{\text{MSE}} = \frac{\sum_{i=1}^N \sum_{h \in \{1, 2, 4\}} m_{i, t+h} \left(z_{i, t+h} - \hat{z}_{i, t+h}\right)^2}{\sum_{i=1}^N \sum_{h \in \{1, 2, 4\}} m_{i, t+h}}
    \label{eq:exp2-masked-loss}
\end{equation}
where $m_{i,t+h} \in \{0,1\}$ is the observation mask of the target, and the sums also run over the windows in a mini-batch. Forecasts are transformed back to raw counts for evaluation, and MAE, RMSE and MASE are computed over the same observed targets. Missing targets are therefore excluded from both the objective and the reporting tables.
```

**Search:** check the rest of the thesis for references to `eq:exp2-masked-loss` that call it an MAE loss, e.g. in Chapter 4 or Chapter 7.

### 6-06 [CORRECTION] Line 150: why only 48 nodes are evaluated (T2-U)

The six nodes all stopped reporting before the test period started; their last observations fall between September 2024 and November 2025. They have 32k–61k
training targets, so a MASE scale exists for each of them. The stated reason is therefore wrong. MAE is also a median over 48 nodes, not 54.

- OLD: `While all $N = 54$ synchronised stations participate in joint spatial message passing and are evaluated on global MAE, exactly 6 peripheral stations lack complete historical cycles to compute the in-sample weekly seasonal-naive scale denominator required for MASE; consequently, MASE evaluations and paired Diebold--Mariano tests are reported across the $N = 48$ stations possessing valid seasonal reference baselines.`
- NEW: `All $N = 54$ stations take part in spatial message passing during training and inference. However, six stations (\nolinkurl{NclJohnDobsonStNorthumberlandRdNorth}, \nolinkurl{NclSideStCrownPosadaNorth}, \nolinkurl{NclSideStCrownPosadaSouth}, \nolinkurl{NclJesmondDeneRdBentonBank}, \nolinkurl{NclNinthAveSecondAve} and \nolinkurl{NclHeatonParkViewHeatonParkRd}) stopped reporting between September 2024 and November 2025 and have no observed targets in the test split. All test metrics (MAE, RMSE and MASE), node medians and Diebold--Mariano tests are therefore computed over the $N = 48$ stations with observed test targets. The six stations still contribute inputs to their neighbours' message passing, as zero-filled counts flagged by the mask channel, throughout the test period.`

**Limitation:** Part E.2, item 6 (test-period coverage).

### 6-07 [CORRECTION] Fig. 6.2 caption (line 186) and line 195: Euclidean, not walking-corridor adjacency (T2-P)

The code builds every topology, including Gabriel, from **haversine (straight-line) distances**. The walking network enters only through the
edge features.

- Line 186, OLD: `The Gabriel graph preserves strict planar walking corridor adjacency, while unconstrained topologies reveal citywide preferential routing.`
- Line 186, NEW: `The Gabriel graph links each sensor only to its nearest straight-line neighbours (a planar proximity graph), whereas the fully connected and learned topologies allow attention between any pair of sensors.`
- Line 195, OLD: `The Gabriel graph represents a sparse, non-crossing physical prior with median in-degree 3.0 (range: 1--5), matching the natural connectivity of planar pedestrian walking corridors.`
- Line 195, NEW: `The Gabriel graph is a sparse, non-crossing proximity prior with median in-degree 3.0 (range: 1--5). It is constructed from straight-line distances, so an edge does not guarantee a direct walking route; walking-network information enters the model only through the edge attributes (Section~\ref{subsec:exp2-edge-features}).`

### 6-08 [CORRECTION] Line 198: learned adjacency formula (T2-P)

From `src/graph_learner.py:211-250`, the learned adjacency:
- uses **one** embedding E (a linear map from the 15 static node features to 32 dimensions), with the formula $\mathrm{ReLU}(\mathrm{ReLU}(EE^\top) - 0.5)$;
- is symmetric and static after training;
- is **not** free of geography, because the static features include distance to the city centre and every learned edge carries the physical edge attributes;
- has 1,756 non-self edges for seed 42, but 1,696 and 1,762 for seeds 43 and 44.

Replace the item text after `\textbf{Learned Adjacency (\texttt{gat\_learned}, Figure~\ref{fig:topol-learned}):}` with:

```latex
An end-to-end learned graph. A linear layer maps each node's 15 static descriptors (including network centrality measures and distance to the city centre) to an embedding $\mathbf{e}_i \in \mathbb{R}^{32}$, and the adjacency is $\mathbf{A}_{\text{learned}} = \mathrm{ReLU}\!\left(\mathrm{ReLU}(\mathbf{E}\mathbf{E}^{\top}) - 0.5\right)$, where $\mathbf{E} \in \mathbb{R}^{N \times 32}$ stacks the embeddings. The matrix is symmetric, is learned jointly with the forecaster, and is fixed once training ends (it does not vary with the input window). The 0.5 threshold sparsifies the graph to 1,696--1,762 directed non-self edges across the three seeds (1,756 for seed~42, shown in Figure~\ref{fig:topol-learned}). Each learned edge carries the same nine physical edge attributes as the static graphs plus the learned weight, so the model is not free of geographic information; it is free only of a predefined topology.
```

**Follow-on edits:** "without coordinate priors" appears at line 272, and "parameterised without physical coordinates" at line 509.
- Line 272: replace `without coordinate priors` with `without a predefined topology`.
- Line 509: see 6-18.

### 6-09 [CORRECTION] Lines 205–210: edge-attribute vector (T2-P)

The thesis vector does not exist in the code. Δx, Δy, θ, log-distance and is_bridge are **not** implemented. The actual nine attributes are listed in
`_exp2_verification.md` §2.

Replace Eq. `exp2-edge-dim` and the paragraph that follows (lines 205–210) with:

```latex
To evaluate whether walking-network geometry provides a useful inductive bias, each directed edge in the Gabriel, $k$NN and FC graphs carries a 9-dimensional attribute vector $\mathbf{a}_{ij} \in \mathbb{R}^9$ derived from the OpenStreetMap walking network:
\begin{equation}
    \mathbf{a}_{ij} = \left[ \tilde{d}^{\,\text{net}}_{ij},\; \tilde{d}^{\,\text{euc}}_{ij},\; c_{ij},\; f^{\text{ped}}_{ij},\; f^{\text{res}}_{ij},\; f^{\text{ter}}_{ij},\; f^{\text{sec}}_{ij},\; f^{\text{pri}}_{ij},\; f^{\text{ser}}_{ij} \right]^{\top}
    \label{eq:exp2-edge-dim}
\end{equation}
where $\tilde{d}^{\,\text{net}}_{ij}$ is the shortest walking-network path length and $\tilde{d}^{\,\text{euc}}_{ij}$ the straight-line (haversine) distance, both divided by the largest shortest-path length in the network; $c_{ij} = d^{\,\text{euc}}_{ij} / d^{\,\text{net}}_{ij} \in (0, 1]$ is the route directness (1 for a straight route, lower for a circuitous one); and $f^{(\cdot)}_{ij}$ are the fractions of the shortest-path length on six collapsed OpenStreetMap \texttt{highway} classes: pedestrian (pedestrian, footway, path, steps, corridor), residential (residential, living street), tertiary (tertiary, unclassified), secondary, primary (primary, trunk) and service (service, track, bridleway, cycleway and untagged ways). The six fractions sum to one. Self-loops carry a zero vector except $c_{ii} = 1$.
```

**Follow-on edit (line 518):** `(distance, bearing, path class)` → `(network and straight-line distance, route directness and road-class composition)`.

### 6-10 [BOTH] Lines 217–233: training protocol (T2-N)

The code differs from the thesis in four places:
- **No weight decay.** The optimiser is plain `Adam(params, lr)`.
- **Early stopping** monitors pooled masked validation MSE on $z$-scored targets over all horizons, not node-median validation MAE at $h=1$. Champion *selection* does use node-median validation MAE at $h=1$.
- **The grid search has no artefacts.** None exist for the search described at lines 227–233. Only `spatial_full.yaml` and a smoke config exist, and every model uses the same fixed block.
- **The epoch cap was reached** by all three GAT-LSTM seeds, all three kNN-8 seeds and all three gabriel_plain seeds.

**(a) Line 217.**
- OLD: `To ensure absolute experimental parity across all topological variants and model families, the identical training configuration was strictly and uniformly enforced across every single one of the 46 experimental runs:`
- NEW: `All trainable models in the 46 runs share one fixed training configuration:`

**(b) Line 219.**
- OLD: `\item \textbf{Optimisation:} Adam optimiser ($\beta_1 = 0.9, \beta_2 = 0.999$, weight decay $10^{-4}$) with initial learning rate $\eta = 10^{-3}$;`
- NEW: `\item \textbf{Optimisation:} Adam optimiser ($\beta_1 = 0.9, \beta_2 = 0.999$, no weight decay) with initial learning rate $\eta = 10^{-3}$, minimising the masked MSE of Equation~\eqref{eq:exp2-masked-loss};`

**(c) Line 221.**
- OLD: `\item \textbf{Early Stopping:} Maximum 60 epochs with a patience of 8 validation epochs based on node-median validation MAE at $h=1$;`
- NEW: `\item \textbf{Early Stopping:} Maximum 60 epochs, with a patience of 8 epochs on the masked validation MSE pooled over nodes and horizons. All three GAT-LSTM (Gabriel) seeds, and the $k$NN-8 and unconditioned-Gabriel GAT-Transformer seeds, reached the 60-epoch limit without triggering early stopping;`

**(d) Line 224, append:** `The champion configuration was selected on the validation split, by the lowest seed-mean node-median validation MAE at $h=1$.`

**(e) Lines 227–233 [AUTHOR].**
- Comment out the whole block with `%`.
- Precede it with `% TODO(author): no artefacts of this grid search exist in configs/ or artifacts/. Restore this block only if the search logs can be supplied.`
- Insert in its place:

```latex
These hyperparameters were fixed \emph{a priori} from common practice for models of this size rather than tuned; in particular, the temporal baselines and all graph variants share the same settings, and no topology-specific tuning was performed.
```

**Limitation:** Part E.2, items 3 and 4.

### 6-11 [CORRECTION] Lines 245–247: baseline definitions

- **Line 245, OLD:** `Predicts the historical mean for node $i$ conditioned on the specific day-of-week and 15-minute time-of-day bin across the training split.`
- **Line 245, NEW:** `Predicts the training-split mean for node $i$ in the same hour-of-week bin (168 hourly bins).`
- **Line 247:** append a new sentence after the item: `For both seasonal-naive baselines, if the reference value is unobserved, the forecast falls back to the last observed value (persistence).`

### 6-12 [BOTH] Lines 199, 254–255 and 262: the "identical parameters" claims (T2-O, T2-R)

The temporal baselines (`lstm`, `transformer`) are **per-node models with a single linear head**, not GAT models run on the identity graph. The GAT
models add a residual GATv2 stack and a 2-layer MLP decoder (`param_counts.csv`). For example, GAT-LSTM has 74,115 parameters against 51,907 for LSTM, i.e. 43% more.
The only matched control is `gat_transformer|identity`, with 89,411 parameters against 90,563 for `gat_transformer|gabriel`. The 1,152-parameter difference is the edge-attribute projection.

**(a) Line 199.**
- OLD: `passing the full spatio-temporal GAT architecture through the identity graph completely disables cross-sensor spatial aggregation while preserving model parameters, isolating the exact capability of the temporal backbone.`
- NEW: `passing the GAT-Transformer through the identity graph disables cross-sensor aggregation while keeping the architecture otherwise unchanged (89,411 versus 90,563 trainable parameters; the difference is the edge-attribute projection, which has no inputs on a self-loop-only graph).`

**(b) Line 251, append:** `These baselines use a single linear output head, whereas the spatio-temporal models use a residual GATv2 stack followed by a two-layer MLP decoder; the comparison in Table~\ref{tab:exp2-spatial-gain} therefore measures the combined effect of spatial message passing and this additional capacity (Panel~C isolates message passing alone).`

**(c) Line 262.**
- OLD: `This severs cross-sensor message passing while retaining identical GAT parameters, providing an exact topological ablation control.`
- NEW: `This severs cross-sensor message passing while retaining the same architecture (apart from the edge-attribute projection), making it the closest available control for the effect of message passing.`

### 6-13 [CORRECTION + T2-O] Tables 6.3 and 6.4: parameter-count column and MASE caption

**(a) Table 6.3 (`tab:exp2-main-results`).**
- **Caption, OLD:** `MASE is computed relative to the in-sample weekly seasonal-naive scale (evaluated across the 48 non-sparse nodes).`
- **Caption, NEW:** `MASE is scaled by each node's in-sample MAE of the daily seasonal-naive forecast (lag 96 steps). All metrics are medians over the 48 nodes with observed test targets. \#Params is the number of trainable parameters.`
- **Columns:** change `\begin{tabular}{llcccc}` to `\begin{tabular}{llccccr}`. Add header cell `& \textbf{\#Params}`, change `\multicolumn{6}` to `\multicolumn{7}`, and append these cells to the rows:

| Row | Append |
|---|---|
| gat\_lstm Gabriel | `& 74{,}115` |
| gat\_learned | `& 91{,}203` |
| gat\_transformer kNN-8 | `& 90{,}563` |
| gat\_transformer FC | `& 90{,}563` |
| gat\_transformer Gabriel | `& 90{,}563` |
| gat\_transformer Gabriel (Unconditioned) | `& 89{,}411` |
| gat\_plain | `& 31{,}235` |
| gat\_transformer Identity | `& 89{,}411` |

**(b) Table 6.4 (`tab:exp2-baselines-results`).** Same column change. Append these cells:

| Row | Append |
|---|---|
| lstm | `& 51{,}907` |
| transformer | `& 68{,}355` |
| each of the four heuristic rows | `& 0` |

Also add this sentence to the caption: `MASE uses the same daily seasonal-naive scale as Table~\ref{tab:exp2-main-results}.`

**(c) Table 6.4, line 327:** `Historical Mean (Day $\times$ 15-min bin)` → `Historical Mean (hour-of-week bin)`.

### 6-14 [CORRECTION] Lines 336 and 338: tier ranges

- **Line 336, OLD:** `Every learned neural model reduces the naive error floor by $25.8\%\text{--}38.5\%$ at $h=1$.`
- **Line 336, NEW:** `Relative to the strongest naive baseline (weekly seasonal naive, MAE 6.392), every learned model reduces $h=1$ MAE, by between $25.8\%$ (\texttt{gat\_plain}) and $31.9\%$ (\texttt{gat\_lstm}).`
- **Line 338:** `($\text{MAE} \approx 4.35\text{--}5.2$)` → `($\text{MAE} \approx 4.35\text{--}5.5$)`. The spatial tier includes `gat_plain` at 5.516 for $h=4$.

### 6-15 [CORRECTION] Fig. 6.3 caption, line 356: MASE scale is daily, not weekly

The code computes `|raw[t] − raw[t−96]|` over training steps (`evaluate.py:22-42`). This was verified against the artefacts to a relative error of 4×10⁻⁸.

- OLD: `The MASE scaling denominator is computed in-sample using weekly seasonal-naive differencing ($m = 672$ steps, 7 days $\times$ 96 steps/day); this in-sample baseline scaling explains why the test-split Seasonal Naive weekly median falls below 1.0 (0.865).`
- NEW: `The MASE denominator is each node's in-sample MAE of the daily seasonal-naive forecast ($m = 96$ steps), computed over the training split; a MASE below 1 therefore means a forecast is more accurate on the test split than the daily seasonal-naive forecast was in-sample, which is why the weekly seasonal-naive baseline attains a median of 0.865.`

**Search:** look for "weekly seasonal-naive scale", "m = 672" or "672" used in connection with MASE elsewhere in the thesis, especially Chapter 4 (metric definitions) and Chapter 7, and correct any you find.

### 6-16 [BOTH] Lines 363–395: spatial gain (add Panel C and the paired figure)

**(a) Table 6.4 (`tab:exp2-spatial-gain`): add Panel C before `\bottomrule`.** This compares two models with the same architecture, with and without edges, so it isolates message passing.

```latex
        \midrule
        \multicolumn{6}{l}{\textbf{Panel C: GAT-Transformer (Gabriel) versus GAT-Transformer (Identity graph, matched architecture)}} \\
        $h = 1$ (15\,min) & 4.622 $\pm$ 0.019 & 4.401 $\pm$ 0.015 & $-0.221$ & \textbf{$-4.8\%$} & $0.598 \pm 0.001 \to 0.583 \pm 0.002$ \\
        $h = 2$ (30\,min) & 5.276 $\pm$ 0.039 & 4.817 $\pm$ 0.042 & $-0.459$ & \textbf{$-8.7\%$} & $0.634 \pm 0.002 \to 0.612 \pm 0.001$ \\
        $h = 4$ (60\,min) & 5.620 $\pm$ 0.024 & 5.217 $\pm$ 0.023 & $-0.403$ & \textbf{$-7.2\%$} & $0.692 \pm 0.006 \to 0.653 \pm 0.001$ \\
```

The identity-graph MASE values come from `aggregated/model_comparison.csv`. The Gabriel values match Table 6.4 Panel B.

**(b) Caption, append:** `Panels~A and~B compare against per-node temporal baselines with a linear output head, so they combine the effect of message passing with that of the additional GAT and decoder layers (74,115 versus 51,907 parameters for Panel~A). Panel~C compares architecturally matched models with and without cross-node edges, and isolates message passing alone. No GAT-LSTM run on the identity graph is available.`

**(c) Line 389.**
- OLD: `incorporating the Gabriel graph GAT yields a robust $5.4\%\text{--}11.8\%$ error reduction over the isolated temporal backbones.`
- NEW: `incorporating the Gabriel graph GAT reduces node-median MAE by $5.4\%\text{--}11.8\%$ relative to the temporal-only backbones, and by $4.8\%\text{--}8.7\%$ relative to the architecturally matched identity-graph control (Panel~C).`

**(d) After line 393 (the end of the itemize), insert:**

```latex
Because node-median values can hide heterogeneity, Figure~\ref{fig:exp2-paired-gain} shows the paired per-node gain. For the LSTM backbone, the median per-node gain rises from $3.7\%$ at $h=1$ to $5.0\%$ at $h=2$ and $6.4\%$ at $h=4$, and the GAT improves on 46, 46 and 47 of the 48 nodes respectively. For the Transformer backbone, the corresponding medians are $3.1\%$, $4.2\%$ and $5.1\%$, with 45, 46 and 48 of 48 nodes improving. The per-node median gain therefore increases monotonically with horizon. The peak at $h=2$ in Table~\ref{tab:exp2-spatial-gain} is a property of the node-median MAE (a ratio of medians), not of the typical node.
```

**(e) Line 395.** Soften the kinematic explanation to a hypothesis, because the per-node gain does *not* plateau.
- OLD: `This finding reflects the physical kinematics of pedestrian movement:`
- NEW: `One interpretation is based on pedestrian kinematics:`
- OLD: `causing the marginal lift from immediate topological neighbours to plateau as broader diurnal and schedule transitions re-assert prominence.`
- NEW: `which may limit the additional value of immediate neighbours. We did not test this mechanism directly, and the per-node gains in Figure~\ref{fig:exp2-paired-gain} continue to rise at $h=4$.`

### 6-17 [CORRECTION] Lines 397–437: Diebold–Mariano tests (T2-M)

The current Table 6.5 has five problems:
- it tests `gat_transformer|gabriel`, which is not the champion;
- it reports one seed per row, and the seed differs between rows (42 vs 43);
- it uses squared-error loss, in counts², but does not say so;
- at $h=1$ it applies no autocorrelation correction;
- it is described as "definitive".

New tests have been run (`dm_champion_seedavg.csv`):
- **Models:** champion `gat_lstm|gabriel` against each comparator.
- **Forecasts:** averaged over seeds 42–44 for every trained model.
- **Loss:** absolute error.
- **Variance:** the repository's DM implementation, with $h-1$ autocovariance lags and the Harvey–Leybourne–Newbold correction.
- **Multiple testing:** BH-FDR across 48 nodes per comparison × horizon.
- **Robustness check:** Newey–West (Bartlett) with a 96-step (1-day) lag. The proportion of significant nodes changes by ≤ 2.1 pp.

**(a) Section title, line 397:** already `Diebold--Mariano`; no change.

**(b) Replace line 400 with:**

```latex
To test whether these gains are consistent at the level of individual sensors, we applied the Diebold--Mariano (DM) test \citep{dieboldComparingPredictiveAccuracy2002} at each of the 48 evaluated nodes, comparing the champion \texttt{gat\_lstm | gabriel} with each comparator. For every trained model, the forecasts of the three seeds were first averaged, so that each test compares two three-member ensembles. The loss is the absolute error on observed test targets, so loss differences are in pedestrian counts. The long-run variance uses $h-1$ autocovariance lags with the small-sample correction of \citet{harveyTestingEqualityPrediction1997}. Because the 15-minute loss differential is autocorrelated beyond $h-1$ lags, we repeated every test with a Newey--West (Bartlett) variance estimator \citep{neweySimplePositiveSemidefinite1987} with a one-day (96-step) lag; the proportion of significant nodes changed by at most 2.1 percentage points. $p$-values were adjusted across the 48 nodes for each comparison and horizon with the Benjamini--Hochberg false discovery rate (\gls{fdr}) procedure \citep{benjaminiRoyalStatisticalSociety}.
```

**(c) Replace the whole `table` environment at lines 402–429 with:**

```latex
\begin{table}[htbp]
    \centering
    \small
    \caption[Diebold--Mariano tests: champion versus comparators]{Per-node Diebold--Mariano tests of the champion \texttt{gat\_lstm | gabriel} against each comparator ($N = 48$ nodes, test split). Trained models use seed-averaged forecasts (seeds 42--44); heuristic baselines are deterministic. Loss is absolute error. ``Better'' is the share of nodes with a lower mean absolute error for the champion. ``Sig.\ better'' and ``Sig.\ worse'' are the shares of nodes where the difference is significant after Benjamini--Hochberg correction ($\alpha = 0.05$) in favour of the champion or the comparator, respectively. $\Delta$MAE is the mean over nodes of the per-node difference in MAE (champion minus comparator, counts per 15~min); negative values favour the champion.}
    \label{tab:exp2-dm-summary}
    \resizebox{\linewidth}{!}{%
    \begin{tabular}{lcccccc}
        \toprule
        \textbf{Comparator} & \textbf{Horizon} & \textbf{Better} & \textbf{Sig.\ better} & \textbf{Sig.\ worse} & \textbf{Median DM} & \textbf{$\Delta$MAE} \\
        \midrule
        \multirow{3}{*}{\shortstack[l]{\texttt{lstm}\\(temporal only)}}
        & $h=1$ & 95.8\% & 91.7\% & 4.2\% & $-8.93$ & $-0.19$ \\
        & $h=2$ & 95.8\% & 95.8\% & 2.1\% & $-11.05$ & $-0.34$ \\
        & $h=4$ & 97.9\% & 97.9\% & 0.0\% & $-11.26$ & $-0.54$ \\
        \midrule
        \multirow{3}{*}{\shortstack[l]{\texttt{transformer}\\(temporal only)}}
        & $h=1$ & 91.7\% & 89.6\% & 6.3\% & $-10.08$ & $-0.26$ \\
        & $h=2$ & 95.8\% & 91.7\% & 2.1\% & $-11.03$ & $-0.40$ \\
        & $h=4$ & 100.0\% & 95.8\% & 0.0\% & $-10.91$ & $-0.64$ \\
        \midrule
        \multirow{3}{*}{\shortstack[l]{\texttt{gat\_transformer}\\(identity graph)}}
        & $h=1$ & 93.8\% & 91.7\% & 2.1\% & $-9.68$ & $-0.23$ \\
        & $h=2$ & 97.9\% & 93.8\% & 2.1\% & $-10.60$ & $-0.37$ \\
        & $h=4$ & 100.0\% & 97.9\% & 0.0\% & $-11.13$ & $-0.59$ \\
        \midrule
        \multirow{3}{*}{\shortstack[l]{\texttt{gat\_transformer}\\(Gabriel)}}
        & $h=1$ & 64.6\% & 47.9\% & 20.8\% & $-1.96$ & $-0.04$ \\
        & $h=2$ & 70.8\% & 54.2\% & 16.7\% & $-2.83$ & $-0.05$ \\
        & $h=4$ & 68.8\% & 54.2\% & 14.6\% & $-2.51$ & $-0.06$ \\
        \midrule
        \multirow{3}{*}{\shortstack[l]{\texttt{snaive\_weekly}\\(heuristic)}}
        & $h=1$ & 100.0\% & 100.0\% & 0.0\% & $-33.24$ & $-2.92$ \\
        & $h=2$ & 100.0\% & 100.0\% & 0.0\% & $-24.61$ & $-2.57$ \\
        & $h=4$ & 100.0\% & 97.9\% & 0.0\% & $-16.87$ & $-2.12$ \\
        \midrule
        \multirow{3}{*}{\shortstack[l]{\texttt{persistence}\\(heuristic)}}
        & $h=1$ & 100.0\% & 100.0\% & 0.0\% & $-32.21$ & $-2.09$ \\
        & $h=2$ & 100.0\% & 100.0\% & 0.0\% & $-29.54$ & $-2.71$ \\
        & $h=4$ & 100.0\% & 100.0\% & 0.0\% & $-29.13$ & $-3.26$ \\
        \bottomrule
    \end{tabular}%
    }
\end{table}
```

**(d) Replace lines 431–437, from `The DM test results` to `systemic across the network.`, with:**

```latex
The DM results (Table~\ref{tab:exp2-dm-summary}) show that the spatial gain is consistent across nodes:
\begin{itemize}
    \item Against the temporal-only LSTM, the champion is significantly more accurate at $91.7\%$, $95.8\%$ and $97.9\%$ of nodes for $h = 1, 2, 4$, and significantly less accurate at no more than two nodes ($4.2\%$) at any horizon.
    \item Against the temporal-only Transformer and the architecturally matched identity-graph GAT-Transformer, the champion is significantly more accurate at $89.6\%\text{--}97.9\%$ of nodes.
    \item Against the GAT-Transformer on the same Gabriel graph, the picture is mixed: the champion is significantly better at $47.9\%\text{--}54.2\%$ of nodes but significantly worse at $14.6\%\text{--}20.8\%$, with a mean difference of only $0.04\text{--}0.06$ counts. The choice of temporal backbone therefore matters far less than the presence of spatial edges.
\end{itemize}
Consistent with this, the per-node seed-mean MAE is lower under the champion than under the temporal LSTM at 46 of the 48 nodes at $h=1$ (mean reduction $3.9\%$; Figure~\ref{fig:exp2-paired-gain}). Because the DM tests compare seed-averaged ensembles, they measure the consistency of the gain across nodes rather than the variability due to random initialisation, which Table~\ref{tab:exp2-main-results} reports separately.
```

**(e)** Insert the paired-gain figure (Part D.4) directly after this paragraph.

**Bibliography:** two new entries are needed; see Part D.6.

### 6-18 [CORRECTION] Lines 449–518: distance decay (T2-Q, Fig. 6.5)

Current state:
- The Fig. 6.5 caption claims degree-normalised α, but the images and the r values in the text are raw α.
- Raw α is confounded with in-degree: Gabriel in-degree ranges from 1 to 5, and α sums to 1 per node.
- All attention exports come from **seed 42**, the **final GATv2 layer**, averaged over **4 heads** and **256 random test windows**.
- The exports exist only for GAT-Transformer and gat_learned. The champion GAT-LSTM has none.

Normalised values are computed as α̃ = α / mean α over the destination's non-self neighbours:

| Topology | r(α̃, d) | p | Mean α̃ <300 m | 300–800 m | ≥800 m |
|---|---|---|---|---|---|
| Gabriel | −0.36 | 5×10⁻⁶ | 1.12 | 0.97 | 0.82 |
| FC | −0.42 | <10⁻¹⁰⁰ | 2.44 | 1.48 | 0.87 |
| Gabriel (no edge attributes) | +0.05 | 0.56 | 0.99 | 1.00 | 1.01 |
| kNN-8 | −0.05 | 0.23 | 1.08 | 0.97 | 0.98 |
| Learned | +0.02 | 0.40 | 0.89 | 0.91 | 1.02 |

**(a) Line 449.**
- OLD: `To answer this, we extracted the test-set attention coefficients $\alpha_{ij}$ emitted by the \gls{gatv2} layer and computed their correlation with inter-sensor Euclidean distances $d(i, j)$:`
- NEW: `To answer this, we extracted the attention coefficients $\alpha_{ij}$ of the final \gls{gatv2} layer, averaged over the four heads and 256 randomly sampled test windows, for the GAT-Transformer variants and the learned-graph model (attention was not exported for the GAT-LSTM champion). The exports are from seed~42. Because the attention weights into each node sum to one, raw $\alpha_{ij}$ is mechanically larger for nodes with few neighbours. We therefore also report the degree-normalised weight $\tilde{\alpha}_{ij} = \alpha_{ij} / \bar{\alpha}_{i}$, where $\bar{\alpha}_{i}$ is the mean attention over node $i$'s non-self neighbours; $\tilde{\alpha}_{ij} = 1$ corresponds to uniform attention. Both are correlated with the straight-line inter-sensor distance $d(i,j)$:`

**(b) Table 6.6 (lines 451–470).** Keep the existing columns and add one column, $r(\tilde{\alpha}, d)$, after $r(\alpha, \log d)$.
- **Tabular spec:** change `{lccccccc}` to `{lcccccccc}`.
- **Header:** add `& $r(\tilde{\alpha}, d)$`.
- **Footnote:** `\multicolumn{8}` → `\multicolumn{9}`.

Row values to insert:

| Row | $r(\tilde{\alpha}, d)$ cell |
|---|---|
| gabriel | `$\mathbf{-0.36}$` |
| fc | `$\mathbf{-0.42}$` |
| gabriel\_plain | `$+0.05$` |
| knn8 | `$-0.05$` |
| learned | `$+0.02$` |

**Caption, append:** `Attention is from seed~42 (final layer, mean over four heads and 256 test windows); the seed-stability column uses all three seeds. $\tilde{\alpha}$ is attention divided by the node's mean non-self attention, which removes the effect of in-degree; mean $\alpha$ by distance band is raw.`

**(c) Replace the figure at lines 472–503** with the new four-panel figure (Part D.3). Keep the label `fig:exp2-distance-decay` so that existing `\ref`s still resolve. The subfigure labels `fig:dist-decay-gabriel`, `-knn`, `-fc` and `-learned` disappear, so update the references in 6-18(d).

**(d) Replace lines 505–509** (the list header and items 1–3) with:

```latex
The evidence for distance decay depends on the topology (Table~\ref{tab:exp2-distance-decay} and Figure~\ref{fig:exp2-distance-decay}):
\begin{enumerate}
    \item \textbf{Moderate decay on the Gabriel graph:} Raw attention correlates with distance at $r = -0.52$, but part of this reflects in-degree. After degree normalisation the correlation is $r = -0.36$ ($p < 0.001$): neighbours within $300~\text{m}$ receive on average $1.12$ times the uniform share, and those beyond $800~\text{m}$ $0.82$ times.
    \item \textbf{Locality recovered in the fully connected graph:} With all 53 candidates available and no geometric pruning, normalised attention still decays with distance ($r = -0.42$): sensors within $300~\text{m}$ receive $2.44$ times the uniform share, compared with $0.87$ times beyond $800~\text{m}$. Since the edge attributes include both distances, this shows that the model learns to use distance information when it is available, rather than rediscovering proximity from flow patterns alone.
    \item \textbf{No decay on the $k$NN-8 and learned graphs:} Normalised attention is unrelated to distance on the $k$NN-8 graph ($r = -0.05$, $p = 0.23$) and on the learned graph ($r = +0.02$, $p = 0.40$). On $k$NN-8, every candidate neighbour is already among the eight nearest, which may leave little distance contrast to exploit; the learned graph's attention is close to uniform (mean normalised entropy $0.97$). Attention therefore does not reflect geography in every configuration.
\end{enumerate}
```

**(e) Line 510, item 4:** keep it, but change `\item \textbf{Alignment with Urban Infrastructure Corridors:} Inspecting the highest-weighted edges reveals` to `\item \textbf{Highest-weighted edges:} On the Gabriel graph, the highest raw attention weights fall on`.

**(f) Lines 516–518: edge features (T2-T).** Replace line 518 with:

```latex
Removing the edge attributes leaves accuracy almost unchanged at $h=1$ (MAE 4.401 versus 4.423, $0.5\%$) and costs $2.0\text{--}2.1\%$ at $h=2$ and $h=4$ (Table~\ref{tab:exp2-main-results}). Its effect on attention is larger: without edge attributes, normalised attention shows no distance decay ($r = +0.05$, $p = 0.56$, compared with $-0.36$ with them), and mean normalised attention entropy rises from $0.94$ to $0.99$ (1.0 is uniform). The raw correlation of $-0.21$ for \texttt{gabriel\_plain} in Table~\ref{tab:exp2-distance-decay} is therefore entirely an in-degree effect. Edge attributes thus make attention distance-dependent and somewhat more selective, but even with them attention remains close to uniform (entropy $0.94$), and their contribution to accuracy is modest.
```

Also apply the 6-09 follow-on edit for `(distance, bearing, path class)`.

### 6-19 [BOTH] Lines 523–552: density, stability and "collinear equifinality" (T2-Q, T2-R)

**(a) Line 523.**
- **Reference:** `(Figure~\ref{fig:exp2-seed-stability-mae})` → `(Table~\ref{tab:exp2-main-results})`.
- **Backbone:** after `Gabriel (4.401), $k$NN-8 (4.392), and Fully-Connected (4.396)`, insert `(all GAT-Transformer)`.

**(b) Line 529.** The receptive-field argument conflicts with Table 6.7: a third layer does not make the model worse.
- OLD: `The sparse Gabriel graph (median in-degree 3.0) constrains long-range multi-step message propagation. Over a 60-minute window, pedestrian flows and crowd movements travel across broader urban walking districts; a sparse planar graph with a 2-layer GNN can aggregate information only from a 2-hop radius, starving multi-step predictions of broader regional context.`
- NEW: `One explanation is receptive field: with a median in-degree of 3.0, a two-layer GAT on the Gabriel graph aggregates information only from a two-hop neighbourhood, whereas the $k$NN-8, FC and learned graphs reach distant sensors in a single hop. The capacity ablation is consistent with this only in part: a third GAT layer on the Gabriel graph improves $h=4$ MAE by just $1.0\%$ (Table~\ref{tab:exp2-capacity-ablation}), much less than the $5.2\%$ gap to the FC graph, which suggests that direct one-hop access matters more than depth. We did not test this directly.`

**(c) Fig. 6.7 caption (line 534).** gabriel\_plain's high stability is degenerate (uniform attention), as the table footnote already says.
- OLD: `Sparse planar graphs (\texttt{gabriel}, $\bar{\rho} = 0.78$) and unconditioned planar graphs (\texttt{gabriel\_plain}, $\bar{\rho} = 0.90$) yield consistent, reproducible attention allocations, whereas unconstrained dense topologies (\texttt{fc}, $\bar{\rho} = 0.28$) collapse to erratic importance distributions due to collinear equifinality among surrogate corridor nodes.`
- NEW: `Attention on the Gabriel graph is reproducible across seeds ($\bar{\rho} = 0.78$). The unconditioned Gabriel graph scores higher ($\bar{\rho} = 0.90$) only because its attention is close to uniform. Attention on the fully connected graph is much less reproducible ($\bar{\rho} = 0.28$).`

**(d) Line 538.** `Yet this multi-step accuracy gain is accompanied by a severe decline in cross-seed attention reproducibility` → `However, this multi-step accuracy gain is accompanied by lower cross-seed reproducibility of attention`.

**(e) Line 542.** `Cross-seed rank correlation plummets to $\rho = 0.28$. Across different seeds, the model selects entirely different subsets of cross-city nodes to achieve the same loss minimum.` → `Cross-seed rank correlation falls to $\rho = 0.28$: different seeds distribute attention over substantially different subsets of distant nodes while reaching similar accuracy.`

**(f) Line 543.** `operating as an uninterpretable global mixer rather than reflecting localised urban dynamics.` → `so that the learned graph behaves more like a global averaging operator than a localised one.` Also add before the closing `\end{itemize}`: `(The learned graph's cross-seed stability was not computed, because its edge set differs between seeds.)`

**(g) Lines 546–552. "Collinear equifinality" is untested; recast it as a hypothesis.**
- Line 546, OLD: `Rather than interpreting this divergence as an inherent, immutable theoretical trade-off, closer inspection reveals it to be an \textbf{empirical artefact of finite network scale ($N = 54$) and severe information concentration} within urban pedestrian networks:`
- Line 546, NEW: `We propose, as a hypothesis not tested here, that this divergence reflects the small network ($N = 54$) rather than an inherent trade-off:`
- Line 549: replace `the optimisation landscape exhibits extensive \emph{equifinality}` with `attention may be weakly identifiable (several attention allocations giving near-identical loss)`.
- Line 549: replace `The low seed stability ($\rho = 0.28$) is thus a direct manifestation of over-parameterised surrogate competition on a small node sample.` with `The low seed stability ($\rho = 0.28$) is consistent with this explanation, although we did not measure the collinearity between candidate neighbours directly.`
- Line 550: replace `The sparse Gabriel graph avoids this equifinality by using physical walking network geometry as a strict regulariser` with `The sparse Gabriel graph may avoid this problem by restricting each node to its nearest straight-line neighbours`.
- Line 552: replace `For current finite deployments ($N \approx 50$), geometric pruning provides an inductive bias: it sacrifices marginal multi-step capacity to prevent collinear equifinality, delivering operationally dependable, interpretable attention weights suitable for infrastructure monitoring.` with `For deployments of this size ($N \approx 50$), geometric pruning trades a small amount of accuracy at $h=4$ ($5.4\%$ higher MAE than FC for GAT-Transformer) for attention weights that are more reproducible across seeds.`
- Rename every remaining occurrence of "collinear equifinality", including in the subsection title at line 520 and the summary at line 671, to "attention non-identifiability".

**Limitation:** Part E.2, item 7.

### 6-20 [STYLE] Lines 559 and 564

- Line 559, subsection title: already changed by the author to `The Inductive Role of Temporal Sequence Encoders`; no change.
- Line 564: "decisive value" no longer appears; no change.

### 6-21 [CORRECTION] Lines 569–574: backbone comparison (T2-R)

The `lstm` and `transformer` baselines are per-node models, not runs on the identity topology. "Statistical parity" was never tested: the $h=1$ gap
of 0.112 is several times the seed SDs. No GAT-LSTM runs exist on kNN-8 or FC.

Replace lines 569–574 with:

```latex
Without spatial edges, the per-node LSTM and Transformer baselines achieve similar accuracy at longer horizons, but not at $h=1$:
\begin{itemize}
    \item Horizon $h=1$: LSTM MAE $4.597 \pm 0.011$ vs Transformer MAE $4.709 \pm 0.090$ ($\Delta = 0.112$);
    \item Horizon $h=4$: LSTM MAE $5.605 \pm 0.014$ vs Transformer MAE $5.646 \pm 0.008$ ($\Delta = 0.041$).
\end{itemize}
When paired with the Gabriel GAT, \texttt{gat\_lstm} has a lower node-median MAE than \texttt{gat\_transformer} at every horizon (4.350 vs 4.401 at $h=1$; 5.051 vs 5.217 at $h=4$). At the node level, however, the advantage is mixed: the champion is significantly better at about half of the nodes and significantly worse at $15\text{--}21\%$ of them (Table~\ref{tab:exp2-dm-summary}). Because GAT-LSTM was run only on the Gabriel graph, we cannot say whether the LSTM backbone would also be preferable on denser topologies.
```

### 6-22 [CORRECTION] Lines 576–608 and Table 6.7: capacity ablation (T2-R, T2-N)

**The capacity ablations (`gat_h1`, `gat_h8`, `gat_l1`, `gat_l3`) are GAT-Transformer runs** (manifest `temporal_encoder: transformer`).
Table 6.7 compares them with a "Champion" row of 4.350/4.671/5.051, which is **GAT-LSTM**. The matched reference is
`gat_transformer|gabriel` (K = 4, L = 2), at 4.401/4.817/5.217.

Against the correct reference:
- the 3-layer and 8-head variants are slightly **better**;
- the conclusions "L = 3 degrades through oversmoothing" and "no improvement from 8 heads" are wrong.

Seed-mean ± SD values, recomputed from `runs/*/results.csv`:

| Config | h=1 | h=2 | h=4 | Params |
|---|---|---|---|---|
| K=1, L=2 | 4.400 ± 0.009 | 4.870 ± 0.022 | 5.223 ± 0.035 | 90,563 |
| **K=4, L=2 (reference)** | **4.401 ± 0.015** | **4.817 ± 0.042** | **5.217 ± 0.023** | 90,563 |
| K=8, L=2 | 4.408 ± 0.019 | 4.759 ± 0.040 | 5.169 ± 0.014 | 90,563 |
| K=4, L=1 | 4.449 ± 0.016 | 4.991 ± 0.056 | 5.293 ± 0.014 | 81,539 |
| K=4, L=3 | 4.390 ± 0.046 | 4.773 ± 0.029 | 5.167 ± 0.039 | 99,587 |

The head count does not change the parameter count: the hidden width is split across the heads.

**(a) Replace Table 6.7 (lines 581–601) with:**

```latex
\begin{table}[htbp]
    \centering
    \small
    \caption[Ablation of GAT heads and layer depth]{Ablation of GATv2 attention heads ($K$) and layer depth ($L$) for the GAT-Transformer on the Gabriel graph. Values are test-split node-median MAE (counts per 15~min), mean $\pm$ 1~SD over three seeds. The reference configuration ($K=4$, $L=2$) is \texttt{gat\_transformer | gabriel} from Table~\ref{tab:exp2-main-results}. The hidden width is split across heads, so $K$ does not change the number of trainable parameters.}
    \label{tab:exp2-capacity-ablation}
    \begin{tabular}{lccccr}
        \toprule
        \textbf{Ablation} & \textbf{Configuration} & \textbf{MAE ($h=1$)} & \textbf{MAE ($h=2$)} & \textbf{MAE ($h=4$)} & \textbf{\#Params} \\
        \midrule
        \multirow{3}{*}{\shortstack[l]{Heads\\($L = 2$)}}
        & $K = 1$ & 4.400 $\pm$ 0.009 & 4.870 $\pm$ 0.022 & 5.223 $\pm$ 0.035 & 90{,}563 \\
        & $K = 4$ (reference) & 4.401 $\pm$ 0.015 & 4.817 $\pm$ 0.042 & 5.217 $\pm$ 0.023 & 90{,}563 \\
        & $K = 8$ & 4.408 $\pm$ 0.019 & 4.759 $\pm$ 0.040 & 5.169 $\pm$ 0.014 & 90{,}563 \\
        \midrule
        \multirow{3}{*}{\shortstack[l]{Depth\\($K = 4$)}}
        & $L = 1$ & 4.449 $\pm$ 0.016 & 4.991 $\pm$ 0.056 & 5.293 $\pm$ 0.014 & 81{,}539 \\
        & $L = 2$ (reference) & 4.401 $\pm$ 0.015 & 4.817 $\pm$ 0.042 & 5.217 $\pm$ 0.023 & 90{,}563 \\
        & $L = 3$ & 4.390 $\pm$ 0.046 & 4.773 $\pm$ 0.029 & 5.167 $\pm$ 0.039 & 99{,}587 \\
        \bottomrule
    \end{tabular}
\end{table}
```

**(b) Line 579.** `systematic hyperparameter ablations were executed on the Gabriel GAT architecture across attention heads and layer depths:` → `we varied the number of attention heads and GAT layers of the GAT-Transformer on the Gabriel graph (the ablation was not repeated for the GAT-LSTM champion):`

**(c) Replace lines 603–608**, from `The capacity ablations` to `topological connectivity.`, with:

```latex
The capacity ablations (Table~\ref{tab:exp2-capacity-ablation}) show that a single GAT layer is clearly insufficient ($+1.1\%$ to $+3.6\%$ MAE relative to the reference), but beyond two layers and four heads the returns are small:
\begin{itemize}
    \item At $h=1$, all variants with $L \ge 2$ lie within $0.011$ counts of the reference, inside one seed standard deviation.
    \item At $h=2$ and $h=4$, eight heads and three layers are slightly \emph{better} than the reference, by $0.9\text{--}1.2\%$ ($0.044\text{--}0.058$ counts), which is comparable to one to two seed standard deviations. There is therefore no evidence of oversmoothing at three layers.
\end{itemize}
These changes are small compared with the effect of topology at $h=4$ (FC is $5.2\%$ better than Gabriel for the same backbone). The evidence therefore suggests that, at this network size, connectivity matters more than GAT capacity, although capacity is not strictly saturated.
```

**(d)** Replacing lines 603–608 removes the sentence at line 608 that referenced Figure `fig:exp2-seed-stability-mae`, and 6-19(a) redirects the line 523 reference to the table. Keep the figure referenced: append this sentence to the end of the new paragraph in (c): `Figure~\ref{fig:exp2-seed-stability-mae} shows the cross-seed variability of every configuration.`

### 6-23 [BOTH] Lines 625–637: completeness and "spatial error propagation" (T2-S)

New evidence (Figure `fig_ch6_completeness`, $h=1$, n = 48):
- **Own completeness vs gain:** completeness predicts MASE (r = −0.33, p = 0.021) but **not** spatial gain (r = +0.10, p = 0.51).
- **Neighbour completeness vs gain:** bivariate r = −0.15 (p = 0.30); partial r controlling for own completeness = −0.29 (p = 0.048).

The error-propagation hypothesis predicts a **positive** relationship: sparser neighbours should mean less gain. The weak partial association runs in the **opposite** direction and would not survive correction for multiple tests. The propagation mechanism is therefore not supported.

**(a) Line 625, append:** `(Figure~\ref{fig:exp2-node-map}a).`

**(b) Line 631, append:** `The correlation strengthens with horizon ($r = -0.37$ at $h=2$ and $-0.39$ at $h=4$). Figure~\ref{fig:exp2-completeness} shows this and the related analyses below.`

**(c) Line 633, last sentence.**
- OLD: `In the multi-node regime of Experiment~2, however, data completeness emerges as a direct, systemic driver of network-wide error.`
- NEW: `In Experiment~2, completeness is also associated with error across nodes, although the association is moderate ($r = -0.33$) and, being cross-sectional, does not establish a causal effect.`

**(d) Replace lines 635–637** (the "fundamental reason" and "spatial error propagation" paragraphs) with:

```latex
One mechanism specific to graph models would be \emph{spatial error propagation}: a sparse node's zero-filled inputs are aggregated into its neighbours' representations, so that missingness harms not only the node itself but also its neighbours. If this were important, nodes with sparse neighbours should gain less from message passing. The data do not support this (Figure~\ref{fig:exp2-completeness}). A node's own completeness is unrelated to its spatial gain ($r = +0.10$, $p = 0.51$), and the mean completeness of its Gabriel neighbours shows no bivariate association with gain ($r = -0.15$, $p = 0.30$). After controlling for the node's own completeness, the partial correlation is $-0.29$ ($p = 0.048$), but its sign is the opposite of that predicted by error propagation, and it would not survive correction for the several tests reported here. Within the $\ge 30\%$ completeness range retained in this chapter, we therefore find no evidence that sparse neighbours reduce the benefit of message passing. Whether this changes once sparser nodes are included is examined in Chapter~\ref{ch:exp3}.
```

**Limitation:** Part E.2, item 8.

### 6-24 [CORRECTION] Lines 642–645: core vs periphery (T2-T)

**(a) Line 642.** `Mapping performance geographically illuminates distinct operational regimes:` → `Figure~\ref{fig:exp2-node-map} maps per-node MASE and spatial gain. Two groups stand out:`

**(b) Line 645.** Ravenswood Road (0.730) ranks third-worst, ahead of Red Hall Drive (0.699).
- OLD: `Adelaide Terrace ($\text{MASE} = 0.773$), Sandyford Road ($\text{MASE} = 0.737$), and Red Hall Drive ($\text{MASE} = 0.699$).`
- NEW: `Adelaide Terrace ($\text{MASE} = 0.773$), Sandyford Road ($\text{MASE} = 0.737$) and Ravenswood Road ($\text{MASE} = 0.730$).`

**(c) Line 644 [AUTHOR].** "high signal-to-noise ratios, and tightly coupled opposing directional footways" is not computed anywhere. Soften to `These locations have high, regular pedestrian volumes.` The Quayside under the High Level Bridge has a node MAE of 8.54 counts.

### 6-25 [CORRECTION] Lines 651–659: Falconar St and Barras Bridge

Values checked against the seed-mean per-node results:

| Node | LSTM MAE | GAT-LSTM MAE | Change | Own completeness | Gabriel in-degree | Mean neighbour completeness |
|---|---|---|---|---|---|---|
| Falconar St | 13.84 | 14.73 | +6.4% | 38.6% (9th-lowest of 48) | **5 (maximum)** | 55.1% |
| Barras Bridge | 14.56 | 15.26 | +4.8% | 33.6% (2nd-lowest of 48) | **5 (maximum)** | 55.2% |

For comparison, the median own completeness across the 48 nodes is 53.6%.

Three corrections follow:
- the absolute MAEs at lines 653–654 are wrong;
- the nodes are **not** low-degree boundary nodes;
- their neighbours are **not** sparse.

The gain varies with horizon:

| Node | $h=1$ | $h=2$ | $h=4$ |
|---|---|---|---|
| Falconar St | −6.4% | −5.3% | −1.3% |
| Barras Bridge | −4.8% | −1.6% | **+1.1%** |

**(a) Line 651.** `exactly two nodes exhibited slight performance degradation under the spatial champion compared to the pure temporal baseline:` → `two nodes had higher error under the spatial champion than under the temporal-only LSTM at $h=1$ (Figure~\ref{fig:exp2-node-map}b):`

**(b) Line 653.** `(increasing from $4.81$ to $5.12$ counts)` → `(from $13.84$ to $14.73$ counts)`

**(c) Line 654.** `(increasing from $11.23$ to $11.77$ counts)` → `(from $14.56$ to $15.26$ counts)`

**(d) Replace lines 657–659 with:**

```latex
Both sensors have low observation completeness: \texttt{FalconarSt} has $38.6\%$, and \texttt{BarrasBridge} has $33.6\%$, the second lowest among the 48 evaluated nodes. Their neighbourhoods, however, are not sparse. Both have five Gabriel neighbours, the maximum in-degree in the graph. The mean completeness of those neighbours ($55.1\%$ and $55.2\%$) is slightly above the median completeness of the evaluated nodes ($53.6\%$). The degradation is also horizon-dependent: at $h=4$, \texttt{BarrasBridge} benefits from the GAT ($+1.1\%$) and \texttt{FalconarSt} degrades by only $1.3\%$. We therefore cannot attribute these two cases to corrupted neighbour inputs. A plausible alternative is that, for a node with sparse training data of its own, the additional GAT and decoder parameters are less well constrained.

With two affected nodes out of 48, this is a limited observation rather than a systematic failure mode. It nevertheless motivates the question addressed in Chapter~\ref{ch:exp3}: how a graph model should handle nodes, and neighbours, whose observations are missing.
```

### 6-26 [CORRECTION] Lines 666–678: chapter summary and bridge

- **Line 666:** `established a rigorous empirical benchmark` → `established an empirical benchmark`.
- **Line 668, item 1:** replace with:
  `\item \textbf{Spatial Information Gives a Consistent Gain:} Coupling a \gls{gatv2} encoder with temporal backbones reduces node-median MAE by $5.4\%\text{--}11.8\%$ relative to temporal-only models, and by $4.8\%\text{--}8.7\%$ relative to an architecturally matched model without cross-node edges. Diebold--Mariano tests on seed-averaged forecasts find the champion significantly more accurate than each temporal-only baseline at $89.6\%\text{--}97.9\%$ of the 48 evaluated nodes (Benjamini--Hochberg corrected).`
- **Line 669, item 2:** replace with:
  `\item \textbf{Spatial Gain Grows with Horizon:} The median per-node gain over the temporal LSTM rises from $3.7\%$ at $h=1$ to $6.4\%$ at $h=4$. The node-median MAE gap peaks at $h=2$ ($-11.8\%$ for the LSTM backbone, $-8.6\%$ for the Transformer).`
- **Line 670, item 3:** replace with:
  `\item \textbf{Attention Partly Reflects Geography:} After correcting for in-degree, attention decays with distance on the Gabriel ($r = -0.36$) and fully connected ($r = -0.42$) graphs, but not on the $k$NN-8 or learned graphs. Edge attributes are needed for this distance dependence, although they change accuracy by only $0.5\text{--}2.1\%$.`
- **Line 671, item 4:** `suffer from interpretability collapse and cross-seed instability ($\rho = 0.28$ vs $0.78$ for Gabriel) due to collinear equifinality among surrogate corridor nodes in finite networks.` → `produce attention that is much less reproducible across seeds ($\rho = 0.28$ for FC versus $0.78$ for Gabriel), which we attribute, as a hypothesis, to non-identifiable attention among similar candidate neighbours.`
- **Line 672, item 5:** replace with:
  `\item \textbf{Completeness and Error:} Nodes with lower completeness have higher error ($r = -0.33$), but we found no evidence that sparse neighbours reduce the benefit of message passing within the retained $\ge 30\%$ completeness range. Two of 48 nodes were less accurate with the GAT at $h=1$.`
- **Line 676.** The author has already softened this, but the new second sentence asserts a mechanism that 6-23 shows was not isolated.
  - OLD: `its failure analysis reveals an operational limitation: standard forward-predictive graph architectures lack native mechanisms to isolate or reconstruct missing neighbour observations during sensor dropouts. When an adjacent sensor drops out, spatial message passing can propagate corrupted representations across graph neighbourhoods.`
  - NEW: `it did so only after excluding 27 sparse stations, and the forecasting models have no mechanism to reconstruct missing observations, either at a node or at its neighbours. Whether missing neighbour inputs degrade a node's forecasts was not isolated in this experiment (Section~\ref{subsec:exp2-completeness-driver}).`
- **Line 678.** Already softened by the author ("investigates this challenge", "assessing whether"); no change.
- **[AUTHOR]** Check the bridge against the current RIoT v5/v6 results before sending.

### 6-27 [CORRECTION] Lines 111 and 148: split terminology

See 5-28. Apply the two Chapter 6 rows of that table here.

### 6-28 [LIMITATION] Line 144: shared metric (T2-V)

Append to the end of line 144, before the itemize:

` For orientation only, the Chapter~\ref{ch:exp1} LSTM-MV has a median per-stream MAE of 4.32 counts at a 2-hour horizon, against 4.35 counts per station for the Experiment~2 champion at 15 minutes; the differences in unit, horizon and test period listed below mean that these values should not be compared directly.`

---

## D. New figures: LaTeX

Copy the PDFs listed in Section 0, item 5. Every figure below is referenced in the text by an edit in Part B or Part C.

### D.0 Chapter 5, Fig. 5.6 (replacement)

Covered by 5-10:
- Same file name, new content from `ch5_v2_figures/threshold_cvrmse_lstm_mv_h8.{png,pdf}`.
- Keep the old file as `threshold_cvrmse_lstm_mv_h8_mixed_v1v2.png`.
- No change to the `\includegraphics` line.

### D.1 Chapter 6: node map (h = 1)

- **Placement:** insert directly after `\end{itemize}` at line 646 (end of `subsec:exp2-core-periphery`).
- **Referenced by:** 6-24(a) and 6-25(a).

```latex
\begin{figure}[htbp]
    \centering
    \includegraphics[width=\linewidth]{figures/experiment2/fig_ch6_node_map_h1.pdf}
    \caption[Per-node champion performance and spatial gain at $h=1$]{Per-node test performance of the champion GAT-LSTM (Gabriel) at $h=1$ (15~min). (a)~\gls{mase}. (b)~Percentage MAE reduction relative to the temporal-only LSTM, from seed-averaged test MAE; blue nodes benefit from spatial message passing. Grey lines are Gabriel edges. Hollow circles are included nodes without test targets ($n = 6$); crosses are candidate sensors excluded for completeness below 30\%. Only Falconar Street ($-6.4\%$) and Barras Bridge ($-4.8\%$) are less accurate with the GAT. Basemap \copyright~OpenStreetMap contributors.}
    \label{fig:exp2-node-map}
\end{figure}
```

### D.2 Chapter 6: completeness

- **Placement:** directly after the paragraph that ends at line 631 (after the equation block), before line 633.
- **Referenced by:** 6-23(b).
- **Note:** panel (c) is exploratory. The caption says so, matching the wording of 6-23(d).

```latex
\begin{figure}[htbp]
    \centering
    \includegraphics[width=\linewidth]{figures/experiment2/fig_ch6_completeness.pdf}
    \caption[Completeness, forecast error and spatial gain]{Per-node relationships at $h=1$ for the champion GAT-LSTM (Gabriel), $n = 48$. (a)~Lower completeness is associated with higher \gls{mase} (Pearson $r = -0.33$, $p = 0.021$). (b)~Completeness is unrelated to the spatial gain over the temporal-only LSTM ($r = +0.10$, $p = 0.51$). (c)~The mean completeness of a node's Gabriel neighbours shows no marginal association with its spatial gain ($r = -0.15$, $p = 0.30$); after controlling for the node's own completeness, the partial correlation is $-0.29$ ($p = 0.048$, uncorrected for multiple comparisons; exploratory).}
    \label{fig:exp2-completeness}
\end{figure}
```

### D.3 Chapter 6: distance decay (replaces Fig. 6.5)

- **Placement:** replace the whole `figure` environment at lines 472–503, from `\begin{figure}[htbp]` to `\end{figure}`.
- **Before replacing,** confirm that line 472 is `\begin{figure}[htbp]` and line 503 is `\end{figure}`.
- **Keep** the label `fig:exp2-distance-decay`.
- **Referenced by:** 6-18(d), and by any existing `\ref`.
- The four old PNGs (`attention_distance_*.png`) stay in the figures folder, unreferenced; do not delete them.

```latex
\begin{figure}[htbp]
    \centering
    \includegraphics[width=\linewidth]{figures/experiment2/fig_ch6_distance_decay.pdf}
    \caption[Degree-normalised attention versus inter-sensor distance]{\gls{gatv2} attention against straight-line inter-sensor distance for four topologies. Each weight is divided by the mean weight over the destination node's non-self neighbours, so that $\tilde{\alpha} = 1$ (dotted line) corresponds to uniform attention; this removes the effect of in-degree and of the self-loop share. Red lines show medians over eight distance-quantile bins. Attention decays with distance on the Gabriel ($r = -0.36$) and fully connected ($r = -0.42$) graphs but not on the $k$NN-8 ($r = -0.05$, $p = 0.23$) or learned ($r = +0.02$, $p = 0.40$) graphs. Attention is from seed~42 (final layer, mean over four heads and 256 test windows) of the GAT-Transformer for the Gabriel, $k$NN-8 and fully connected graphs and of the learned-adjacency GAT; no attention was exported for the GAT-LSTM champion.}
    \label{fig:exp2-distance-decay}
\end{figure}
```

### D.4 Chapter 6: paired per-node gain

- **Placement:** directly after the DM paragraph inserted by 6-17(d), i.e. after the sentence ending `which Table~\ref{tab:exp2-main-results} reports separately.`
- **Referenced by:** 6-16 and 6-17(d).

```latex
\begin{figure}[htbp]
    \centering
    \includegraphics[width=\linewidth]{figures/experiment2/fig_ch6_paired_gain.pdf}
    \caption[Per-node spatial gain by horizon]{Per-node spatial gain of the Gabriel GAT over its temporal-only counterpart (GAT-LSTM versus LSTM, left; GAT-Transformer versus Transformer, right), from seed-averaged test MAE. Each point is one node ($n = 48$); boxes show the median and interquartile range. Numbers above each box count the nodes that improved: 46, 46 and 47 of 48 for the LSTM backbone and 45, 46 and 48 for the Transformer backbone at $h = 1, 2, 4$. The median gain for the LSTM backbone rises from $3.7\%$ ($h=1$) to $5.0\%$ ($h=2$) and $6.4\%$ ($h=4$).}
    \label{fig:exp2-paired-gain}
\end{figure}
```

### D.5 Chapter 6: node map (h = 4), optional

The figure is not required. If used, place it in an appendix and reference it from the D.1 caption by appending `The corresponding map for $h=4$ is shown in Figure~\ref{fig:exp2-node-map-h4}.`

```latex
\begin{figure}[htbp]
    \centering
    \includegraphics[width=\linewidth]{figures/experiment2/fig_ch6_node_map_h4.pdf}
    \caption[Per-node champion performance and spatial gain at $h=4$]{As Figure~\ref{fig:exp2-node-map}, for $h=4$ (1~h). \gls{mase} ranges from 0.456 to 1.038 (median 0.651); the median gain over the temporal-only LSTM is $6.4\%$, and only Falconar Street is less accurate with the GAT ($-1.3\%$). Basemap \copyright~OpenStreetMap contributors.}
    \label{fig:exp2-node-map-h4}
\end{figure}
```

If it is not used, do not add the sentence to D.1. An undefined reference must not be left.

### D.6 Bibliography entries

Add the following to `shared/bibliography/references.bib`.
- The first two are required by 6-17.
- The third is required only for the citation in E.2, item 7. If you prefer not to add it, delete the `\citep{...}` there.

```bibtex
@article{harveyTestingEqualityPrediction1997,
  author  = {Harvey, David and Leybourne, Stephen and Newbold, Paul},
  title   = {Testing the Equality of Prediction Mean Squared Errors},
  journal = {International Journal of Forecasting},
  year    = {1997},
  volume  = {13},
  number  = {2},
  pages   = {281--291},
  doi     = {10.1016/S0169-2070(96)00719-4}
}

@article{neweySimplePositiveSemidefinite1987,
  author  = {Newey, Whitney K. and West, Kenneth D.},
  title   = {A Simple, Positive Semi-Definite, Heteroskedasticity and Autocorrelation Consistent Covariance Matrix},
  journal = {Econometrica},
  year    = {1987},
  volume  = {55},
  number  = {3},
  pages   = {703--708},
  doi     = {10.2307/1913610}
}

@inproceedings{jainAttentionNotExplanation2019,
  author    = {Jain, Sarthak and Wallace, Byron C.},
  title     = {Attention is not Explanation},
  booktitle = {Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies},
  year      = {2019},
  pages     = {3543--3556},
  doi       = {10.18653/v1/N19-1357}
}
```

**[AUTHOR]** Check the DOIs before submission; they were written from memory and could not be verified offline.

---

## E. Limitations sections (ready to paste)

Neither chapter currently has a Limitations section. Both sections below are new, and each item ends by naming the Part B/C edit it comes from. Items that are only limitations, with no change to the text, are marked **[LIMITATION only]** in the mapping tables. Every other item also has a correction in the text ([BOTH]).

### E.1 Chapter 5

**Placement:** insert immediately before `\section{Chapter Summary}` (line 549).

```latex
\section{Limitations}
\label{sec:exp1-limitations}

The results of this chapter are subject to the following limitations, which are revisited in Section~\ref{sec:disc-limitations}.
\begin{enumerate}
    \item \textbf{Single training seed.} Every neural model was trained once (seed 42), so differences between architectures of a few per cent in median error cannot be separated from initialisation variance.
    \item \textbf{Inactive masking.} Input windows are drawn only from gap-free runs, so the masked Transformer variants never see a missing step and are functionally identical to the standard Transformers. This experiment therefore provides no evidence for or against mask-aware learning.
    \item \textbf{Leakage in the KNN imputer.} The $k$-NN imputer is fitted on all observed steps of a stream, including those in the validation and test periods, and fills gaps of any length. Imputed input values in the test period can therefore draw on observations from the same period, which may make the KNN arm's test errors optimistic.
    \item \textbf{Horizon-specific, sample-sensitive threshold.} The $\approx 3$-hour breakpoint is supported at $h=8$ but not at $h=24$ ($\Delta\text{AIC} = 0.39$); its upper confidence bound moves from $3.4$ to $4.9$~hours when three fragmented streams are removed, and only 8 of 110 streams have average sequences longer than 6~hours.
    \item \textbf{Cross-sectional associations only.} The quality--error relationships are rank correlations across 110 directional streams from 53 locations; the two directions of a location are not independent observations. Sequence length and completeness are strongly correlated ($\rho = 0.78$), and a random forest using the quality features does not predict per-sensor error (out-of-bag $R^2 \le 0.04$). The analysis therefore does not establish that improving continuity would reduce error at a given sensor.
    \item \textbf{Unequal evaluation sets between arms.} Because the split is applied to each contiguous run, interpolation, which lengthens runs, changes which windows reach the test set and from which periods; the interpolation arms are therefore not evaluated on the same test windows as the un-imputed arm. The matched-pairs analysis removes this difference, but 90.6\% of matched steps fall between 13:00 and 22:59~UTC, so it says little about night and early-morning performance.
    \item \textbf{No event covariates.} The models use only the target series and calendar features. Surges linked to football fixtures (Heber Street) and campus events (Urban Science Building) are not predictable from these inputs, and the worst-performing streams are dominated by a handful of such steps.
\end{enumerate}
```

| Item | Source edit | Classification |
|---|---|---|
| 1 | 5-03 | [BOTH] |
| 2 | 5-04 | [BOTH] |
| 3 | 5-05 | [BOTH] |
| 4 | 5-11 | [BOTH] |
| 5 | 5-01, 5-08, 5-22 | [BOTH] |
| 6 | 5-02, 5-16, 5-23 | [LIMITATION only] for the arm difference; correction for the hourly claims |
| 7 | 5-17, 5-18 | [LIMITATION only] |

### E.2 Chapter 6

**Placement:** insert immediately before `\section{Chapter Summary and Bridge to Chapter 7}` (line 662, preceded by a `% ====` rule at line 661). Insert above the rule.

The item numbers are fixed, because Part C refers to items 1, 3, 4, 6, 7 and 8.

```latex
\section{Limitations}
\label{sec:exp2-limitations}

The results of this chapter are subject to the following limitations, which are revisited in Section~\ref{sec:disc-limitations}.
\begin{enumerate}
    \item \textbf{Untested completeness threshold.} The $\ge 30\%$ completeness threshold, which excluded 27 of 81 candidate stations, was chosen a priori; no threshold sweep or all-station forecasting run was performed, so its effect on accuracy is unknown. Chapter~\ref{ch:exp3} retains all stations through masked reconstruction.
    \item \textbf{Euclidean graph construction.} The Gabriel, $k$NN-8 and fully connected graphs are built from straight-line distances. The walking network enters only through the edge attributes, so topologies defined on network distance were not evaluated.
    \item \textbf{No hyperparameter tuning.} All trainable models, temporal and spatial, share one fixed configuration (learning rate $10^{-3}$, batch size 32, dropout 0.1, hidden size 64). Rankings could change under model-specific tuning, and the capacity ablation is reported on the test split, so it should be read as descriptive rather than as model selection.
    \item \textbf{Training-length cap.} Training was capped at 60 epochs; all three seeds of the champion, and the $k$NN-8 and plain-Gabriel GAT-Transformers, reached the cap without early stopping and may be under-trained.
    \item \textbf{Imperfect controls.} The parameter-matched control without cross-node edges (identity graph) exists only for the Transformer backbone. The GAT-LSTM differs from the temporal LSTM in its decoder (a two-layer MLP rather than a linear head) as well as in message passing, and has $43\%$ more parameters (74,115 versus 51,907), so its gain over the LSTM does not isolate message passing. The topology, depth and head ablations were run only on the GAT-Transformer.
    \item \textbf{Test-period coverage.} The test period runs from 2 December 2025 to 7 June 2026 and contains no summer or early-autumn data. Six of the 54 stations stopped reporting before it began, so all per-node results are based on 48 stations, although all 54 take part in message passing.
    \item \textbf{Limited attention evidence.} Attention weights were exported for one seed (42) and only for the GAT-Transformer and learned-adjacency models; no attention is available for the GAT-LSTM champion. Attention weights are not necessarily faithful measures of feature importance \citep{jainAttentionNotExplanation2019}, and the non-identifiability explanation offered for the low seed stability of the fully connected graph was not tested.
    \item \textbf{Cross-sectional completeness analysis.} The completeness--error relationship is a correlation across 48 stations within the retained $\ge 30\%$ range. The neighbour-completeness result ($p = 0.048$) is one of several exploratory tests and does not survive correction for multiple comparisons.
    \item \textbf{Ensemble-based significance tests.} The Diebold--Mariano tests compare seed-averaged forecasts, i.e.\ three-member ensembles, against each other and against deterministic baselines. They measure how consistently the gain holds across nodes, not its robustness to random initialisation.
    \item \textbf{No external baselines.} Established spatio-temporal models, including DCRNN \citep{liDiffusionConvolutionalRecurrent2018}, STGCN \citep{yuSpatioTemporalGraphConvolutional2018}, Graph WaveNet \citep{wuGraphWaveNetDeep2019} and AGCRN \citep{baiAdaptiveGraphConvolutional2020}, were not evaluated, and GATv2 was not compared with the original GAT. The results establish the value of spatial information within one model family, not the champion's position relative to the literature.
    \item \textbf{Limited comparability with Chapter~\ref{ch:exp1}.} This chapter reports MAE and \gls{mase} per station at 15--60~min horizons, whereas Chapter~\ref{ch:exp1} reports RMSE and CV-RMSE per directional stream at 2 and 6~h horizons, over different test periods. The two sets of results therefore cannot be compared quantitatively.
\end{enumerate}
```

| Item | Source edit | Classification |
|---|---|---|
| 1 | 6-02 | [BOTH] |
| 2 | 6-07 | [BOTH] (local text corrected; limitation disclosed) |
| 3 | 6-10, 6-22 | [BOTH]; the grid-search text is an [AUTHOR] item |
| 4 | 6-10 | [LIMITATION only] |
| 5 | 6-12, 6-21, 6-22 | [BOTH] |
| 6 | 6-03, 6-06 | [BOTH] |
| 7 | 6-18, 6-19 | [BOTH] |
| 8 | 6-23 | [BOTH] |
| 9 | 6-17 | [LIMITATION only] |
| 10 | 6-13 | [LIMITATION only] (T2-O) |
| 11 | 5-27, 6-28 | [LIMITATION only] (T2-V) |

---

## F. Tier 3 and residual items (optional, not needed for the first draft)

1. **Ch5 Fig. 5.5 heatmap:** regenerate it from v2 (see 5-08(a)). It currently has unknown provenance.
2. **Ch5 RF feature-importance figure:** exists but is unused (T2-F). With $R^2 \approx 0$, feature importances are not meaningful. Leave it out.
3. **Ch6 seasonal-naive fallback:** the seasonal-naive baselines fall back to persistence when the seasonal reference step is unobserved (`models/temporal.py:75-117`). One sentence at Ch6 line 245 would disclose this. The `hist_avg` binning is already corrected in 6-12.
4. **Ch6 DM robustness:** with Newey–West (96-lag) variance, the share of nodes where the champion is significantly better changes by at most 2.1 percentage points in any comparison (`dm_champion_seedavg.csv`, columns `*_nw96`). A footnote is enough if 6-17 does not already mention it.
5. **Ch6 h=4 node map:** see D.5.
6. **Consistency pass:** after applying everything, search both chapters for these words and soften any that remain:
   - "definitive", "incontrovertible", "profound", "critical", "crucially", "rigorous", "directly refutes", "systematically", "fundamental".

---

## G. Author verification list (cannot be resolved from the repository)

| # | Item | Where | What to check |
|---|---|---|---|
| G1 | Grid search for Experiment 2 hyperparameters | Ch6 lines 226–232, 6-10 | No configs or artefacts exist. Supply them, or accept the "fixed a priori" wording |
| G2 | ANPR node in the pedestrian graph | Ch6 line 47 | `MowbrayStStratfordRdANPR` appears among the 54 nodes, but line 47 says ANPR sensors are excluded. Confirm it is a pedestrian counter, or correct line 47 |
| G3 | Heber Street "duplicate" | Ch5 line 449, 5-17 | Check the raw database for 23 Feb 2025 16:00 |
| G4 | Fig. 5.12 dates | Ch5 line 455, 5-18 | Were 21 and 28 April 2026 fixture days? |
| G5 | $R^2 = 0.81$ "preliminary study" | Ch5 line 502, 5-22 | No source in the repository |
| G6 | Table 5.4 run version | 5-07 | Confirm that moving to v2 is acceptable (the brief assumes yes) |
| G7 | $t = 0.31$ regression | Ch5 line 284, 5-08 | No script. Re-run values are given |
| G8 | Spring seasonal claim | Ch5 line 404 | Not traced to an artefact |
| G9 | Heber "93% / 141 weekend days" | Ch5 line 459, 5-18 | The table says 91.38% |
| G10 | Cluster distance correlation | Ch5 line 326, 5-14 | $-0.037$ on mixed data; recompute on v2 |
| G11 | Fig. 5.3 Panel C and Fig. 6.1 labels | 5-02, 5-28 | Scripts under your edit still say "daily" or "alternating-day" |
| G12 | Fig. 5.1 framework image | Ch5 line 19 | Not inspected visually |
| G13 | BibTeX DOIs | D.6 | Verify online |
| G14 | Bridge to Chapter 7 | Ch6 lines 676–678 | Check against current RIoT v5/v6 results |
| G15 | "Weekly MASE" / $m = 672$ elsewhere | Ch4, Ch7 | The Experiment 2 code uses the daily scale ($m = 96$). Search other chapters for the weekly description |
| G16 | `eq:exp2-masked-loss` | Ch4, Ch7 | If 6-xx removes or renames this equation label, check references in other chapters |
| G17 | Linear/spline gap limit | 5-05 | `max_gap_steps: 8` (2 h) in `configs/quality/quality_full_v2.yaml`; confirm this is the intended description |

