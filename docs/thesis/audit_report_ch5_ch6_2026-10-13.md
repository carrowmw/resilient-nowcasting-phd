# Second Audit: Chapters 5 & 6 — Readiness for First-Draft Submission

**Date:** 13 October 2026
**Intended location:** append to `full/audit_report.md`. The `full/` folder was mounted read-only during the audit, so this file sits next to it.
**Scope:** `chapters/chapter5_experiment1_single_node.tex` (538 lines), `chapters/chapter6_experiment2_multi_node.tex` (646 lines), all figures included by both chapters, and cross-references into Chapters 4 and 7.
**Method:** Both chapters were read in full. Every number in the prose was checked against its own tables and figures. Every included figure was inspected visually, and every `\ref`/`\gls` was checked (all resolve). The thesis was **not compiled**: no LaTeX toolchain is available in the audit environment. Bibliography keys were not checked because no `.bib` file is in the repository.

## Verdict

| | Ready to send? | Effort to get there |
|---|---|---|
| **Chapter 5** | **Not yet.** It is close. The structure, tables and narrative are in place. However, several headline numbers are wrong or inconsistent with the chapter's own tables and figures, and one (24.4-fold coverage) is a calculation error. | About 1 day: number corrections, tone pass, figure label fixes. |
| **Chapter 6** | **Not yet.** One figure caption does not match its image, there are sign errors in the results, one claim is contradicted by the chapter's own leaderboard, and seed variance is never reported. It also needs 2–3 more figures. | About 2–3 days, including regenerating or creating figures. |

Neither chapter has a fatal design flaw for a *first* draft. But the Tier 1 items below are things a supervisor will spot on a first read, and they would undermine trust in the rest. Fix Tier 1 before sending. Tier 2 can be listed as "known issues" in the covering email if time is short.

---

## Tier 1 — Must fix before sending (factual / internal-consistency errors)

### Cross-chapter

**X1. Experiment 1 is described as interpolated with 70/15/15 splits in three places, but Chapter 5's headline results are un-imputed, with alternating-day micro-splits.**
- Chapter 4, around lines 290–294 (caption and text: "interpolation", "synthetic gap-filling").
- Chapter 6, line 111 (Fig. 6.1 caption) and line 141 ("chronological splits with single-node interpolation").
- Chapter 6, Fig. 6.1 *image*: the left panel is titled "Per-Stream 70/15/15% Splits" and shows "Interpolated" segments.

Fix: either regenerate Fig. 6.1's left panel from the un-imputed alternating-day split, or relabel it as the KNN/interpolated variant and say so. Then align Chapter 4 and Chapter 6 line 141 with the Chapter 5 protocol.

**X2. Chapter 6, line 600, misstates the Chapter 5 result.**
```diff
- ... a breakpoint of 3.0 hours (W = 12 steps) ... completeness played a secondary role ...
+ ... a breakpoint in mean contiguous sequence length of approximately 3 h (12 steps at 15-min resolution; the Chapter 5 input window was W = 24) ... completeness was also strongly associated with error (Spearman ρ = −0.58, Table 5.4) ...
```
Chapter 5 uses W = 24; 3 h is the sequence-length breakpoint, not W. Table 5.4 shows completeness (ρ = −0.58) correlating *more* strongly than sequence length (ρ = −0.54), so "secondary" contradicts Chapter 5.

### Chapter 5

| # | Line | Problem | Fix |
|---|---|---|---|
| C5-1 | 504, 520, 537; Fig. 5.15 Panel D title | **The "24.4-fold coverage expansion" is a calculation error.** 1,221,570 ÷ 50,051 compares KNN's *total* windows with the *matched intersection*, not with the un-imputed model's total. Panel D's own green bars show about 333k (un-imputed) vs about 1,222k (KNN), i.e. **≈3.7×**. | Recompute as KNN total ÷ un-imputed total. Update the table row, the text, the summary (line 537) and the figure title. Also rephrase line 504's "only capable of producing 50,051 test forecasts", because 50,051 is the intersection, not the un-imputed total. |
| C5-2 | 14 | "over 20%" — GRU improvement is 19.5%. | "approximately 20%" or give per-model values. |
| C5-3 | 158 | MV-vs-UV CV-RMSE gains are misstated. Actual gaps: 5.5–6.4 pp (h=8) and 5.1–6.8 pp (h=24). The stated 6.4–6.8 range holds only for the recurrent models at h=24. | Restate per horizon, in pp. |
| C5-4 | 193 | "significant on 92–100% of sensors" — Table 5.3 shows 91.8%, 87.3% and 79.1%. | Quote the table range. Replace "statistically indistinguishable" with "not significantly different at α = 0.05". State that there is no multiplicity correction (or apply BH, as Chapter 6 does). Table 5.3 also has no Transformer comparisons, even though Trans-MV / Trans-Masked-MV are best on some metrics. |
| C5-5 | 336–338, 535; Fig. 5.9 | **The cold-start claim is not supported by the matched-pairs analysis.** On matched timestamps (Fig. 5.15 Panel B, UTC), the un-imputed 08:00 spike persists (RMSE ≈ 39 vs KNN ≈ 27), and KNN is *worse* from 09:00 to 13:00 (≈44–47 vs ≈16–32). The 74% reduction in Fig. 5.9 compares different test samples. Line 503 also notes that the un-imputed model has almost no matched predictions between 00:00 and 08:00, so the 08:00 un-imputed windows are a selected subset. The LSTM is stateless per window, so "pre-warming recurrent hidden states" is not the right mechanism either. The 74% is the weekday figure (weekend ≈ 79%). | Reframe the result as an un-imputed-sample artefact plus a missing-input effect, and cite Panel B. Remove "successfully eliminates" from line 535. State the timezone (UTC; BST shifts summer peaks by 1 h). |
| C5-6 | 379 | "Summer exhibits the highest predictability across both model variants" — Table 5.5 / Fig. 5.10: un-imputed Spring 66.4% < Summer 66.8% (a tie); KNN Autumn 64.6% < Summer 67.7%. | "Summer CV-RMSE is among the lowest for both variants (66.8% and 67.7%)". |
| C5-7 | 457 | "asymptotic floor between 35% and 40%" — Fig. 5.13's LOWESS line flattens at about 45–50% (un-imputed) and about 55–68% (KNN). | Quote the values from the figure, or the median CV-RMSE for sensors with more than 20 counts per 15 min. |
| C5-8 | 340 | "weekend error steadily escalates into the late afternoon (16:00–17:00, ≈36.4)" — Fig. 5.9 shows a single 16:00 spike (35.5 / 36.5) dropping to ≈15–20 at 17:00. | "peaks sharply at 16:00 UTC". |
| C5-9 | 322 | Calls KNN the "top-performing" imputation strategy; per Table 5.1 it is best only on MAPE, which the chapter later calls pathological (line 479). Table 5.1 also shows night-zero with the best CV-RMSE at h=24, yet the text calls it "harmful". | Justify choosing KNN on other grounds (coverage, plausibility) and say so; correct the night-zero description. |
| C5-10 | 67–73; Fig. 5.1 | The text lists 4 harmonics (8 variables), but the ablation (Table 5.4, lines 236–240) has **5** (half-day, day, week, quarter, year). The framework figure says "8 Harmonic Features". The count of three cyclical timestamp variables also does not add up. | Reconcile to 5 harmonics × sin/cos = 10, or whatever the code uses. Update Fig. 5.1. |
| C5-11 | 414, 436 | "at or above 3.0 h" / "necessary condition" — two of the five best sensors have mean sequence lengths of 2.88 h and 2.95 h (Table 5.6, lines 399, 401). | "approximately 3 h or more"; downgrade "necessary condition" to "associated with" (see also T2-C). |
| C5-12 | 533 | "110 pedestrian monitoring stations" — these are **streams** (AGENTS.md §4). | "110 directional streams from 53 sensor locations". |
| C5-13 | 484 | "previous models predicting absolute RMSE achieved R² > 0.80" — no such model is reported in the chapter. | Report the RMSE-target R² in the text or a table, or delete the sentence. |
| C5-14 | 303 | "strongly tied to urban geography" — the heatmap (Fig. 5.5) shows ρ(dist_to_city_centre, CV-RMSE) = −0.01. | Soften; attribute the cluster difference to volume and sequence length rather than location. |
| C5-15 | 131 | "reduced … by 17% … and by 17%" reads awkwardly. | Give the two metrics explicitly. |

### Chapter 6

| # | Line | Problem | Fix |
|---|---|---|---|
| C6-1 | 507–509; Fig. 6.6 | **The caption does not match the image.** The caption describes Spearman correlations of attention weights across seeds (ρ = 0.78 / 0.28). The image `seed_stability.png` shows **node-median MAE ± seed error bars for every configuration and horizon**. | Either generate the correct attention-stability figure (e.g. a bar of cross-seed ρ per topology), or keep this image, recaption it as "seed variability of MAE", and cite it where seed variance is discussed (it is the only seed-variance evidence in the chapter; see C6-2). Ideally do both. |
| C6-2 | throughout; 579 | **No seed standard deviations are reported anywhere**, yet line 579 refers to "seed SD". The "champion" margins are as small as 0.006 MAE. The error bars in `seed_stability.png` show that the SDs exist. | Add ± SD (n = 3 seeds) to Table 6.3 and the spatial-gain table. Avoid ranking configurations whose difference is smaller than the seed SD. |
| C6-3 | 620–621 | Sign error: the two degraded sensors get *worse* by +6.4% and +4.8% (MAE increases), but the text writes them as negatives/reductions. | Correct the sign. Then line 626: with n = 2, "fundamental vulnerability" → "a possible vulnerability, observed at two boundary nodes". |
| C6-4 | 536–538 | "gat_plain is the single worst-performing neural architecture … cannot compensate" — `gat_plain` beats both temporal-only models at h=2 (5.076 vs LSTM 5.28 / Transformer 5.27) and h=4 (5.516 vs 5.605 / ≈5.65). This is visible in Fig. 6.6. | "gat_plain is the weakest *spatial* configuration, but still outperforms the temporal-only baselines at h ≥ 2". Note this actually *supports* the value of spatial context. |
| C6-5 | 329 | "28–48%" reduction range — the table values give about 25.8–38.5%. | Recompute. |
| C6-6 | 382, 636 | "spatial lift roughly doubles / expands across horizons" holds only for LSTM from h=1 to h=2. Transformer goes from 6.5% to 8.6%, and both dip at h=4. | Describe the per-backbone trajectory. Also fix the Fig. 6.3 caption ("divergence expanding at longer horizons"). |
| C6-7 | 233 | "twelve model configurations", but 13 are listed (14 with gabriel_plain). | Recount. |
| C6-8 | 129, 384 | 150–500 m at 1.3 m/s ≈ 2–6.5 min, not 5–15 min. This weakens the "upstream arrivals" argument at h = 2–4. Walking speeds are uncited. | Fix the arithmetic, cite a walking-speed source, and soften line 384. |
| C6-9 | 596 | "p < 0.02" — for r = −0.33 with n = 48, p ≈ 0.022. It is also unclear which model this is, and the test is Pearson whereas Chapter 5 uses Spearman. | "p ≈ 0.02 (Pearson, n = 48, <model>, h = <h>)", or switch to Spearman for consistency. |
| C6-10 | 95–97 vs 120 | Line 95 says gaps are zero/mean-filled, causing "conflation", but line 120's input definition includes a mask channel. These contradict each other. It is also not stated whether missing targets are excluded from the MAE. | State the actual input (with or without mask) and the loss/metric masking. |
| C6-11 | 216 | $H$ is used for both forecast horizon and hidden dimension. | Rename the hidden dimension $d_h$. |
| C6-12 | splits section | Split step counts sum to 119,667, not T = 119,712, and the date boundaries are off by about 100 steps. | Reconcile; state whether boundary windows are dropped. |
| C6-13 | 389 | Cite key `benjaminiRoyalStatisticalSociety` looks malformed (Zotero/BBT key from the journal name). | Check that it renders as Benjamini & Hochberg (1995). |
| C6-14 | 137 | Addresses "Readers and examiners" directly. | "Direct numerical comparison between Experiments 1 and 2 is not meaningful because …" |
| C6-15 | Fig. 6.3(a) y-axis | Labelled "raw **vehicle** counts". | "pedestrian counts". |

---

## Tier 2 — Methodological issues supervisors are likely to raise

These do not all need fixing before a first draft. Either address them or acknowledge them in a short **Limitations** subsection. **Neither chapter currently has one**, and AGENTS.md §2 requires it.

### Chapter 5

- **T2-A. Volume confound in the core finding.** Fig. 5.5 shows ρ(avg_seq_len, mean_count) = 0.86. Event-driven sensors report more continuously when footfall is high, so sequence length may partly be a proxy for volume. CV-RMSE still correlates with volume (ρ ≈ −0.42, and the Fig. 5.13 LOWESS shows it), so line 459's "eliminating scale variance" overclaims. *Fix:* report partial correlations (or a regression with log volume) for sequence length and completeness, at both horizons. Line 274 claims partial correlations but does not report them.
- **T2-B. Unequal test samples.** Table 5.1 (imputation strategies), the seasonal table and the h=8 vs h=24 comparisons each evaluate different windows. This also explains why h=24 RMSE is lower than h=8 RMSE, which currently goes unexplained. The matched-pairs section (line 489) admits this. Add one sentence early in the chapter signalling it, and caveat Table 5.1 accordingly. Linear interpolation lowering RMSE while raising CV-RMSE by about 10 pp is a sample-composition effect.
- **T2-C. Threshold model.** Report the breakpoint CI (Fig. 5.6 shows roughly 1.9–3.4 h) and both slopes. Note that the fit is driven by a few outliers (225%, 187%) and that very few points lie beyond 6 h. Also reconcile the "3-hour context" interpretation with the design: every input window is already 6 h (W = 24) of contiguous data.
- **T2-D. Heber Street (Fig. 5.12).** The un-imputed panel is dominated by a single reading of about **2,240 counts in 15 min**. That is far outside the sensor's range elsewhere (≤ 260) and probably a sensor fault, not a stadium surge. If so, 187.5% CV-RMSE reflects one corrupted point, and the "stadium anomaly" narrative (lines 426–440) needs re-checking. Also, the epistemic/aleatoric framing (440) is wrong: fixture-driven surges are predictable from a fixture calendar (a missing covariate), not irreducible noise.
- **T2-E. Worst sensor not discussed.** UrbanScienceBuildingSquareMid (224.6%) is the true worst case.
- **T2-F. RF meta-model.** The "quality" features include volume. Cross-validation is not grouped by location, so the two directions of one site can leak across folds. The RF feature-importance figure exists but is unused (see Figures).
- **T2-G. Matched pairs.** The two arms use different split schemes. 108 vs 110 sensors is unexplained. "Valid windows" is undefined. Line 526 quotes the smallest penalty (4.3 pp) while the pooled figures are about +14 pp CV-RMSE and +20% RMSE. Report all three.
- **T2-H. KNN imputation is not specified** (k, distance, donors). If the donors are other sensors, the experiment is no longer strictly single-node; say so.
- **T2-I. Extreme events (Fig. 5.14).** Comparing absolute RMSE between >3σ and ±1σ periods is scale-confounded by construction, so the 5.0× is largely mechanical. Use relative error, or say so explicitly. The 1–3σ band is silently excluded. Line 479's "denominator in the hundreds or thousands" does not match mean volumes of 5–180.
- **T2-J. Clusters.** The clustering method, k selection and any significance test are not described.
- **T2-K.** Masked Transformers (line 85) are moot if all windows are fully contiguous; explain what the mask does in this setting. No seeds or variance are reported for Experiment 1. Add the number of seeds or state that it is a single run.

### Chapter 6

- **T2-L. Missingness collapse / 30% threshold** (lines 63, 95–99) is asserted, not shown. Either add a small threshold sweep (or the N = 81 run, which Chapter 7 line 52 implies exists), or present it explicitly as a hypothesis that Chapter 7 tests.
- **T2-M. DM table** compares `gat_transformer|gabriel` (not the champion) with `lstm`, uses a single seed that differs by row (42 vs 43), and does not state the loss-difference units. Line 420 says "definitive". The summary's "65–90%" mixes comparisons, and "46/48" may refer to a different comparison. *Fix:* run DM on the champion vs each baseline, either for every seed or on seed-averaged forecasts.
- **T2-N. Tuning asymmetry.** Hyperparameters were tuned only on Gabriel, and it is unclear whether the baselines were tuned. The capacity-ablation values are test-split numbers, so there is a test-set selection risk. State that the validation split was used, or caveat.
- **T2-O. Literature baselines.** None of DCRNN, Graph WaveNet, STGCN or AGCRN is evaluated. There is no GAT-v1 vs GATv2 ablation and no parameter counts, although line 27 criticises the literature for exactly this omission. At minimum, add a parameter-count column and acknowledge the missing baselines.
- **T2-P. Graph geometry.** The Gabriel graph is built on Euclidean coordinates, but the line 179 caption implies walking-corridor adjacency, and line 19 criticises Euclidean distance. $c_\text{direct}$ is undefined and the $r_\text{class}$ encoding is unclear. $D_\text{in} = 3$ does not add up given sin/cos time-of-day and day-of-week encodings.
- **T2-Q. Attention interpretation.**
  - kNN-8 (r = −0.03) and learned (r = +0.08) show *no* distance decay, which is not discussed, yet the summary says attention "reflects geography".
  - Gabriel normalised entropy is 0.94 (near-uniform).
  - Gabriel in-degree varies from 1 to 5, so raw α is confounded with degree (α ≈ 1/|N(i)| under uniform attention). Plot α·|N(i)| instead.
  - Line 516: "entirely different" overclaims for ρ = 0.28.
  - "Collinear equifinality" is untested; either cite Beven or use "non-identifiability", and label it a hypothesis.
- **T2-R.** Line 503's receptive-field argument conflicts with L = 3 performing worse (line 580 blames oversmoothing). Line 543's "statistical parity" is untested, and the LSTM/Transformer baselines are not identity-topology runs. Line 548 compares gat_lstm on kNN/FC, but those runs do not exist. Line 582 overclaims capacity.
- **T2-S. Spatial error propagation** (lines 602, 626, 639) is not isolated. Regress node error, or spatial gain, on *neighbours'* mean completeness while controlling for the node's own completeness.
- **T2-T.** Edge features change accuracy by only 0.5–2%, so "essential" overclaims. Line 497 cites the seed figure where Table 6.3 / Fig. 6.3 is meant. Line 609 mentions "mapping performance geographically" but there is no map (see Figures).
- **T2-U.** Test period is Dec–Jun only (no summer), and 6 of 54 nodes are excluded from the DM tests. Fig. 6.1's right panel shows several nodes with training data but no val/test coverage, which is probably why. Say so explicitly at line 143.
- **T2-V. No shared metric with Chapter 5.** Chapter 5 uses RMSE/MAPE/CV-RMSE and Chapter 6 uses MAE/MASE. Report one metric, e.g. MAE, in both so the reader can connect the experiments, even with the caveat at line 137.

---

## Tier 3 — Tone, style, notation (quick pass)

**Chapter 5 hyperbole** (AGENTS.md §2):
- Words to replace: "newly evaluated", "remarkably" / "remarkable" (414), "architecturally significant", "universal necessity" (158), "astronomical", "massive" (432, 479), "cleanly", "structurally incapable" (473), "unlocks" (537), "incontrovertible" and "rigorous" (491), "proves" / "proved" (479).
- Line 34: soften "virtually absent".
- Line 30: "survivor bias" → "survivorship bias".
- Line 381: "Artifacts" → "Artefacts".

**Chapter 5 notation and consistency:**
- Use $\times$, not "x" (5.0x, 24.4x).
- Write CV-RMSE differences as **pp**, not %. This applies to the text, Fig. 5.15 Panel C's x-axis and the Panel D title.
- The MAPE equation (around line 477) lacks ×100%, and its $N$ clashes with $N$ = number of sensors.
- Make naming consistent: KNN vs $k$-NN, and Naïve vs Naive.
- "Bidirectional" (line 12) is used nowhere else.
- WAPE is recommended (479) but never reported; either report it or drop the recommendation.

**Chapter 6:**
- Hyperbole to replace: "definitive" (420), "decisively refute" (538), "indispensable" (526), "fundamental vulnerability" (626), "directly resolves" (645).
- Tone down the marketing language in the Chapter 7 bridge, and check it against the current RIoT v5/v6 results before sending.
- Use an en-dash in "Diebold--Mariano" consistently (line 389 uses a hyphen).
- Remove code/artefact paths from the prose (`in_edge_cache = True`, line 77).
- Table 6.3: make bold/underline consistent (bold = best, underline = second, per horizon).
- Line 23: cite AGCRN and MTGNN. Line 29's claim is uncited.

**Deferred from the first audit:** the self-citation `morris-wiltshireQualityAwareAIoTEnabledWireless2026` (Chapter 5, line 10), pending the publication fee. This has now been published.

---

## Figures

### Do the chapters need more figures?

- **Chapter 5 (15 figures): no.** It has enough. If anything, the cluster radar (Fig. 5.7) and the cluster boxplot (Fig. 5.8) could be merged into one two-panel figure. Optional: an RF feature-importance plot, but only if it is regenerated with CV-RMSE as the target and location-grouped CV (`rf_importance_lstm_mv_h*.png` exist but are unused and may target RMSE).
- **Chapter 6 (5 figures): yes, add 2–3.**
  1. **Map of per-node MASE and spatial gain** (two panels on a basemap, with Gabriel edges). This is high priority: line 609 already promises it, and it is the obvious figure for a spatial chapter. It also makes the FalconarSt/BarrasBridge boundary story visible.
  2. **Completeness vs MASE / spatial-gain scatter** (one point per node, with r and p). This supports line 596 and T2-S.
  3. **Distance decay for all topologies.** Extend Fig. 6.5 to four panels (Gabriel, kNN-8, FC, learned) using `attention_distance_*` images that already exist. Plot α·|N(i)| rather than raw α. This shows the absence of decay for kNN-8/learned honestly.
  4. *(Optional)* A paired per-node gain plot (GAT − LSTM, per node, per horizon). The current boxplots (Fig. 6.3) show the GAT and temporal boxes overlapping almost entirely, which visually undersells a consistent paired gain.

### Figures that need updating

| Figure | File | Issue | Action |
|---|---|---|---|
| 5.1 | `quality_aware_framework.pdf` | Shows "LSTM / Transformer" (no GRU) and "8 Harmonic Features" (should be 10, see C5-10). No imputation branch. Monospace font. | Update the content; match the thesis font. |
| 5.2 | `dataloader_example_training_sequence.png` | The caption's colours do not match the image (the covariate is green). It does not make clear that the target is a single point at t+h. | Fix the caption or the colours; annotate "point forecast at t+h". |
| 5.3 | `experiment1_split_comparison.pdf` | Panel C shows up to 77 windows/day, but the maximum for a contiguous day is 65, implying runs across midnight despite the "daytime" label. Red/green palette. | Verify the counts; use a colourblind-safe palette. |
| 5.4 | `a1_diurnal_vs_error.png` | Code name "h8" in the title. | Clean the title. |
| 5.5 | `ext_correlation_heatmap.png` | Raw variable names. Full symmetric matrix. Horizon not stated. Includes POI/transit/distance variables never discussed. | Mask the upper triangle, rename variables, state h; either discuss or drop the extra variables. Delete the duplicate `correlation_heatmap.png`. |
| 5.6 | `threshold_cvrmse_lstm_mv_h8.png` | Raw title "lstm_mv, horizon 8 steps". Breakpoint CI not reported in the text. | Use a clean title (LSTM-MV, h = 8 (2 h)); report the CI and slopes in the text. |
| 5.7 | `ext_cluster_radar.png` | Raw variable names; the "−1.0" label overlaps. | Rename variables; fix the label. Delete the duplicate `cluster_radar.png`. |
| 5.8 | `a3_cvrmse_by_cluster.png` | **Axis says "Cluster 0 / Cluster 1"** instead of Suburban / Urban Core. | Relabel (0 = Suburban, 1 = Urban Core). |
| 5.10 | `exp_h_seasonal_error_side_by_side.png` | Legend title "interpolation" (raw). | Use "Imputation" or remove the legend title. |
| 5.11, 5.12 | `exp_i_case_study_*.png` | Raw stream IDs in titles. **The un-imputed and KNN panels show different time periods** (e.g. Shields: Jul–Sep 2023 vs Dec 2025–Jan 2026), so they cannot be compared visually. The un-imputed panels span months with 6-h snippets, which is unreadable. Heber's y-axis is dominated by the 2,240 outlier. | Plot both models over the **same matched dates** (from the matched-pairs set), a few days long. For Heber, either clip the y-axis (and annotate the outlier) or remove the outlier if it is a fault (T2-D). Shields' predicted spikes recur at about 16:00 daily, consistent with Fig. 5.9; one sentence explaining this will pre-empt "how does a 2-h-ahead model hit a one-step spike?". |
| 5.14 | `exp_k_extreme_events_side_by_side.png` | Absolute RMSE (scale-confounded, T2-I). | Add a relative-error panel, or switch to CV-RMSE. |
| 5.15 | `exp_l_matched_pairs_comparison.png` | Panel D title: **"24.4x" is wrong (≈3.7× from its own bars)** and "+4.3%" should be pp; "The Operational Dilemma" is a rhetorical title. Panel D's twin-axis side-by-side bars are hard to read. Panel C axis "[%]" should be pp. | Recompute; split Panel D into two simple bars or a small table; use pp throughout. |
| 6.1 | `sensor_availability_exp1_vs_exp2.pdf` | Left panel shows the **interpolated 70/15/15** protocol, not Chapter 5's un-imputed alternating-day split (X1). The right-panel note "56 stream slots absent (27 of 81 stations…)" mixes streams and stations on one axis. | Regenerate the left panel; reword the note ("110 streams vs 54 station nodes; 27 of 81 candidate stations excluded at < 30% completeness"). |
| 6.2 | `attention_graph_*.png` | Raw titles ("gat_transformer on gabriel — mean attention (seed 42)"). No basemap, scale bar or edge-weight colour bar. FC and learned panels are unreadable hairballs. Single seed. | Clean the titles; add a scale bar and colour bar; for FC/learned, show only the top-k edges per node (e.g. k = 3) or use an adjacency heatmap. State the seed in the caption. |
| 6.3 | `model_comparison_mae.png`, `_mase.png` | **"raw vehicle counts"**. The MASE y-label is clipped at the top. "GAT-Trans (Plain Gab)" is truncated. The seasonal period of the MASE scaling is not stated (S-Naive medians below 1 suggest in-sample scaling; say so). | Fix the labels; state the MASE seasonality and scaling period in the caption. |
| 6.5 | `attention_distance_*.png` | Raw lowercase titles/axes; raw α (degree-confounded for Gabriel). The subfigure labels `fig:dist-decay-gabriel`/`-fc` are never `\ref`'d. | Clean the labels; plot α·|N(i)|; extend to 4 panels (see above). |
| 6.6 | `seed_stability.png` | **The image does not match the caption** (C6-1). Raw configuration names on the x-axis; error bars too small to see at this scale. | Regenerate, or recaption and add a second figure; use display names. |

**Housekeeping:** non-figure files are sitting in the figure folders: `reviewer_*_response.pdf`, `mdpi_sensors.pdf`, `template.pdf`. There are also probable duplicates: `correlation_heatmap.png` (duplicates `ext_`) and `cluster_radar.png` (duplicates `ext_`). Move or remove them before sharing the source.

---

## Suggested order of work

1. **Chapter 5 numbers** (C5-1 to C5-15), plus X2. About half a day.
2. **Chapter 6 numbers and contradictions** (C6-1 to C6-15). About half a day, plus regenerating Fig. 6.6.
3. **X1**: align the Experiment 1 protocol description in Chapters 4 and 6 and regenerate the left panel of Fig. 6.1.
4. **Figure label and title fixes** (Figs 5.4–5.8, 5.10–5.12, 5.15, 6.2, 6.3, 6.5). Mostly regenerating with display names.
5. **New Chapter 6 figures**: map, completeness scatter, four-panel distance decay.
6. **Short Limitations subsection** in each chapter covering the Tier 2 items not fixed.
7. **Tone pass** (Tier 3).

Items 1–4 are the minimum for sending as a first draft. Items 5–7 can follow in the revision cycle if time is short, but it is worth saying so in the covering note.
