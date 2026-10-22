# Chapter 6 — new figures (generated 2026-10-13)

Recommended in `../audit_report_ch5_ch6_2026-10-13.md`. Regenerate from the repository root with:

    python docs/thesis/ch6_new_figures/make_ch6_new_figures.py

Every figure is saved as both PDF (vector, for LaTeX) and PNG (300 dpi). The computed statistics are in
`figure_stats.md`. Data come from `artifacts/spatial_full`: test split, per-node metrics averaged over
seeds 42–44. Spatial gain = 100 × (MAE_temporal − MAE_GAT) / MAE_temporal, computed per node.
Six included nodes have no test targets, so every per-node statistic uses n = 48.

| File | Content |
|---|---|
| `fig_ch6_node_map_h1.{pdf,png}` | (a) GAT-LSTM test MASE and (b) spatial gain over the temporal-only LSTM, h = 1, on the OSM basemap with Gabriel edges |
| `fig_ch6_node_map_h4.{pdf,png}` | Same, h = 4 (supplementary or appendix) |
| `fig_ch6_completeness.{pdf,png}` | Completeness vs MASE; completeness vs gain; Gabriel-neighbour completeness vs gain |
| `fig_ch6_distance_decay.{pdf,png}` | Degree-normalised attention vs distance for Gabriel, kNN-8, FC and learned adjacency |
| `fig_ch6_paired_gain.{pdf,png}` | Per-node gain distributions by horizon, for the LSTM and Transformer backbones |

## Suggested captions

**Node map (h = 1).** Per-node test performance of the champion GAT-LSTM (Gabriel) at $h=1$ (15 min).
(a) MASE. (b) Percentage MAE reduction relative to the temporal-only LSTM; blue nodes benefit from
spatial message passing. Grey lines are Gabriel edges. Hollow circles are included nodes with no test
targets ($n=6$); crosses are sensors excluded for completeness below 30%. Only Falconar St ($-6.4\%$)
and Barras Bridge ($-4.8\%$) are worse with the GAT.

**Completeness.** Per-node relationships at $h=1$ ($n=48$). (a) Lower completeness is associated with
higher MASE (Pearson $r=-0.33$, $p=0.021$). (b) Completeness is unrelated to spatial gain
($r=+0.10$, $p=0.51$). (c) Mean completeness of Gabriel neighbours shows no marginal association with
gain ($r=-0.15$, $p=0.30$). After controlling for the node's own completeness, the partial
correlation is $-0.29$ ($p=0.048$).

**Distance decay.** Attention against inter-sensor distance. Each weight is normalised by the mean
weight over the destination node's non-self neighbours, so 1 corresponds to uniform attention; this
removes the effect of in-degree and of the self-loop share. Red lines show medians over eight
distance quantile bins. Distance decay is present for Gabriel ($r=-0.36$) and fully connected
($r=-0.42$) graphs. It is absent for kNN-8 ($r=-0.05$, $p=0.23$) and learned adjacency
($r=+0.02$, $p=0.40$).

**Paired gain.** Per-node spatial gain of the GAT (Gabriel) over its matched temporal-only model,
using seed-averaged test MAE. Each point is one node ($n=48$); boxes show the IQR and median. The
numbers above each box count the nodes that improved.

## Points that affect the thesis text

1. **Completeness–MASE correlation:** $r=-0.334$, **$p=0.021$** (GAT-LSTM, $h=1$). The text currently
   says $p<0.02$. The correlation strengthens with horizon: $r=-0.365$ at $h=2$ and $-0.392$ at $h=4$.
2. **Sign error, ch. 6 lines ~620–621:** Falconar St (−6.4%) and Barras Bridge (−4.8%) are the only
   nodes where the GAT is *worse* than the temporal-only LSTM at $h=1$.
3. **Breadth of the gain:** 46/48 nodes improve at $h=1$ and $h=2$, and 47/48 at $h=4$ (LSTM backbone). The
   Transformer backbone gives 45/48, 46/48 and 48/48. Median gain rises with horizon: 3.7% → 5.0% → 6.4% (LSTM).
4. **Distance-decay r values:** the raw-α values reproduce the thesis figures (Gabriel −0.52, kNN-8
   −0.03, FC −0.42, learned +0.08). After degree normalisation, Gabriel falls to **−0.36**, so part
   of the raw value reflects degree and self-loop share rather than distance. FC is
   unchanged (−0.42). kNN-8 (−0.05) and learned (+0.02, Spearman +0.10) still show no decay. The text
   should either cite the normalised values or say that the raw values are degree-confounded.
5. **Panel (c) is marginal:** the partial correlation ($p=0.048$) is one test among several and does not
   survive any multiple-comparison correction. Present it as exploratory, or omit panel (c).

## Caveats

- The attention exports (`aggregated/attention_distance_*.csv`) come from seed 42: final GAT layer,
  mean over 4 heads and 256 test windows. This was confirmed from the Experiment 2 source code on branch
  `feature/ch6-spatial-gat-rebuild` (see `_exp2_verification.md`). No attention was exported for the
  GAT-LSTM champion.
- The decay panels use GAT-Transformer runs for Gabriel, kNN-8 and FC because those are the
  exported files. The champion is GAT-LSTM. Say so in the text.
- The basemap tiles come from the existing cache, `artifacts/cache/tiles_osm`. OSM attribution is
  printed under the map figures.

## Other files in this folder

| File | Content |
|---|---|
| `dm_champion_seedavg.py` | Diebold–Mariano tests of the champion (gat_lstm\|gabriel) against every baseline: seed-averaged forecasts, absolute-error loss, HLN correction, BH correction across nodes, Newey–West (96-lag) robustness |
| `dm_champion_seedavg.csv` | Summary per comparison and horizon (n = 48 nodes) |
| `dm_champion_seedavg_pernode.csv` | Per-node DM statistics and p-values |
| `param_counts.csv` | Trainable parameter counts for every Experiment 2 configuration |
| `_exp2_verification.md` | Experiment 2 code and artefact facts, with file:line references |
| `_exp1_verification.md` | Experiment 1 (Chapter 5) facts: v1/v2 run sets, re-computed statistics |

Related outputs:
- `../ch5_v2_figures/`: v2 replacement for Ch5 Fig. 5.6 (`make_fig5_6_threshold_v2.py`) and the Experiment 1
  verification scripts (`exp1_verification_scripts/`).
- `../implementation_ch5_ch6_tier2_figures_2026-10-13.md`: the implementation brief that places these figures
  (Part D) and applies the remaining Tier 2 edits.
