# Chapter 6 new figures — summary statistics

Test split; per-node metrics averaged over available seeds (3 for trainable models, 1 for heuristics). Spatial gain = 100 × (MAE_temporal − MAE_spatial) / MAE_temporal, per node.

### Node map (h=1)
- Nodes with test targets: 48 of 54 included; no test targets: NclHeatonParkViewHeatonParkRd, NclJesmondDeneRdBentonBank, NclJohnDobsonStNorthumberlandRdNorth, NclNinthAveSecondAve, NclSideStCrownPosadaNorth, NclSideStCrownPosadaSouth
- MASE range 0.352–0.773 (median 0.581)
- Spatial gain range -6.4% to 13.7% (median 3.7%); nodes worse with GAT: NclFalconarStLoversLn (-6.4%), NclBarrasBridgeCivicCentre (-4.8%)

### Node map (h=4)
- Nodes with test targets: 48 of 54 included; no test targets: NclHeatonParkViewHeatonParkRd, NclJesmondDeneRdBentonBank, NclJohnDobsonStNorthumberlandRdNorth, NclNinthAveSecondAve, NclSideStCrownPosadaNorth, NclSideStCrownPosadaSouth
- MASE range 0.456–1.038 (median 0.651)
- Spatial gain range -1.3% to 18.5% (median 6.4%); nodes worse with GAT: NclFalconarStLoversLn (-1.3%)

### Completeness scatter (h=1, n=48)
- (a) completeness vs MASE: Pearson r=-0.334 (p=0.021); Spearman rho=-0.370 (p=0.010)
- (b) completeness vs spatial gain: Pearson r=+0.098 (p=0.507); Spearman rho=+0.004 (p=0.980)
- (c) neighbour completeness vs gain: Pearson r=-0.152 (p=0.303); partial r controlling own completeness=-0.290 (p=0.048)

### Distance decay (attention exports in artifacts/spatial_full/aggregated)
- gat_transformer_gabriel: 156 edges, median in-degree 3.0; raw alpha Pearson r=-0.519 (p=3.8e-12); relative alpha Pearson r=-0.356 (p=5.2e-06), Spearman rho=-0.351 (p=7.1e-06)
- gat_transformer_knn8: 544 edges, median in-degree 9.5; raw alpha Pearson r=-0.033 (p=0.44); relative alpha Pearson r=-0.052 (p=0.23), Spearman rho=-0.083 (p=0.054)
- gat_transformer_fc: 2862 edges, median in-degree 53.0; raw alpha Pearson r=-0.419 (p=6.5e-122); relative alpha Pearson r=-0.415 (p=9.9e-120), Spearman rho=-0.449 (p=2.9e-142)
- gat_learned_identity: 1756 edges, median in-degree 33.5; raw alpha Pearson r=+0.080 (p=0.00076); relative alpha Pearson r=+0.020 (p=0.4), Spearman rho=+0.097 (p=4.4e-05)

### Paired per-node spatial gain (seed-mean test MAE)
- LSTM backbone, h=1: median 3.7%, mean 3.9%, IQR 2.5–5.5%, improved 46/48 nodes, min -6.4% (NclFalconarStLoversLn)
- LSTM backbone, h=2: median 5.0%, mean 5.8%, IQR 3.4–7.1%, improved 46/48 nodes, min -5.3% (NclFalconarStLoversLn)
- LSTM backbone, h=4: median 6.4%, mean 7.4%, IQR 3.6–9.4%, improved 47/48 nodes, min -1.3% (NclFalconarStLoversLn)
- Transformer backbone, h=1: median 3.1%, mean 3.5%, IQR 2.1–4.8%, improved 45/48 nodes, min -2.1% (NclJesmondRdAllSaintsCemeteryNorth)
- Transformer backbone, h=2: median 4.2%, mean 5.1%, IQR 2.6–7.1%, improved 46/48 nodes, min -1.5% (NclBarrasBridgeCivicCentre)
- Transformer backbone, h=4: median 5.1%, mean 6.9%, IQR 3.1–9.3%, improved 48/48 nodes, min 0.1% (NclBarrasBridgeCivicCentre)
