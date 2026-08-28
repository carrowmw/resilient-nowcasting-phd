# MDPI Sensors Manuscript Revision Plan

Based on the newly generated V2 results (using the dataset without artificial zero-imputation), several sections of the existing `mdpi_sensors.tex` manuscript contain outdated findings and require substantial rewriting. The shift from a 12-sensor subset with artificial zero-filling to a robust 110-sensor dataset fundamentally changes the core narrative of the paper.

Below is a section-by-section breakdown of the necessary changes.

## 1. Abstract & Introduction
- **Current state**: Mentions that average sequence lengths below 20 hours produce higher prediction errors, and completeness paradox.
- **Required changes**: 
    - Update the sequence length threshold from 20 hours to **~3 hours** (half a diurnal cycle).
    - Emphasise the shift in sample size (from 12 to 110 sensors).
    - Introduce the spatial aspect (transit-heavy urban core vs. suburban) which is now a major finding.

## 2. Section 3.1: Data Collection & Preprocessing
- **Current state**: Mentions regular nighttime gaps but implies the dataset was processed as-is.
- **Required changes**: 
    - Explicitly state the methodological decision to leave `NaN` values intact, avoiding artificial zero-imputation which previously corrupted the sequence lengths.
    - Update the summary statistics (average sequence length is ~3 hours, not 8-35 hours).
    - **Figure 1 (preprocessed_data_combined.png)**: Keep, but ensure the narrative surrounding it reflects the true short-sequence nature of the data.
    - **Interpolation Context**: Include the results from the various interpolation attempts (zero-fill, linear, kNN, spline) here or reference them to demonstrate why advanced imputation did not meaningfully overcome the data gap challenges (e.g., highlighting the RMSE vs MAPE trade-offs and how night-zero filling was actively harmful at 24h horizons).

## 3. Section 3.5: Model Training
- **Current state**: States that the experiment was conducted on "twelve sensors selected for their superior data completeness".
- **Required changes**: 
    - Update to state that the experiment was conducted on **110 viable sensors** across the network.
    - Mention the new criteria for viability ($\ge$ 500 training windows).
    - Update the interpolation experiment description to include Linear, Spline, and KNN interpolation, rather than just nighttime zero-fill.

## 4. Section 3.7: Hyper-parameter Tuning
- **Current state**: Table 2 lists hyperparameters for the 12-sensor models.
- **Required changes**: 
    - Update Table 2 with the new hyperparameters generated for the V2 run.
    - Rewrite the accompanying text explaining the architecture differences based on the new tuning outcomes.

## 5. Section 4.1: Univariate / Multivariate Experiment
- **Current state**: Concludes that Univariate (UV) outperformed Multivariate (MV) at the 2-hour horizon, and MV was only marginally better at the 6-hour horizon. Discusses the "completeness paradox" using a 20-hour vs 35-hour threshold.
- **Required changes**: 
    - **Complete Rewrite**. The V2 data shows LSTM-MV dominates across *both* horizons. 
    - Replace Tables 3 and 4 with the new baseline performance tables (LSTM-MV vs LSTM-UV vs Baselines).
    - Introduce the **Diebold-Mariano (DM) tests** which statistically prove the LSTM-MV's superiority on over 91% of sensors.
    - Remove the old Figure 3 (`rmse_completeness_bar.png` and `rmse_ave_sequences_bar.png`). Replace with the **Segmented Regression Threshold plot** showing the exact 3-hour breakpoint.
    - Discuss the resolution of the "completeness paradox" using the new partial correlation analysis, proving that completeness is a spurious metric.

## 6. Section 4.2: Interpolation Experiment
- **Current state**: Concludes that interpolation yields negligible improvements and introduces artificial patterns.
- **Required changes**: 
    - Update Table 5 (`raw_int_mv24_comparison`) to compare Baseline vs Zero-fill vs Linear vs Spline vs KNN.
    - Acknowledge that while zero-fill is harmful, **Linear Interpolation provides a 17% reduction in RMSE** at the 2-hour horizon. Discuss the RMSE vs. MAPE trade-off.
    - Regenerate and replace Figure 4 (`w_rmse_ave_sequences_bar.png`).

## 7. Section 4.3: Predictive Modelling Performance
- **Current state**: Random Forest meta-model achieved an $R^2$ of 0.35. Standard deviation of sequence length was the most important predictor.
- **Required changes**: 
    - **Major Update**. The RF meta-model now achieves an **$R^2$ of 0.81**.
    - Update the feature importance narrative: **Sequence Count** (fragmentation) is now overwhelmingly the most important predictor, not standard deviation.
    - Replace Figure 5 (`w_mv24_pdp.png` and `w_mv24_feature_importance.png`) with the new RF feature importance plots.
    - Add the **OLS interaction analysis** to prove that sequence fragmentation hurts performance regardless of pedestrian volume.

## 8. New Section: Spatial and Contextual Analysis
- **Current state**: Does not exist.
- **Required changes**: 
    - Add a new section detailing the **Location-Based Clustering** (Urban Core vs Suburban).
    - Insert the **Radar Chart** to visualise the clusters.
    - Discuss the **Error vs Distance** relationship and the highly significant **Mann-Whitney U test** proving that high-transit areas have stronger diurnal predictability.

## 9. Section 5: Discussion & Conclusion
- **Current state**: Heavily references the 20-hour threshold and the unreliability of completeness.
- **Required changes**: 
    - Re-centre the discussion on the true 3-hour requirement for LSTMs.
    - Emphasise the success of identifying spatial (transit/POI) proxies for data quality.
    - Remove caveats about the small sample size (12 sensors), as the study is now statistically robust (110 sensors).

---
### Summary of Figure/Table Replacements

| Current Figure/Table | Status | New Figure/Table |
| :--- | :--- | :--- |
| Table 1 (Hyperparameters) | UPDATE | Update with V2 hyperparameters |
| Table 2 (Hyperparameter comparison) | UPDATE | Update with V2 optimal architectures |
| Tables 3 \& 4 (MV vs UV) | REPLACE | Replace with Baseline vs Models (LSTM, Naive, ARIMA) |
| Figure 3 (RMSE vs completeness) | REPLACE | Replace with Segmented Regression (3-hour threshold) |
| Table 5 (Raw vs Int comparison) | REPLACE | Replace with 5-strategy Interpolation comparison |
| Figure 4 (Sequence length comparison) | REPLACE | Replace with Diurnal vs Error scatter plot |
| Figure 5 (PDP and Feature Importance) | REPLACE | Replace with V2 RF Feature Importance |
| *New* | ADD | Diebold-Mariano Statistical Tests Table |
| *New* | ADD | Cluster Profile Table & Radar Chart |
| *New* | ADD | 2D Temporal Missingness Heatmap |
| *New* | ADD | Distance vs Error plot |
