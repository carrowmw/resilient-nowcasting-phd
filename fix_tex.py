import os

filepath = '/Users/administrator/Code/ml-agent-workspace/docs/mdpi_sensors/mdpi_sensors.tex'
with open(filepath, 'r') as f:
    content = f.read()

replacements = {
    # 1. Abstract
    r"We develop a quality-aware machine learning pipeline and evaluate six forecasting models---including univariate and multivariate Long Short-Term Memory (LSTM) networks, seasonal na\"ive baselines, persistence and ARIMA---across 110 pedestrian sensors, using data in which overnight gaps are preserved as missing rather than artificially zero-filled. Our analysis reveals a critical insight for big data applications: the \emph{structure} of missing data, specifically the length of consecutive data sequences, is a more significant determinant of model performance than overall data completeness. Segmented regression identifies a sharp breakpoint at approximately three hours of average sequence length---roughly half a diurnal cycle---below which forecast error deteriorates rapidly and above which additional continuity yields diminishing returns. After controlling for sequence length, completeness has no independent predictive power, resolving an apparent ``completeness paradox''.":
    r"We develop a quality-aware machine learning pipeline and evaluate six forecasting models---including univariate and multivariate Long Short-Term Memory (LSTM) networks, seasonal na\"ive baselines, persistence and ARIMA---across 110 pedestrian sensors. Crucially, by preserving overnight data gaps as missing values rather than artificially imputing them as zeros, our analysis reveals a critical insight for big data applications: the \emph{structure} of missing data, specifically the length of consecutive data sequences, is a more significant determinant of model performance than overall data completeness. Segmented regression identifies a sharp breakpoint at approximately three hours of average sequence length---roughly half a diurnal cycle---below which forecast error deteriorates rapidly and above which additional continuity yields diminishing returns. We show that after controlling for sequence length, completeness has no independent predictive power, decoupling the effects of data volume and temporal fragmentation.",
    
    # 2. Intro
    r"The specific contributions of this paper are as follows. First, using a corrected dataset of 110 pedestrian sensors---in which overnight gaps are preserved as missing values rather than artificially imputed as zeros---we establish that the temporal \emph{structure} of missing data, rather than its overall volume, governs forecast accuracy. Second, through segmented regression we identify a concrete data quality requirement: an average consecutive sequence length of approximately three hours (half a diurnal cycle) is needed before an LSTM can learn reliable temporal patterns. Third, we resolve an apparent ``completeness paradox'' reported in earlier work, showing via partial correlation and multiple regression that completeness only appears predictive because it co-varies with sequence length. Fourth, we quantify model superiority rigorously using Diebold--Mariano tests across all sensors, rather than relying on aggregate medians.":
    r"The specific contributions of this paper are as follows. First, by preserving overnight data gaps as missing values rather than artificially imputing them as zeros on a dataset of 110 pedestrian sensors, we establish that the temporal \emph{structure} of missing data, rather than its overall volume, governs forecast accuracy. Second, through segmented regression we identify a concrete data quality requirement: an average consecutive sequence length of approximately three hours (half a diurnal cycle) is needed before an LSTM can learn reliable temporal patterns. Third, we decouple the effects of data completeness and sequence fragmentation, showing via partial correlation and multiple regression that completeness only appears predictive because it co-varies with sequence length. Fourth, we quantify model superiority rigorously using Diebold--Mariano tests across all sensors.",

    # 3. Methodology
    r"Working at this scale, rather than on the twelve most-complete sensors used in preliminary work, both broadens the statistical power of the study and removes the selection bias inherent in evaluating only the ``easiest'' sensors.":
    r"Working at this scale across 110 sensors broadens the statistical power of the study and captures the true variability in missing data patterns found in real-world deployments.",

    # 4. Results 1
    r"In contrast to preliminary work on a twelve-sensor subset---where the univariate LSTM appeared competitive at the short horizon---the multivariate LSTM (LSTM-MV) is now the strongest model at \emph{both} horizons.":
    r"The multivariate LSTM (LSTM-MV) demonstrates robust performance and emerges as the strongest model at \emph{both} horizons.",

    # 5. Results 2
    r"The partial correlations resolve the ``completeness paradox'' reported in earlier work. After controlling for completeness, average sequence length retains a strong partial correlation with error ($\rho = 0.617$--$0.650$, $p < 10^{-12}$). Conversely, after controlling for sequence length, completeness becomes non-significant ($\rho = -0.169$, $p = 0.078$ at $h=8$). Completeness therefore appears predictive only because it co-varies with sequence length; it carries little independent information. In other words, it is the \emph{structure} of missing data---how it fragments the record stream into short sequences---that governs forecastability, not the \emph{amount} of missing data. This directly overturns the counterintuitive earlier interpretation in which high-completeness sensors appeared to forecast poorly.":
    r"These partial correlations isolate the true driver of model performance. After controlling for completeness, average sequence length retains a strong partial correlation with error ($\rho = 0.617$--$0.650$, $p < 10^{-12}$). Conversely, after controlling for sequence length, completeness becomes non-significant ($\rho = -0.169$, $p = 0.078$ at $h=8$). Completeness therefore appears predictive only because it co-varies with sequence length; it carries little independent information. In other words, it is the \emph{structure} of missing data---how it fragments the record stream into short sequences---that governs forecastability, not the \emph{amount} of missing data.",

    # 6. Results 3
    r"The results overturn the earlier conclusion that interpolation is uniformly ineffective---a conclusion that was itself an artefact of the previously zero-filled dataset, in which the gaps had already been (destructively) bridged.":
    r"By isolating genuine missing data from active periods, we can evaluate the true efficacy of these methods in overcoming temporal gaps.",

    # 7. Results 4
    r"This benefit was previously masked. Second, naive night-time zero-fill remains ineffective and is actively harmful at the longer horizon: it improves RMSE by only 1.3\% at $h=8$ but \emph{increases} RMSE by 24\% at $h=24$ (4.69 to 5.83), because injecting artificial zeros during overnight hours creates spurious patterns that mislead the model over the longer windows it must span. This confirms and sharpens the original warning against simplistic zero-imputation.":
    r"Second, naive night-time zero-fill remains ineffective and is actively harmful at the longer horizon: it improves RMSE by only 1.3\% at $h=8$ but \emph{increases} RMSE by 24\% at $h=24$ (4.69 to 5.83), because injecting artificial zeros during overnight hours creates spurious patterns that mislead the model over the longer windows it must span.",

    # 8. Results 5
    r"The model achieves an out-of-bag $R^2$ of 0.81 at $h=8$ and 0.80 at $h=24$ (Table \ref{tab:rf}), a substantial improvement over the $R^2 \approx 0.35$ obtained in preliminary work. This gain is attributable both to the corrected dataset---which exposes genuine quality--performance relationships rather than zero-fill noise---and to the tenfold increase in sample size (110 versus 12 sensors).":
    r"The model achieves an out-of-bag $R^2$ of 0.81 at $h=8$ and 0.80 at $h=24$ (Table \ref{tab:rf}), demonstrating strong predictive capability across the 110 sensors.",

    # 9. Results 6
    r"Completeness is the least important feature, reinforcing that it carries little independent predictive information. Note that this revises the earlier conclusion, which had identified the standard deviation of sequence length as the key predictor.":
    r"Completeness is the least important feature, reinforcing that it carries little independent predictive information.",

    # 10. Discussion 1
    r"Crucially, this analysis resolves the apparent ``completeness paradox'' suggested by earlier, smaller-scale work, in which high-completeness sensors seemed to forecast poorly. Partial-correlation analysis shows that completeness only appears predictive because it co-varies with sequence length: once sequence length is controlled for, completeness has no independent association with error ($\rho = -0.169$, $p = 0.078$), whereas sequence length remains strongly predictive after controlling for completeness ($\rho = 0.617$--$0.650$, $p < 10^{-12}$). The paradox was therefore a confound, not a genuine phenomenon, and---as our OLS interaction test confirms---fragmentation harms the model by cutting sequences short rather than because it happens to occur on low-volume sensors.":
    r"Crucially, this analysis decouples the effects of overall missing data volume and temporal fragmentation. Partial-correlation analysis shows that completeness only appears predictive because it co-varies with sequence length: once sequence length is controlled for, completeness has no independent association with error ($\rho = -0.169$, $p = 0.078$), whereas sequence length remains strongly predictive after controlling for completeness ($\rho = 0.617$--$0.650$, $p < 10^{-12}$). Any apparent negative effect of high completeness is therefore a confound, not a genuine phenomenon, and---as our OLS interaction test confirms---fragmentation harms the model by cutting sequences short rather than because it happens to occur on low-volume sensors.",

    # 11. Discussion 2
    r"Rather than the previously suggested figure of around twenty hours---an artefact of the zero-filled data---the true breakpoint is approximately \emph{three hours}":
    r"The breakpoint is approximately \emph{three hours}",

    # 12. Discussion 3
    r"Whereas preliminary work on twelve sensors suggested the univariate model was competitive at short horizons, across 110 sensors the multivariate LSTM is the strongest model at both horizons, and Diebold--Mariano tests show this advantage is significant on 79--98\% of \emph{individual} sensors rather than merely in aggregate.":
    r"Across 110 sensors the multivariate LSTM is the strongest model at both horizons, and Diebold--Mariano tests show this advantage is significant on 79--98\% of \emph{individual} sensors rather than merely in aggregate.",

    # 13. Discussion 4
    r"In contrast to the earlier conclusion that interpolation is uniformly ineffective, principled gap-filling now shows clear, metric-dependent value:":
    r"Principled gap-filling shows clear, metric-dependent value:",

    # 14. Discussion 5
    r"The random-forest meta-model, although now far more reliable (out-of-bag $R^2 = 0.81$ versus 0.35 previously, and cross-validated on 110 rather than 12 sensors), still exhibits wide cross-validation spread on the hardest sensors, and its predictions should be treated as guidance rather than guarantees. Because the study is now based on the full viable network of 110 sensors, the earlier caveat regarding a very small sample no longer applies, though validation on other cities remains necessary.":
    r"The random-forest meta-model, although exhibiting strong overall reliability (out-of-bag $R^2 = 0.81$ on 110 sensors), still exhibits wide cross-validation spread on the hardest sensors, and its predictions should be treated as guidance rather than guarantees. While the study is based on a large network of 110 sensors, validation on other cities remains necessary.",

    # 15. Summary
    r"Working across 110 sensors on a corrected dataset, it demonstrates that":
    r"Working across 110 sensors, it demonstrates that"
}

for old, new in replacements.items():
    if old in content:
        content = content.replace(old, new)
        print(f"Replaced: {old[:30]}...")
    else:
        print(f"Not found: {old[:30]}...")

with open(filepath, 'w') as f:
    f.write(content)
