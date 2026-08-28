import re

filepath = '/Users/administrator/Code/ml-agent-workspace/docs/mdpi_sensors/mdpi_sensors.tex'
with open(filepath, 'r') as f:
    content = f.read()

target = r"""\caption{Comparison of tuned hyper-parameters for different experiments. The window size was consistently selected as 24 time steps (6 hours). All models selected the same batch size. Hidden dimensions and number of layers vary between experiments. The smallest model is UV8 and the largest is UV24.}\label{tab:hyperparameter_comparison}
\begin{tabular}{@{}lrrrr@{}}
\toprule
& \textbf{UV-24} & \textbf{MV-24} & \textbf{UV-8} & \textbf{MV-8} \\
\midrule
\texttt{window\_size} & 24 & 24 & 24 & 24 \\
\texttt{horizon} & 24 & 24 & 8 & 8 \\
\texttt{stride} & 1 & 1 & 1 & 1 \\
\texttt{batch\_size} & 32 & 32 & 32 & 32 \\
\texttt{model\_type} & lstm & lstm & lstm & lstm \\
\texttt{hidden\_dim} & 256 & 64 & 64 & 64 \\
\texttt{num\_layers} & 2 & 2 & 1 & 2 \\
\texttt{dropout} & 0.064 & 0.053 & 0.072 & 0.173 \\
\texttt{learning\_rate} & 0.00011 & 0.00024 & 0.00382 & 0.00012 \\
\texttt{scheduler\_step\_size} & 7 & 4 & 4 & 3 \\
\texttt{scheduler\_gamma} & 0.518 & 0.564 & 0.440 & 0.802 \\
\bottomrule
\end{tabular}
\end{table}

The resulting hyperparameters from the prior optimisation phase, detailed in Table \ref{tab:hyperparameter_comparison}, reveal distinct architectural differences among the models that were subsequently applied to the 110-sensor network. The UV-8 model emerged as the most compact, utilising a single layer with 64 hidden dimensions. In contrast, MV models maintained consistent architectures with 64 hidden dimensions across two layers. The UV-24 model proved to be the largest, implementing 256 hidden dimensions across two layers. This configuration may raise concerns about potential over-parameterisation, as the UV-24 model contains four times the hidden units of the MV-24 model despite processing fewer input features (1 versus 12). Additionally, the MV-8 model's elevated dropout rate of 17.3\% (compared to 5-7\% in other models) suggests a higher susceptibility to overfitting, necessitating more aggressive regularisation."""

replacement = r"""\caption{Hyper-parameters adopted for the univariate and multivariate LSTM models. Both architectures share identical window size, batch size, dropout, and learning rate schedules. Models were evaluated at horizons $h=8$ and $h=24$ using these constant architectures.}\label{tab:hyperparameter_comparison}
\begin{tabular}{@{}lrr@{}}
\toprule
\textbf{Parameter} & \textbf{LSTM-UV} & \textbf{LSTM-MV} \\
\midrule
\texttt{window\_size} & 24 & 24 \\
\texttt{batch\_size} & 128 & 128 \\
\texttt{hidden\_dim} & 64 & 64 \\
\texttt{num\_layers} & 1 & 2 \\
\texttt{dropout} & 0.1 & 0.1 \\
\texttt{learning\_rate} & 0.001 & 0.0005 \\
\texttt{scheduler\_step\_size} & 5 & 5 \\
\texttt{scheduler\_gamma} & 0.7 & 0.7 \\
\bottomrule
\end{tabular}
\end{table}

The hyper-parameters established for the LSTM models are detailed in Table \ref{tab:hyperparameter_comparison}. The chosen architectures remain remarkably consistent across univariate and multivariate variants, with both employing 64 hidden dimensions, a batch size of 128, and a dropout rate of 0.1. The primary architectural distinction lies in the depth of the networks: the multivariate model requires two LSTM layers to process the complex interactions between raw counts and the 11 engineered temporal features, whereas the univariate model, relying solely on raw counts, is constrained to a single layer. The multivariate model also employs a slightly more conservative learning rate ($0.0005$ versus $0.001$). Importantly, these unified architectures were held constant when evaluating performance across both the short ($h=8$) and long ($h=24$) forecasting horizons, ensuring that any performance variations are strictly attributable to the forecasting horizon and input features rather than model-specific tuning artefacts."""

if target in content:
    content = content.replace(target, replacement)
    print("Success")
    with open(filepath, 'w') as f:
        f.write(content)
else:
    print("Target not found")
