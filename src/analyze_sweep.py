import os
import re
import argparse
import pandas as pd
import numpy as np
import optuna
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import spearmanr
import logging

# Set up simple logging for the analyzer
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def parse_sweep_logs(log_dir="logs"):
    """
    Parses Optuna sweep logs to extract hyperparameter trials and their final validation loss.
    """
    trials = []
    if not os.path.exists(log_dir):
        logger.error(f"Log directory {log_dir} not found.")
        return pd.DataFrame()

    log_files = [f for f in os.listdir(log_dir) if f.startswith("optuna_sweep") and f.endswith(".log")]
    
    # Regex patterns for parameters and loss
    param_patterns = {
        "lr": r"lr: ([\d\.e-]+)",
        "w_forecast": r"w_forecast: ([\d\.e-]+)",
        "w_impute": r"w_impute: ([\d\.e-]+)",
        "sparsity_threshold": r"sparsity_threshold: ([\d\.e-]+)",
        "mask_prob": r"mask_prob: ([\d\.e-]+)",
        "num_layers": r"num_layers: (\d+)",
        "embedding_dim": r"embedding_dim: (\d+)",
        "hidden_dim": r"hidden_dim: (\d+)",
        "batch_size": r"batch_size: (\d+)",
        "val_loss": r"Best validation loss: ([\d\.e-]+)"
    }

    for log_file in log_files:
        path = os.path.join(log_dir, log_file)
        with open(path, "r") as f:
            content = f.read()
            
            # Extract the best trial reported at the end of each log
            trial_data = {}
            found_any = False
            for param, pattern in param_patterns.items():
                match = re.search(pattern, content)
                if match:
                    trial_data[param] = float(match.group(1))
                    found_any = True
            
            if found_any and "val_loss" in trial_data:
                trials.append(trial_data)
                
    return pd.DataFrame(trials)

def load_study_data(storage: str, study_name: str):
    """
    Loads trial data from an Optuna SQLite database.
    """
    try:
        study = optuna.load_study(study_name=study_name, storage=storage)
        
        trials = []
        for trial in study.trials:
            if trial.state != optuna.trial.TrialState.COMPLETE:
                continue
            
            trial_data = trial.params.copy()
            trial_data["val_loss"] = trial.value
            trials.append(trial_data)
            
        return pd.DataFrame(trials)
    except Exception as e:
        logger.error(f"Failed to load study data from {storage}: {e}")
        return pd.DataFrame()

def analyze_importance(df):
    if df.empty or len(df) < 2:
        print("\n[!] Not enough trial data to perform analysis. Currently found only", len(df), "trials.")
        return
    
    X = df.drop(columns=["val_loss"])
    # Handle object columns if any
    X = pd.get_dummies(X, drop_first=True)
    y = df["val_loss"]
    
    # 1. Feature Importance via Random Forest
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X, y)
    
    importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
    
    # 2. Correlation Analysis
    correlations = {}
    for col in X.columns:
        corr, _ = spearmanr(X[col], y)
        correlations[col] = corr
    
    corr_series = pd.Series(correlations).sort_values()

    print("\n" + "="*40)
    print("HYPERPARAMETER IMPORTANCE (Random Forest)")
    print("="*40)
    print(importances.to_string())
    print("\n" + "="*40)
    print("PARAMETER CORRELATION WITH LOSS (Spearman)")
    print("(Negative = Parameter increase reduces loss)")
    print("(Positive = Parameter increase increases loss)")
    print("="*40)
    print(corr_series.to_string())
    print("="*40)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze Optuna sweep results")
    parser.add_argument("--study-name", type=str, help="Name of the Optuna study to analyze")
    parser.add_argument("--storage", type=str, help="SQLite database connection string (e.g., sqlite:///optuna.db)")
    args = parser.parse_args()

    if args.storage and args.study_name:
        logger.info(f"Loading data for study '{args.study_name}' from {args.storage}...")
        trial_df = load_study_data(args.storage, args.study_name)
    else:
        logger.info("No database provided. Falling back to log analysis...")
        trial_df = parse_sweep_logs()
    
    if not trial_df.empty:
        logger.info(f"Successfully extracted {len(trial_df)} data points.")
        analyze_importance(trial_df)
    else:
        logger.error("No trial data found.")
