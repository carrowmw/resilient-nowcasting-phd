import os
import ast
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

def parse_sweep_logs(log_path=None, log_dir="logs"):
    """
    Parses Optuna sweep logs to extract hyperparameter trials and their final validation loss.
    Supports:
    1. New explicit multi-trial logging format.
    2. Legacy 'Best Trial' summary fallback.
    3. Robust 'scattered' log parsing (extracting parameters from initialization lines).
    """
    all_trials = []
    
    if log_path:
        log_files = [log_path] if os.path.exists(log_path) else []
    elif os.path.exists(log_dir):
        log_files = [os.path.join(log_dir, f) for f in os.listdir(log_dir) if f.startswith("optuna_sweep") and f.endswith(".log")]
    else:
        logger.error(f"Log source not found: {log_path or log_dir}")
        return pd.DataFrame()

    # Regex patterns for various formats
    # 1. New multi-trial format
    trial_start_pattern = r"Trial (\d+) parameters: (\{.*\})"
    trial_end_pattern = r"Trial (\d+) finished with val_loss: ([\d\.e-]+)"
    
    # 2. Scattered/Legacy patterns (flexible for : or =)
    param_patterns = {
        "lr": r"lr[:=]\s*([\d\.e-]+)",
        "w_forecast": r"w_forecast[:=]\s*([\d\.e-]+)",
        "w_impute": r"w_impute[:=]\s*([\d\.e-]+)",
        "sparsity_threshold": r"sparsity_threshold[:=]\s*([\d\.e-]+)",
        "mask_prob": r"mask_prob[:=]\s*([\d\.e-]+)",
        "num_layers": r"num_layers[:=]\s*(\d+)",
        "embedding_dim": r"embedding_dim[:=]\s*(\d+)",
        "hidden_dim": r"hidden_dim[:=]\s*(\d+)",
        "batch_size": r"batch_size[:=]\s*(\d+)",
        "val_loss_summary": r"Best validation loss: ([\d\.e-]+)",
        "val_loss_epoch": r"val_loss[:=]\s*([\d\.e-]+)"
    }

    for path in log_files:
        with open(path, "r") as f:
            content = f.read()
            lines = content.splitlines()
            
            # 1. Attempt Modern Format First
            file_trials = {} # tid -> data
            modern_count = 0
            for line in lines:
                start_match = re.search(trial_start_pattern, line)
                if start_match:
                    tid, params_str = start_match.groups()
                    try: file_trials[tid] = ast.literal_eval(params_str)
                    except: continue
                
                end_match = re.search(trial_end_pattern, line)
                if end_match:
                    tid, val = end_match.groups()
                    if tid in file_trials:
                        file_trials[tid]["val_loss"] = float(val)
                        all_trials.append(file_trials[tid])
                        modern_count += 1
                        del file_trials[tid]
            
            if modern_count > 0:
                continue

            # 2. Fallback to Scattered Parsing (Split by trial markers)
            # "Loading RealIoTDataset" is the best separator as it marks the start of setup
            trial_chunks = re.split(r"(Loading RealIoTDataset)", content)
            
            # Recombine chunks (re.split with groups keeps the delimiter)
            recombined = []
            for i in range(1, len(trial_chunks), 2):
                recombined.append(trial_chunks[i] + trial_chunks[i+1])
            
            # If no dataset loading markers, try "Model initialised"
            if not recombined:
                trial_chunks = re.split(r"(Model initialised)", content)
                for i in range(1, len(trial_chunks), 2):
                    recombined.append(trial_chunks[i] + trial_chunks[i+1])

            for chunk in recombined:
                trial_data = {}
                found_any = False
                min_loss = float('inf')
                
                for param, pattern in param_patterns.items():
                    matches = re.findall(pattern, chunk)
                    if matches:
                        if param == "val_loss_epoch":
                            losses = [float(m) for m in matches]
                            min_loss = min(min_loss, min(losses))
                        elif param == "val_loss_summary":
                            trial_data["val_loss"] = float(matches[-1])
                        else:
                            # Take the last match for the parameter in this chunk
                            # Special case: embedding_dim, hidden_dim, num_layers 
                            # might appear in multiple lines, take the most recent
                            trial_data[param] = float(matches[-1])
                        found_any = True
                
                if "val_loss" not in trial_data and min_loss != float('inf'):
                    trial_data["val_loss"] = min_loss
                
                if found_any and "val_loss" in trial_data:
                    all_trials.append(trial_data)
            
            # 3. Last Resort: Single best trial summary at end of file (if not already found)
            if not recombined:
                legacy_data = {}
                found_legacy = False
                for param, pattern in param_patterns.items():
                    match = re.search(pattern, content)
                    if match:
                        if param in ["val_loss_summary", "val_loss_epoch"]:
                            legacy_data["val_loss"] = float(match.group(1))
                        else:
                            legacy_data[param] = float(match.group(1))
                        found_legacy = True
                
                if found_legacy and "val_loss" in legacy_data:
                    all_trials.append(legacy_data)
                    
    return pd.DataFrame(all_trials)

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
        if not df.empty:
            print("Extracted data columns:", df.columns.tolist())
        return
    
    # Drop columns that are entirely NaN
    df = df.dropna(axis=1, how='all')
    
    if "val_loss" not in df.columns:
        print("[!] No validation loss found in data.")
        return

    X = df.drop(columns=["val_loss"])
    y = df["val_loss"]
    
    # Drop rows where target is NaN
    valid_idx = y.notna()
    X = X[valid_idx]
    y = y[valid_idx]
    
    # Handle NaNs in features (impute with mean or drop)
    X = X.fillna(X.mean())
    
    # Handle object columns if any
    X = pd.get_dummies(X, drop_first=True)
    
    if X.empty:
        print("[!] No feature data available for importance analysis.")
        return
    
    # 1. Feature Importance via Random Forest
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X, y)
    
    importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
    
    # 2. Correlation Analysis
    correlations = {}
    for col in X.columns:
        if X[col].nunique() > 1: # Spearman is undefined for constant columns
            corr, _ = spearmanr(X[col], y)
            correlations[col] = corr
        else:
            correlations[col] = np.nan
    
    corr_series = pd.Series(correlations).sort_values()

    print("\n" + "="*40)
    print(f"HYPERPARAMETER IMPORTANCE ({len(df)} trials)")
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
    parser.add_argument("--log-file", type=str, help="Path to a specific log file to analyze")
    parser.add_argument("--log-dir", type=str, default="logs", help="Directory containing log files (default: logs)")
    args = parser.parse_args()

    if args.storage and args.study_name:
        logger.info(f"Loading data for study '{args.study_name}' from {args.storage}...")
        trial_df = load_study_data(args.storage, args.study_name)
    else:
        source = args.log_file if args.log_file else args.log_dir
        logger.info(f"Analyzing logs from {source}...")
        trial_df = parse_sweep_logs(log_path=args.log_file, log_dir=args.log_dir)
    
    if not trial_df.empty:
        logger.info(f"Successfully extracted {len(trial_df)} data points.")
        analyze_importance(trial_df)
    else:
        logger.error("No trial data found.")
