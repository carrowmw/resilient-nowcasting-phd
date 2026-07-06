#!/usr/bin/env python3
import argparse
import os
import sys
import yaml
import optuna
import pandas as pd
from typing import Optional, List, Dict, Any

def setup_logger():
    import logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    return logging.getLogger(__name__)

logger = setup_logger()

def set_nested_value(d: dict, keys: list, value):
    """Recursively sets a value in a nested dictionary."""
    for key in keys[:-1]:
        d = d.setdefault(key, {})
    d[keys[-1]] = value

def list_studies(storage: str):
    """Lists all studies in the specified storage."""
    try:
        summaries = optuna.get_all_study_summaries(storage)
        if not summaries:
            print(f"No studies found in {storage}")
            return

        print(f"\nStudies in {storage}:")
        print(f"{'Name':<40} {'Trials':<10} {'Best Value':<15}")
        print("-" * 65)
        for s in summaries:
            best_value = f"{s.best_trial.value:.6f}" if s.best_trial is not None else "N/A"
            print(f"{s.study_name:<40} {s.n_trials:<10} {best_value:<15}")
    except (optuna.exceptions.InvalidStateError, RuntimeError, OSError) as e:
        logger.error(f"Failed to list studies: {e}")

def analyze_study(storage: str, study_name: str):
    """Performs detailed analysis of a specific study."""
    try:
        study = optuna.load_study(study_name=study_name, storage=storage)
        
        completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        
        if not completed_trials:
            print(f"No completed trials found in study '{study_name}'")
            return

        print(f"\n" + "="*60)
        print(f"ANALYSIS: {study_name} ({len(completed_trials)} completed trials)")
        print("="*60)

        # 1. Best Trial & Metadata
        best_trial = study.best_trial
        print(f"\nBEST TRIAL: #{best_trial.number}")
        print(f"Value: {best_trial.value:.6f}")
        
        git_commit = best_trial.user_attrs.get("git_commit", "N/A")
        git_branch = best_trial.user_attrs.get("branch", "N/A")
        print(f"Artifact Twinning (Git): {git_branch} @ {git_commit[:8] if git_commit != 'N/A' else 'N/A'}")
        
        print("\nBest Parameters:")
        for k, v in best_trial.params.items():
            print(f"  - {k}: {v}")

        # 2. Hyperparameter Importance
        if len(completed_trials) > 1:
            try:
                importances = optuna.importance.get_param_importances(study)
                print("\nHyperparameter Importance:")
                for param, importance in importances.items():
                    print(f"  - {param:<30}: {importance:.4f}")
            except (ValueError, RuntimeError) as e:
                logger.warning(f"Could not calculate importance: {e}")

        print("="*60 + "\n")

    except (KeyError, ValueError, RuntimeError, OSError) as e:
        logger.error(f"Failed to analyze study: {e}")

def promote_study(storage: str, study_name: str, config_path: str):
    """Updates the config YAML with the best parameters from the study."""
    try:
        study = optuna.load_study(study_name=study_name, storage=storage)
        best_params = study.best_params
        
        if not best_params:
            logger.error("No best parameters found to promote.")
            return

        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        print(f"\nPromoting best parameters to {config_path}...")
        for param_name, value in best_params.items():
            keys = param_name.split('.')
            if len(keys) > 1:
                set_nested_value(config, keys, value)
                print(f"  - Updated {param_name} -> {value}")
            else:
                # Handle legacy/top-level parameters if they exist
                # This logic should match src/tune.py mapping
                if param_name in ["lr", "batch_size"]:
                    set_nested_value(config, ["training", param_name], value)
                elif param_name in ["w_forecast", "w_impute"]:
                    set_nested_value(config, ["loss", param_name], value)
                elif param_name in ["sparsity_threshold", "num_layers", "embedding_dim", "hidden_dim"]:
                    set_nested_value(config, ["model", param_name], value)
                elif param_name == "mask_prob":
                    set_nested_value(config, ["data", param_name], value)
                else:
                    config[param_name] = value
                print(f"  - Updated {param_name} -> {value}")

        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        logger.info(f"Successfully promoted parameters to {config_path}")

    except (KeyError, ValueError, OSError) as e:
        logger.error(f"Failed to promote study: {e}")

def main():
    parser = argparse.ArgumentParser(description="Optuna Study Analysis & Management")
    parser.add_argument("--storage", type=str, default="sqlite:///optuna.db", help="SQLite storage string")
    
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # List command
    subparsers.add_parser("list", help="List all studies in the database")
    
    # Analyze command
    analyze_parser = subparsers.add_parser("analyze", help="Analyze a specific study")
    analyze_parser.add_argument("study_name", type=str, help="Name of the study to analyze")
    
    # Promote command
    promote_parser = subparsers.add_parser("promote", help="Promote best parameters to a config file")
    promote_parser.add_argument("study_name", type=str, help="Name of the study")
    promote_parser.add_argument("--config", type=str, default="configs/train/train.yaml", help="Path to config file")

    args = parser.parse_args()

    if not os.path.exists(args.storage.replace("sqlite:///", "")):
        # Check if DB file exists before trying to load it
        db_file = args.storage.replace("sqlite:///", "")
        if args.command in ["list", "analyze", "promote"]:
            logger.warning(f"Database file '{db_file}' not found.")
            # If listing, we can just stop. If other commands, they will fail anyway.
            if args.command == "list": return

    if args.command == "list":
        list_studies(args.storage)
    elif args.command == "analyze":
        analyze_study(args.storage, args.study_name)
    elif args.command == "promote":
        promote_study(args.storage, args.study_name, args.config)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
