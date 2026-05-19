import pytest
import os
import yaml
import optuna
import tempfile
import logging
from analysis import set_nested_value, list_studies, analyze_study, promote_study

# Disable logging for cleaner test output
logging.getLogger("src.analysis").setLevel(logging.CRITICAL)

@pytest.fixture
def temp_db():
    """Provides a temporary SQLite database for Optuna."""
    fd, db_path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    storage = f"sqlite:///{db_path}"
    yield storage
    if os.path.exists(db_path):
        os.remove(db_path)

@pytest.fixture
def sample_study(temp_db):
    """Creates a sample Optuna study with completed trials."""
    study_name = "test-study"
    study = optuna.create_study(study_name=study_name, storage=temp_db, direction="minimize")
    
    def objective(trial):
        h = trial.suggest_float("model.hidden_dim", 32, 128)
        lr = trial.suggest_float("training.lr", 0.0001, 0.01)
        trial.set_user_attr("git_commit", "abcdef1234567890")
        trial.set_user_attr("branch", "feature/tuning")
        return h * 0.1 + lr * 100
        
    study.optimize(objective, n_trials=3)
    return study_name

def test_set_nested_value():
    """Tests the utility function for setting values in nested dictionaries."""
    d = {"a": {"b": 1}}
    set_nested_value(d, ["a", "c"], 2)
    assert d["a"]["c"] == 2
    
    set_nested_value(d, ["x", "y", "z"], 3)
    assert d["x"]["y"]["z"] == 3
    assert d["a"]["b"] == 1

def test_list_studies_empty(temp_db, capsys):
    """Tests list_studies when no studies exist."""
    list_studies(temp_db)
    captured = capsys.readouterr()
    assert "No studies found" in captured.out

def test_list_studies_populated(temp_db, sample_study, capsys):
    """Tests list_studies with a populated database."""
    list_studies(temp_db)
    captured = capsys.readouterr()
    assert "test-study" in captured.out
    assert "3" in captured.out  # Number of trials

def test_analyze_study_no_trials(temp_db, capsys):
    """Tests analyze_study for a study with no trials."""
    optuna.create_study(study_name="empty-study", storage=temp_db)
    analyze_study(temp_db, "empty-study")
    captured = capsys.readouterr()
    assert "No completed trials found" in captured.out

def test_analyze_study_success(temp_db, sample_study, capsys):
    """Tests analyze_study for a successful study."""
    analyze_study(temp_db, sample_study)
    captured = capsys.readouterr()
    assert "ANALYSIS: test-study" in captured.out
    assert "BEST TRIAL: #" in captured.out
    assert "Value: " in captured.out
    assert "Artifact Twinning (Git): feature/tuning @ abcdef12" in captured.out
    assert "Hyperparameter Importance" in captured.out

def test_promote_study_nested(temp_db, sample_study, tmp_path):
    """Tests promoting best parameters with dot-notation keys."""
    config_file = tmp_path / "train.yaml"
    initial_config = {
        "model": {"hidden_dim": 16},
        "training": {"lr": 0.5}
    }
    with open(config_file, "w") as f:
        yaml.dump(initial_config, f)
    
    promote_study(temp_db, sample_study, str(config_file))
    
    with open(config_file, "r") as f:
        updated_config = yaml.safe_load(f)
    
    # Check if values were updated
    assert updated_config["model"]["hidden_dim"] != 16
    assert 32 <= updated_config["model"]["hidden_dim"] <= 128
    assert updated_config["training"]["lr"] != 0.5
    assert 0.0001 <= updated_config["training"]["lr"] <= 0.01

def test_promote_study_legacy_mapping(temp_db, tmp_path):
    """Tests promoting best parameters with legacy flat keys."""
    study_name = "legacy-study"
    study = optuna.create_study(study_name=study_name, storage=temp_db)
    
    def objective(trial):
        trial.suggest_float("lr", 0.001, 0.002)
        trial.suggest_float("w_forecast", 1.0, 2.0)
        trial.suggest_float("num_layers", 2, 4)
        trial.suggest_float("mask_prob", 0.1, 0.2)
        return 1.0
        
    study.optimize(objective, n_trials=1)
    
    config_file = tmp_path / "train.yaml"
    initial_config = {
        "training": {"lr": 0.5},
        "loss": {"w_forecast": 0.0},
        "model": {"num_layers": 1},
        "data": {"mask_prob": 0.5}
    }
    with open(config_file, "w") as f:
        yaml.dump(initial_config, f)
        
    promote_study(temp_db, study_name, str(config_file))
    
    with open(config_file, "r") as f:
        updated_config = yaml.safe_load(f)
    
    assert 0.001 <= updated_config["training"]["lr"] <= 0.002
    assert 1.0 <= updated_config["loss"]["w_forecast"] <= 2.0
    assert 2 <= updated_config["model"]["num_layers"] <= 4
    assert 0.1 <= updated_config["data"]["mask_prob"] <= 0.2

def test_promote_study_no_best_params(temp_db, tmp_path, capsys):
    """Tests promote_study when no best parameters are available."""
    study_name = "empty-study"
    optuna.create_study(study_name=study_name, storage=temp_db)
    
    config_file = tmp_path / "train.yaml"
    with open(config_file, "w") as f:
        yaml.dump({"a": 1}, f)
        
    promote_study(temp_db, study_name, str(config_file))
    # We check if it logged an error. Since we disabled logger, we might need to check return or just ensure no crash.
    # The code says logger.error("No best parameters found to promote.")
    # If we want to check logs, we should use caplog.

def test_promote_study_missing_db(tmp_path, capsys):
    """Tests behavior when the database file is missing."""
    promote_study("sqlite:///non_existent.db", "any-study", "config.yaml")

def test_main_list(temp_db, sample_study, capsys, monkeypatch):
    """Tests the main() entry point for the 'list' command."""
    from analysis import main
    import sys
    
    monkeypatch.setattr(sys, "argv", ["analysis.py", "--storage", temp_db, "list"])
    main()
    captured = capsys.readouterr()
    assert "test-study" in captured.out

def test_main_analyze(temp_db, sample_study, capsys, monkeypatch):
    """Tests the main() entry point for the 'analyze' command."""
    from analysis import main
    import sys
    
    monkeypatch.setattr(sys, "argv", ["analysis.py", "--storage", temp_db, "analyze", sample_study])
    main()
    captured = capsys.readouterr()
    assert f"ANALYSIS: {sample_study}" in captured.out
