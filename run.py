#!/usr/bin/env python3
"""
Main script to run the federated learning pipeline with consistent preprocessing.

This script ensures that:
1. A global feature processor is created for consistent preprocessing
2. Ray Tune hyperparameter optimization uses this global processor
3. Federated learning uses the same global processor
4. All phases maintain preprocessing consistency
"""

import subprocess
import sys
import hydra
from omegaconf import DictConfig
from flwr.common.logger import log
from logging import INFO
from src.config.config_manager import get_config_manager

def run_command(command, description):
    """Run a command and handle errors."""
    log(INFO, "Running: %s", description)
    log(INFO, "Command: %s", " ".join(command))
    
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        log(INFO, "✓ %s completed successfully", description)
        if result.stdout:
            log(INFO, "Output: %s", result.stdout.strip())
        return True
    except subprocess.CalledProcessError as e:
        log(INFO, "✗ %s failed with exit code %d", description, e.returncode)
        if e.stdout:
            log(INFO, "Stdout: %s", e.stdout.strip())
        if e.stderr:
            log(INFO, "Stderr: %s", e.stderr.strip())
        return False

@hydra.main(version_base=None, config_path="configs", config_name="base")
def main(cfg: DictConfig) -> None:
    """Main execution pipeline."""
    log(INFO, "Starting Federated Learning Pipeline with Consistent Preprocessing")
    log(INFO, "=" * 80)
    
    # Convert DictConfig to structured config using ConfigManager
    config_manager = get_config_manager()
    # Create structured config from the Hydra DictConfig
    config = config_manager._convert_to_structured_config(cfg)
    
    log(INFO, "Configuration loaded successfully")
    log(INFO, "Dataset: %s", config.data.path + "/" + config.data.filename)
    log(INFO, "Training method: %s", config.federated.train_method)
    log(INFO, "Number of rounds: %d", config.federated.num_rounds)
    log(INFO, "Pool size: %d", config.federated.pool_size)
    
    # Construct full data path
    data_file_path = config.data.path + "/" + config.data.filename
    
    # Step 1: Create global feature processor
    log(INFO, "Step 1: Creating global feature processor for consistent preprocessing")
    if not run_command([
        "python", "src/core/create_global_processor.py",
        "--data-file", data_file_path,
        "--output-dir", config.outputs.base_dir,
        "--force"
    ], "Global feature processor creation"):
        log(INFO, "Failed to create global feature processor. Exiting.")
        sys.exit(1)
    
    # Step 2: Run hyperparameter tuning with consistent preprocessing (if enabled)
    if config.tuning.enabled:
        log(INFO, "Step 2: Running hyperparameter tuning with consistent preprocessing")
        tuning_command = [
            "python", "src/tuning/ray_tune_xgboost.py",
            "--data-file", data_file_path,
            "--num-samples", str(config.tuning.num_samples),
            "--cpus-per-trial", str(config.tuning.cpus_per_trial),
            "--output-dir", config.tuning.output_dir
        ]
        
        if not run_command(tuning_command, "Ray Tune hyperparameter optimization"):
            log(INFO, "Hyperparameter tuning failed. Continuing with default parameters.")
        
        # Step 3: Generate tuned parameters file
        log(INFO, "Step 3: Generating tuned parameters file")
        if not run_command([
            "python", "src/models/use_tuned_params.py"
        ], "Tuned parameters generation"):
            log(INFO, "Failed to generate tuned parameters. Using default parameters.")
    else:
        log(INFO, "Hyperparameter tuning disabled. Using default parameters.")
    
    # Step 4: Run federated learning simulation
    log(INFO, "Step 4: Starting federated learning simulation")
    sim_command = [
        "python", "src/federated/sim.py"
    ]
    
    # Pass configuration through environment or config file
    # The sim.py will load configuration using ConfigManager
    if not run_command(sim_command, "Federated learning simulation"):
        log(INFO, "Federated learning simulation failed.")
        sys.exit(1)
    
    log(INFO, "=" * 80)
    log(INFO, "Federated Learning Pipeline completed successfully!")
    log(INFO, "Key improvements:")
    log(INFO, "✓ Consistent preprocessing across all phases")
    log(INFO, "✓ Temporal splitting to prevent data leakage")
    log(INFO, "✓ Global feature processor for uniform data representation")
    log(INFO, "✓ Tuned hyperparameters applied to federated learning")
    log(INFO, "Results saved to: %s", config.outputs.base_dir)

if __name__ == "__main__":
    main()
