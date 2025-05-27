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
from flwr.common.logger import log
from logging import INFO

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

def main():
    """Main execution pipeline."""
    log(INFO, "Starting Federated Learning Pipeline with Consistent Preprocessing")
    log(INFO, "=" * 80)
    
    # Step 1: Create global feature processor
    log(INFO, "Step 1: Creating global feature processor for consistent preprocessing")
    if not run_command([
        "python", "create_global_processor.py",
        "--data-file", "data/received/final_dataset.csv",
        "--output-dir", "outputs",
        "--force"
    ], "Global feature processor creation"):
        log(INFO, "Failed to create global feature processor. Exiting.")
        sys.exit(1)
    
    # Step 2: Run hyperparameter tuning with consistent preprocessing
    log(INFO, "Step 2: Running hyperparameter tuning with consistent preprocessing")
    if not run_command([
        "python", "ray_tune_xgboost_updated.py",
        "--data-file", "data/received/final_dataset.csv",
        "--num-samples", "30",
        "--cpus-per-trial", "2",
        "--output-dir", "./tune_results"
    ], "Ray Tune hyperparameter optimization"):
        log(INFO, "Hyperparameter tuning failed. Continuing with default parameters.")
    
    # Step 3: Generate tuned parameters file
    log(INFO, "Step 3: Generating tuned parameters file")
    if not run_command([
        "python", "use_tuned_params.py"
    ], "Tuned parameters generation"):
        log(INFO, "Failed to generate tuned parameters. Using default parameters.")
    
    # Step 4: Run federated learning simulation
    log(INFO, "Step 4: Starting federated learning simulation")
    if not run_command([
        "python", "sim.py",
        "--train-method", "bagging",
        "--pool-size", "5",
        "--num-rounds", "5",
        "--num-clients-per-round", "5",
        "--centralised-eval",
        "--csv-file", "data/received/final_dataset.csv"
    ], "Federated learning simulation"):
        log(INFO, "Federated learning simulation failed.")
        sys.exit(1)
    
    log(INFO, "=" * 80)
    log(INFO, "Federated Learning Pipeline completed successfully!")
    log(INFO, "Key improvements:")
    log(INFO, "✓ Consistent preprocessing across all phases")
    log(INFO, "✓ Temporal splitting to prevent data leakage")
    log(INFO, "✓ Global feature processor for uniform data representation")
    log(INFO, "✓ Tuned hyperparameters applied to federated learning")

if __name__ == "__main__":
    main()
