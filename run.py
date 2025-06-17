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
import os
import hydra
from omegaconf import DictConfig
from src.config.config_manager import get_config_manager
from src.utils.enhanced_logging import setup_enhanced_logging

def run_command(command, description, enhanced_logger):
    """Run a command and handle errors with enhanced logging."""
    # Set up environment with PYTHONPATH to include project root
    env = os.environ.copy()
    project_root = os.path.dirname(os.path.abspath(__file__))
    if 'PYTHONPATH' in env:
        env['PYTHONPATH'] = f"{project_root}:{env['PYTHONPATH']}"
    else:
        env['PYTHONPATH'] = project_root
    
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True, env=env)
        enhanced_logger.step_success(description, output=result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        error_msg = f"Command failed with exit code {e.returncode}"
        if e.stderr:
            error_msg += f"\nStderr: {e.stderr.strip()}"
        enhanced_logger.step_error(description, error_msg, e.returncode)
        return False

@hydra.main(version_base=None, config_path="configs", config_name="base")
def main(cfg: DictConfig) -> None:
    """Main execution pipeline."""
    # Setup enhanced logging
    enhanced_logger = setup_enhanced_logging()
    
    # Convert DictConfig to structured config using ConfigManager
    config_manager = get_config_manager()
    # Create structured config from the Hydra DictConfig
    config = config_manager._convert_to_structured_config(cfg)  # pylint: disable=protected-access
    
    # Start pipeline with enhanced logging
    enhanced_logger.pipeline_start(config)
    
    # Construct full data path
    data_file_path = config.data.path + "/" + config.data.filename
    
    # Step 1: Create global feature processor
    enhanced_logger.step_start(
        "global_processor", 
        "Creating global feature processor for consistent preprocessing",
        f"python src/core/create_global_processor.py --data-file {data_file_path} --output-dir {config.outputs.base_dir} --force"
    )
    
    if not run_command([
        "python", "src/core/create_global_processor.py",
        "--data-file", data_file_path,
        "--output-dir", config.outputs.base_dir,
        "--force"
    ], "global_processor", enhanced_logger):
        enhanced_logger.step_error("global_processor", "Failed to create global feature processor")
        sys.exit(1)
    
    # Step 2: Run hyperparameter tuning with consistent preprocessing (if enabled)
    if config.tuning.enabled:
        enhanced_logger.step_start(
            "hyperparameter_tuning",
            "Running hyperparameter tuning with consistent preprocessing",
            f"python src/tuning/ray_tune_xgboost.py --data-file {data_file_path} --num-samples {config.tuning.num_samples} --cpus-per-trial {config.tuning.cpus_per_trial} --output-dir {config.tuning.output_dir}"
        )
        
        tuning_command = [
            "python", "src/tuning/ray_tune_xgboost.py",
            "--data-file", data_file_path,
            "--num-samples", str(config.tuning.num_samples),
            "--cpus-per-trial", str(config.tuning.cpus_per_trial),
            "--output-dir", config.tuning.output_dir
        ]
        
        if not run_command(tuning_command, "hyperparameter_tuning", enhanced_logger):
            enhanced_logger.step_error("hyperparameter_tuning", "Hyperparameter tuning failed. Continuing with default parameters.")
        
        # Step 3: Generate tuned parameters file
        enhanced_logger.step_start(
            "generate_tuned_params",
            "Generating tuned parameters file",
            "python src/models/use_tuned_params.py"
        )
        
        if not run_command([
            "python", "src/models/use_tuned_params.py"
        ], "generate_tuned_params", enhanced_logger):
            enhanced_logger.step_error("generate_tuned_params", "Failed to generate tuned parameters. Using default parameters.")
    else:
        enhanced_logger.logger.info("🔧 Hyperparameter tuning disabled. Using default parameters.")
    
    # Step 4: Run federated learning simulation
    enhanced_logger.step_start(
        "federated_learning",
        "Starting federated learning simulation",
        "python src/federated/sim.py"
    )
    
    sim_command = [
        "python", "src/federated/sim.py"
    ]
    
    # Pass configuration through environment or config file
    # The sim.py will load configuration using ConfigManager
    if not run_command(sim_command, "federated_learning", enhanced_logger):
        enhanced_logger.step_error("federated_learning", "Federated learning simulation failed.")
        sys.exit(1)
    
    # Pipeline completion
    enhanced_logger.pipeline_complete(config.outputs.base_dir)

if __name__ == "__main__":
    main()
