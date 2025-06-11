"""
Unified hyperparameter tuning interface for both XGBoost and Random Forest models.

This module provides a single entry point for hyperparameter optimization
that automatically selects the appropriate tuning strategy based on the model type.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional

from src.config.config_manager import ConfigManager
from src.tuning.ray_tune_xgboost import EnhancedXGBoostTrainer
from src.tuning.ray_tune_random_forest import EnhancedRandomForestTrainer

logger = logging.getLogger(__name__)


class UnifiedTuner:
    """
    Unified tuner that handles hyperparameter optimization for different model types.
    
    Automatically selects the appropriate tuning strategy based on the configured model type.
    """
    
    def __init__(self, config_manager: ConfigManager):
        """
        Initialize the unified tuner.
        
        Args:
            config_manager: Configured ConfigManager instance
        """
        self.config_manager = config_manager
        self.config = config_manager.config
        self.model_type = config_manager.get_model_type()
        
        logger.info("Initialized UnifiedTuner for model type: %s", self.model_type)
    
    def tune_hyperparameters(self, 
                           data_file: Optional[str] = None,
                           test_file: Optional[str] = None,
                           num_samples: Optional[int] = None,
                           cpus_per_trial: Optional[int] = None,
                           max_concurrent_trials: Optional[int] = None,
                           output_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Perform hyperparameter tuning for the configured model type.
        
        Args:
            data_file: Path to data file (defaults to config data path)
            test_file: Path to test file (optional)
            num_samples: Number of tuning samples (defaults to config value)
            cpus_per_trial: CPUs per trial (defaults to config value)
            max_concurrent_trials: Max concurrent trials (defaults to config value)
            output_dir: Output directory (defaults to config value)
            
        Returns:
            Best hyperparameter configuration
        """
        # Use config defaults if not provided
        if data_file is None:
            data_file = str(self.config_manager.get_data_path())
        if num_samples is None:
            num_samples = self.config.tuning.num_samples
        if cpus_per_trial is None:
            cpus_per_trial = self.config.tuning.cpus_per_trial
        if max_concurrent_trials is None:
            max_concurrent_trials = self.config.tuning.max_concurrent_trials
        if output_dir is None:
            output_dir = self.config.tuning.output_dir
        
        logger.info("Starting hyperparameter tuning for %s", self.model_type)
        logger.info("Data file: %s", data_file)
        logger.info("Num samples: %s", num_samples)
        logger.info("CPUs per trial: %s", cpus_per_trial)
        logger.info("Output dir: %s", output_dir)
        
        if self.model_type == "xgboost":
            return self._tune_xgboost(
                data_file, test_file, num_samples, cpus_per_trial, 
                max_concurrent_trials, output_dir
            )
        if self.model_type == "random_forest":
            return self._tune_random_forest(
                data_file, num_samples, cpus_per_trial, 
                max_concurrent_trials, output_dir
            )
        else:
            raise ValueError(f"Unsupported model type for tuning: {self.model_type}")
    
    def _tune_xgboost(self, 
                     data_file: str,
                     test_file: Optional[str],
                     num_samples: int,
                     cpus_per_trial: int,
                     max_concurrent_trials: int,
                     output_dir: str) -> Dict[str, Any]:
        """Tune XGBoost hyperparameters."""
        logger.info("Running XGBoost hyperparameter tuning")
        
        trainer = EnhancedXGBoostTrainer(
            data_file=data_file,
            test_data_file=test_file,
            num_samples=num_samples,
            max_concurrent_trials=max_concurrent_trials
        )
        
        # Load and prepare data
        trainer.load_and_prepare_data()
        
        # Run tuning
        trainer.tune_hyperparameters()
        
        # Get best parameters
        xgb_best_params = trainer.best_params
        
        # Save results
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        results_file = output_path / "best_params.json"
        import json
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(xgb_best_params, f, indent=2)
        
        logger.info("XGBoost tuning completed. Best params saved to %s", results_file)
        return xgb_best_params
    
    def _tune_random_forest(self,
                          data_file: str,
                          num_samples: int,
                          cpus_per_trial: int,
                          max_concurrent_trials: int,
                          output_dir: str) -> Dict[str, Any]:
        """Tune Random Forest hyperparameters."""
        logger.info("Running Random Forest hyperparameter tuning")
        
        trainer = EnhancedRandomForestTrainer(
            data_file=data_file,
            output_dir=output_dir
        )
        
        # Run tuning
        rf_best_params = trainer.tune_hyperparameters(
            num_samples=num_samples,
            max_concurrent_trials=max_concurrent_trials,
            cpus_per_trial=cpus_per_trial
        )
        
        # Save results
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        results_file = output_path / "best_params.json"
        import json
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(rf_best_params, f, indent=2)
        
        logger.info("Random Forest tuning completed. Best params saved to %s", results_file)
        return rf_best_params


def run_tuning_from_config(config_name: str = "base",
                          experiment: Optional[str] = None,
                          overrides: Optional[list] = None) -> Dict[str, Any]:
    """
    Run hyperparameter tuning using configuration.
    
    Args:
        config_name: Base configuration name
        experiment: Experiment override
        overrides: Additional configuration overrides
        
    Returns:
        Best hyperparameter configuration
    """
    # Load configuration
    config_manager = ConfigManager()
    config_manager.load_config(config_name, experiment, overrides)
    
    # Only run tuning if enabled
    if not config_manager.is_tuning_enabled():
        logger.warning("Tuning is disabled in configuration. Skipping hyperparameter optimization.")
        return {}
    
    # Create and run tuner
    tuner = UnifiedTuner(config_manager)
    return tuner.tune_hyperparameters()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Unified hyperparameter tuning")
    parser.add_argument("--config", default="base", help="Configuration file name")
    parser.add_argument("--experiment", help="Experiment configuration")
    parser.add_argument("--data-file", help="Override data file path")
    parser.add_argument("--test-file", help="Test file path")
    parser.add_argument("--num-samples", type=int, help="Number of tuning samples")
    parser.add_argument("--output-dir", help="Output directory")
    
    args = parser.parse_args()
    
    # Build overrides from command line arguments
    overrides = []
    if args.data_file:
        overrides.append(f"data.path={Path(args.data_file).parent}")
        overrides.append(f"data.filename={Path(args.data_file).name}")
    if args.num_samples:
        overrides.append(f"tuning.num_samples={args.num_samples}")
    if args.output_dir:
        overrides.append(f"tuning.output_dir={args.output_dir}")
    
    # Enable tuning
    overrides.append("tuning.enabled=true")
    
    # Run tuning
    try:
        tuning_results = run_tuning_from_config(
            config_name=args.config,
            experiment=args.experiment,
            overrides=overrides
        )
        print("Tuning completed successfully!")
        print(f"Best parameters: {tuning_results}")
    except Exception as e:
        logger.error("Tuning failed: %s", e)
        raise 