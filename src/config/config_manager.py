"""
Configuration Management System for FL-CML-Pipeline

This module provides a centralized, type-safe configuration management system
using Hydra for loading YAML configurations with experiment overrides.
"""

from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
import logging
from omegaconf import DictConfig, OmegaConf
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra


@dataclass
class DataConfig:  # pylint: disable=too-many-instance-attributes
    """Data-related configuration."""
    path: str
    filename: str
    train_test_split: float
    stratified: bool
    temporal_window_size: int
    seed: int


@dataclass
class ModelParamsConfig:  # pylint: disable=too-many-instance-attributes
    """XGBoost model parameters configuration."""
    objective: str
    num_class: int
    eta: float
    max_depth: int
    min_child_weight: int
    gamma: float
    subsample: float
    colsample_bytree: float
    colsample_bylevel: float
    nthread: int
    tree_method: str
    eval_metric: List[str]
    max_delta_step: int
    reg_alpha: float
    reg_lambda: float
    base_score: float
    scale_pos_weight: float
    grow_policy: str
    normalize_type: str
    random_state: int
    # Optional parameters for specific experiments
    num_boost_round: Optional[int] = None


@dataclass
class ModelConfig:
    """Model configuration."""
    type: str
    num_local_rounds: int
    params: ModelParamsConfig


@dataclass
class FederatedConfig:  # pylint: disable=too-many-instance-attributes
    """Federated learning configuration."""
    train_method: str
    pool_size: int
    num_rounds: int
    num_clients_per_round: int
    num_evaluate_clients: int
    centralised_eval: bool
    num_partitions: int
    partitioner_type: str
    test_fraction: float
    scaled_lr: bool
    num_cpus_per_client: int
    # Optional fields for experiment overrides
    fraction_fit: Optional[float] = None
    fraction_evaluate: Optional[float] = None


@dataclass
class SchedulerConfig:
    """Ray Tune scheduler configuration."""
    type: str
    max_t: int
    grace_period: int
    reduction_factor: int


@dataclass
class TuningConfig:  # pylint: disable=too-many-instance-attributes
    """Hyperparameter tuning configuration."""
    enabled: bool
    num_samples: int
    cpus_per_trial: int
    max_concurrent_trials: int
    output_dir: str
    scheduler: SchedulerConfig


@dataclass
class GlobalProcessorConfig:
    """Global processor configuration."""
    force_recreate: bool
    output_dir: str


@dataclass
class PreprocessingConfig:
    """Preprocessing configuration."""
    consistent_across_phases: bool
    global_processor_path: Optional[str] = None


@dataclass
class PipelineConfig:
    """Pipeline execution configuration."""
    steps: List[str]
    global_processor: GlobalProcessorConfig
    preprocessing: PreprocessingConfig


@dataclass
class LoggingConfig:
    """Logging configuration."""
    level: str
    format: str
    file: str


@dataclass
class OutputsConfig:  # pylint: disable=too-many-instance-attributes
    """Output configuration."""
    base_dir: str
    create_timestamped_dirs: bool
    save_results_pickle: bool
    save_model: bool
    generate_visualizations: bool
    experiment_name: Optional[str] = None


@dataclass
class EarlyStoppingConfig:
    """Early stopping configuration."""
    enabled: bool
    patience: int
    min_delta: float


@dataclass
class FlConfig:  # pylint: disable=too-many-instance-attributes
    """Main configuration container."""
    data: DataConfig
    model: ModelConfig
    federated: FederatedConfig
    tuning: TuningConfig
    pipeline: PipelineConfig
    logging: LoggingConfig
    outputs: OutputsConfig
    early_stopping: EarlyStoppingConfig


class ConfigManager:
    """
    Centralized configuration manager using Hydra.
    
    Provides type-safe access to configuration with experiment overrides.
    """
    
    def __init__(self, config_dir: Optional[str] = None):
        """
        Initialize ConfigManager.
        
        Args:
            config_dir: Directory containing configuration files.
                       Defaults to 'configs' in project root.
        """
        self._config: Optional[FlConfig] = None
        self._raw_config: Optional[DictConfig] = None
        self._config_dir = config_dir
        self._logger = logging.getLogger(__name__)
        
    def load_config(self, config_name: str = "base", 
                   experiment: Optional[str] = None,
                   overrides: Optional[List[str]] = None) -> FlConfig:
        """
        Load configuration using Hydra.
        
        Args:
            config_name: Base configuration name (default: "base")
            experiment: Experiment name for overrides (e.g., "bagging", "cyclic")
            overrides: Additional configuration overrides
            
        Returns:
            Loaded and validated FlConfig instance
            
        Example:
            # Load base config
            config = manager.load_config()
            
            # Load bagging experiment config
            config = manager.load_config(experiment="bagging")
            
            # Load with custom overrides
            config = manager.load_config(
                experiment="dev",
                overrides=["tuning.enabled=true", "federated.num_rounds=10"]
            )
        """
        try:
            # Clear any existing Hydra instance
            if GlobalHydra().is_initialized():
                GlobalHydra.instance().clear()
            
            # Determine config directory
            if self._config_dir is None:
                # Default to configs directory in project root
                project_root = Path(__file__).parent.parent.parent
                self._config_dir = str(project_root / "configs")
            
            # Initialize Hydra with config directory
            with initialize_config_dir(config_dir=self._config_dir, version_base=None):
                # Build config overrides
                config_overrides = []
                if experiment:
                    # Use +experiment= syntax to append experiment to defaults
                    config_overrides.append(f"+experiment={experiment}")
                if overrides:
                    config_overrides.extend(overrides)
                
                # Load configuration
                self._raw_config = compose(config_name=config_name, overrides=config_overrides)
                
                # Convert to structured config
                self._config = self._convert_to_structured_config(self._raw_config)
                
                self._logger.info("Configuration loaded successfully: %s", config_name)
                if experiment:
                    self._logger.info("Experiment override applied: %s", experiment)
                if overrides:
                    self._logger.info("Custom overrides applied: %s", overrides)
                
                return self._config
                
        except Exception as e:
            self._logger.error("Failed to load configuration: %s", e)
            raise
    
    def _convert_to_structured_config(self, cfg: DictConfig) -> FlConfig:
        """Convert DictConfig to structured dataclass configuration."""
        try:
            # Convert nested configurations
            data_config = DataConfig(**cfg.data)
            
            model_params = ModelParamsConfig(**cfg.model.params)
            model_config = ModelConfig(
                type=cfg.model.type,
                num_local_rounds=cfg.model.num_local_rounds,
                params=model_params
            )
            
            federated_config = FederatedConfig(**cfg.federated)
            
            scheduler_config = SchedulerConfig(**cfg.tuning.scheduler)
            tuning_config = TuningConfig(
                enabled=cfg.tuning.enabled,
                num_samples=cfg.tuning.num_samples,
                cpus_per_trial=cfg.tuning.cpus_per_trial,
                max_concurrent_trials=cfg.tuning.max_concurrent_trials,
                output_dir=cfg.tuning.output_dir,
                scheduler=scheduler_config
            )
            
            global_processor_config = GlobalProcessorConfig(**cfg.pipeline.global_processor)
            preprocessing_config = PreprocessingConfig(**cfg.pipeline.preprocessing)
            pipeline_config = PipelineConfig(
                steps=cfg.pipeline.steps,
                global_processor=global_processor_config,
                preprocessing=preprocessing_config
            )
            
            logging_config = LoggingConfig(**cfg.logging)
            outputs_config = OutputsConfig(**cfg.outputs)
            early_stopping_config = EarlyStoppingConfig(**cfg.early_stopping)
            
            return FlConfig(
                data=data_config,
                model=model_config,
                federated=federated_config,
                tuning=tuning_config,
                pipeline=pipeline_config,
                logging=logging_config,
                outputs=outputs_config,
                early_stopping=early_stopping_config
            )
            
        except Exception as e:
            self._logger.error("Failed to convert configuration to structured format: %s", e)
            raise
    
    @property
    def config(self) -> FlConfig:
        """Get the current configuration."""
        if self._config is None:
            raise RuntimeError("Configuration not loaded. Call load_config() first.")
        return self._config
    
    @property
    def raw_config(self) -> DictConfig:
        """Get the raw Hydra DictConfig."""
        if self._raw_config is None:
            raise RuntimeError("Configuration not loaded. Call load_config() first.")
        return self._raw_config
    
    def get_model_params_dict(self) -> Dict[str, Any]:
        """Get XGBoost model parameters as dictionary."""
        model_params = self.config.model.params
        params_dict = {
            'objective': model_params.objective,
            'num_class': model_params.num_class,
            'eta': model_params.eta,
            'max_depth': model_params.max_depth,
            'min_child_weight': model_params.min_child_weight,
            'gamma': model_params.gamma,
            'subsample': model_params.subsample,
            'colsample_bytree': model_params.colsample_bytree,
            'colsample_bylevel': model_params.colsample_bylevel,
            'nthread': model_params.nthread,
            'tree_method': model_params.tree_method,
            'eval_metric': 'mlogloss',
            'max_delta_step': model_params.max_delta_step,
            'reg_alpha': model_params.reg_alpha,
            'reg_lambda': model_params.reg_lambda,
            'base_score': model_params.base_score,
            'scale_pos_weight': model_params.scale_pos_weight,
            'grow_policy': model_params.grow_policy,
            'normalize_type': model_params.normalize_type,
            'random_state': model_params.random_state
        }
        
        # Add optional parameters if present
        if model_params.num_boost_round is not None:
            params_dict['num_boost_round'] = model_params.num_boost_round
            
        return params_dict
    
    def get_data_path(self) -> Path:
        """Get the full data file path."""
        return Path(self.config.data.path) / self.config.data.filename
    
    def is_tuning_enabled(self) -> bool:
        """Check if hyperparameter tuning is enabled."""
        return self.config.tuning.enabled
    
    def get_experiment_name(self) -> str:
        """Get experiment name for output organization."""
        if hasattr(self.config.outputs, 'experiment_name') and self.config.outputs.experiment_name:
            return self.config.outputs.experiment_name
        return f"{self.config.federated.train_method}_experiment"
    
    def should_create_timestamped_dirs(self) -> bool:
        """Check if timestamped output directories should be created."""
        return self.config.outputs.create_timestamped_dirs
    
    def update_config_value(self, key_path: str, value: Any) -> None:
        """
        Update a configuration value dynamically.
        
        Args:
            key_path: Dot-notation path to the config value (e.g., "tuning.enabled")
            value: New value to set
        """
        if self._raw_config is None:
            raise RuntimeError("Configuration not loaded. Call load_config() first.")
        
        OmegaConf.set(self._raw_config, key_path, value)
        # Reload structured config
        self._config = self._convert_to_structured_config(self._raw_config)
        self._logger.info("Updated configuration: %s = %s", key_path, value)
    
    def print_config(self) -> None:
        """Print current configuration in a readable format."""
        if self._raw_config is None:
            raise RuntimeError("Configuration not loaded. Call load_config() first.")
        
        print("\n" + "="*60)
        print("FL-CML-Pipeline Configuration")
        print("="*60)
        print(OmegaConf.to_yaml(self._raw_config))
        print("="*60 + "\n")
    
    def save_config(self, output_path: Union[str, Path]) -> None:
        """Save current configuration to a YAML file."""
        if self._raw_config is None:
            raise RuntimeError("Configuration not loaded. Call load_config() first.")
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            OmegaConf.save(self._raw_config, f)
        
        self._logger.info("Configuration saved to: %s", output_path)


# Global configuration manager instance
_config_manager = None  # pylint: disable=global-statement

def get_config_manager() -> ConfigManager:
    """Get the global ConfigManager instance."""
    global _config_manager  # pylint: disable=global-statement
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager

def load_config(config_name: str = "base", 
               experiment: Optional[str] = None,
               overrides: Optional[List[str]] = None) -> FlConfig:
    """Convenience function to load configuration."""
    manager = get_config_manager()
    return manager.load_config(config_name, experiment, overrides) 