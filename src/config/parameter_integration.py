"""
Parameter integration utilities for ConfigManager.

This module extends the ConfigManager with parameter mapping capabilities,
enabling seamless model type switching while preserving configuration integrity.
"""

import logging
from typing import Dict, Any, Optional, Union, Tuple

from src.config.config_manager import ConfigManager
from src.utils.parameter_mapping import UnifiedParameterManager, ModelType

logger = logging.getLogger(__name__)


class ParameterIntegratedConfigManager(ConfigManager):
    """
    Enhanced ConfigManager with parameter mapping capabilities.
    
    This class extends the base ConfigManager to support seamless parameter
    conversion between different model types while maintaining configuration
    consistency.
    """
    
    def __init__(self):
        """Initialize the parameter-integrated config manager."""
        super().__init__()
        self._param_manager = UnifiedParameterManager()
        self._original_model_type = None
        self._original_params = None
    
    def load_config(self, config_name: str = "base", 
                   experiment: Optional[str] = None,
                   overrides: Optional[list] = None) -> None:
        """
        Load configuration and store original parameters for conversion.
        
        Args:
            config_name: Base configuration name
            experiment: Name of the experiment configuration to load
            overrides: Additional configuration overrides
        """
        super().load_config(config_name, experiment, overrides)
        
        # Store original configuration for reference
        self._original_model_type = self.get_model_type()
        self._original_params = self.get_model_params_dict().copy()
        
        logger.info("Loaded configuration with model type: %s", self._original_model_type)
    
    def switch_model_type(self, new_model_type: Union[str, ModelType],
                         preserve_equivalent_params: bool = True) -> Dict[str, Any]:
        """
        Switch to a different model type with parameter conversion.
        
        Args:
            new_model_type: Target model type to switch to
            preserve_equivalent_params: Whether to preserve equivalent parameters
            
        Returns:
            New model parameters after conversion
        """
        if isinstance(new_model_type, str):
            new_model_type = ModelType(new_model_type.lower())
        
        current_model_type = ModelType(self.get_model_type().lower())
        
        if current_model_type == new_model_type:
            logger.info("Model type already set to %s", new_model_type.value)
            return self.get_model_params_dict()
        
        logger.info("Switching model type from %s to %s", 
                   current_model_type.value, new_model_type.value)
        
        # Get current parameters
        current_params = self.get_model_params_dict()
        
        # Convert parameters using the unified parameter manager
        if preserve_equivalent_params:
            converted_params = self._param_manager.convert_parameters(
                current_params, current_model_type, new_model_type
            )
        else:
            # Use default parameters for the new model type
            converted_params = self._param_manager.get_default_parameters(new_model_type)
        
        # Update the configuration
        self._update_model_config(new_model_type.value, converted_params)
        
        logger.info("Successfully switched to %s with %d parameters", 
                   new_model_type.value, len(converted_params))
        
        return converted_params
    
    def get_cross_compatible_params(self, target_model_type: Union[str, ModelType]) -> Dict[str, Any]:
        """
        Get parameters that are compatible with the target model type.
        
        Args:
            target_model_type: Target model type for compatibility
            
        Returns:
            Parameters compatible with the target model type
        """
        if isinstance(target_model_type, str):
            target_model_type = ModelType(target_model_type.lower())
        
        current_model_type = ModelType(self.get_model_type().lower())
        current_params = self.get_model_params_dict()
        
        if current_model_type == target_model_type:
            return current_params.copy()
        
        # Convert parameters
        compatible_params = self._param_manager.convert_parameters(
            current_params, current_model_type, target_model_type
        )
        
        logger.info("Generated cross-compatible parameters for %s", target_model_type.value)
        return compatible_params
    
    def validate_current_params(self) -> Tuple[bool, str]:
        """
        Validate current model parameters.
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        model_type = self.get_model_type()
        params = self.get_model_params_dict()
        
        return self._param_manager.validate_parameters(params, model_type)
    
    def restore_original_config(self) -> None:
        """Restore the original configuration loaded from file."""
        if self._original_model_type is None or self._original_params is None:
            raise RuntimeError("No original configuration to restore. Load config first.")
        
        logger.info("Restoring original configuration: %s", self._original_model_type)
        self._update_model_config(self._original_model_type, self._original_params)
    
    def create_experiment_config(self, model_type: Union[str, ModelType],
                               base_params: Optional[Dict[str, Any]] = None,
                               experiment_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Create a new experiment configuration for the specified model type.
        
        Args:
            model_type: Model type for the experiment
            base_params: Base parameters to start with (uses defaults if None)
            experiment_name: Name for the experiment
            
        Returns:
            Complete experiment configuration
        """
        if isinstance(model_type, str):
            model_type = ModelType(model_type.lower())
        
        # Get unified configuration
        if base_params:
            config = self._param_manager.create_unified_config(base_params, model_type)
        else:
            config = self._param_manager.get_default_parameters(model_type)
        
        # Update the current configuration
        self._update_model_config(model_type.value, config)
        
        # Set experiment name if provided
        if experiment_name:
            self.update_config_value("outputs.experiment_name", experiment_name)
        
        logger.info("Created experiment config for %s with %d parameters", 
                   model_type.value, len(config))
        
        return config
    
    def get_tuning_compatible_params(self) -> Dict[str, Any]:
        """
        Get parameters suitable for hyperparameter tuning.
        
        Returns:
            Parameters with tuning-friendly values and ranges
        """
        model_type = self.get_model_type()
        params = self.get_model_params_dict()
        
        if model_type == "xgboost":
            return self._get_xgboost_tuning_params(params)
        if model_type == "random_forest":
            return self._get_random_forest_tuning_params(params)
        raise ValueError(f"Tuning not supported for model type: {model_type}")
    
    def apply_tuned_params(self, tuned_params: Dict[str, Any]) -> None:
        """
        Apply tuned parameters to the current configuration.
        
        Args:
            tuned_params: Dictionary of tuned hyperparameters
        """
        model_type = self.get_model_type()
        current_params = self.get_model_params_dict()
        
        # Merge tuned parameters with current parameters
        updated_params = current_params.copy()
        updated_params.update(tuned_params)
        
        # Validate the updated parameters
        is_valid, error_msg = self._param_manager.validate_parameters(updated_params, model_type)
        if not is_valid:
            logger.warning("Tuned parameters validation failed: %s", error_msg)
        
        # Update the configuration
        self._update_model_config(model_type, updated_params)
        
        logger.info("Applied %d tuned parameters to %s model", 
                   len(tuned_params), model_type)
    
    def export_for_framework(self, target_framework: str = "auto") -> Dict[str, Any]:
        """
        Export configuration in format suitable for the target framework.
        
        Args:
            target_framework: Target framework ('xgboost', 'sklearn', 'auto')
            
        Returns:
            Framework-specific configuration
        """
        model_type = self.get_model_type()
        params = self.get_model_params_dict()
        
        if target_framework == "auto":
            target_framework = "xgboost" if model_type == "xgboost" else "sklearn"
        
        # Create framework-specific export
        export_config = {
            "model_type": model_type,
            "framework": target_framework,
            "parameters": params.copy(),
            "federated_config": {
                "num_rounds": self.config.federated.num_rounds,
                "num_clients_per_round": self.config.federated.num_clients_per_round,
                "train_method": self.config.federated.train_method
            },
            "data_config": {
                "path": str(self.get_data_path()),
                "test_fraction": self.config.federated.test_fraction
            }
        }
        
        logger.info("Exported configuration for %s framework", target_framework)
        return export_config
    
    def _update_model_config(self, model_type: str, params: Dict[str, Any]) -> None:
        """Update the model configuration with new type and parameters."""
        # Update model type
        self.update_config_value("model.type", model_type)
        
        # Update individual parameters
        for key, value in params.items():
            param_path = f"model.params.{key}"
            self.update_config_value(param_path, value)
    
    def _get_xgboost_tuning_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Get XGBoost parameters suitable for tuning."""
        tuning_params = {}
        
        # Parameters commonly tuned for XGBoost
        tunable_params = [
            'eta', 'max_depth', 'min_child_weight', 'gamma', 'subsample', 
            'colsample_bytree', 'reg_alpha', 'reg_lambda', 'num_boost_round'
        ]
        
        for param in tunable_params:
            if param in params:
                tuning_params[param] = params[param]
        
        return tuning_params
    
    def _get_random_forest_tuning_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Get Random Forest parameters suitable for tuning."""
        tuning_params = {}
        
        # Parameters commonly tuned for Random Forest
        tunable_params = [
            'n_estimators', 'max_depth', 'min_samples_split', 'min_samples_leaf',
            'max_features', 'min_impurity_decrease'
        ]
        
        for param in tunable_params:
            if param in params:
                tuning_params[param] = params[param]
        
        return tuning_params


class ConfigurationPreset:
    """
    Predefined configuration presets for common use cases.
    """
    
    @staticmethod
    def get_quick_test_config(model_type: Union[str, ModelType]) -> Dict[str, Any]:
        """Get configuration optimized for quick testing."""
        if isinstance(model_type, str):
            model_type = ModelType(model_type.lower())
        
        if model_type == ModelType.XGBOOST:
            return {
                "eta": 0.3,
                "max_depth": 6,
                "min_child_weight": 1,
                "num_boost_round": 10,
                "objective": "multi:softprob",
                "num_class": 11,
                "eval_metric": "mlogloss"
            }
        elif model_type == ModelType.RANDOM_FOREST:
            return {
                "n_estimators": 10,
                "max_depth": 5,
                "min_samples_split": 2,
                "min_samples_leaf": 1,
                "random_state": 42
            }
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
    
    @staticmethod
    def get_production_config(model_type: Union[str, ModelType]) -> Dict[str, Any]:
        """Get configuration optimized for production use."""
        if isinstance(model_type, str):
            model_type = ModelType(model_type.lower())
        
        if model_type == ModelType.XGBOOST:
            return {
                "eta": 0.05,
                "max_depth": 8,
                "min_child_weight": 5,
                "gamma": 0.5,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "num_boost_round": 500,
                "objective": "multi:softprob",
                "num_class": 11,
                "eval_metric": "mlogloss",
                "reg_alpha": 0.1,
                "reg_lambda": 1.0
            }
        elif model_type == ModelType.RANDOM_FOREST:
            return {
                "n_estimators": 200,
                "max_depth": 15,
                "min_samples_split": 5,
                "min_samples_leaf": 2,
                "max_features": "sqrt",
                "random_state": 42,
                "class_weight": "balanced"
            }
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
    
    @staticmethod
    def get_tuning_config(model_type: Union[str, ModelType]) -> Dict[str, Any]:
        """Get configuration optimized for hyperparameter tuning."""
        if isinstance(model_type, str):
            model_type = ModelType(model_type.lower())
        
        if model_type == ModelType.XGBOOST:
            return {
                "eta": 0.1,
                "max_depth": 6,
                "min_child_weight": 3,
                "gamma": 0.1,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "num_boost_round": 100,
                "objective": "multi:softprob",
                "num_class": 11,
                "eval_metric": "mlogloss"
            }
        elif model_type == ModelType.RANDOM_FOREST:
            return {
                "n_estimators": 50,
                "max_depth": 10,
                "min_samples_split": 2,
                "min_samples_leaf": 1,
                "max_features": "sqrt",
                "random_state": 42
            }
        else:
            raise ValueError(f"Unsupported model type: {model_type}")


# Convenience functions

def create_config_manager_with_mapping() -> ParameterIntegratedConfigManager:
    """Create a new parameter-integrated config manager."""
    return ParameterIntegratedConfigManager()


def switch_config_model_type(config_manager: ConfigManager, 
                           new_model_type: Union[str, ModelType]) -> Dict[str, Any]:
    """
    Switch model type for an existing config manager.
    
    Args:
        config_manager: Existing ConfigManager instance
        new_model_type: Target model type
        
    Returns:
        New model parameters after conversion
    """
    if not isinstance(config_manager, ParameterIntegratedConfigManager):
        raise TypeError("ConfigManager must be ParameterIntegratedConfigManager for model switching")
    
    return config_manager.switch_model_type(new_model_type)


def validate_config_compatibility(config_manager: ConfigManager, 
                                target_model_type: Union[str, ModelType]) -> Tuple[bool, str]:
    """
    Validate if current configuration is compatible with target model type.
    
    Args:
        config_manager: ConfigManager instance
        target_model_type: Target model type to check compatibility
        
    Returns:
        Tuple of (is_compatible, message)
    """
    try:
        current_type = config_manager.get_model_type()
        current_params = config_manager.get_model_params_dict()
        
        if isinstance(target_model_type, str):
            target_model_type = ModelType(target_model_type.lower())
        
        if ModelType(current_type.lower()) == target_model_type:
            return True, "Configuration is already for target model type"
        
        # Try to convert parameters to check compatibility
        param_manager = UnifiedParameterManager()
        converted_params = param_manager.convert_parameters(
            current_params, current_type, target_model_type
        )
        
        # Validate converted parameters
        is_valid, error_msg = param_manager.validate_parameters(converted_params, target_model_type)
        
                if is_valid:
            return True, "Configuration is compatible and can be converted"
        
        return False, f"Conversion validation failed: {error_msg}"
            
    except (ValueError, TypeError, AttributeError) as e:
        return False, f"Compatibility check failed: {str(e)}" 