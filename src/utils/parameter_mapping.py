"""
Parameter mapping utilities for seamless model type switching.

This module provides utilities to convert parameters between different model types
(XGBoost, Random Forest) and offers unified parameter interfaces for the federated
learning pipeline.
"""

import logging
from typing import Dict, Any, Optional, Union, Tuple
from abc import ABC, abstractmethod
from enum import Enum

logger = logging.getLogger(__name__)


class ModelType(Enum):
    """Supported model types."""
    XGBOOST = "xgboost"
    RANDOM_FOREST = "random_forest"


class ParameterCategory(Enum):
    """Categories of parameters for mapping."""
    TREE_STRUCTURE = "tree_structure"
    REGULARIZATION = "regularization"
    LEARNING = "learning"
    SAMPLING = "sampling"
    PERFORMANCE = "performance"
    RANDOM_STATE = "random_state"


class BaseParameterMapper(ABC):
    """Abstract base class for parameter mappers."""
    
    @abstractmethod
    def map_to_target(self, source_params: Dict[str, Any], 
                     target_type: ModelType) -> Dict[str, Any]:
        """Map parameters from source to target model type."""
        pass
    
    @abstractmethod
    def get_default_params(self) -> Dict[str, Any]:
        """Get default parameters for this model type."""
        pass
    
    @abstractmethod
    def validate_params(self, params: Dict[str, Any]) -> Tuple[bool, str]:
        """Validate parameters for this model type."""
        pass


class XGBoostParameterMapper(BaseParameterMapper):
    """Parameter mapper for XGBoost models."""
    
    # Default XGBoost parameters
    DEFAULT_PARAMS = {
        "objective": "multi:softprob",
        "num_class": 11,
        "eta": 0.05,
        "max_depth": 8,
        "min_child_weight": 5,
        "gamma": 0.5,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "colsample_bylevel": 0.8,
        "nthread": 16,
        "tree_method": "hist",
        "eval_metric": "mlogloss",
        "max_delta_step": 1,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "base_score": 0.5,
        "scale_pos_weight": 1.0,
        "grow_policy": "depthwise",
        "normalize_type": "tree",
        "random_state": 42,
        "num_boost_round": 100
    }
    
    def get_default_params(self) -> Dict[str, Any]:
        """Get default XGBoost parameters."""
        return self.DEFAULT_PARAMS.copy()
    
    def map_to_target(self, source_params: Dict[str, Any], 
                     target_type: ModelType) -> Dict[str, Any]:
        """Map XGBoost parameters to target model type."""
        if target_type == ModelType.XGBOOST:
            return source_params.copy()
        elif target_type == ModelType.RANDOM_FOREST:
            return self._map_to_random_forest(source_params)
        else:
            raise ValueError(f"Unsupported target model type: {target_type}")
    
    def _map_to_random_forest(self, xgb_params: Dict[str, Any]) -> Dict[str, Any]:
        """Map XGBoost parameters to Random Forest parameters."""
        rf_params = {}
        
        # Tree structure parameters
        if "max_depth" in xgb_params:
            rf_params["max_depth"] = xgb_params["max_depth"]
        
        if "min_child_weight" in xgb_params:
            # Map min_child_weight to min_samples_leaf (rough approximation)
            rf_params["min_samples_leaf"] = max(1, int(xgb_params["min_child_weight"]))
        
        # Sampling parameters
        if "subsample" in xgb_params:
            # Use subsample to determine max_samples for Random Forest
            if xgb_params["subsample"] < 1.0:
                rf_params["max_samples"] = xgb_params["subsample"]
        
        if "colsample_bytree" in xgb_params:
            # Map colsample_bytree to max_features
            col_sample = xgb_params["colsample_bytree"]
            if col_sample <= 0.5:
                rf_params["max_features"] = "sqrt"
            elif col_sample <= 0.8:
                rf_params["max_features"] = col_sample
            else:
                rf_params["max_features"] = "auto"
        
        # Number of estimators (from num_boost_round)
        if "num_boost_round" in xgb_params:
            # Scale down from XGBoost rounds to RF estimators
            rf_params["n_estimators"] = max(10, xgb_params["num_boost_round"] // 2)
        
        # Random state
        if "random_state" in xgb_params:
            rf_params["random_state"] = xgb_params["random_state"]
        
        # Performance parameters
        if "nthread" in xgb_params:
            rf_params["n_jobs"] = xgb_params["nthread"]
        
        # Set reasonable defaults for unmapped parameters
        rf_params.setdefault("criterion", "gini")
        rf_params.setdefault("bootstrap", True)
        rf_params.setdefault("oob_score", False)
        rf_params.setdefault("class_weight", "balanced")
        rf_params.setdefault("min_samples_split", 5)
        rf_params.setdefault("min_weight_fraction_leaf", 0.0)
        rf_params.setdefault("max_leaf_nodes", None)
        rf_params.setdefault("min_impurity_decrease", 0.0)
        rf_params.setdefault("warm_start", False)
        
        logger.info("Mapped XGBoost parameters to Random Forest: %d -> %d parameters", 
                   len(xgb_params), len(rf_params))
        
        return rf_params
    
    def validate_params(self, params: Dict[str, Any]) -> Tuple[bool, str]:
        """Validate XGBoost parameters."""
        required_params = ["objective", "num_class"]
        
        for param in required_params:
            if param not in params:
                return False, f"Missing required parameter: {param}"
        
        # Validate parameter ranges
        if "eta" in params and not (0.0 < params["eta"] <= 1.0):
            return False, "eta must be in range (0, 1]"
        
        if "max_depth" in params and params["max_depth"] <= 0:
            return False, "max_depth must be positive"
        
        if "subsample" in params and not (0.0 < params["subsample"] <= 1.0):
            return False, "subsample must be in range (0, 1]"
        
        return True, "Parameters are valid"


class RandomForestParameterMapper(BaseParameterMapper):
    """Parameter mapper for Random Forest models."""
    
    # Default Random Forest parameters
    DEFAULT_PARAMS = {
        "n_estimators": 100,
        "max_depth": 10,
        "min_samples_split": 5,
        "min_samples_leaf": 2,
        "max_features": "sqrt",
        "criterion": "gini",
        "bootstrap": True,
        "oob_score": False,
        "n_jobs": -1,
        "random_state": 42,
        "class_weight": "balanced",
        "max_samples": None,
        "min_weight_fraction_leaf": 0.0,
        "max_leaf_nodes": None,
        "min_impurity_decrease": 0.0,
        "warm_start": False
    }
    
    def get_default_params(self) -> Dict[str, Any]:
        """Get default Random Forest parameters."""
        return self.DEFAULT_PARAMS.copy()
    
    def map_to_target(self, source_params: Dict[str, Any], 
                     target_type: ModelType) -> Dict[str, Any]:
        """Map Random Forest parameters to target model type."""
        if target_type == ModelType.RANDOM_FOREST:
            return source_params.copy()
        elif target_type == ModelType.XGBOOST:
            return self._map_to_xgboost(source_params)
        else:
            raise ValueError(f"Unsupported target model type: {target_type}")
    
    def _map_to_xgboost(self, rf_params: Dict[str, Any]) -> Dict[str, Any]:
        """Map Random Forest parameters to XGBoost parameters."""
        xgb_params = {}
        
        # Tree structure parameters
        if "max_depth" in rf_params:
            xgb_params["max_depth"] = rf_params["max_depth"]
        
        if "min_samples_leaf" in rf_params:
            # Map min_samples_leaf to min_child_weight
            xgb_params["min_child_weight"] = rf_params["min_samples_leaf"]
        
        # Sampling parameters
        if "max_samples" in rf_params and rf_params["max_samples"] is not None:
            if isinstance(rf_params["max_samples"], float):
                xgb_params["subsample"] = rf_params["max_samples"]
        
        if "max_features" in rf_params:
            # Map max_features to colsample_bytree
            max_feat = rf_params["max_features"]
            if max_feat == "sqrt":
                xgb_params["colsample_bytree"] = 0.5
            elif max_feat == "log2":
                xgb_params["colsample_bytree"] = 0.3
            elif isinstance(max_feat, float):
                xgb_params["colsample_bytree"] = max_feat
            elif isinstance(max_feat, int):
                # Assume this is a fraction for now
                xgb_params["colsample_bytree"] = 0.8
            else:
                xgb_params["colsample_bytree"] = 0.8
        
        # Number of estimators
        if "n_estimators" in rf_params:
            # Scale up from RF estimators to XGBoost rounds
            xgb_params["num_boost_round"] = rf_params["n_estimators"] * 2
        
        # Random state
        if "random_state" in rf_params:
            xgb_params["random_state"] = rf_params["random_state"]
        
        # Performance parameters
        if "n_jobs" in rf_params:
            xgb_params["nthread"] = rf_params["n_jobs"] if rf_params["n_jobs"] > 0 else 16
        
        # Set XGBoost-specific defaults for unmapped parameters
        xgb_params.setdefault("objective", "multi:softprob")
        xgb_params.setdefault("num_class", 11)
        xgb_params.setdefault("eta", 0.05)
        xgb_params.setdefault("gamma", 0.5)
        xgb_params.setdefault("colsample_bylevel", 0.8)
        xgb_params.setdefault("tree_method", "hist")
        xgb_params.setdefault("eval_metric", "mlogloss")
        xgb_params.setdefault("max_delta_step", 1)
        xgb_params.setdefault("reg_alpha", 0.1)
        xgb_params.setdefault("reg_lambda", 1.0)
        xgb_params.setdefault("base_score", 0.5)
        xgb_params.setdefault("scale_pos_weight", 1.0)
        xgb_params.setdefault("grow_policy", "depthwise")
        xgb_params.setdefault("normalize_type", "tree")
        
        logger.info("Mapped Random Forest parameters to XGBoost: %d -> %d parameters", 
                   len(rf_params), len(xgb_params))
        
        return xgb_params
    
    def validate_params(self, params: Dict[str, Any]) -> Tuple[bool, str]:
        """Validate Random Forest parameters."""
        required_params = ["n_estimators"]
        
        for param in required_params:
            if param not in params:
                return False, f"Missing required parameter: {param}"
        
        # Validate parameter ranges
        if "n_estimators" in params and params["n_estimators"] <= 0:
            return False, "n_estimators must be positive"
        
        if "max_depth" in params and params["max_depth"] is not None and params["max_depth"] <= 0:
            return False, "max_depth must be positive or None"
        
        if "min_samples_split" in params and params["min_samples_split"] < 2:
            return False, "min_samples_split must be at least 2"
        
        if "min_samples_leaf" in params and params["min_samples_leaf"] < 1:
            return False, "min_samples_leaf must be at least 1"
        
        return True, "Parameters are valid"


class UnifiedParameterManager:
    """
    Unified parameter manager for seamless model type switching.
    
    This class provides a high-level interface for parameter conversion,
    validation, and management across different model types.
    """
    
    def __init__(self):
        """Initialize the unified parameter manager."""
        self._mappers = {
            ModelType.XGBOOST: XGBoostParameterMapper(),
            ModelType.RANDOM_FOREST: RandomForestParameterMapper()
        }
        self._current_model_type = None
        self._current_params = None
    
    def get_mapper(self, model_type: Union[str, ModelType]) -> BaseParameterMapper:
        """Get parameter mapper for the specified model type."""
        if isinstance(model_type, str):
            model_type = ModelType(model_type.lower())
        
        if model_type not in self._mappers:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        return self._mappers[model_type]
    
    def convert_parameters(self, source_params: Dict[str, Any], 
                          source_type: Union[str, ModelType],
                          target_type: Union[str, ModelType]) -> Dict[str, Any]:
        """
        Convert parameters from one model type to another.
        
        Args:
            source_params: Parameters in source model format
            source_type: Source model type
            target_type: Target model type
            
        Returns:
            Parameters converted to target model format
        """
        if isinstance(source_type, str):
            source_type = ModelType(source_type.lower())
        if isinstance(target_type, str):
            target_type = ModelType(target_type.lower())
        
        if source_type == target_type:
            return source_params.copy()
        
        source_mapper = self.get_mapper(source_type)
        
        # Validate source parameters
        is_valid, error_msg = source_mapper.validate_params(source_params)
        if not is_valid:
            logger.warning("Source parameters validation failed: %s", error_msg)
        
        # Convert parameters
        converted_params = source_mapper.map_to_target(source_params, target_type)
        
        # Validate converted parameters
        target_mapper = self.get_mapper(target_type)
        is_valid, error_msg = target_mapper.validate_params(converted_params)
        if not is_valid:
            logger.warning("Converted parameters validation failed: %s", error_msg)
        
        logger.info("Successfully converted parameters from %s to %s", 
                   source_type.value, target_type.value)
        
        return converted_params
    
    def get_default_parameters(self, model_type: Union[str, ModelType]) -> Dict[str, Any]:
        """Get default parameters for the specified model type."""
        mapper = self.get_mapper(model_type)
        return mapper.get_default_params()
    
    def validate_parameters(self, params: Dict[str, Any], 
                           model_type: Union[str, ModelType]) -> Tuple[bool, str]:
        """Validate parameters for the specified model type."""
        mapper = self.get_mapper(model_type)
        return mapper.validate_params(params)
    
    def create_unified_config(self, base_params: Dict[str, Any],
                             model_type: Union[str, ModelType]) -> Dict[str, Any]:
        """
        Create a unified configuration that can be easily converted between model types.
        
        This method takes a base parameter set and enriches it with defaults
        to create a comprehensive configuration.
        
        Args:
            base_params: Base parameters to start with
            model_type: Model type for the configuration
            
        Returns:
            Unified configuration dictionary
        """
        if isinstance(model_type, str):
            model_type = ModelType(model_type.lower())
        
        # Get default parameters for the model type
        default_params = self.get_default_parameters(model_type)
        
        # Merge base parameters with defaults (base parameters take priority)
        unified_config = default_params.copy()
        unified_config.update(base_params)
        
        # Store current state
        self._current_model_type = model_type
        self._current_params = unified_config.copy()
        
        logger.info("Created unified config for %s with %d parameters", 
                   model_type.value, len(unified_config))
        
        return unified_config
    
    def switch_model_type(self, new_model_type: Union[str, ModelType]) -> Dict[str, Any]:
        """
        Switch to a different model type, converting current parameters.
        
        Args:
            new_model_type: Target model type to switch to
            
        Returns:
            Parameters converted to the new model type
        """
        if self._current_model_type is None or self._current_params is None:
            raise RuntimeError("No current configuration. Create a unified config first.")
        
        if isinstance(new_model_type, str):
            new_model_type = ModelType(new_model_type.lower())
        
        if new_model_type == self._current_model_type:
            return self._current_params.copy()
        
        # Convert parameters
        converted_params = self.convert_parameters(
            self._current_params, 
            self._current_model_type, 
            new_model_type
        )
        
        # Update current state
        self._current_model_type = new_model_type
        self._current_params = converted_params
        
        return converted_params
    
    def get_current_config(self) -> Optional[Tuple[ModelType, Dict[str, Any]]]:
        """Get the current model type and parameters."""
        if self._current_model_type is None or self._current_params is None:
            return None
        return self._current_model_type, self._current_params.copy()
    
    def export_config_for_framework(self, model_type: Union[str, ModelType],
                                   params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Export configuration in the format expected by the target framework.
        
        Args:
            model_type: Target model type
            params: Parameters to export (uses current if None)
            
        Returns:
            Configuration dictionary ready for use with the target framework
        """
        if params is None:
            if self._current_params is None:
                raise RuntimeError("No parameters available. Provide params or create a config first.")
            params = self._current_params
        
        if isinstance(model_type, str):
            model_type = ModelType(model_type.lower())
        
        # Validate parameters
        is_valid, error_msg = self.validate_parameters(params, model_type)
        if not is_valid:
            logger.warning("Parameter validation failed: %s", error_msg)
        
        # Create framework-specific configuration
        config = {
            "type": model_type.value,
            "params": params.copy()
        }
        
        logger.info("Exported config for %s with %d parameters", 
                   model_type.value, len(params))
        
        return config


# Convenience functions for easy usage

def convert_xgboost_to_random_forest(xgb_params: Dict[str, Any]) -> Dict[str, Any]:
    """Convert XGBoost parameters to Random Forest parameters."""
    manager = UnifiedParameterManager()
    return manager.convert_parameters(xgb_params, ModelType.XGBOOST, ModelType.RANDOM_FOREST)


def convert_random_forest_to_xgboost(rf_params: Dict[str, Any]) -> Dict[str, Any]:
    """Convert Random Forest parameters to XGBoost parameters."""
    manager = UnifiedParameterManager()
    return manager.convert_parameters(rf_params, ModelType.RANDOM_FOREST, ModelType.XGBOOST)


def get_unified_defaults(model_type: Union[str, ModelType]) -> Dict[str, Any]:
    """Get default parameters for the specified model type."""
    manager = UnifiedParameterManager()
    return manager.get_default_parameters(model_type)


def create_cross_compatible_config(base_params: Dict[str, Any], 
                                  primary_model: Union[str, ModelType],
                                  secondary_model: Union[str, ModelType]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Create configurations for both model types that are cross-compatible.
    
    Args:
        base_params: Base parameters to start with
        primary_model: Primary model type
        secondary_model: Secondary model type
        
    Returns:
        Tuple of (primary_config, secondary_config)
    """
    manager = UnifiedParameterManager()
    
    # Create unified config for primary model
    primary_config = manager.create_unified_config(base_params, primary_model)
    
    # Convert to secondary model
    secondary_config = manager.convert_parameters(primary_config, primary_model, secondary_model)
    
    return primary_config, secondary_config 