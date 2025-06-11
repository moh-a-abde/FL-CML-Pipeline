"""
Model factory for creating different types of models in the federated learning pipeline.

This module provides a factory pattern for instantiating models based on 
configuration, enabling easy switching between XGBoost, Random Forest, and 
future model implementations.
"""

import logging
from typing import Any, Dict, Type

from .base_model import BaseModel
from .random_forest_model import RandomForestModel

logger = logging.getLogger(__name__)


class ModelFactory:
    """Factory for creating models based on configuration."""
    
    # Registry of available models
    _models = {
        'random_forest': RandomForestModel,
        'randomforest': RandomForestModel,  # Alternative naming
        'rf': RandomForestModel,  # Short name
    }
    
    @classmethod
    def register_model(cls, name: str, model_class: Type[BaseModel]) -> None:
        """
        Register a new model type.
        
        Args:
            name: Name to register the model under
            model_class: Model class that implements BaseModel interface
        """
        if not issubclass(model_class, BaseModel):
            raise ValueError(f"Model class {model_class} must inherit from BaseModel")
            
        cls._models[name.lower()] = model_class
        logger.info("Registered model type '%s' -> %s", name, model_class.__name__)
    
    @classmethod
    def get_available_models(cls) -> Dict[str, Type[BaseModel]]:
        """Get dictionary of all available model types."""
        return cls._models.copy()
    
    @classmethod
    def create_model(cls, model_type: str, params: Dict[str, Any]) -> BaseModel:
        """
        Create a model instance based on type and parameters.
        
        Args:
            model_type: Type of model to create (e.g., 'xgboost', 'random_forest')
            params: Dictionary of model hyperparameters
            
        Returns:
            Initialized model instance
            
        Raises:
            ValueError: If model_type is not supported
        """
        model_type_lower = model_type.lower()
        
        if model_type_lower not in cls._models:
            available = ', '.join(cls._models.keys())
            raise ValueError(
                f"Unknown model type '{model_type}'. "
                f"Available models: {available}"
            )
        
        model_class = cls._models[model_type_lower]
        logger.info("Creating %s with parameters: %s", model_class.__name__, params)
        
        try:
            return model_class(params)
        except Exception as e:
            logger.error("Failed to create %s: %s", model_class.__name__, e)
            raise
    
    @classmethod
    def create_from_config(cls, config: Dict[str, Any]) -> BaseModel:
        """
        Create a model from a configuration dictionary.
        
        Expected config format:
        {
            "type": "random_forest",
            "params": {
                "n_estimators": 100,
                "max_depth": 10,
                ...
            }
        }
        
        Args:
            config: Configuration dictionary with 'type' and 'params' keys
            
        Returns:
            Initialized model instance
        """
        if 'type' not in config:
            raise ValueError("Configuration must contain 'type' key")
        
        if 'params' not in config:
            raise ValueError("Configuration must contain 'params' key")
        
        model_type = config['type']
        params = config['params']
        
        return cls.create_model(model_type, params)


# Try to import and register XGBoost model if available
try:
    from .xgboost_model import XGBoostModel
    ModelFactory.register_model('xgboost', XGBoostModel)
    ModelFactory.register_model('xgb', XGBoostModel)
except ImportError:
    logger.warning("XGBoostModel not available - XGBoost models will not be supported")


def get_model_for_config(config: Dict[str, Any]) -> BaseModel:
    """
    Convenience function to create a model from configuration.
    
    Args:
        config: Model configuration dictionary
        
    Returns:
        Initialized model instance
    """
    return ModelFactory.create_from_config(config)


def get_available_model_types() -> list:
    """
    Get list of available model type names.
    
    Returns:
        List of available model type strings
    """
    return list(ModelFactory.get_available_models().keys()) 