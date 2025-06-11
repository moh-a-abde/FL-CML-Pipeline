"""
Abstract base model interface for federated learning pipeline.

This module defines the interface that all models (XGBoost, Random Forest, etc.)
must implement to be compatible with the federated learning framework.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple, Union, Optional
import numpy as np
import pandas as pd


class BaseModel(ABC):
    """Abstract base class for all models in the federated learning pipeline."""
    
    def __init__(self, params: Dict[str, Any]):
        """
        Initialize the model with hyperparameters.
        
        Args:
            params: Dictionary of model hyperparameters
        """
        self.params = params
        self.model = None
        self._is_trained = False
    
    @abstractmethod
    def fit(self, 
            X: Union[np.ndarray, pd.DataFrame], 
            y: Union[np.ndarray, pd.Series],
            X_val: Optional[Union[np.ndarray, pd.DataFrame]] = None,
            y_val: Optional[Union[np.ndarray, pd.Series]] = None,
            **kwargs) -> 'BaseModel':
        """
        Train the model on the provided data.
        
        Args:
            X: Training features
            y: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            **kwargs: Additional training parameters
            
        Returns:
            Self for method chaining
        """
        raise NotImplementedError("Subclasses must implement fit method")
    
    @abstractmethod
    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Make predictions on the provided data.
        
        Args:
            X: Features to predict on
            
        Returns:
            Predicted class labels
        """
        raise NotImplementedError("Subclasses must implement predict method")
    
    @abstractmethod
    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Make probability predictions on the provided data.
        
        Args:
            X: Features to predict on
            
        Returns:
            Predicted class probabilities
        """
        raise NotImplementedError("Subclasses must implement predict_proba method")
    
    @abstractmethod
    def serialize(self) -> bytes:
        """
        Serialize the model to bytes for transmission in federated learning.
        
        Returns:
            Serialized model as bytes
        """
        raise NotImplementedError("Subclasses must implement serialize method")
    
    @abstractmethod
    def deserialize(self, model_bytes: bytes) -> 'BaseModel':
        """
        Deserialize model from bytes.
        
        Args:
            model_bytes: Serialized model bytes
            
        Returns:
            Self with loaded model
        """
        raise NotImplementedError("Subclasses must implement deserialize method")
    
    @abstractmethod
    def save_model(self, filepath: str) -> None:
        """
        Save model to disk.
        
        Args:
            filepath: Path to save the model
        """
        raise NotImplementedError("Subclasses must implement save_model method")
    
    @abstractmethod
    def load_model(self, filepath: str) -> 'BaseModel':
        """
        Load model from disk.
        
        Args:
            filepath: Path to load the model from
            
        Returns:
            Self with loaded model
        """
        raise NotImplementedError("Subclasses must implement load_model method")
    
    @abstractmethod
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance scores.
        
        Returns:
            Dictionary mapping feature names to importance scores
        """
        raise NotImplementedError("Subclasses must implement get_feature_importance method")
    
    @abstractmethod
    def update_from_model(self, other_model: 'BaseModel', **kwargs) -> 'BaseModel':
        """
        Update this model using another model (for federated aggregation).
        
        Args:
            other_model: Another model to update from
            **kwargs: Additional parameters for update strategy
            
        Returns:
            Self with updated model
        """
        raise NotImplementedError("Subclasses must implement update_from_model method")
    
    @property
    def is_trained(self) -> bool:
        """Check if model has been trained."""
        return self._is_trained
    
    @property
    def model_type(self) -> str:
        """Get the model type identifier."""
        return self.__class__.__name__.replace('Model', '').lower()
    
    def get_params(self) -> Dict[str, Any]:
        """Get model hyperparameters."""
        return self.params.copy()
    
    def set_params(self, **params) -> 'BaseModel':
        """Set model hyperparameters."""
        self.params.update(params)
        return self 