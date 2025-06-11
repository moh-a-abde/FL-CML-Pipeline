"""
Random Forest implementation for federated learning pipeline.

This module provides a Random Forest classifier that implements the BaseModel
interface for compatibility with the federated learning framework.
"""

import pickle
import logging
from typing import Any, Dict, Union, Optional
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

from .base_model import BaseModel

logger = logging.getLogger(__name__)


class RandomForestModel(BaseModel):
    """Random Forest implementation for federated learning."""
    
    def __init__(self, params: Dict[str, Any]):
        """
        Initialize Random Forest model with hyperparameters.
        
        Args:
            params: Dictionary of Random Forest hyperparameters
        """
        super().__init__(params)
        
        # Extract Random Forest specific parameters
        self.rf_params = self._extract_rf_params(params)
        self.model = RandomForestClassifier(**self.rf_params)
        
        # Store training history for federated learning
        self.training_history = []
        
    def _extract_rf_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Extract Random Forest specific parameters from config."""
        rf_params = {}
        
        # Core Random Forest parameters
        rf_params['n_estimators'] = params.get('n_estimators', 100)
        rf_params['max_depth'] = params.get('max_depth', None)
        rf_params['min_samples_split'] = params.get('min_samples_split', 2)
        rf_params['min_samples_leaf'] = params.get('min_samples_leaf', 1)
        rf_params['max_features'] = params.get('max_features', 'sqrt')
        rf_params['criterion'] = params.get('criterion', 'gini')
        rf_params['bootstrap'] = params.get('bootstrap', True)
        rf_params['oob_score'] = params.get('oob_score', False)
        rf_params['n_jobs'] = params.get('n_jobs', -1)
        rf_params['random_state'] = params.get('random_state', 42)
        rf_params['class_weight'] = params.get('class_weight', None)
        rf_params['max_samples'] = params.get('max_samples', None)
        rf_params['min_weight_fraction_leaf'] = params.get('min_weight_fraction_leaf', 0.0)
        rf_params['max_leaf_nodes'] = params.get('max_leaf_nodes', None)
        rf_params['min_impurity_decrease'] = params.get('min_impurity_decrease', 0.0)
        rf_params['warm_start'] = params.get('warm_start', False)
        
        return rf_params
    
    def fit(self, 
            X: Union[np.ndarray, pd.DataFrame], 
            y: Union[np.ndarray, pd.Series],
            X_val: Optional[Union[np.ndarray, pd.DataFrame]] = None,
            y_val: Optional[Union[np.ndarray, pd.Series]] = None,
            **kwargs) -> 'RandomForestModel':
        """
        Train the Random Forest model.
        
        Args:
            X: Training features
            y: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            **kwargs: Additional training parameters
            
        Returns:
            Self for method chaining
        """
        logger.info(f"Training Random Forest with {X.shape[0]} samples, {X.shape[1]} features")
        
        # Convert pandas to numpy if needed
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values
            
        # Train the model
        self.model.fit(X, y)
        self._is_trained = True
        
        # Calculate training metrics
        train_pred = self.model.predict(X)
        train_pred_proba = self.model.predict_proba(X)
        train_accuracy = accuracy_score(y, train_pred)
        train_f1 = f1_score(y, train_pred, average='weighted')
        
        # Calculate validation metrics if validation data provided
        val_accuracy = None
        val_f1 = None
        if X_val is not None and y_val is not None:
            if isinstance(X_val, pd.DataFrame):
                X_val = X_val.values
            if isinstance(y_val, pd.Series):
                y_val = y_val.values
                
            val_pred = self.model.predict(X_val)
            val_pred_proba = self.model.predict_proba(X_val)
            val_accuracy = accuracy_score(y_val, val_pred)
            val_f1 = f1_score(y_val, val_pred, average='weighted')
        
        # Store training metrics
        training_round = {
            'train_accuracy': train_accuracy,
            'train_f1': train_f1,
            'val_accuracy': val_accuracy,
            'val_f1': val_f1,
            'n_estimators': self.model.n_estimators,
            'oob_score': getattr(self.model, 'oob_score_', None) if self.rf_params.get('oob_score', False) else None
        }
        self.training_history.append(training_round)
        
        logger.info(f"Training completed: Train Acc={train_accuracy:.4f}, Train F1={train_f1:.4f}")
        if val_accuracy is not None:
            logger.info(f"Validation: Val Acc={val_accuracy:.4f}, Val F1={val_f1:.4f}")
            
        return self
    
    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Make predictions on the provided data."""
        if not self._is_trained:
            raise ValueError("Model must be trained before making predictions")
            
        if isinstance(X, pd.DataFrame):
            X = X.values
            
        return self.model.predict(X)
    
    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Make probability predictions on the provided data."""
        if not self._is_trained:
            raise ValueError("Model must be trained before making predictions")
            
        if isinstance(X, pd.DataFrame):
            X = X.values
            
        return self.model.predict_proba(X)
    
    def serialize(self) -> bytes:
        """Serialize the model to bytes for transmission in federated learning."""
        if not self._is_trained:
            raise ValueError("Cannot serialize untrained model")
            
        try:
            return pickle.dumps(self.model)
        except Exception as e:
            logger.error(f"Failed to serialize Random Forest model: {e}")
            raise
    
    def deserialize(self, model_bytes: bytes) -> 'RandomForestModel':
        """Deserialize model from bytes."""
        try:
            self.model = pickle.loads(model_bytes)
            self._is_trained = True
            logger.info("Successfully deserialized Random Forest model")
            return self
        except Exception as e:
            logger.error(f"Failed to deserialize Random Forest model: {e}")
            raise
    
    def save_model(self, filepath: str) -> None:
        """Save model to disk."""
        if not self._is_trained:
            raise ValueError("Cannot save untrained model")
            
        try:
            # Save using joblib for better scikit-learn compatibility
            import joblib
            joblib.dump(self.model, filepath)
            logger.info(f"Model saved to {filepath}")
        except ImportError:
            # Fallback to pickle if joblib not available
            with open(filepath, 'wb') as f:
                pickle.dump(self.model, f)
            logger.info(f"Model saved to {filepath} (using pickle)")
        except Exception as e:
            logger.error(f"Failed to save model to {filepath}: {e}")
            raise
    
    def load_model(self, filepath: str) -> 'RandomForestModel':
        """Load model from disk."""
        try:
            # Try joblib first
            try:
                import joblib
                self.model = joblib.load(filepath)
            except ImportError:
                # Fallback to pickle
                with open(filepath, 'rb') as f:
                    self.model = pickle.load(f)
                    
            self._is_trained = True
            logger.info(f"Model loaded from {filepath}")
            return self
        except Exception as e:
            logger.error(f"Failed to load model from {filepath}: {e}")
            raise
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores."""
        if not self._is_trained:
            raise ValueError("Model must be trained to get feature importance")
            
        # Get feature importances from the trained model
        importances = self.model.feature_importances_
        
        # Create dictionary with feature names (if available) or indices
        feature_names = getattr(self, 'feature_names_', None)
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(len(importances))]
            
        return dict(zip(feature_names, importances))
    
    def update_from_model(self, other_model: 'RandomForestModel', **kwargs) -> 'RandomForestModel':
        """
        Update this model using another model (for federated aggregation).
        
        For Random Forest, this combines trees from both models to create
        a larger ensemble.
        """
        if not isinstance(other_model, RandomForestModel):
            raise ValueError("Can only update from another RandomForestModel")
            
        if not other_model.is_trained:
            raise ValueError("Other model must be trained")
            
        strategy = kwargs.get('strategy', 'combine_trees')
        
        if strategy == 'combine_trees':
            # Combine trees from both models
            self_trees = list(self.model.estimators_)
            other_trees = list(other_model.model.estimators_)
            
            # Create new model with combined trees
            combined_trees = self_trees + other_trees
            
            # Update model parameters
            self.model.estimators_ = np.array(combined_trees)
            self.model.n_estimators = len(combined_trees)
            
            logger.info(f"Combined {len(self_trees)} + {len(other_trees)} = {len(combined_trees)} trees")
            
        elif strategy == 'replace':
            # Simply replace this model with the other model
            self.model = other_model.model
            self._is_trained = other_model.is_trained
            
        else:
            raise ValueError(f"Unknown update strategy: {strategy}")
            
        return self
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get detailed information about the trained model."""
        if not self._is_trained:
            return {"status": "untrained"}
            
        info = {
            "status": "trained",
            "n_estimators": self.model.n_estimators,
            "max_depth": self.model.max_depth,
            "n_features": self.model.n_features_in_ if hasattr(self.model, 'n_features_in_') else None,
            "n_classes": self.model.n_classes_ if hasattr(self.model, 'n_classes_') else None,
            "oob_score": getattr(self.model, 'oob_score_', None),
            "training_rounds": len(self.training_history),
            "feature_importances": self.get_feature_importance() if self._is_trained else None
        }
        
        return info 