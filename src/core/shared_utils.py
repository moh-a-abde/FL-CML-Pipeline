"""
Shared Utilities for FL-CML-Pipeline

This module provides centralized implementations of commonly used functionality
to eliminate code duplication across the federated learning pipeline.

Key Components:
- DMatrixFactory: Centralized XGBoost DMatrix creation
- MetricsCalculator: Centralized metrics calculation for classification
- XGBoostParamsBuilder: Centralized parameter management for XGBoost

Created for Phase 3: Code Deduplication
"""

from typing import Dict, List, Optional, Union, Tuple, Any
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, log_loss
)
from flwr.common.logger import log
from flwr.common.typing import Scalar
from logging import INFO, WARNING, ERROR
import pickle
import os
from dataclasses import dataclass

# Type aliases for clarity
Features = Union[np.ndarray, pd.DataFrame]
Labels = Union[np.ndarray, pd.Series]
ClientMetrics = List[Tuple[int, Dict[str, float]]]


@dataclass
class MetricsResult:
    """Container for classification metrics results."""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    mlogloss: Optional[float] = None
    confusion_matrix: Optional[np.ndarray] = None
    classification_report: Optional[str] = None
    raw_metrics: Optional[Dict[str, float]] = None


class DMatrixFactory:
    """
    Centralized XGBoost DMatrix creation with consistent configuration.
    
    Eliminates code duplication across modules and ensures consistent
    handling of missing values, logging, and validation.
    """
    
    @staticmethod
    def create_dmatrix(
        features: Features,
        labels: Optional[Labels] = None,
        handle_missing: bool = True,
        feature_names: Optional[List[str]] = None,
        weights: Optional[np.ndarray] = None,
        validate: bool = True,
        log_details: bool = True
    ) -> xgb.DMatrix:
        """
        Create XGBoost DMatrix with consistent handling and validation.
        
        Args:
            features: Feature data (numpy array or pandas DataFrame)
            labels: Target labels (optional for prediction-only usage)
            handle_missing: Whether to replace inf values with nan
            feature_names: Optional feature names for the DMatrix
            weights: Optional sample weights for training
            validate: Whether to validate input data integrity
            log_details: Whether to log detailed creation information
            
        Returns:
            xgb.DMatrix: Properly configured DMatrix object
            
        Raises:
            ValueError: If validation fails or incompatible data provided
            TypeError: If unsupported data types provided
        """
        if log_details:
            log(INFO, "[DMatrixFactory] Creating DMatrix...")
        
        # Convert to numpy arrays for consistent processing
        if isinstance(features, pd.DataFrame):
            feature_names = feature_names or list(features.columns)
            features_array = features.values
        else:
            features_array = np.asarray(features)
            
        # Handle labels conversion
        labels_array = None
        if labels is not None:
            if isinstance(labels, (pd.Series, pd.DataFrame)):
                labels_array = labels.values.ravel()
            else:
                labels_array = np.asarray(labels).ravel()
                
        # Validation checks
        if validate:
            DMatrixFactory._validate_input_data(features_array, labels_array, weights)
            
        # Handle missing values
        if handle_missing:
            features_array = np.where(np.isinf(features_array), np.nan, features_array)
            
        # Log data statistics
        if log_details:
            log(INFO, f"[DMatrixFactory] Features shape: {features_array.shape}")
            if labels_array is not None:
                log(INFO, f"[DMatrixFactory] Labels shape: {labels_array.shape}")
                log(INFO, f"[DMatrixFactory] Labels dtype: {labels_array.dtype}")
                unique_labels, counts = np.unique(labels_array, return_counts=True)
                log(INFO, f"[DMatrixFactory] Unique labels: {unique_labels.tolist()}")
                log(INFO, f"[DMatrixFactory] Label counts: {counts.tolist()}")
            else:
                log(INFO, "[DMatrixFactory] No labels provided (prediction mode)")
                
            if weights is not None:
                log(INFO, f"[DMatrixFactory] Sample weights shape: {weights.shape}")
                
        # Create DMatrix with appropriate parameters
        dmatrix_kwargs = {
            'data': features_array,
            'missing': np.nan  # Consistent missing value handling
        }
        
        if labels_array is not None:
            dmatrix_kwargs['label'] = labels_array
            
        if feature_names is not None:
            dmatrix_kwargs['feature_names'] = feature_names
            
        if weights is not None:
            dmatrix_kwargs['weight'] = weights
            
        # Create the DMatrix
        dmatrix = xgb.DMatrix(**dmatrix_kwargs)
        
        if log_details:
            log(INFO, f"[DMatrixFactory] Created DMatrix: {dmatrix.num_row()} rows, {dmatrix.num_col()} features")
            
        return dmatrix
    
    @staticmethod
    def _validate_input_data(
        features: np.ndarray, 
        labels: Optional[np.ndarray] = None, 
        weights: Optional[np.ndarray] = None
    ) -> None:
        """Validate input data for DMatrix creation."""
        if features.size == 0:
            raise ValueError("Features array is empty")
            
        if features.ndim != 2:
            raise ValueError(f"Features must be 2D array, got shape {features.shape}")
            
        if labels is not None:
            if len(labels) != features.shape[0]:
                raise ValueError(
                    f"Labels length ({len(labels)}) doesn't match "
                    f"features rows ({features.shape[0]})"
                )
                
        if weights is not None:
            if len(weights) != features.shape[0]:
                raise ValueError(
                    f"Weights length ({len(weights)}) doesn't match "
                    f"features rows ({features.shape[0]})"
                )
            if np.any(weights < 0):
                raise ValueError("Sample weights must be non-negative")
    
    @staticmethod
    def create_weighted_dmatrix(
        base_dmatrix: xgb.DMatrix,
        weights: np.ndarray,
        feature_names: Optional[List[str]] = None
    ) -> xgb.DMatrix:
        """
        Create a weighted DMatrix from an existing DMatrix.
        
        Common pattern in federated learning for class balancing.
        
        Args:
            base_dmatrix: Original DMatrix to extract data from
            weights: Sample weights to apply
            feature_names: Optional feature names
            
        Returns:
            New DMatrix with applied weights
        """
        return DMatrixFactory.create_dmatrix(
            features=base_dmatrix.get_data(),
            labels=base_dmatrix.get_label(),
            weights=weights,
            feature_names=feature_names or base_dmatrix.feature_names,
            log_details=False  # Avoid duplicate logging
        )


class MetricsCalculator:
    """
    Centralized metrics calculation with consistent implementation.
    
    Provides unified interface for calculating classification metrics
    across federated learning, hyperparameter tuning, and evaluation.
    """
    
    # Standard class names for UNSW-NB15 dataset (can be overridden)
    DEFAULT_CLASS_NAMES = [
        'Normal', 'Generic', 'Exploits', 'Reconnaissance', 'Fuzzers',
        'DoS', 'Analysis', 'Backdoor', 'Backdoors', 'Worms', 'Shellcode'
    ]
    
    @staticmethod
    def calculate_classification_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_pred_proba: Optional[np.ndarray] = None,
        class_names: Optional[List[str]] = None,
        prefix: str = "",
        calculate_per_class: bool = False,
        return_confusion_matrix: bool = True
    ) -> MetricsResult:
        """
        Calculate comprehensive classification metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Prediction probabilities (optional)
            class_names: Class names for reporting
            prefix: Prefix for metric keys
            calculate_per_class: Whether to calculate per-class metrics
            return_confusion_matrix: Whether to include confusion matrix
            
        Returns:
            MetricsResult object with all calculated metrics
        """
        # Ensure arrays are proper format
        y_true = np.asarray(y_true).astype(int)
        y_pred = np.asarray(y_pred).astype(int)
        
        if class_names is None:
            class_names = MetricsCalculator.DEFAULT_CLASS_NAMES
            
        # Calculate core metrics with consistent parameters
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        # Calculate log loss if probabilities provided
        mlogloss = None
        if y_pred_proba is not None:
            try:
                # Ensure probabilities are valid
                if y_pred_proba.ndim == 2 and y_pred_proba.shape[1] > 1:
                    mlogloss = log_loss(y_true, y_pred_proba, labels=np.unique(y_true))
                else:
                    log(WARNING, "[MetricsCalculator] Invalid probability shape for log_loss calculation")
            except Exception as e:
                log(WARNING, f"[MetricsCalculator] Could not calculate log_loss: {e}")
                
        # Calculate confusion matrix
        conf_matrix = None
        if return_confusion_matrix:
            try:
                conf_matrix = confusion_matrix(y_true, y_pred)
            except Exception as e:
                log(WARNING, f"[MetricsCalculator] Could not calculate confusion matrix: {e}")
                
        # Generate classification report
        class_report = None
        try:
            # Filter class names to match actual labels
            unique_labels = np.unique(np.concatenate([y_true, y_pred]))
            filtered_names = [class_names[i] for i in unique_labels if i < len(class_names)]
            
            class_report = classification_report(
                y_true, y_pred,
                labels=unique_labels,
                target_names=filtered_names,
                zero_division=0
            )
        except Exception as e:
            log(WARNING, f"[MetricsCalculator] Could not generate classification report: {e}")
            
        # Create raw metrics dictionary
        raw_metrics = {
            f"{prefix}accuracy": accuracy,
            f"{prefix}precision": precision,
            f"{prefix}recall": recall,
            f"{prefix}f1": f1
        }
        
        if mlogloss is not None:
            raw_metrics[f"{prefix}mlogloss"] = mlogloss
            
        # Add per-class metrics if requested
        if calculate_per_class:
            per_class_metrics = MetricsCalculator._calculate_per_class_metrics(
                y_true, y_pred, prefix
            )
            raw_metrics.update(per_class_metrics)
            
        return MetricsResult(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            mlogloss=mlogloss,
            confusion_matrix=conf_matrix,
            classification_report=class_report,
            raw_metrics=raw_metrics
        )
    
    @staticmethod
    def _calculate_per_class_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        prefix: str = ""
    ) -> Dict[str, float]:
        """Calculate per-class precision, recall, and f1 scores."""
        per_class_metrics = {}
        unique_classes = np.unique(y_true)
        
        for class_idx in unique_classes:
            class_prefix = f"{prefix}class_{class_idx}_"
            
            # Binary classification metrics for this class
            y_true_binary = (y_true == class_idx).astype(int)
            y_pred_binary = (y_pred == class_idx).astype(int)
            
            if np.sum(y_true_binary) > 0:  # Only if class exists in true labels
                per_class_metrics[f"{class_prefix}precision"] = precision_score(
                    y_true_binary, y_pred_binary, zero_division=0
                )
                per_class_metrics[f"{class_prefix}recall"] = recall_score(
                    y_true_binary, y_pred_binary, zero_division=0
                )
                per_class_metrics[f"{class_prefix}f1"] = f1_score(
                    y_true_binary, y_pred_binary, zero_division=0
                )
                
        return per_class_metrics
    
    @staticmethod
    def aggregate_client_metrics(
        client_metrics: ClientMetrics,
        log_details: bool = True
    ) -> Dict[str, float]:
        """
        Aggregate metrics from multiple federated learning clients.
        
        Args:
            client_metrics: List of (num_examples, metrics_dict) tuples
            log_details: Whether to log aggregation details
            
        Returns:
            Dictionary of aggregated metrics
        """
        if not client_metrics:
            log(WARNING, "[MetricsCalculator] No client metrics provided for aggregation")
            return {}
            
        total_examples = sum([num for num, _ in client_metrics])
        
        if log_details:
            log(INFO, f"[MetricsCalculator] Aggregating metrics from {len(client_metrics)} clients")
            log(INFO, f"[MetricsCalculator] Total examples: {total_examples}")
            
        # Find common metrics across all clients
        common_metrics = set(client_metrics[0][1].keys())
        for _, metrics in client_metrics[1:]:
            common_metrics &= set(metrics.keys())
            
        aggregated = {}
        
        # Aggregate each metric using weighted average
        for metric_name in common_metrics:
            if metric_name == "confusion_matrix":
                # Special handling for confusion matrices
                aggregated[metric_name] = MetricsCalculator._aggregate_confusion_matrices(
                    client_metrics, total_examples
                )
            else:
                # Weighted average for scalar metrics
                weighted_sum = sum([
                    metrics[metric_name] * num 
                    for num, metrics in client_metrics
                    if metric_name in metrics
                ])
                aggregated[metric_name] = weighted_sum / total_examples
                
                if log_details:
                    individual_values = [
                        metrics[metric_name] for _, metrics in client_metrics
                        if metric_name in metrics
                    ]
                    log(INFO, f"[MetricsCalculator] {metric_name}: {individual_values} -> {aggregated[metric_name]:.4f}")
                    
        return aggregated
    
    @staticmethod
    def _aggregate_confusion_matrices(
        client_metrics: ClientMetrics,
        total_examples: int
    ) -> Optional[List[List[float]]]:
        """Aggregate confusion matrices from multiple clients."""
        aggregated_matrix = None
        
        for num, metrics in client_metrics:
            if "confusion_matrix" in metrics:
                matrix = metrics["confusion_matrix"]
                if aggregated_matrix is None:
                    # Initialize with zeros
                    aggregated_matrix = [[0.0 for _ in range(len(matrix[0]))] 
                                       for _ in range(len(matrix))]
                
                # Add weighted contribution
                for i in range(len(matrix)):
                    for j in range(len(matrix[0])):
                        aggregated_matrix[i][j] += matrix[i][j] * num / total_examples
                        
        return aggregated_matrix


class XGBoostParamsBuilder:
    """
    Centralized XGBoost parameter management with priority handling.
    
    Provides consistent parameter building across different components
    while respecting configuration hierarchy.
    """
    
    # Default parameters for UNSW-NB15 multi-class classification
    DEFAULT_PARAMS = {
        'objective': 'multi:softprob',
        'num_class': 11,
        'eval_metric': 'mlogloss',
        'learning_rate': 0.05,
        'max_depth': 6,
        'min_child_weight': 1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0,
        'reg_lambda': 1,
        'gamma': 0,
        'scale_pos_weight': 1.0,
        'seed': 42,
        'verbosity': 0
    }
    
    @staticmethod
    def build_params(
        config_manager=None,
        overrides: Optional[Dict[str, Any]] = None,
        use_tuned: bool = False,
        base_params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Build XGBoost parameters with priority handling.
        
        Priority (highest to lowest):
        1. overrides - Direct parameter overrides
        2. config_manager - Parameters from configuration
        3. tuned_params - Previously tuned parameters
        4. base_params - Custom base parameters
        5. DEFAULT_PARAMS - Built-in defaults
        
        Args:
            config_manager: ConfigManager instance for parameter retrieval
            overrides: Direct parameter overrides (highest priority)
            use_tuned: Whether to load tuned parameters
            base_params: Custom base parameters
            
        Returns:
            Complete XGBoost parameter dictionary
        """
        # Start with defaults
        params = XGBoostParamsBuilder.DEFAULT_PARAMS.copy()
        
        # Apply base parameters if provided
        if base_params:
            params.update(base_params)
            
        # Apply tuned parameters if requested
        if use_tuned:
            try:
                tuned_params = XGBoostParamsBuilder._load_tuned_params()
                if tuned_params:
                    params.update(tuned_params)
                    log(INFO, "[XGBoostParamsBuilder] Applied tuned parameters")
            except Exception as e:
                log(WARNING, f"[XGBoostParamsBuilder] Could not load tuned parameters: {e}")
                
        # Apply config manager parameters
        if config_manager:
            try:
                config_params = config_manager.get_model_params_dict()
                if config_params:
                    params.update(config_params)
                    log(INFO, "[XGBoostParamsBuilder] Applied ConfigManager parameters")
            except Exception as e:
                log(WARNING, f"[XGBoostParamsBuilder] Could not get ConfigManager parameters: {e}")
                
        # Apply direct overrides (highest priority)
        if overrides:
            params.update(overrides)
            log(INFO, f"[XGBoostParamsBuilder] Applied parameter overrides: {list(overrides.keys())}")
            
        # Validate and ensure required parameters
        params = XGBoostParamsBuilder._validate_params(params)
        
        return params
    
    @staticmethod
    def _load_tuned_params() -> Optional[Dict[str, Any]]:
        """Load previously tuned parameters if available."""
        # Try to import and load tuned parameters
        try:
            from src.models.use_tuned_params import load_tuned_params
            return load_tuned_params()
        except ImportError:
            log(WARNING, "[XGBoostParamsBuilder] Could not import tuned parameters module")
            return None
        except Exception as e:
            log(WARNING, f"[XGBoostParamsBuilder] Error loading tuned parameters: {e}")
            return None
    
    @staticmethod
    def _validate_params(params: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and fix parameter values."""
        # Ensure required parameters exist
        required_params = ['objective', 'num_class']
        for param in required_params:
            if param not in params:
                params[param] = XGBoostParamsBuilder.DEFAULT_PARAMS[param]
                log(WARNING, f"[XGBoostParamsBuilder] Missing required parameter {param}, using default")
                
        # Convert numeric parameters to appropriate types
        int_params = ['num_class', 'max_depth', 'min_child_weight', 'seed', 'verbosity']
        for param in int_params:
            if param in params:
                try:
                    params[param] = int(params[param])
                except (ValueError, TypeError):
                    log(WARNING, f"[XGBoostParamsBuilder] Invalid {param} value, using default")
                    params[param] = XGBoostParamsBuilder.DEFAULT_PARAMS.get(param, 0)
                    
        # Validate float parameters
        float_params = ['learning_rate', 'subsample', 'colsample_bytree', 'reg_alpha', 'reg_lambda', 'gamma']
        for param in float_params:
            if param in params:
                try:
                    params[param] = float(params[param])
                    # Ensure valid ranges
                    if param in ['subsample', 'colsample_bytree'] and not (0 < params[param] <= 1):
                        raise ValueError(f"{param} must be in (0, 1]")
                except (ValueError, TypeError):
                    log(WARNING, f"[XGBoostParamsBuilder] Invalid {param} value, using default")
                    params[param] = XGBoostParamsBuilder.DEFAULT_PARAMS.get(param, 1.0)
                    
        return params


# Utility functions for backward compatibility and convenience

def create_dmatrix(
    features: Features,
    labels: Optional[Labels] = None,
    handle_missing: bool = True
) -> xgb.DMatrix:
    """
    Convenience function for DMatrix creation.
    
    Simple wrapper around DMatrixFactory.create_dmatrix for common usage.
    """
    return DMatrixFactory.create_dmatrix(
        features=features,
        labels=labels,
        handle_missing=handle_missing
    )


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_pred_proba: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Convenience function for metrics calculation.
    
    Returns raw metrics dictionary for compatibility with existing code.
    """
    result = MetricsCalculator.calculate_classification_metrics(
        y_true=y_true,
        y_pred=y_pred,
        y_pred_proba=y_pred_proba
    )
    return result.raw_metrics


def build_xgb_params(
    config_manager=None,
    overrides: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Convenience function for parameter building.
    
    Simple wrapper around XGBoostParamsBuilder.build_params for common usage.
    """
    return XGBoostParamsBuilder.build_params(
        config_manager=config_manager,
        overrides=overrides
    ) 