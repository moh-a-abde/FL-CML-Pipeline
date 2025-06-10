"""
Unit tests for shared utilities module (Phase 3: Code Deduplication)

Tests the centralized DMatrix creation, metrics calculation, and XGBoost
parameter building functionality to ensure reliability before migration.
"""

import pytest
import numpy as np
import pandas as pd
import xgboost as xgb
from unittest.mock import Mock, patch

# Import the modules we're testing
from src.core.shared_utils import (
    DMatrixFactory, MetricsCalculator, XGBoostParamsBuilder,
    MetricsResult, create_dmatrix, calculate_metrics, build_xgb_params
)


class TestDMatrixFactory:
    """Test cases for DMatrixFactory class."""
    
    def setup_method(self):
        """Set up test data."""
        # Create sample data
        self.features = np.random.rand(100, 10)
        self.labels = np.random.randint(0, 3, 100)
        self.weights = np.random.rand(100)
        
        # Create DataFrame version
        self.features_df = pd.DataFrame(
            self.features, 
            columns=[f'feature_{i}' for i in range(10)]
        )
        self.labels_series = pd.Series(self.labels)
    
    def test_create_basic_dmatrix(self):
        """Test basic DMatrix creation with minimal parameters."""
        dmatrix = DMatrixFactory.create_dmatrix(
            features=self.features,
            labels=self.labels,
            log_details=False  # Reduce test output
        )
        
        assert isinstance(dmatrix, xgb.DMatrix)
        assert dmatrix.num_row() == 100
        assert dmatrix.num_col() == 10
        assert len(dmatrix.get_label()) == 100
    
    def test_create_dmatrix_without_labels(self):
        """Test DMatrix creation for prediction (no labels)."""
        dmatrix = DMatrixFactory.create_dmatrix(
            features=self.features,
            labels=None,
            log_details=False
        )
        
        assert isinstance(dmatrix, xgb.DMatrix)
        assert dmatrix.num_row() == 100
        assert dmatrix.num_col() == 10


class TestMetricsCalculator:
    """Test cases for MetricsCalculator class."""
    
    def setup_method(self):
        """Set up test data."""
        self.y_true = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0])
        self.y_pred = np.array([0, 1, 1, 0, 1, 2, 0, 2, 2, 1])
        self.y_pred_proba = np.array([
            [0.8, 0.1, 0.1],
            [0.2, 0.7, 0.1],
            [0.1, 0.6, 0.3],
            [0.9, 0.05, 0.05],
            [0.3, 0.6, 0.1],
            [0.1, 0.2, 0.7],
            [0.8, 0.1, 0.1],
            [0.1, 0.1, 0.8],
            [0.2, 0.3, 0.5],
            [0.1, 0.8, 0.1]
        ])
    
    def test_calculate_basic_metrics(self):
        """Test basic metrics calculation."""
        result = MetricsCalculator.calculate_classification_metrics(
            y_true=self.y_true,
            y_pred=self.y_pred
        )
        
        assert isinstance(result, MetricsResult)
        assert 0.0 <= result.accuracy <= 1.0
        assert 0.0 <= result.precision <= 1.0
        assert 0.0 <= result.recall <= 1.0
        assert 0.0 <= result.f1_score <= 1.0
    
    def test_aggregate_client_metrics(self):
        """Test client metrics aggregation."""
        # Simulate metrics from 3 clients
        client_metrics = [
            (50, {"accuracy": 0.8, "precision": 0.75, "f1": 0.77}),
            (30, {"accuracy": 0.7, "precision": 0.72, "f1": 0.71}),
            (20, {"accuracy": 0.9, "precision": 0.88, "f1": 0.89})
        ]
        
        aggregated = MetricsCalculator.aggregate_client_metrics(client_metrics, log_details=False)
        
        # Check that all metrics are aggregated
        assert "accuracy" in aggregated
        assert "precision" in aggregated
        assert "f1" in aggregated


class TestXGBoostParamsBuilder:
    """Test cases for XGBoostParamsBuilder class."""
    
    def test_build_default_params(self):
        """Test building parameters with defaults only."""
        params = XGBoostParamsBuilder.build_params()
        
        # Check required parameters exist
        assert "objective" in params
        assert "num_class" in params
        assert params["objective"] == "multi:softprob"
        assert params["num_class"] == 11
    
    def test_build_params_with_overrides(self):
        """Test parameter building with overrides."""
        overrides = {
            "max_depth": 8,
            "learning_rate": 0.1,
            "custom_param": "test_value"
        }
        
        params = XGBoostParamsBuilder.build_params(overrides=overrides)
        
        # Check overrides are applied
        assert params["max_depth"] == 8
        assert params["learning_rate"] == 0.1
        assert params["custom_param"] == "test_value"


class TestConvenienceFunctions:
    """Test convenience wrapper functions."""
    
    def setup_method(self):
        """Set up test data."""
        self.features = np.random.rand(50, 5)
        self.labels = np.random.randint(0, 3, 50)
        self.y_true = np.array([0, 1, 2, 0, 1])
        self.y_pred = np.array([0, 1, 1, 0, 2])
    
    def test_create_dmatrix_convenience(self):
        """Test convenience create_dmatrix function."""
        dmatrix = create_dmatrix(
            features=self.features,
            labels=self.labels
        )
        
        assert isinstance(dmatrix, xgb.DMatrix)
        assert dmatrix.num_row() == 50
        assert dmatrix.num_col() == 5
    
    def test_calculate_metrics_convenience(self):
        """Test convenience calculate_metrics function."""
        metrics = calculate_metrics(
            y_true=self.y_true,
            y_pred=self.y_pred
        )
        
        assert isinstance(metrics, dict)
        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1" in metrics
    
    def test_build_xgb_params_convenience(self):
        """Test convenience build_xgb_params function."""
        params = build_xgb_params()
        
        assert isinstance(params, dict)
        assert "objective" in params
        assert "num_class" in params
        assert params["objective"] == "multi:softprob"


if __name__ == "__main__":
    pytest.main([__file__])
