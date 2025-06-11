"""
Federated learning strategies package.

This package contains specialized strategies for different model types
in federated learning scenarios.
"""

from .random_forest_strategy import (
    RandomForestFedAvg,
    RandomForestBagging,
    create_random_forest_strategy
)

__all__ = [
    'RandomForestFedAvg',
    'RandomForestBagging', 
    'create_random_forest_strategy'
]
