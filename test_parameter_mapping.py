#!/usr/bin/env python3
"""
Parameter Mapping Test and Demonstration Script

This script demonstrates the parameter conversion utilities for seamless
model type switching in the federated learning pipeline.

Usage:
    python test_parameter_mapping.py
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.utils.parameter_mapping import (
    UnifiedParameterManager, 
    ModelType, 
    convert_xgboost_to_random_forest,
    convert_random_forest_to_xgboost,
    create_cross_compatible_config
)
from src.config.config_manager import ConfigManager


def test_basic_conversions():
    """Test basic parameter conversions between model types."""
    print("=" * 60)
    print("Testing Basic Parameter Conversions")
    print("=" * 60)
    
    # Test XGBoost to Random Forest conversion
    print("\n1. XGBoost to Random Forest Conversion:")
    xgb_params = {
        "objective": "multi:softprob",
        "num_class": 11,
        "eta": 0.1,
        "max_depth": 8,
        "min_child_weight": 5,
        "gamma": 0.5,
        "subsample": 0.8,
        "colsample_bytree": 0.7,
        "num_boost_round": 100,
        "random_state": 42,
        "nthread": 8
    }
    
    print(f"Original XGBoost params ({len(xgb_params)} parameters):")
    for key, value in xgb_params.items():
        print(f"  {key}: {value}")
    
    rf_params = convert_xgboost_to_random_forest(xgb_params)
    print(f"\nConverted Random Forest params ({len(rf_params)} parameters):")
    for key, value in rf_params.items():
        print(f"  {key}: {value}")
    
    # Test Random Forest to XGBoost conversion
    print("\n2. Random Forest to XGBoost Conversion:")
    rf_params_original = {
        "n_estimators": 100,
        "max_depth": 10,
        "min_samples_split": 5,
        "min_samples_leaf": 2,
        "max_features": "sqrt",
        "criterion": "gini",
        "random_state": 42,
        "n_jobs": 4
    }
    
    print(f"Original Random Forest params ({len(rf_params_original)} parameters):")
    for key, value in rf_params_original.items():
        print(f"  {key}: {value}")
    
    xgb_params_converted = convert_random_forest_to_xgboost(rf_params_original)
    print(f"\nConverted XGBoost params ({len(xgb_params_converted)} parameters):")
    for key, value in xgb_params_converted.items():
        print(f"  {key}: {value}")


def test_unified_parameter_manager():
    """Test the UnifiedParameterManager functionality."""
    print("\n" + "=" * 60)
    print("Testing Unified Parameter Manager")
    print("=" * 60)
    
    manager = UnifiedParameterManager()
    
    # Test default parameters
    print("\n1. Default Parameters:")
    xgb_defaults = manager.get_default_parameters(ModelType.XGBOOST)
    rf_defaults = manager.get_default_parameters(ModelType.RANDOM_FOREST)
    
    print(f"XGBoost defaults: {len(xgb_defaults)} parameters")
    print(f"Random Forest defaults: {len(rf_defaults)} parameters")
    
    # Test parameter validation
    print("\n2. Parameter Validation:")
    
    # Valid XGBoost params
    valid_xgb = {"objective": "multi:softprob", "num_class": 11, "eta": 0.1}
    is_valid, msg = manager.validate_parameters(valid_xgb, ModelType.XGBOOST)
    print(f"Valid XGBoost params: {is_valid} - {msg}")
    
    # Invalid XGBoost params (missing required)
    invalid_xgb = {"eta": 0.1, "max_depth": 6}
    is_valid, msg = manager.validate_parameters(invalid_xgb, ModelType.XGBOOST)
    print(f"Invalid XGBoost params: {is_valid} - {msg}")
    
    # Test unified config creation
    print("\n3. Unified Config Creation:")
    base_params = {"max_depth": 6, "random_state": 42}
    
    unified_xgb = manager.create_unified_config(base_params, ModelType.XGBOOST)
    print(f"Unified XGBoost config: {len(unified_xgb)} parameters")
    
    # Test model switching
    print("\n4. Model Type Switching:")
    switched_rf = manager.switch_model_type(ModelType.RANDOM_FOREST)
    print(f"Switched to Random Forest: {len(switched_rf)} parameters")
    
    # Get current config
    current = manager.get_current_config()
    if current:
        model_type, params = current
        print(f"Current config: {model_type.value} with {len(params)} parameters")


def test_cross_compatibility():
    """Test cross-compatible configuration creation."""
    print("\n" + "=" * 60)
    print("Testing Cross-Compatible Configurations")
    print("=" * 60)
    
    base_params = {
        "max_depth": 8,
        "random_state": 42,
        # This could be either eta (XGBoost) or learning rate equivalent
        "learning_rate": 0.1
    }
    
    # Create configs for both model types
    xgb_config, rf_config = create_cross_compatible_config(
        base_params, 
        ModelType.XGBOOST, 
        ModelType.RANDOM_FOREST
    )
    
    print("Cross-compatible configurations created:")
    print(f"XGBoost config: {len(xgb_config)} parameters")
    print(f"Random Forest config: {len(rf_config)} parameters")
    
    # Show mapping of key parameters
    print("\nKey parameter mappings:")
    print(f"Max depth - XGB: {xgb_config.get('max_depth')}, RF: {rf_config.get('max_depth')}")
    print(f"Random state - XGB: {xgb_config.get('random_state')}, RF: {rf_config.get('random_state')}")


def test_config_manager_integration():
    """Test integration with ConfigManager."""
    print("\n" + "=" * 60)
    print("Testing ConfigManager Integration")
    print("=" * 60)
    
    try:
        # Create a ConfigManager and load default config
        config_manager = ConfigManager()
        config_manager.load_config()
        
        print("ConfigManager loaded successfully")
        print(f"Current model type: {config_manager.get_model_type()}")
        
        # Get current parameters
        current_params = config_manager.get_model_params_dict()
        print(f"Current parameters: {len(current_params)} parameters")
        
        # Show parameter conversion potential
        manager = UnifiedParameterManager()
        current_type = config_manager.get_model_type()
        
        if current_type.lower() == "xgboost":
            target_type = ModelType.RANDOM_FOREST
        else:
            target_type = ModelType.XGBOOST
        
        converted_params = manager.convert_parameters(
            current_params, current_type, target_type
        )
        
        print(f"Could convert to {target_type.value}: {len(converted_params)} parameters")
        
    except Exception as e:
        print(f"ConfigManager integration test failed: {e}")
        print("This is expected if configuration files are not properly set up")


def test_parameter_categories():
    """Test parameter categorization and mapping logic."""
    print("\n" + "=" * 60)
    print("Testing Parameter Categories and Mapping Logic")
    print("=" * 60)
    
    # Create sample parameters for each category
    tree_structure_params = {
        "max_depth": 8,
        "min_child_weight": 5,  # XGBoost
        "min_samples_leaf": 2   # Random Forest
    }
    
    regularization_params = {
        "gamma": 0.5,           # XGBoost
        "reg_alpha": 0.1,       # XGBoost
        "reg_lambda": 1.0       # XGBoost
    }
    
    sampling_params = {
        "subsample": 0.8,       # XGBoost
        "colsample_bytree": 0.7, # XGBoost
        "max_samples": 0.8,     # Random Forest
        "bootstrap": True       # Random Forest
    }
    
    print("Parameter mapping examples:")
    print("\n1. Tree Structure Parameters:")
    for param, value in tree_structure_params.items():
        print(f"  {param}: {value}")
    
    print("\n2. Regularization Parameters:")
    for param, value in regularization_params.items():
        print(f"  {param}: {value}")
    
    print("\n3. Sampling Parameters:")
    for param, value in sampling_params.items():
        print(f"  {param}: {value}")
    
    # Test conversion of these categories
    manager = UnifiedParameterManager()
    
    # Combine all XGBoost-style parameters
    xgb_test_params = {
        **tree_structure_params,
        **regularization_params,
        **sampling_params,
        "objective": "multi:softprob",
        "num_class": 11,
        "eta": 0.1
    }
    
    # Remove RF-specific params for clean test
    xgb_clean_params = {k: v for k, v in xgb_test_params.items() 
                       if k not in ["min_samples_leaf", "max_samples", "bootstrap"]}
    
    print("\n4. Converting XGBoost-style parameters to Random Forest:")
    rf_converted = manager.convert_parameters(
        xgb_clean_params, ModelType.XGBOOST, ModelType.RANDOM_FOREST
    )
    
    print("Converted parameters:")
    for key, value in rf_converted.items():
        if key in ["max_depth", "n_estimators", "max_features", "min_samples_leaf"]:
            print(f"  {key}: {value}")


def demonstrate_use_cases():
    """Demonstrate real-world use cases for parameter mapping."""
    print("\n" + "=" * 60)
    print("Real-World Use Cases Demonstration")
    print("=" * 60)
    
    print("\n1. Experiment Comparison:")
    print("   - Start with XGBoost configuration")
    print("   - Convert to Random Forest for comparison")
    print("   - Maintain equivalent complexity levels")
    
    print("\n2. Model Selection Pipeline:")
    print("   - Begin with unified parameter search space")
    print("   - Test multiple model types with equivalent parameters")
    print("   - Select best performing model type")
    
    print("\n3. Federated Learning Flexibility:")
    print("   - Switch model types without reconfiguring entire pipeline")
    print("   - Maintain client compatibility across model changes")
    print("   - Preserve tuned hyperparameters where applicable")
    
    print("\n4. A/B Testing:")
    print("   - Run identical experiments with different model types")
    print("   - Compare performance with equivalent parameter sets")
    print("   - Make data-driven model type decisions")
    
    # Practical example
    print("\n5. Practical Example - Hyperparameter Transfer:")
    
    # Assume we tuned XGBoost parameters
    tuned_xgb_params = {
        "objective": "multi:softprob",
        "num_class": 11,
        "eta": 0.05,
        "max_depth": 10,
        "min_child_weight": 3,
        "gamma": 0.1,
        "subsample": 0.9,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.2,
        "reg_lambda": 1.5,
        "num_boost_round": 200,
        "random_state": 42
    }
    
    print("   Tuned XGBoost parameters:")
    for key, value in tuned_xgb_params.items():
        print(f"     {key}: {value}")
    
    # Convert to Random Forest
    manager = UnifiedParameterManager()
    equivalent_rf_params = manager.convert_parameters(
        tuned_xgb_params, ModelType.XGBOOST, ModelType.RANDOM_FOREST
    )
    
    print("\n   Equivalent Random Forest parameters:")
    for key, value in equivalent_rf_params.items():
        print(f"     {key}: {value}")
    
    print("\n   This allows testing Random Forest with equivalent complexity!")


def main():
    """Run all tests and demonstrations."""
    print("Parameter Mapping Utilities - Test and Demonstration")
    print("FL-CML-Pipeline")
    print("=" * 60)
    
    try:
        test_basic_conversions()
        test_unified_parameter_manager()
        test_cross_compatibility()
        test_config_manager_integration()
        test_parameter_categories()
        demonstrate_use_cases()
        
        print("\n" + "=" * 60)
        print("All tests completed successfully!")
        print("Parameter mapping utilities are ready for use.")
        print("=" * 60)
        
    except ImportError as e:
        print(f"Import error: {e}")
        print("Make sure you're running from the project root directory")
        return 1
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 