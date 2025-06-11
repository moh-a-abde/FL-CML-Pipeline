# Parameter Mapping Utilities Guide

This guide explains how to use the parameter mapping utilities for seamless model type switching in the FL-CML-Pipeline.

## Overview

The parameter mapping utilities provide a unified interface for converting parameters between different model types (XGBoost, Random Forest) while preserving equivalent functionality and complexity. This enables:

- **Seamless Model Switching**: Change model types without manual parameter reconfiguration
- **Cross-Model Experimentation**: Compare different model types with equivalent parameter sets
- **Hyperparameter Transfer**: Apply tuned parameters from one model type to another
- **Unified Configuration Management**: Manage parameters across model types with a single interface

## Core Components

### 1. UnifiedParameterManager

The main class for parameter conversion and management.

```python
from src.utils.parameter_mapping import UnifiedParameterManager, ModelType

# Create manager
manager = UnifiedParameterManager()

# Convert parameters between model types
xgb_params = {"eta": 0.1, "max_depth": 8, "objective": "multi:softprob", "num_class": 11}
rf_params = manager.convert_parameters(xgb_params, ModelType.XGBOOST, ModelType.RANDOM_FOREST)

# Get default parameters for any model type
defaults = manager.get_default_parameters(ModelType.RANDOM_FOREST)

# Create unified configuration
unified_config = manager.create_unified_config(base_params, ModelType.XGBOOST)

# Switch model types while preserving parameters
new_params = manager.switch_model_type(ModelType.RANDOM_FOREST)
```

### 2. Parameter Mappers

Individual mappers for each model type handle specific conversion logic.

```python
from src.utils.parameter_mapping import XGBoostParameterMapper, RandomForestParameterMapper

# XGBoost mapper
xgb_mapper = XGBoostParameterMapper()
xgb_defaults = xgb_mapper.get_default_params()
rf_params = xgb_mapper.map_to_target(xgb_params, ModelType.RANDOM_FOREST)

# Random Forest mapper  
rf_mapper = RandomForestParameterMapper()
rf_defaults = rf_mapper.get_default_params()
xgb_params = rf_mapper.map_to_target(rf_params, ModelType.XGBOOST)
```

### 3. Convenience Functions

Quick conversion functions for common use cases.

```python
from src.utils.parameter_mapping import (
    convert_xgboost_to_random_forest,
    convert_random_forest_to_xgboost,
    create_cross_compatible_config
)

# Quick conversions
rf_params = convert_xgboost_to_random_forest(xgb_params)
xgb_params = convert_random_forest_to_xgboost(rf_params)

# Cross-compatible configurations
xgb_config, rf_config = create_cross_compatible_config(
    base_params, ModelType.XGBOOST, ModelType.RANDOM_FOREST
)
```

## Parameter Mapping Logic

### Tree Structure Parameters

| Concept | XGBoost | Random Forest | Mapping Logic |
|---------|---------|---------------|---------------|
| Tree Depth | `max_depth` | `max_depth` | Direct mapping |
| Min Samples | `min_child_weight` | `min_samples_leaf` | Approximate conversion |
| Split Control | `gamma` | `min_impurity_decrease` | Approximate conversion |

### Sampling Parameters

| Concept | XGBoost | Random Forest | Mapping Logic |
|---------|---------|---------------|---------------|
| Row Sampling | `subsample` | `max_samples` | Direct mapping |
| Feature Sampling | `colsample_bytree` | `max_features` | Categorical mapping |
| Bootstrap | N/A | `bootstrap` | Default true for RF |

### Regularization Parameters

| Concept | XGBoost | Random Forest | Mapping Logic |
|---------|---------|---------------|---------------|
| L1 Regularization | `reg_alpha` | N/A | Not directly mappable |
| L2 Regularization | `reg_lambda` | N/A | Not directly mappable |
| Tree Regularization | `gamma` | `min_impurity_decrease` | Approximate |

### Learning Parameters

| Concept | XGBoost | Random Forest | Mapping Logic |
|---------|---------|---------------|---------------|
| Learning Rate | `eta` | N/A | Not applicable |
| Number of Rounds | `num_boost_round` | `n_estimators` | Scale conversion |

## Usage Examples

### Example 1: Basic Parameter Conversion

```python
# Start with XGBoost parameters
xgb_params = {
    "objective": "multi:softprob",
    "num_class": 11,
    "eta": 0.1,
    "max_depth": 8,
    "min_child_weight": 5,
    "subsample": 0.8,
    "colsample_bytree": 0.7,
    "num_boost_round": 100,
    "random_state": 42
}

# Convert to Random Forest
from src.utils.parameter_mapping import convert_xgboost_to_random_forest
rf_params = convert_xgboost_to_random_forest(xgb_params)

print(rf_params)
# Output:
# {
#     'max_depth': 8,
#     'min_samples_leaf': 5,
#     'max_samples': 0.8,
#     'max_features': 0.7,
#     'n_estimators': 50,
#     'random_state': 42,
#     'criterion': 'gini',
#     'bootstrap': True,
#     'oob_score': False,
#     'class_weight': 'balanced',
#     'min_samples_split': 5,
#     'n_jobs': 16
# }
```

### Example 2: ConfigManager Integration

```python
from src.config.config_manager import ConfigManager
from src.utils.parameter_mapping import UnifiedParameterManager, ModelType

# Load configuration
config_manager = ConfigManager()
config_manager.load_config(experiment="bagging")

# Get current parameters
current_params = config_manager.get_model_params_dict()
current_type = config_manager.get_model_type()

# Convert to different model type
param_manager = UnifiedParameterManager()
converted_params = param_manager.convert_parameters(
    current_params, current_type, ModelType.RANDOM_FOREST
)

# Use converted parameters in federated learning
# (Parameters are now compatible with Random Forest models)
```

### Example 3: Hyperparameter Transfer

```python
# Assume you've tuned XGBoost parameters using Ray Tune
tuned_xgb_params = {
    "eta": 0.05,
    "max_depth": 10,
    "min_child_weight": 3,
    "gamma": 0.1,
    "subsample": 0.9,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.2,
    "reg_lambda": 1.5,
    "num_boost_round": 200
}

# Transfer to Random Forest for comparison
manager = UnifiedParameterManager()
equivalent_rf_params = manager.convert_parameters(
    tuned_xgb_params, ModelType.XGBOOST, ModelType.RANDOM_FOREST
)

# Now you can run equivalent Random Forest experiments
# without starting hyperparameter tuning from scratch
```

### Example 4: A/B Testing with Model Types

```python
from src.utils.parameter_mapping import create_cross_compatible_config

# Define base parameters for experiment
base_params = {
    "max_depth": 8,
    "random_state": 42,
    # Parameters that make sense for both models
}

# Create equivalent configurations for both model types
xgb_config, rf_config = create_cross_compatible_config(
    base_params, ModelType.XGBOOST, ModelType.RANDOM_FOREST
)

# Run A/B test
# - Group A: Uses XGBoost with xgb_config
# - Group B: Uses Random Forest with rf_config
# - Compare results with equivalent complexity
```

### Example 5: Federated Learning Model Switching

```python
from src.federated.generic_client import GenericFederatedClient
from src.config.config_manager import ConfigManager

# Load configuration for XGBoost
config_manager = ConfigManager()
config_manager.load_config(experiment="bagging")  # XGBoost config

# Create federated client
client = GenericFederatedClient(
    client_id=0,
    config_manager=config_manager
)

# Later, switch to Random Forest without changing federated setup
# The generic client automatically handles parameter conversion
config_manager.update_config_value("model.type", "random_forest")

# Convert parameters automatically
param_manager = UnifiedParameterManager()
current_params = config_manager.get_model_params_dict()
rf_params = param_manager.convert_parameters(
    current_params, ModelType.XGBOOST, ModelType.RANDOM_FOREST
)

# Update parameters in config
for key, value in rf_params.items():
    config_manager.update_config_value(f"model.params.{key}", value)
```

## Advanced Features

### Parameter Validation

```python
manager = UnifiedParameterManager()

# Validate parameters for specific model type
params = {"n_estimators": 100, "max_depth": 10}
is_valid, message = manager.validate_parameters(params, ModelType.RANDOM_FOREST)

if not is_valid:
    print(f"Parameter validation failed: {message}")
```

### Configuration Presets

```python
from src.config.parameter_integration import ConfigurationPreset

# Get preset configurations for quick testing
quick_test_xgb = ConfigurationPreset.get_quick_test_config(ModelType.XGBOOST)
production_rf = ConfigurationPreset.get_production_config(ModelType.RANDOM_FOREST)
tuning_ready_xgb = ConfigurationPreset.get_tuning_config(ModelType.XGBOOST)
```

### Enhanced ConfigManager

```python
from src.config.parameter_integration import ParameterIntegratedConfigManager

# Use enhanced config manager with parameter mapping
config_manager = ParameterIntegratedConfigManager()
config_manager.load_config(experiment="bagging")

# Switch model types seamlessly
new_params = config_manager.switch_model_type(ModelType.RANDOM_FOREST)

# Get cross-compatible parameters
compatible_params = config_manager.get_cross_compatible_params(ModelType.XGBOOST)

# Restore original configuration
config_manager.restore_original_config()
```

## Best Practices

### 1. Parameter Equivalence

When converting parameters, understand that:
- Some parameters have direct equivalents (e.g., `max_depth`)
- Others require approximation (e.g., `min_child_weight` → `min_samples_leaf`)
- Some are model-specific and don't translate (e.g., `eta` for XGBoost)

### 2. Validation

Always validate converted parameters:

```python
# Convert parameters
converted_params = manager.convert_parameters(source_params, source_type, target_type)

# Validate before use
is_valid, message = manager.validate_parameters(converted_params, target_type)
if not is_valid:
    print(f"Validation failed: {message}")
    # Handle validation failure
```

### 3. Experimentation Workflow

1. Start with one model type and tune parameters
2. Convert to other model types for comparison
3. Fine-tune converted parameters if needed
4. Select best performing model type
5. Use final parameters for production

### 4. Federated Learning Integration

- Use `GenericFederatedClient` for automatic model type handling
- Convert parameters at the configuration level, not client level
- Ensure all clients use the same model type in a given round

## Limitations and Considerations

### Parameter Mapping Limitations

1. **Not Perfect Equivalence**: Parameter conversion provides reasonable approximations, not perfect equivalents
2. **Model-Specific Features**: Some parameters (like XGBoost's boosting) don't have Random Forest equivalents
3. **Performance Differences**: Converted parameters may need fine-tuning for optimal performance

### Performance Considerations

1. **Converted Parameters**: May not be optimal for the target model type
2. **Fine-Tuning**: Consider additional tuning after conversion
3. **Validation**: Always validate performance with converted parameters

### Model-Specific Behaviors

1. **XGBoost**: Gradient boosting with complex regularization
2. **Random Forest**: Ensemble of decision trees with different characteristics
3. **Feature Importance**: Different calculation methods between models

## Testing the Implementation

Run the test script to verify functionality:

```bash
python test_parameter_mapping.py
```

This will demonstrate:
- Basic parameter conversions
- Unified parameter manager functionality
- Cross-compatible configuration creation
- ConfigManager integration
- Real-world use cases

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure you're running from the project root directory
2. **Parameter Validation Failures**: Check that required parameters are present
3. **Configuration Loading Errors**: Verify config files exist and are properly formatted

### Debug Mode

Enable debug logging to see parameter conversion details:

```python
import logging
logging.getLogger('src.utils.parameter_mapping').setLevel(logging.DEBUG)
```

## Conclusion

The parameter mapping utilities provide a powerful foundation for seamless model type switching in the FL-CML-Pipeline. They enable:

- Flexible experimentation across model types
- Efficient hyperparameter transfer
- Simplified federated learning configuration
- Consistent parameter management

Use these utilities to build more flexible and maintainable machine learning pipelines that can adapt to different model requirements while preserving configuration consistency. 