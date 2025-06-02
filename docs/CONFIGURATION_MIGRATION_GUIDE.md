# Configuration Migration Guide

This guide explains how to migrate from the legacy constants system to the new ConfigManager-based configuration system.

## Overview

The FL-CML-Pipeline has migrated from hardcoded constants and argument parsers to a centralized configuration management system using Hydra. This provides better maintainability, reproducibility, and flexibility.

## Before (Legacy System)

### Old way - Using hardcoded constants:
```python
# ❌ OLD - Don't use this anymore
from src.config.legacy_constants import BST_PARAMS, NUM_LOCAL_ROUND

# Using hardcoded parameters
model_params = BST_PARAMS
num_rounds = NUM_LOCAL_ROUND
```

### Old way - Using argument parsers:
```python
# ❌ OLD - Don't use this anymore  
from src.config.legacy_constants import client_args_parser, server_args_parser

args = client_args_parser()
train_method = args.train_method
```

## After (New ConfigManager System)

### New way - Using ConfigManager:
```python
# ✅ NEW - Use this approach
from src.config.config_manager import ConfigManager

# Initialize and load configuration
config_manager = ConfigManager()
config_manager.load_config()

# Get model parameters
model_params = config_manager.get_model_params_dict()
num_rounds = config_manager.config.model.num_local_rounds

# Get federated learning settings
train_method = config_manager.config.federated.train_method
pool_size = config_manager.config.federated.pool_size
```

### Alternative - Automatic initialization:
```python
# ✅ NEW - ConfigManager with automatic initialization
config_manager = ConfigManager()
config_manager.load_config()  # or load_config(experiment="dev")

# Access nested configuration
data_config = config_manager.config.data
model_config = config_manager.config.model
federated_config = config_manager.config.federated
```

## Configuration Structure

The new system uses YAML configuration files organized as follows:

```
configs/
├── base.yaml              # Base configuration
└── experiment/
    ├── dev.yaml          # Development settings
    ├── cyclic.yaml       # Cyclic training
    └── production.yaml   # Production settings
```

### Key Configuration Sections

#### Model Configuration (replaces BST_PARAMS)
```yaml
model:
  type: xgboost
  num_local_rounds: 20  # replaces NUM_LOCAL_ROUND
  params:               # replaces BST_PARAMS
    objective: "multi:softprob"
    num_class: 11
    eta: 0.05
    max_depth: 8
    # ... other XGBoost parameters
```

#### Federated Learning Configuration (replaces argument parsers)
```yaml
federated:
  train_method: "bagging"    # replaces --train-method
  pool_size: 5              # replaces --pool-size
  num_rounds: 5             # replaces --num-rounds
  num_clients_per_round: 5  # replaces --num-clients-per-round
  # ... other federated settings
```

## Migration Examples

### 1. Client Code Migration

**Before:**
```python
from src.config.legacy_constants import BST_PARAMS, client_args_parser

args = client_args_parser()
params = BST_PARAMS.copy()

# Use params and args...
```

**After:**
```python
from src.config.config_manager import ConfigManager

config_manager = ConfigManager()
config_manager.load_config()

params = config_manager.get_model_params_dict()
train_method = config_manager.config.federated.train_method
partitioner_type = config_manager.config.federated.partitioner_type
```

### 2. Server Code Migration

**Before:**
```python
from src.config.legacy_constants import server_args_parser

args = server_args_parser()
pool_size = args.pool_size
num_rounds = args.num_rounds
```

**After:**
```python
from src.config.config_manager import ConfigManager

config_manager = ConfigManager()
config_manager.load_config()

pool_size = config_manager.config.federated.pool_size
num_rounds = config_manager.config.federated.num_rounds
```

### 3. Model Parameter Access

**Before:**
```python
from src.config.legacy_constants import BST_PARAMS

bst = xgb.Booster(params=BST_PARAMS)
num_class = BST_PARAMS['num_class']
```

**After:**
```python
from src.config.config_manager import ConfigManager

config_manager = ConfigManager()
config_manager.load_config()

model_params = config_manager.get_model_params_dict()
bst = xgb.Booster(params=model_params)
num_class = model_params['num_class']
```

## Best Practices

### 1. Initialize ConfigManager Once
```python
# ✅ Good - Initialize once and reuse
config_manager = ConfigManager()
config_manager.load_config()

# Pass config_manager to functions that need it
def train_model(config_manager):
    params = config_manager.get_model_params_dict()
    # ...
```

### 2. Use Experiment Configurations
```python
# ✅ Load specific experiment configuration
config_manager = ConfigManager()
config_manager.load_config(experiment="dev")  # For development
config_manager.load_config(experiment="production")  # For production
```

### 3. Override Configuration Programmatically
```python
# ✅ Override specific values
config_manager = ConfigManager()
config_manager.load_config()

# Override for testing
config_manager.config.model.num_local_rounds = 5
config_manager.config.federated.num_rounds = 2
```

## Testing with ConfigManager

### Unit Tests
```python
def test_model_training():
    config_manager = ConfigManager()
    config_manager.load_config(experiment="dev")  # Use dev config for testing
    
    model_params = config_manager.get_model_params_dict()
    assert model_params['num_class'] == 11
    # ... rest of test
```

### Integration Tests
```python
def test_federated_setup():
    config_manager = ConfigManager()
    config_manager.load_config()
    
    # Test with actual configuration values
    pool_size = config_manager.config.federated.pool_size
    assert pool_size > 0
```

## Troubleshooting

### Common Issues and Solutions

1. **Configuration not loaded error:**
   ```python
   # ❌ Problem
   config_manager = ConfigManager()
   params = config_manager.get_model_params_dict()  # RuntimeError
   
   # ✅ Solution
   config_manager = ConfigManager()
   config_manager.load_config()  # Load first!
   params = config_manager.get_model_params_dict()
   ```

2. **Missing experiment configuration:**
   ```python
   # ❌ Problem
   config_manager.load_config(experiment="nonexistent")
   
   # ✅ Solution - Use existing experiment or fall back to base
   config_manager.load_config(experiment="dev")  # or omit for base config
   ```

3. **Import errors for legacy constants:**
   ```python
   # ❌ Problem - legacy_constants.py cleaned up
   from src.config.legacy_constants import BST_PARAMS  # ImportError or empty
   
   # ✅ Solution - Use ConfigManager
   from src.config.config_manager import ConfigManager
   config_manager = ConfigManager()
   config_manager.load_config()
   model_params = config_manager.get_model_params_dict()
   ```

## Benefits of the New System

1. **Centralized Configuration**: All settings in one place (YAML files)
2. **Environment-specific Settings**: Different configs for dev/prod
3. **Type Safety**: Validated configuration with dataclasses
4. **Reproducibility**: Configuration saved with experiment outputs
5. **Flexibility**: Easy to override settings programmatically
6. **Maintainability**: No more scattered hardcoded constants

## File Changes Summary

### Files Cleaned Up
- `src/config/legacy_constants.py` - Deprecated constants and parsers removed
- Test files updated to use ConfigManager

### Files Updated  
- `src/federated/utils.py` - Uses ConfigManager for model parameters
- `src/federated/client_utils.py` - Uses ConfigManager for model parameters
- `src/models/use_tuned_params.py` - Uses ConfigManager for defaults
- Entry points (`run.py`, `server.py`, `client.py`, `sim.py`) - Use ConfigManager

### New Configuration Files
- `configs/base.yaml` - Base configuration with all settings
- `configs/experiment/*.yaml` - Experiment-specific overrides

This migration ensures the codebase is more maintainable and follows modern configuration management best practices. 