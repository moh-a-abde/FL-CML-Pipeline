# Phase 3 Implementation Guide: Code Deduplication

## 🚀 Quick Start Guide

This guide provides practical instructions for implementing Phase 3 code deduplication using the centralized shared utilities.

**Status**: ✅ **READY TO BEGIN** - Shared utilities created and tested

---

## 📦 Available Shared Utilities

### 1. DMatrixFactory - Centralized DMatrix Creation

**Location**: `src.core.shared_utils.DMatrixFactory`

**Key Methods**:
- `create_dmatrix()` - Main DMatrix creation with full options
- `create_weighted_dmatrix()` - Create weighted DMatrix from existing one

**Usage Examples**:
```python
from src.core.shared_utils import DMatrixFactory

# Basic usage (replaces most current patterns)
dmatrix = DMatrixFactory.create_dmatrix(
    features=train_features,
    labels=train_labels,
    handle_missing=True
)

# For prediction (no labels)
predict_dmatrix = DMatrixFactory.create_dmatrix(
    features=test_features,
    labels=None
)

# With sample weights (federated learning pattern)
weighted_dmatrix = DMatrixFactory.create_weighted_dmatrix(
    base_dmatrix=existing_dmatrix,
    weights=sample_weights
)
```

### 2. MetricsCalculator - Centralized Metrics Calculation

**Location**: `src.core.shared_utils.MetricsCalculator`

**Key Methods**:
- `calculate_classification_metrics()` - Comprehensive metrics calculation
- `aggregate_client_metrics()` - Federated learning metrics aggregation

**Usage Examples**:
```python
from src.core.shared_utils import MetricsCalculator

# Basic metrics calculation (replaces scattered sklearn calls)
result = MetricsCalculator.calculate_classification_metrics(
    y_true=y_true,
    y_pred=y_pred_labels,
    y_pred_proba=y_pred_proba  # Optional
)

# Access individual metrics
accuracy = result.accuracy
precision = result.precision
f1_score = result.f1_score
raw_dict = result.raw_metrics  # For compatibility

# Federated learning aggregation
aggregated = MetricsCalculator.aggregate_client_metrics(
    client_metrics=[(num_examples, metrics_dict), ...]
)
```

### 3. XGBoostParamsBuilder - Centralized Parameter Management

**Location**: `src.core.shared_utils.XGBoostParamsBuilder`

**Key Methods**:
- `build_params()` - Build parameters with priority handling

**Usage Examples**:
```python
from src.core.shared_utils import XGBoostParamsBuilder

# Use ConfigManager parameters (highest priority)
params = XGBoostParamsBuilder.build_params(
    config_manager=config_manager
)

# With specific overrides
params = XGBoostParamsBuilder.build_params(
    config_manager=config_manager,
    overrides={"max_depth": 8, "learning_rate": 0.1}
)

# With tuned parameters
params = XGBoostParamsBuilder.build_params(
    config_manager=config_manager,
    use_tuned=True
)
```

---

## 🔄 Migration Patterns

### Pattern 1: Replace DMatrix Creation

**BEFORE** (Scattered across modules):
```python
# Various inconsistent patterns found:
train_data = xgb.DMatrix(train_features, label=train_labels, missing=np.nan)
test_data = xgb.DMatrix(test_features, label=test_labels)
dmatrix = xgb.DMatrix(x, label=y)
```

**AFTER** (Centralized):
```python
from src.core.shared_utils import DMatrixFactory

train_data = DMatrixFactory.create_dmatrix(train_features, train_labels)
test_data = DMatrixFactory.create_dmatrix(test_features, test_labels)
dmatrix = DMatrixFactory.create_dmatrix(x, y)
```

### Pattern 2: Replace Metrics Calculation

**BEFORE** (Repeated calculations):
```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

precision = precision_score(y_true, y_pred_labels, average='weighted', zero_division=0)
recall = recall_score(y_true, y_pred_labels, average='weighted', zero_division=0) 
f1 = f1_score(y_true, y_pred_labels, average='weighted', zero_division=0)
accuracy = accuracy_score(y_true, y_pred_labels)
```

**AFTER** (Centralized):
```python
from src.core.shared_utils import MetricsCalculator

result = MetricsCalculator.calculate_classification_metrics(y_true, y_pred_labels)
precision = result.precision
recall = result.recall  
f1 = result.f1_score
accuracy = result.accuracy
```

### Pattern 3: Replace Parameter Building

**BEFORE** (Scattered parameter logic):
```python
# Different parameter handling across modules
params = {
    'objective': 'multi:softprob',
    'num_class': 11,
    # ... various patterns
}
if config_manager:
    params.update(config_manager.get_model_params_dict())
```

**AFTER** (Centralized):
```python
from src.core.shared_utils import XGBoostParamsBuilder

params = XGBoostParamsBuilder.build_params(config_manager=config_manager)
```

---

## 📋 File-by-File Migration Plan

### Step 1: Update `src/core/dataset.py` ✅ READY

**Target Function**: `transform_dataset_to_dmatrix()`

**Current Code Location**: Lines 476-500
```python
def transform_dataset_to_dmatrix(data, processor: FeatureProcessor = None, is_training: bool = False):
    # ... preprocessing logic ...
    
    # REPLACE THIS SECTION:
    dmatrix = xgb.DMatrix(x, label=y)
    
    # WITH:
    from src.core.shared_utils import DMatrixFactory
    dmatrix = DMatrixFactory.create_dmatrix(
        features=x,
        labels=y,
        handle_missing=True
    )
```

### Step 2: Update `src/federated/sim.py` ✅ READY

**Target Function**: `get_client_fn()` inline DMatrix creation

**Current Code Location**: Lines 58-94
```python
# REPLACE:
train_dmatrix = xgb.DMatrix(x_train, label=y_train)
valid_dmatrix = xgb.DMatrix(x_valid, label=y_valid)

# WITH:
from src.core.shared_utils import DMatrixFactory
train_dmatrix = DMatrixFactory.create_dmatrix(x_train, y_train, log_details=False)
valid_dmatrix = DMatrixFactory.create_dmatrix(x_valid, y_valid, log_details=False)
```

### Step 3: Update `src/federated/client.py` ✅ READY

**Target Function**: `load_data()`

**Current Code Location**: Lines 38-85
```python
# REPLACE:
train_data = transform_dataset_to_dmatrix(client_dataset, processor=processor, is_training=True)
test_data = transform_dataset_to_dmatrix(test_dataset, processor=processor, is_training=False)

# WITH: (Already using centralized function, just ensure it uses DMatrixFactory internally)
```

### Step 4: Update `src/tuning/ray_tune_xgboost.py` ✅ READY

**Target Function**: `train_with_config()`

**Current Code Location**: Lines 409-447
```python
# REPLACE:
train_dmatrix = xgb.DMatrix(train_features, label=train_labels)
test_dmatrix = xgb.DMatrix(test_features, label=test_labels)

# WITH:
from src.core.shared_utils import DMatrixFactory
train_dmatrix = DMatrixFactory.create_dmatrix(train_features, train_labels, log_details=False)
test_dmatrix = DMatrixFactory.create_dmatrix(test_features, test_labels, log_details=False)
```

### Step 5: Update `src/federated/client_utils.py` ✅ READY

**Target Method**: `XgbClient.fit()` weighted DMatrix creation

**Current Code Location**: Lines 290
```python
# REPLACE:
dtrain_weighted = xgb.DMatrix(self.train_dmatrix.get_data(), label=y_train, weight=sample_weights, feature_names=self.train_dmatrix.feature_names)

# WITH:
from src.core.shared_utils import DMatrixFactory
dtrain_weighted = DMatrixFactory.create_weighted_dmatrix(
    base_dmatrix=self.train_dmatrix,
    weights=sample_weights
)
```

**Target Method**: `XgbClient.evaluate()` metrics calculation

**Current Code Location**: Lines 322-442
```python
# REPLACE scattered metrics calculations:
precision = precision_score(y_true, y_pred_labels, average='weighted', zero_division=0)
recall = recall_score(y_true, y_pred_labels, average='weighted', zero_division=0)
f1 = f1_score(y_true, y_pred_labels, average='weighted', zero_division=0)
accuracy = accuracy_score(y_true, y_pred_labels)

# WITH:
from src.core.shared_utils import MetricsCalculator
result = MetricsCalculator.calculate_classification_metrics(
    y_true=y_true,
    y_pred=y_pred_labels,
    y_pred_proba=y_pred_proba
)
precision = result.precision
recall = result.recall
f1 = result.f1_score
accuracy = result.accuracy
mlogloss = result.mlogloss
```

---

## 🧪 Testing Strategy

### Before Each Migration
1. **Run affected module** to ensure no regressions
2. **Test DMatrix creation** with sample data
3. **Verify metrics consistency** with known test cases

### Integration Testing
```python
# Test script to verify migration
python -c "
from src.core.shared_utils import DMatrixFactory, MetricsCalculator
import numpy as np

# Test DMatrix creation
features = np.random.rand(100, 10)
labels = np.random.randint(0, 3, 100)
dmatrix = DMatrixFactory.create_dmatrix(features, labels, log_details=False)
print(f'✅ DMatrix: {dmatrix.num_row()} x {dmatrix.num_col()}')

# Test metrics calculation
y_true = np.array([0, 1, 2, 0, 1])
y_pred = np.array([0, 1, 1, 0, 2])
result = MetricsCalculator.calculate_classification_metrics(y_true, y_pred)
print(f'✅ Metrics: acc={result.accuracy:.3f}, f1={result.f1_score:.3f}')
"
```

### Full System Test
```bash
# Test critical paths still work
python run.py --help                    # Should work
python src/federated/server.py --help   # Should work
python src/federated/client.py --help   # Should work
```

---

## ⚠️ Critical Preservation Checklist

During migration, **ABSOLUTELY PRESERVE**:

✅ **Data Leakage Fixes**: Hybrid temporal-stratified splitting in `dataset.py`
✅ **Hyperparameter Improvements**: Expanded search spaces in Ray Tune
✅ **Consistent Preprocessing**: FeatureProcessor functionality
✅ **Early Stopping**: Ray Tune integration
✅ **ConfigManager Integration**: Phase 2 achievements

**Test Each Migration Against**:
- [ ] All entry points still work (`run.py`, `server.py`, `client.py`, `sim.py`)
- [ ] Ray Tune hyperparameter optimization functions
- [ ] Federated learning training completes successfully
- [ ] Model accuracy does not regress
- [ ] Configuration management remains operational

---

## 📈 Success Verification

After completing each step:

1. **Code Quality Check**:
   ```bash
   # Check for remaining duplicated DMatrix patterns
   grep -r "xgb.DMatrix" src/ --exclude="shared_utils.py"
   
   # Check for remaining duplicated metrics patterns  
   grep -r "precision_score.*recall_score.*f1_score" src/
   ```

2. **Functionality Check**:
   ```python
   # Run a quick federated learning test
   python run.py --config base.yaml --dry-run
   ```

3. **Performance Check**:
   - Time DMatrix creation before/after migration
   - Verify no slowdown in metrics calculation
   - Check memory usage remains consistent

---

## 🎯 Next Steps After Phase 3

Once deduplication is complete:
- **Phase 4**: FL Strategy Classes and Global State Removal
- **Phase 5**: Comprehensive Testing Framework  
- **Phase 6**: Centralized Logging and Monitoring
- **Phase 7**: Documentation and API Polish

---

**Ready to implement! All groundwork is laid and utilities are tested.** 🚀 