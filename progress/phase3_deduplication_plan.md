# Phase 3: Code Deduplication Implementation Plan

## 🎯 Phase 3 Overview
**Objective**: Eliminate code duplication across the FL-CML-Pipeline by creating shared utilities and centralizing repeated functionality.

**Duration**: 1 Week  
**Status**: Ready to Begin  
**Phase 2 Dependencies**: ✅ ConfigManager fully implemented and operational

## 📊 Duplication Analysis Summary

### Critical Areas Identified

#### 1. DMatrix Creation (🔥 High Priority)
**Locations Found** (6+ instances):
- `src/core/dataset.py` - `transform_dataset_to_dmatrix()` 
- `src/federated/sim.py` - `get_client_fn()` inline creation
- `src/federated/client.py` - `load_data()` conversion
- `src/tuning/ray_tune_xgboost.py` - `train_with_config()` 
- `src/federated/client_utils.py` - `fit()` method weighted DMatrix
- `archive/old_implementations/ray_tune_xgboost.py` - multiple locations

**Current Problems**:
- Inconsistent missing value handling (`missing=np.nan` vs not specified)
- Different logging patterns for DMatrix creation
- Repeated feature/label preprocessing logic
- No centralized validation of DMatrix parameters

#### 2. Metrics Calculation (🔥 High Priority)  
**Locations Found** (8+ instances):
- `src/federated/client_utils.py` - `evaluate()` method
- `src/federated/utils.py` - `evaluate_metrics_aggregation()`
- `src/federated/utils.py` - `get_evaluate_fn()` 
- `src/tuning/ray_tune_xgboost.py` - `train_with_config()`
- `src/models/use_saved_model.py` - `evaluate_labeled_data()`
- `src/utils/visualization.py` - `plot_per_class_metrics()`
- `archive/old_implementations/ray_tune_xgboost.py`

**Current Problems**:
- Same metrics calculations (accuracy, precision, recall, f1) repeated everywhere
- Inconsistent `zero_division` parameter handling
- Different averaging strategies across modules
- No centralized metrics validation

#### 3. XGBoost Parameter Handling (🔴 Medium Priority)
**Locations Found** (5+ instances):
- `src/federated/client_utils.py` - parameter loading logic
- `src/tuning/ray_tune_xgboost.py` - parameter building 
- `src/models/use_tuned_params.py` - parameter defaults
- `src/core/dataset.py` - parameter validation
- ConfigManager integration inconsistencies

#### 4. Evaluation Result Processing (🔴 Medium Priority)
**Locations Found** (4+ instances):
- Prediction probability handling
- Confusion matrix generation 
- Classification report formatting
- Results saving and logging

## 🏗️ Implementation Strategy

### Step 1: Create Shared Utilities Module ✅ READY
**Target**: `src/core/shared_utils.py`

#### 1.1 XGBoost DMatrix Factory
```python
class DMatrixFactory:
    """Centralized DMatrix creation with consistent configuration."""
    
    @staticmethod
    def create_dmatrix(
        features: Union[np.ndarray, pd.DataFrame],
        labels: Optional[Union[np.ndarray, pd.Series]] = None,
        handle_missing: bool = True,
        feature_names: Optional[List[str]] = None,
        weights: Optional[np.ndarray] = None,
        validate: bool = True
    ) -> xgb.DMatrix:
        """
        Create XGBoost DMatrix with consistent handling.
        
        Args:
            features: Feature data
            labels: Target labels (optional)
            handle_missing: Replace inf with nan
            feature_names: Optional feature names
            weights: Sample weights (optional)
            validate: Validate input data
            
        Returns:
            xgb.DMatrix with logging and validation
        """
```

#### 1.2 Metrics Calculator  
```python
class MetricsCalculator:
    """Centralized metrics calculation with consistent implementation."""
    
    @staticmethod
    def calculate_classification_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_pred_proba: Optional[np.ndarray] = None,
        class_names: Optional[List[str]] = None,
        prefix: str = ""
    ) -> Dict[str, float]:
        """
        Calculate comprehensive classification metrics.
        
        Returns:
            Dictionary with accuracy, precision, recall, f1, mlogloss, etc.
        """
        
    @staticmethod  
    def aggregate_client_metrics(
        client_metrics: List[Tuple[int, Dict[str, float]]]
    ) -> Dict[str, float]:
        """
        Aggregate metrics from multiple federated clients.
        """
```

#### 1.3 XGBoost Parameter Builder
```python
class XGBoostParamsBuilder:
    """Centralized XGBoost parameter management."""
    
    @staticmethod
    def build_params(
        base_params: Dict[str, Any],
        config_manager: Optional[ConfigManager] = None,
        overrides: Optional[Dict[str, Any]] = None,
        use_tuned: bool = False
    ) -> Dict[str, Any]:
        """
        Build XGBoost parameters with priority handling.
        
        Priority: overrides > config_manager > tuned_params > defaults
        """
```

### Step 2: Replace DMatrix Creation ✅ READY
**Files to Update**:
1. `src/core/dataset.py` - Update `transform_dataset_to_dmatrix()`
2. `src/federated/sim.py` - Replace inline DMatrix creation
3. `src/federated/client.py` - Update `load_data()` 
4. `src/tuning/ray_tune_xgboost.py` - Replace training DMatrix creation
5. `src/federated/client_utils.py` - Update weighted DMatrix creation

**Migration Pattern**:
```python
# OLD: Scattered DMatrix creation
train_data = xgb.DMatrix(train_features, label=train_labels, missing=np.nan)

# NEW: Centralized factory
from src.core.shared_utils import DMatrixFactory
train_data = DMatrixFactory.create_dmatrix(
    features=train_features,
    labels=train_labels,
    handle_missing=True
)
```

### Step 3: Replace Metrics Calculations ✅ READY
**Files to Update**:
1. `src/federated/client_utils.py` - `evaluate()` method
2. `src/federated/utils.py` - `evaluate_metrics_aggregation()`
3. `src/federated/utils.py` - `get_evaluate_fn()`
4. `src/tuning/ray_tune_xgboost.py` - training metrics
5. `src/models/use_saved_model.py` - evaluation metrics

**Migration Pattern**:
```python  
# OLD: Repeated metrics calculation
precision = precision_score(y_true, y_pred_labels, average='weighted', zero_division=0)
recall = recall_score(y_true, y_pred_labels, average='weighted', zero_division=0) 
f1 = f1_score(y_true, y_pred_labels, average='weighted', zero_division=0)
accuracy = accuracy_score(y_true, y_pred_labels)

# NEW: Centralized calculator
from src.core.shared_utils import MetricsCalculator
metrics = MetricsCalculator.calculate_classification_metrics(
    y_true=y_true,
    y_pred=y_pred_labels,
    y_pred_proba=y_pred_proba,
    class_names=class_names
)
```

### Step 4: Consolidate Parameter Handling ✅ READY
**Files to Update**:
1. `src/federated/client_utils.py` - Parameter loading logic
2. `src/tuning/ray_tune_xgboost.py` - Parameter building
3. Remove redundant parameter handling across modules

### Step 5: Testing and Validation ✅ READY
**Test Coverage Required**:
1. `tests/unit/test_shared_utils.py` - Unit tests for all utilities
2. `tests/integration/test_deduplication.py` - Integration testing
3. Verify no functionality regression
4. Performance benchmarking

## 📝 Implementation Checklist

### Pre-Implementation ✅
- [x] Phase 2 (ConfigManager) completed
- [x] Code duplication analysis completed
- [x] Implementation plan documented
- [x] Backup strategy defined

### Step 1: Shared Utilities Creation ⏳
- [ ] Create `src/core/shared_utils.py`
- [ ] Implement `DMatrixFactory` class
- [ ] Implement `MetricsCalculator` class  
- [ ] Implement `XGBoostParamsBuilder` class
- [ ] Add comprehensive docstrings
- [ ] Create unit tests

### Step 2: DMatrix Deduplication ⏳  
- [ ] Update `src/core/dataset.py`
- [ ] Update `src/federated/sim.py`
- [ ] Update `src/federated/client.py`
- [ ] Update `src/tuning/ray_tune_xgboost.py`
- [ ] Update `src/federated/client_utils.py`
- [ ] Test all DMatrix creation paths

### Step 3: Metrics Deduplication ⏳
- [ ] Update `src/federated/client_utils.py`
- [ ] Update `src/federated/utils.py` 
- [ ] Update `src/tuning/ray_tune_xgboost.py`
- [ ] Update `src/models/use_saved_model.py`
- [ ] Verify metrics consistency

### Step 4: Parameter Consolidation ⏳
- [ ] Update parameter handling logic
- [ ] Remove redundant parameter code
- [ ] Test ConfigManager integration

### Step 5: Testing & Validation ⏳
- [ ] Run comprehensive test suite
- [ ] Verify no regression in functionality
- [ ] Performance benchmark comparison
- [ ] Update documentation

### Post-Implementation ⏳
- [ ] Archive old duplicated code
- [ ] Update import statements
- [ ] Clean up deprecated functions
- [ ] Update progress documentation

## 🚨 Critical Preservation Requirements

### Must Preserve (DO NOT BREAK):
1. **Data Leakage Fixes**: Hybrid temporal-stratified splitting
2. **Hyperparameter Improvements**: Expanded search spaces
3. **Consistent Preprocessing**: FeatureProcessor functionality
4. **Early Stopping**: Ray Tune integration
5. **ConfigManager Integration**: Phase 2 achievements

### Testing Validation Points:
1. All entry points (`run.py`, `server.py`, `client.py`, `sim.py`) must work
2. Ray Tune hyperparameter optimization must function
3. Federated learning training must complete successfully
4. Model accuracy must not regress
5. Configuration management must remain operational

## 📈 Success Metrics

### Code Quality:
- [ ] **Zero Code Duplication**: No repeated DMatrix creation or metrics calculation
- [ ] **Consistent API**: Uniform interface across all modules
- [ ] **Improved Maintainability**: Single source of truth for shared functionality
- [ ] **Type Safety**: Full type hints on all shared utilities

### Functionality:
- [ ] **No Performance Regression**: Maintain or improve execution speed
- [ ] **No Accuracy Loss**: Model performance unchanged
- [ ] **All Tests Pass**: 100% test success rate
- [ ] **Documentation Updated**: Clear usage examples

### Architecture:
- [ ] **Clean Dependencies**: Proper import hierarchy
- [ ] **Modular Design**: Well-separated concerns
- [ ] **Extensible Structure**: Easy to add new functionality
- [ ] **Professional Standards**: Industry best practices

## 🔄 Next Phase Preparation

Upon completion of Phase 3, the codebase will be ready for:
- **Phase 4**: FL Strategy Classes and Global State Removal
- **Phase 5**: Comprehensive Testing Framework
- **Phase 6**: Centralized Logging and Monitoring
- **Phase 7**: Documentation and API Polish

---

**Ready to Begin**: ✅ All dependencies satisfied, plan documented, ready for implementation. 