# Federated Learning Pipeline: Actionable Fix Plan

## Problem Summary
The federated learning pipeline shows stagnant metrics across rounds while Ray Tune hyperparameter optimization successfully reduces loss. This indicates a disconnect between hyperparameter optimization and the federated learning implementation.

## Root Causes & Solutions

### 1. Client-Side Training Limitations

#### Issues:
- `NUM_LOCAL_ROUND` fixed at 1, severely limiting client learning
- Excessive regularization preventing effective model learning
- Potential noise features hindering learning

#### Action Items:
- [x] **Increase Local Training Rounds**:
  ```python
  # In utils.py
  NUM_LOCAL_ROUND = 5  # Increase from 1 to 5-10 to allow proper learning
  ```

- [x] **Adjust Regularization Parameters**:
  ```python
  # In utils.py or tuned_params.py
  BST_PARAMS = {
      'max_depth': 6,  # Increase from current value
      'lambda': 0.8,   # Decrease if currently high
      'alpha': 0.8,    # Decrease if currently high
      # Keep other parameters
  }
  ```

- [ ] **Implement Gradual Regularization**:
  ```python
  # In client_utils.py, modify create_xgboost_model function
  def create_xgboost_model(round_num, params, dtrain):
      # Gradually increase regularization over rounds
      if round_num < 5:
          params = params.copy()
          params['lambda'] *= (0.5 + 0.1 * round_num)  # Start with lighter regularization
      return xgb.train(params, dtrain, num_boost_round=NUM_LOCAL_ROUND)
  ```

### 2. Hyperparameter Optimization Integration

#### Issues:
- Ray Tune optimizes `num_boost_round` but this parameter isn't used in FL
- Limited search space with only 5 samples
- Tuned parameters may not transfer properly to FL context

#### Action Items:
- [ ] **Align Ray Tune and FL Parameters**:
  ```python
  # In ray_tune_xgboost.py, ensure num_boost_round is consistently used
  config = {
      "max_depth": tune.choice([3, 4, 5, 6, 8, 10]),
      "learning_rate": tune.loguniform(0.01, 0.3),
      "subsample": tune.uniform(0.5, 1.0),
      "colsample_bytree": tune.uniform(0.5, 1.0),
      "num_boost_round": tune.choice([1, 3, 5, 10]),  # Make this parameter available for tuning
      # other parameters...
  }
  ```

- [ ] **Transfer num_boost_round to FL**:
  ```python
  # In use_tuned_params.py, modify to extract num_boost_round
  def convert_best_config():
      # Existing code...
      if 'num_boost_round' in best_config:
          # Add to separate variable or config
          tuned_rounds = best_config['num_boost_round']
          lines.append(f"NUM_LOCAL_ROUND = {tuned_rounds}\n")
      # Existing code...
  ```

- [ ] **Increase Ray Tune Samples**:
  ```bash
  # In run_ray_tune.sh
  # Increase NUM_SAMPLES from 5 to at least 20
  NUM_SAMPLES=20
  ```

### 3. Metrics Reporting and Aggregation

#### Issues:
- Potential confusion matrix calculation issues
- Type inconsistencies (tuple vs. dictionary) in metrics reporting
- Possible stale model evaluation

#### Action Items:
- [ ] **Fix Metrics Type Inconsistency**:
  ```python
  # In server.py, add defensive type checking and conversion
  def aggregate_metrics(results):
      aggregated_metrics = {}
      for metric_name in expected_metrics:
          values = []
          weights = []
          
          for client_result in results:
              # Handle both tuple and dictionary formats
              if isinstance(client_result, tuple):
                  client_metrics = client_result[1]  # Assuming metrics are in second position
              elif isinstance(client_result, dict):
                  client_metrics = client_result.get('metrics', {})
              else:
                  logger.warning(f"Unexpected result type: {type(client_result)}")
                  continue
                  
              # Extract metric and weight
              if metric_name in client_metrics:
                  values.append(client_metrics[metric_name])
                  weights.append(client_metrics.get('weight', 1.0))
          
          # Calculate weighted average
          if values:
              aggregated_metrics[metric_name] = np.average(values, weights=weights)
      
      return aggregated_metrics
  ```

- [ ] **Add Debug Logging for Model Updates**:
  ```python
  # In server.py, add model parameter tracking
  class XGBoostServer(fl.server.strategy.Strategy):
      def aggregate_fit(self, rnd, results, failures):
          # Existing aggregation code...
          
          # Add debugging to confirm model update
          if self.global_model is not None:
              # Log a hash or sample of model parameters before and after update
              before_update = hash(str(self.global_model.get_dump()[:3]))  # First 3 trees
              # Perform update...
              after_update = hash(str(self.global_model.get_dump()[:3]))
              
              logger.info(f"Round {rnd}: Model update check - Before: {before_update}, After: {after_update}")
              logger.info(f"Round {rnd}: Models different: {before_update != after_update}")
          
          return aggregated_parameters
  ```

- [ ] **Ensure Fresh Model Evaluation**:
  ```python
  # In server.py, ensure evaluation uses the latest model
  def evaluate(self, parameters):
      # Clone the current global model to ensure we're evaluating the latest
      evaluation_model = xgb.Booster()
      evaluation_model.load_model(self.global_model.save_model())
      
      # Rest of evaluation code...
  ```

### 4. Data Handling and Preprocessing

#### Issues:
- Possible non-IID data effects
- Potential preprocessing inconsistencies between Ray Tune and FL

#### Action Items:
- [ ] **Verify Preprocessing Consistency**:
  ```python
  # In client.py and ray_tune_xgboost.py
  # Add version tracking for preprocessing
  def log_preprocessing_config():
      """Log preprocessing parameters to ensure consistency"""
      preproc_config = {
          "drop_columns": COLUMNS_TO_DROP,
          "feature_transformations": FEATURE_TRANSFORMATIONS,
          "encoding_method": ENCODING_METHOD,
          # Add all preprocessing parameters
      }
      logger.info(f"Preprocessing config: {json.dumps(preproc_config)}")
  
  # Call this function at the start of both client training and ray tune
  ```

- [ ] **Test with IID Data Simulation**:
  ```python
  # In sim.py or a new test script
  def run_with_iid_data():
      """Run federated learning with IID data to test if non-IID is the issue"""
      # Load dataset
      dataset = load_dataset()
      
      # Create IID splits instead of the usual client partitioning
      client_datasets = create_iid_partitions(dataset, num_clients=10)
      
      # Run federated learning with these partitions
      # ...
  ```

## Implementation Plan

### Phase 1: Diagnostics & Instrumentation (Day 1)
1. Add extensive logging to trace model updates, parameter changes
2. Add metrics consistency checks
3. Run test FL rounds with debug output to confirm current behavior

### Phase 2: Core Fixes (Days 2-3)
1. Implement changes to `NUM_LOCAL_ROUND` and regularization parameters
2. Fix metrics aggregation and type inconsistencies
3. Align Ray Tune and FL parameter spaces

### Phase 3: Testing & Validation (Days 4-5)
1. Run controlled experiments with new configuration
2. Compare performance with Ray Tune results
3. Conduct IID vs. non-IID tests to isolate data distribution effects

### Phase 4: Refinement (Days 6-7)
1. Fine-tune parameters based on experimental results
2. Implement additional optimizations if needed
3. Document findings and solutions

## Monitoring Plan
- Track metrics across FL rounds to verify improvement
- Compare aggregated model performance with standalone models
- Verify convergence of metrics over time

## Success Criteria
- Metrics show improvement across FL rounds
- Final FL performance approaches Ray Tune's achieved performance
- Confusion matrix shows balanced prediction across classes
- Consistent behavior across multiple runs 