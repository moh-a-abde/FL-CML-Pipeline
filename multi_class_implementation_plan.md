# Multi-Class Classification Implementation Plan

## Current State
- Binary classification (benign vs malicious) using XGBoost
- Features include both categorical and numerical network traffic data
- Using federated learning with Flower framework
- Need to expand to classify specific types of malicious traffic

## Implementation Steps

### 1. Data Preprocessing Modifications (`dataset.py`)
- [ ] Update `preprocess_data()` function:
  ```python
  def preprocess_data(data):
      # ... existing code ...
      if 'label' in df.columns:
          features = df.drop(columns=['label'])
          
          # New label mapping for three classes
          label_mapping = {
              'benign': 0, 
              'dns_tunneling': 1, 
              'icmp_tunneling': 2
          }
          labels_series = df['label'].map(label_mapping)
          
          # Handle unmapped labels
          if labels_series.isnull().any():
              unmapped_labels = df['label'][labels_series.isnull()].unique()
              print(f"Warning: Unmapped labels found: {unmapped_labels}")
              labels_series = labels_series.fillna(-1)
          
          labels = labels_series.astype(int)
          return features, labels
      else:
          return df, None
  ```

### 2. Model Configuration Updates (`client_utils.py`)
- [ ] Update XGBoost parameters in `utils.py`:
  ```python
  BST_PARAMS = {
      'objective': 'multi:softmax',
      'num_class': 3,
      'eval_metric': ['mlogloss', 'merror'],
      'learning_rate': 0.1,
      'max_depth': 6,
      'min_child_weight': 1,
      'subsample': 0.8,
      'colsample_bytree': 0.8,
      'scale_pos_weight': [1.0, 2.0, 2.0]  # Adjust based on class distribution
  }
  ```

- [ ] Modify `evaluate()` method in `XgbClient` class:
  ```python
  def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
      # ... existing model loading code ...
      
      # Generate predictions - No thresholding needed for multi-class
      y_pred_proba = bst.predict(self.valid_dmatrix)
      y_pred_labels = np.argmax(y_pred_proba, axis=1)
      
      # Get ground truth labels
      y_true = self.valid_dmatrix.get_label()
      
      # Compute multi-class metrics
      precision = precision_score(y_true, y_pred_labels, average='weighted')
      recall = recall_score(y_true, y_pred_labels, average='weighted')
      f1 = f1_score(y_true, y_pred_labels, average='weighted')
      accuracy = accuracy_score(y_true, y_pred_labels)
      
      # Multi-class loss calculation
      epsilon = 1e-10
      log_likelihood = -np.log(y_pred_proba[np.arange(len(y_true)), y_true.astype(int)] + epsilon)
      loss = np.mean(log_likelihood)
      
      # Multi-class confusion matrix
      conf_matrix = confusion_matrix(y_true, y_pred_labels)
      
      metrics = {
          "precision": float(precision),
          "recall": float(recall),
          "f1": float(f1),
          "accuracy": float(accuracy),
          "loss": float(loss),
          "confusion_matrix": conf_matrix.tolist(),
          "num_predictions": self.num_val
      }
      
      return metrics
  ```

### 3. Server-Side Updates (`server_utils.py`)
- [ ] Update metrics aggregation:
  ```python
  def evaluate_metrics_aggregation(eval_metrics):
      total_num = sum([num for num, _ in eval_metrics])
      
      # Aggregate evaluation metrics
      metrics_to_aggregate = ['precision', 'recall', 'f1', 'accuracy', 'loss']
      aggregated_metrics = {}
      
      for metric in metrics_to_aggregate:
          if all(metric in metrics for _, metrics in eval_metrics):
              aggregated_metrics[metric] = sum([metrics[metric] * num for num, metrics in eval_metrics]) / total_num
          else:
              aggregated_metrics[metric] = 0.0
      
      # Aggregate confusion matrix
      aggregated_conf_matrix = None
      for num, metrics in eval_metrics:
          conf_matrix = np.array(metrics.get("confusion_matrix", [[0, 0, 0], [0, 0, 0], [0, 0, 0]]))
          if aggregated_conf_matrix is None:
              aggregated_conf_matrix = conf_matrix
          else:
              aggregated_conf_matrix += conf_matrix
      
      aggregated_metrics["confusion_matrix"] = aggregated_conf_matrix.tolist()
      aggregated_metrics["prediction_mode"] = eval_metrics[0][1].get("prediction_mode", False)
      
      return aggregated_metrics["loss"], aggregated_metrics
  ```

- [ ] Update prediction saving:
  ```python
  def save_predictions_to_csv(data, predictions, round_num: int, output_dir: str = None, true_labels=None):
      predictions_dict = {
          'predicted_label': predictions,
          'prediction_type': [
              'benign' if p == 0 else 
              ('dns_tunneling' if p == 1 else 'icmp_tunneling') 
              for p in predictions
          ]
      }
      # ... rest of the function
  ```

### 4. Prediction Updates (`use_saved_model.py`)
- [ ] Update prediction saving:
  ```python
  def save_detailed_predictions(predictions, output_path):
      results_df = pd.DataFrame()
      
      if predictions.ndim > 1 and predictions.shape[1] > 1:
          results_df['raw_probabilities'] = predictions.tolist()
          predicted_labels = np.argmax(predictions, axis=1)
          results_df['predicted_label'] = predicted_labels
          
          results_df['prediction_type'] = [
              'benign' if p == 0 else 
              ('dns_tunneling' if p == 1 else 'icmp_tunneling')
              for p in predicted_labels
          ]
          
          results_df['prediction_score'] = predictions[
              np.arange(len(predicted_labels)), 
              predicted_labels
          ]
      
      results_df.to_csv(output_path, index=False)
      return results_df
  ```

- [ ] Update main evaluation:
  ```python
  def main():
      # ... existing code ...
      if args.has_labels:
          y_pred_labels = np.argmax(raw_predictions, axis=1)
          accuracy = accuracy_score(y_true, y_pred_labels)
          
          cm = confusion_matrix(y_true, y_pred_labels)
          report = classification_report(y_true, y_pred_labels)
          
          log(INFO, f"Accuracy: {accuracy:.4f}")
          log(INFO, f"Confusion Matrix:\n{cm}")
          log(INFO, f"Classification Report:\n{report}")
  ```

## Testing Strategy

### 1. Unit Tests
- [ ] Test label mapping in `dataset.py`
- [ ] Test multi-class metrics calculation
- [ ] Test prediction format and class probabilities
- [ ] Test confusion matrix calculation

### 2. Integration Tests
- [ ] Test full training pipeline with three classes
- [ ] Test federated learning convergence
- [ ] Test model saving and loading
- [ ] Test prediction pipeline

### 3. System Tests
- [ ] End-to-end training and evaluation
- [ ] Performance testing with large datasets
- [ ] Class imbalance handling
- [ ] Error handling and edge cases

## Success Criteria
1. Model successfully trains on three-class data
2. All evaluation metrics properly calculated and reported
3. Predictions include class probabilities
4. Federated learning pipeline works with multiple classes
5. High accuracy in distinguishing between benign and malicious traffic
6. Accurate classification of DNS vs ICMP tunneling attacks
7. Documentation is complete and accurate
8. Test cases pass successfully

## Potential Challenges and Mitigations
1. Class imbalance
   - Solution: Use class weights in model parameters
   - Monitor class distribution in training data
   - Consider data augmentation for underrepresented classes

2. Model complexity
   - Solution: Start with simpler model and gradually increase complexity
   - Monitor training metrics for overfitting
   - Use cross-validation for parameter tuning

3. Federated learning convergence
   - Solution: Adjust learning rate and number of rounds
   - Monitor loss curves across clients
   - Consider using FedAvg with momentum

4. Performance impact
   - Solution: Profile code for bottlenecks
   - Optimize prediction pipeline
   - Consider batch processing for large datasets

## Timeline
1. Data preprocessing: 1 day
2. Model configuration: 1 day
3. Training pipeline: 2 days
4. Prediction and evaluation: 2 days
5. Server-side updates: 1 day
6. Testing and validation: 2 days
7. Documentation: 1 day

Total estimated time: 10 days
