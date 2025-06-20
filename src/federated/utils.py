from typing import Dict, List, Optional
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, log_loss, accuracy_score
from logging import INFO, WARNING
import xgboost as xgb
import pandas as pd
from flwr.common.logger import log
from flwr.common import Parameters, Scalar
from flwr.server.client_manager import SimpleClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.criterion import Criterion
import os
import json
import shutil
from datetime import datetime
import pickle
import numpy as np
# Assuming visualization_utils.py is in the same directory or accessible via PYTHONPATH
from src.utils.visualization import (
    plot_confusion_matrix,
    plot_roc_curves,
    plot_precision_recall_curves,
    plot_class_distribution,
    plot_learning_curves
)
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Global variable to track metrics history for early stopping
METRICS_HISTORY = []

# Authoritative label mapping for UNSW_NB15 dataset (11 classes for engineered dataset)
UNSW_NB15_LABEL_MAPPING = {
    0: 'Normal',
    1: 'Generic', 
    2: 'Exploits',
    3: 'Reconnaissance',
    4: 'Fuzzers',
    5: 'DoS',
    6: 'Analysis',
    7: 'Backdoor',
    8: 'Backdoors',
    9: 'Worms',
    10: 'Shellcode'  # Engineered dataset has 11 classes (0-10)
}

# Helper function to get class names list
def get_class_names_list():
    """Get the list of class names in correct order."""
    return [UNSW_NB15_LABEL_MAPPING[i] for i in range(len(UNSW_NB15_LABEL_MAPPING))]

def setup_output_directory():
    """
    Creates a date and time-based directory structure for outputs.
    
    Returns:
        str: Path to the created output directory
    """
    # Create base outputs directory if it doesn't exist
    base_dir = "outputs"
    os.makedirs(base_dir, exist_ok=True)
    
    # Create date directory
    date_str = datetime.now().strftime("%Y-%m-%d")
    date_dir = os.path.join(base_dir, date_str)
    os.makedirs(date_dir, exist_ok=True)
    
    # Create time directory
    time_str = datetime.now().strftime("%H-%M-%S")
    output_dir = os.path.join(date_dir, time_str)
    os.makedirs(output_dir, exist_ok=True)
    
    # Create .hydra directory
    hydra_dir = os.path.join(output_dir, ".hydra")
    os.makedirs(hydra_dir, exist_ok=True)
    
    # Copy existing .hydra files if they exist
    if os.path.exists(".hydra"):
        for file in os.listdir(".hydra"):
            if file.endswith(".yaml"):
                src_path = os.path.join(".hydra", file)
                dst_path = os.path.join(hydra_dir, file)
                shutil.copy2(src_path, dst_path)
    
    log(INFO, "Created output directory: %s", output_dir)
    return output_dir

def save_results_pickle(results, output_dir):
    """
    Save results dictionary to a pickle file.
    
    Args:
        results (dict): Results to save
        output_dir (str): Directory to save to
    """
    output_path = os.path.join(output_dir, "results.pkl")
    with open(output_path, 'wb') as f:
        pickle.dump(results, f)
    log(INFO, "Saved results to: %s", output_path)

def eval_config(rnd: int, output_dir: str = None) -> Dict[str, str]:
    """
    Return a configuration with global round and output directory.
    
    Args:
        rnd (int): Current round number
        output_dir (str, optional): Output directory path
        
    Returns:
        Dict[str, str]: Configuration dictionary
    """
    # Set prediction_mode to false for rounds 1-10 and true for rounds 11-20
    prediction_mode = "false" if rnd <= 10 else "true"
    
    config = {
        "global_round": str(rnd),
        "prediction_mode": prediction_mode,
    }
    
    # Add output directory if provided
    if output_dir is not None:
        config["output_dir"] = output_dir
        
    return config

def save_evaluation_results(eval_metrics: Dict, round_num: int, output_dir: str = None):
    """
    Save evaluation results for each round.
    
    Args:
        eval_metrics (Dict): Evaluation metrics to save
        round_num (int or str): Round number or identifier
        output_dir (str, optional): Directory to save results to. If None, uses the default results directory.
    """
    # Use default results directory if no output_dir is provided
    if output_dir is None:
        output_dir = "results"
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Format results
    results = {
        'round': round_num,
        'timestamp': datetime.now().isoformat(),
        'metrics': eval_metrics
    }
    
    # Save to file
    output_path = os.path.join(output_dir, f"eval_results_round_{round_num}.json")
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=4)
    
    log(INFO, "Evaluation results saved to: %s", output_path)

def fit_config(rnd: int) -> Dict[str, str]:
    """Return a configuration with global epochs."""
    config = {
        "global_round": str(rnd),
    }
    return config


def evaluate_metrics_aggregation(eval_metrics):
    """
    Aggregate evaluation metrics from multiple clients for multi-class classification.
    
    Args:
        eval_metrics: List of tuples (num_examples, metrics_dict) from each client
        
    Returns:
        tuple: (loss, aggregated_metrics)
    """
    total_num = sum([num for num, _ in eval_metrics])
    
    # Log the raw metrics received from clients
    log(INFO, "Received metrics from %d clients", len(eval_metrics))
    for i, (num, metrics) in enumerate(eval_metrics):
        log(INFO, "Client %d metrics: %s", i+1, metrics.keys())
        if "mlogloss" in metrics:
            log(INFO, "Client %d mlogloss: %f", i+1, metrics["mlogloss"])
    
    # Initialize aggregated metrics dictionary
    metrics_to_aggregate = ['precision', 'recall', 'f1', 'accuracy']
    aggregated_metrics = {}
    
    # Aggregate weighted metrics
    for metric in metrics_to_aggregate:
        if all(metric in metrics for _, metrics in eval_metrics):
            weighted_sum = sum([metrics[metric] * num for num, metrics in eval_metrics])
            aggregated_metrics[metric] = weighted_sum / total_num
        else:
            aggregated_metrics[metric] = 0.0
            log(INFO, "Metric %s not available in all client metrics", metric)
    
    # Aggregate loss (using mlogloss)
    if all("mlogloss" in metrics for _, metrics in eval_metrics):
        client_losses = [metrics["mlogloss"] for _, metrics in eval_metrics]
        log(INFO, "Individual client losses (mlogloss): %s", client_losses)
        loss = sum([metrics["mlogloss"] * num for num, metrics in eval_metrics]) / total_num
        log(INFO, "Aggregated loss calculation: sum(mlogloss*num)=%f, total_num=%d, result=%f",
            sum([metrics["mlogloss"] * num for num, metrics in eval_metrics]), total_num, loss)
    else:
        loss = 0.0
        log(INFO, "Mlogloss not available in all client metrics")
    
    aggregated_metrics["mlogloss"] = loss
    
    # Aggregate confusion matrix
    aggregated_conf_matrix = None
    for num, metrics in eval_metrics:
        if "confusion_matrix" in metrics:
            conf_matrix = metrics["confusion_matrix"]
            if aggregated_conf_matrix is None:
                aggregated_conf_matrix = [[0 for _ in range(len(conf_matrix[0]))] for _ in range(len(conf_matrix))]
            # Add weighted confusion matrix
            for i in range(len(conf_matrix)):
                for j in range(len(conf_matrix[0])):
                    aggregated_conf_matrix[i][j] += conf_matrix[i][j] * num
    
    # Normalize confusion matrix by total examples
    if aggregated_conf_matrix is not None:
        for i in range(len(aggregated_conf_matrix)):
            for j in range(len(aggregated_conf_matrix[0])):
                aggregated_conf_matrix[i][j] /= total_num
        aggregated_metrics["confusion_matrix"] = aggregated_conf_matrix
    
    # Aggregate per-class metrics
    if all("per_class_metrics" in metrics for _, metrics in eval_metrics):
        aggregated_per_class = {}
        for class_id in eval_metrics[0][1]["per_class_metrics"].keys():
            class_metrics = {
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "support": 0
            }
            for num, metrics in eval_metrics:
                if class_id in metrics["per_class_metrics"]:
                    class_data = metrics["per_class_metrics"][class_id]
                    weight = num / total_num
                    class_metrics["precision"] += class_data["precision"] * weight
                    class_metrics["recall"] += class_data["recall"] * weight
                    class_metrics["f1"] += class_data["f1"] * weight
                    class_metrics["support"] += class_data["support"]
            aggregated_per_class[class_id] = class_metrics
        aggregated_metrics["per_class_metrics"] = aggregated_per_class
    
    # Log aggregated metrics
    log(INFO, "Aggregated Metrics Summary:")
    log(INFO, "  Precision (weighted): %.6f", aggregated_metrics["precision"])
    log(INFO, "  Recall (weighted): %.6f", aggregated_metrics["recall"])
    log(INFO, "  F1 Score (weighted): %.6f", aggregated_metrics["f1"])
    log(INFO, "  Accuracy: %.6f", aggregated_metrics["accuracy"])
    log(INFO, "  Loss (mlogloss): %.6f", aggregated_metrics["mlogloss"])
    
    if "per_class_metrics" in aggregated_metrics:
        log(INFO, "Per-class Aggregated Metrics:")
        for class_id, metrics in aggregated_metrics["per_class_metrics"].items():
            log(INFO, "  Class %s:", class_id)
            log(INFO, "    Precision: %.6f", metrics["precision"])
            log(INFO, "    Recall: %.6f", metrics["recall"])
            log(INFO, "    F1 Score: %.6f", metrics["f1"])
            log(INFO, "    Support: %d", metrics["support"])
    
    if aggregated_conf_matrix is not None:
        log(INFO, "Aggregated Confusion Matrix:")
        for row in aggregated_conf_matrix:
            log(INFO, "  %s", " ".join([f"{x:.4f}" for x in row]))
    
    # Add metrics to history for early stopping tracking
    add_metrics_to_history(aggregated_metrics)
    
    # Save aggregated results
    save_evaluation_results(aggregated_metrics, "aggregated")
    
    return loss, aggregated_metrics

def save_predictions_to_csv(data, predictions, round_num: int, output_dir: str = None, true_labels=None, prediction_types=None):
    """
    Save dataset with predictions to CSV in the specified directory.
    
    Args:
        data: Original data
        predictions: Prediction labels (class indices or array of probabilities)
        round_num (int): Round number
        output_dir (str, optional): Directory to save results to. If None, uses the default results directory.
        true_labels (array, optional): True labels if available
        prediction_types (list, optional): List of prediction type strings (e.g., 'Normal', 'Reconnaissance', etc.)
        
    Returns:
        str: Path to the saved CSV file
    """
    # Use default results directory if no output_dir is provided
    if output_dir is None:
        output_dir = "results"
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if predictions is a 2D array (multi-class probabilities)
    if isinstance(predictions, np.ndarray) and len(predictions.shape) > 1:
        log(INFO, "Detected multi-class probability predictions with shape: %s", predictions.shape)
        # Convert probabilities to class labels
        predicted_labels = np.argmax(predictions, axis=1)
    else:
        # Already a list of class indices
        predicted_labels = predictions
    
    # Create predictions DataFrame
    predictions_dict = {
        'predicted_label': predicted_labels,
    }
    
    # Add prediction types if provided
    if prediction_types is not None:
        predictions_dict['prediction_type'] = prediction_types
    else:
        # Use the global authoritative label mapping
        predictions_dict['prediction_type'] = [UNSW_NB15_LABEL_MAPPING.get(int(label), f'unknown_{label}') for label in predicted_labels]
    
    # Add true labels if available
    if true_labels is not None:
        predictions_dict['true_label'] = true_labels
        
        # Generate and save visualizations if we have true labels to compare with
        try:
            class_names = get_class_names_list()
            num_classes = len(class_names)
            
            # Convert to numpy arrays if they're not already
            y_true = np.array(true_labels) if not isinstance(true_labels, np.ndarray) else true_labels
            y_pred = np.array(predicted_labels) if not isinstance(predicted_labels, np.ndarray) else predicted_labels
            
            # Create confusion matrix
            cm = confusion_matrix(y_true, y_pred, labels=range(num_classes))
            cm_path = os.path.join(output_dir, f"confusion_matrix_round_{round_num}.png")
            plot_confusion_matrix(cm, class_names, cm_path)
            
            # Plot class distribution
            dist_path = os.path.join(output_dir, f"class_distribution_round_{round_num}.png")
            plot_class_distribution(y_true, y_pred, class_names, dist_path)
            
            log(INFO, f"Visualizations saved for round {round_num}")
        except Exception as e:
            log(WARNING, f"Error generating visualizations: {e}")
    
    predictions_df = pd.DataFrame(predictions_dict)
    
    # Save predictions
    output_path = os.path.join(output_dir, f"predictions_round_{round_num}.csv")
    predictions_df.to_csv(output_path, index=False)
    log(INFO, "Predictions saved to: %s", output_path)
    
    return output_path

def load_saved_model(model_path, config_manager=None):
    """
    Load a saved XGBoost model from disk.
    
    Args:
        model_path (str): Path to the saved model file (.json or .bin)
        config_manager (ConfigManager, optional): ConfigManager instance for getting model parameters
        
    Returns:
        xgb.Booster: Loaded XGBoost model
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    log(INFO, "Loading model from: %s", model_path)
    
    try:
        # Create a new booster
        bst = xgb.Booster()
        
        # Try to load the model directly
        bst.load_model(model_path)
        log(INFO, "Model loaded successfully")
        return bst
    except Exception as e:
        log(INFO, "Error loading model directly: %s", str(e))
        
        # If direct loading fails, try alternative approaches
        try:
            # Try reading the file as bytes and loading
            with open(model_path, 'rb') as f:
                model_data = f.read()
            
            bst = xgb.Booster()
            bst.load_model(bytearray(model_data))
            log(INFO, "Model loaded successfully using bytearray")
            return bst
        except Exception as e2:
            log(INFO, "Error loading model using bytearray: %s", str(e2))
            
            # If that fails too, try with params from ConfigManager
            try:
                if config_manager is not None:
                    model_params = config_manager.get_model_params_dict()
                    bst = xgb.Booster(params=model_params)
                else:
                    # Fallback to basic params if no ConfigManager available
                    basic_params = {
                        "objective": "multi:softprob",
                        "num_class": 11,
                        "tree_method": "hist"
                    }
                    bst = xgb.Booster(params=basic_params)
                bst.load_model(model_path)
                log(INFO, "Model loaded successfully with params")
                return bst
            except Exception as e3:
                log(INFO, "All loading attempts failed")
                raise ValueError(f"Failed to load model: {str(e)}, {str(e2)}, {str(e3)}")

def predict_with_saved_model(model_path, dmatrix, output_path, config_manager=None):
    # Load the model
    model = load_saved_model(model_path, config_manager)
    
    # Make predictions
    raw_predictions = model.predict(dmatrix)
    
    # Log raw predictions
    log(INFO, "Raw predictions shape: %s", raw_predictions.shape if hasattr(raw_predictions, 'shape') else 'scalar')
    
    # Log distribution of scores
    if hasattr(raw_predictions, 'shape'):
        log(INFO, "Prediction score distribution - Min: %.4f, Max: %.4f, Mean: %.4f", 
            np.min(raw_predictions), np.max(raw_predictions), np.mean(raw_predictions))
    
    # For multi-class with multi:softprob, the raw predictions will be probabilities for each class
    if hasattr(raw_predictions, 'shape') and len(raw_predictions.shape) > 1:
        log(INFO, "Processing multi-class probability predictions with shape: %s", raw_predictions.shape)
        predicted_labels = np.argmax(raw_predictions, axis=1)
        
        # Use the global authoritative label mapping
        # Save predictions to CSV with class names
        predictions_df = pd.DataFrame({
            'predicted_label': predicted_labels,
            'prediction_type': [UNSW_NB15_LABEL_MAPPING.get(int(p), f'unknown_{p}') for p in predicted_labels],
        })
        
        # Add probability columns for each class
        for i in range(raw_predictions.shape[1]):
            predictions_df[f'prob_class_{i}'] = raw_predictions[:, i]
    elif len(raw_predictions.shape) == 1:  # Binary case or multi:softmax
        # Check if this is binary classification or multi-class with direct labels
        if np.max(raw_predictions) <= 1.0 and np.min(raw_predictions) >= 0.0:
            # Binary case
            probabilities = raw_predictions  # Already probabilities
            predicted_labels = (probabilities >= 0.5).astype(int)
            
            # Save predictions to CSV
            predictions_df = pd.DataFrame({
                'predicted_label': predicted_labels,
                'prediction_type': ['benign' if label == 0 else 'malicious' for label in predicted_labels],
                'prediction_score': probabilities
            })
        else:
            # Likely multi:softmax with direct class labels
            predicted_labels = np.round(raw_predictions).astype(int)
            
            # Use the global authoritative label mapping
            # Save predictions to CSV with class names
            predictions_df = pd.DataFrame({
                'predicted_label': predicted_labels,
                'prediction_type': [UNSW_NB15_LABEL_MAPPING.get(int(p), f'unknown_{p}') for p in predicted_labels],
            })
    else:
        # Fallback for unexpected prediction format
        log(WARNING, "Unexpected prediction format. Creating basic predictions DataFrame.")
        predictions_df = pd.DataFrame({
            'raw_prediction': raw_predictions
        })
    
    # Log predicted class distribution
    if 'predicted_label' in predictions_df.columns:
        unique, counts = np.unique(predictions_df['predicted_label'], return_counts=True)
        log(INFO, "Predicted class distribution: %s", dict(zip(unique, counts)))
    
    # Generate visualizations if true labels are available
    try:
        true_labels = dmatrix.get_label()
        if true_labels is not None and 'predicted_label' in predictions_df.columns:
            output_dir = os.path.dirname(output_path)
            num_classes = max(11, np.max(true_labels) + 1)  # Ensure at least 11 classes for the engineered dataset
            class_names = [UNSW_NB15_LABEL_MAPPING.get(i, f'Class_{i}') for i in range(num_classes)]
            
            predicted_labels = predictions_df['predicted_label'].values
            
            # Create confusion matrix
            cm = confusion_matrix(true_labels, predicted_labels, labels=range(num_classes))
            cm_path = os.path.join(output_dir, "final_confusion_matrix.png")
            plot_confusion_matrix(cm, class_names, cm_path)
            
            # Plot class distribution
            dist_path = os.path.join(output_dir, "final_class_distribution.png")
            plot_class_distribution(true_labels, predicted_labels, class_names, dist_path)
            
            # Plot ROC and Precision-Recall Curves (for multi-class)
            if len(raw_predictions.shape) > 1 and raw_predictions.shape[1] >= num_classes:
                roc_path = os.path.join(output_dir, "final_roc_curves.png")
                plot_roc_curves(true_labels, raw_predictions, class_names, roc_path)
                
                pr_path = os.path.join(output_dir, "final_precision_recall_curves.png")
                plot_precision_recall_curves(true_labels, raw_predictions, class_names, pr_path)
            
            log(INFO, "Visualizations saved with final model predictions")
    except Exception as e:
        log(WARNING, f"Error generating visualizations: {e}")
    
    predictions_df.to_csv(output_path, index=False)
    log(INFO, "Predictions saved to: %s", output_path)
    
    return predictions

def get_evaluate_fn(test_data, config_manager=None):
    """Return a function for centralised evaluation."""

    def evaluate_model(
        server_round: int, parameters: Parameters, config: Dict[str, Scalar]
    ):
        if server_round == 0:
            return 0, {}
        else:
            # Get model parameters from ConfigManager if available
            if config_manager is not None:
                model_params = config_manager.get_model_params_dict()
            else:
                # Fallback to basic params if no ConfigManager available
                model_params = {
                    "objective": "multi:softprob",
                    "num_class": 11,
                    "tree_method": "hist"
                }
            
            bst = xgb.Booster(params=model_params)
            para_b = None
            for para in parameters.tensors:
                para_b = bytearray(para)
                break  # Take the first parameter tensor
            
            if para_b is not None:
                bst.load_model(para_b)
            else:
                # No parameters provided, create a new model with default params
                log(WARNING, "No model parameters provided, using fresh model")
                bst = xgb.Booster(params=model_params)
            
            # Get predictions
            y_pred_proba = bst.predict(test_data)
            
            # For multi:softprob, we get probabilities for each class
            # Convert to labels by taking argmax if predictions are probabilities
            if isinstance(y_pred_proba, np.ndarray) and len(y_pred_proba.shape) > 1:
                y_pred_labels = np.argmax(y_pred_proba, axis=1)
                log(INFO, "Converting probability predictions to labels (argmax), shape: %s", y_pred_proba.shape)
            else:
                y_pred_labels = y_pred_proba  # Already labels
            
            # Get true labels
            y_true = test_data.get_label()
            
            # Save dataset with predictions to results directory
            output_path = save_predictions_to_csv(test_data, y_pred_proba, server_round, "results", y_true)
            
            # Compute metrics using the predictions
            predictions = y_pred_labels  # Use the converted labels for metrics
            pred_proba = y_pred_proba    # The original probabilities for plots that need them

            # Ensure pred_proba has the correct shape for multi-class
            if len(pred_proba.shape) == 1 or pred_proba.shape[1] == 1:
                 # If predict gives labels or single class proba, try predict_proba if available
                 try:
                     # Note: XGBoost predict() with multi:softmax directly gives labels.
                     # To get probabilities, the objective might need to be multi:softprob
                     log(WARNING, "Predict output seems 1D, attempting to handle for multi-class probability plots...")
                     if model_params.get('objective') == 'multi:softmax':
                         # Create dummy probabilities centered around the predicted class
                         num_classes = model_params.get('num_class', 11) # Default to 11 if not set
                         pred_proba = np.zeros((len(predictions), num_classes))
                         for i, label in enumerate(predictions):
                             if 0 <= int(label) < num_classes: # Check bounds
                                pred_proba[i, int(label)] = 0.9 # Assign high prob to predicted
                                other_prob = 0.1 / max(1, (num_classes - 1))
                                for j in range(num_classes):
                                    if j != int(label):
                                        pred_proba[i,j] = other_prob
                             else:
                                 log(WARNING, f"Prediction label {label} out of bounds [0, {num_classes-1}]")
                                 # Assign uniform probability as fallback if label is invalid
                                 pred_proba[i, :] = 1.0 / num_classes
                         log(WARNING, "Reconstructed dummy probabilities for multi:softmax. Plots may be inaccurate. Consider using 'multi:softprob' objective for better probability estimates.")
                     else: # Cannot determine probabilities
                         pred_proba = None
                 except AttributeError:
                     log(WARNING, "Could not get probabilities, ROC and PR curves will not be generated.")
                     pred_proba = None
                 except Exception as e:
                     log(WARNING, f"Error processing probabilities: {e}. ROC/PR plots skipped.")
                     pred_proba = None
            elif pred_proba.shape[1] != model_params.get('num_class', 11):
                 log(WARNING, f"Probability shape mismatch ({pred_proba.shape[1]} columns vs {model_params.get('num_class', 11)} classes). Plots may fail.")
                 # Attempt to proceed, but plots requiring probabilities might error out

            # Calculate metrics
            # Ensure y_test is integer type for log_loss if using one-hot encoding
            y_test_int = y_true.astype(int)
            num_classes_actual = model_params.get('num_class', 11)

            if pred_proba is not None and len(pred_proba.shape) > 1 and pred_proba.shape[1] == num_classes_actual:
                 try:
                     # Manual clipping to replace deprecated eps parameter
                     epsilon = 1e-15
                     pred_proba_clipped = np.clip(pred_proba, epsilon, 1 - epsilon)
                     loss = log_loss(y_test_int, pred_proba_clipped, labels=range(num_classes_actual))
                 except ValueError as e:
                     log(WARNING, f"ValueError during log_loss calculation: {e}. Setting loss to high value.")
                     loss = 100.0 # Assign a high loss value
                     log(WARNING, f"y_test unique: {np.unique(y_test_int)}, shape: {y_test_int.shape}")
                     log(WARNING, f"pred_proba shape: {pred_proba.shape}")
                     log(WARNING, f"pred_proba sample: {pred_proba[:5]}")
            else:
                 log(WARNING, "Calculating log_loss using one-hot encoding due to missing/invalid probabilities.")
                 try:
                     predictions_int = np.array(predictions).astype(int)
                     # Manual clipping to replace deprecated eps parameter
                     epsilon = 1e-15
                     one_hot_predictions = np.eye(num_classes_actual)[predictions_int]
                     one_hot_clipped = np.clip(one_hot_predictions, epsilon, 1 - epsilon)
                     loss = log_loss(y_test_int, one_hot_clipped, labels=range(num_classes_actual))
                 except ValueError as e:
                     log(WARNING, f"ValueError during one-hot log_loss calculation: {e}. Setting loss to high value.")
                     loss = 100.0 # Assign a high loss value
                     log(WARNING, f"y_test unique: {np.unique(y_test_int)}, shape: {y_test_int.shape}")
                     log(WARNING, f"predictions unique: {np.unique(predictions)}, shape: {predictions.shape}")


            accuracy = accuracy_score(y_true, predictions)
            precision = precision_score(y_true, predictions, average='weighted', zero_division=0)
            recall = recall_score(y_true, predictions, average='weighted', zero_division=0)
            f1 = f1_score(y_true, predictions, average='weighted', zero_division=0)
            cm = confusion_matrix(y_true, predictions, labels=range(num_classes_actual)) # Ensure labels match num_classes

            log(INFO, f"Centralized eval round {server_round} - Loss: {loss:.4f}, Accuracy: {accuracy:.4f}, F1: {f1:.4f}")

            # --- Generate and Save Plots ---
            output_dir = config.get("output_dir", "results") # Get output dir from config or default
            # Instead of creating a separate plots directory, save directly in output_dir
            log(INFO, f"Saving evaluation plots to: {output_dir}")

            # Update class names to use the global authoritative mapping
            class_names = get_class_names_list()

            # Plot Confusion Matrix
            cm_path = os.path.join(output_dir, f"confusion_matrix_round_{server_round}.png")
            plot_confusion_matrix(cm, class_names[:num_classes_actual], cm_path) # Use actual num_classes

            # Plot Class Distribution
            dist_path = os.path.join(output_dir, f"class_distribution_round_{server_round}.png")
            plot_class_distribution(y_test_int, predictions.astype(int), class_names[:num_classes_actual], dist_path)

            # Plot ROC and Precision-Recall Curves (only if probabilities are available and valid)
            if pred_proba is not None and len(pred_proba.shape) > 1 and pred_proba.shape[1] == num_classes_actual:
                roc_path = os.path.join(output_dir, f"roc_curves_round_{server_round}.png")
                plot_roc_curves(y_test_int, pred_proba, class_names[:num_classes_actual], roc_path)

                pr_path = os.path.join(output_dir, f"precision_recall_curves_round_{server_round}.png")
                plot_precision_recall_curves(y_test_int, pred_proba, class_names[:num_classes_actual], pr_path)
            else:
                 log(WARNING, f"Skipping ROC and PR curve generation due to unavailable/invalid probabilities (shape: {pred_proba.shape if pred_proba is not None else 'None'}).")
            # --- End Plot Generation ---

            # Return metrics
            return loss, {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}

    return evaluate_model



class CyclicClientManager(SimpleClientManager):
    """Provides a cyclic client selection rule."""

    def sample(
        self,
        num_clients: int,
        min_num_clients: Optional[int] = None,
        criterion: Optional[Criterion] = None,
    ) -> List[ClientProxy]:
        """Sample a number of Flower ClientProxy instances."""

        # Block until at least num_clients are connected.
        if min_num_clients is None:
            min_num_clients = num_clients
        self.wait_for(min_num_clients)

        # Sample clients which meet the criterion
        available_cids = list(self.clients)
        if criterion is not None:
            available_cids = [
                cid for cid in available_cids if criterion.select(self.clients[cid])
            ]

        if num_clients > len(available_cids):
            log(
                INFO,
                "Sampling failed: number of available clients"
                " (%s) is less than number of requested clients (%s).",
                len(available_cids),
                num_clients,
            )
            return []

        # Return all available clients
        return [self.clients[cid] for cid in available_cids]

def check_convergence(metrics_history: List[Dict], patience: int = 3, min_delta: float = 0.001) -> bool:
    """
    Check if training has converged based on loss history.
    
    Args:
        metrics_history (List[Dict]): List of metrics from previous rounds
        patience (int): Number of rounds to wait for improvement before stopping
        min_delta (float): Minimum change in loss to be considered an improvement
        
    Returns:
        bool: True if training should stop (converged), False otherwise
    """
    if len(metrics_history) < patience + 1:
        return False
    
    # Extract recent losses (mlogloss)
    recent_losses = []
    for metrics in metrics_history[-(patience + 1):]:
        loss = metrics.get('mlogloss', metrics.get('loss', float('inf')))
        recent_losses.append(loss)
    
    # Calculate improvements between consecutive rounds
    improvements = []
    for i in range(len(recent_losses) - 1):
        improvement = recent_losses[i] - recent_losses[i + 1]
        improvements.append(improvement)
    
    # Check if all recent improvements are below threshold
    converged = all(imp < min_delta for imp in improvements)
    
    if converged:
        log(INFO, "Early stopping triggered: No significant improvement in last %d rounds", patience)
        log(INFO, "Recent losses: %s", recent_losses)
        log(INFO, "Recent improvements: %s", improvements)
    
    return converged

def reset_metrics_history():
    """Reset the global metrics history (useful for new training runs)."""
    global METRICS_HISTORY
    METRICS_HISTORY = []
    log(INFO, "Metrics history reset for new training run")

def add_metrics_to_history(metrics: Dict):
    """Add metrics from current round to history for convergence tracking."""
    global METRICS_HISTORY
    METRICS_HISTORY.append(metrics.copy())
    log(INFO, "Added metrics to history. Total rounds tracked: %d", len(METRICS_HISTORY))

def should_stop_early(patience: int = 3, min_delta: float = 0.001) -> bool:
    """
    Check if early stopping should be triggered based on current metrics history.
    
    Args:
        patience (int): Number of rounds to wait for improvement
        min_delta (float): Minimum improvement threshold
        
    Returns:
        bool: True if training should stop early
    """
    global METRICS_HISTORY
    return check_convergence(METRICS_HISTORY, patience, min_delta)
