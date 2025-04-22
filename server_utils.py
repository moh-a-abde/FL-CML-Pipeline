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
from utils import BST_PARAMS
import os
import json
import shutil
from datetime import datetime
import pickle
import numpy as np
# Assuming visualization_utils.py is in the same directory or accessible via PYTHONPATH
from visualization_utils import (
    plot_confusion_matrix,
    plot_roc_curves,
    plot_precision_recall_curves,
    plot_class_distribution
)

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
    # aggregated_metrics["loss"] = loss  # REMOVED - Keep as "loss" for compatibility
    aggregated_metrics["mlogloss"] = loss  # Store as mlogloss
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
    # Log aggregated metrics
    log(INFO, "Aggregated metrics:")
    log(INFO, "  Precision (weighted): %f", aggregated_metrics["precision"])
    log(INFO, "  Recall (weighted): %f", aggregated_metrics["recall"])
    log(INFO, "  F1 Score (weighted): %f", aggregated_metrics["f1"])
    log(INFO, "  Accuracy: %f", aggregated_metrics["accuracy"])
    log(INFO, "  Loss (mlogloss): %f", aggregated_metrics["loss"])
    if aggregated_conf_matrix is not None:
        log(INFO, "  Confusion Matrix:\n%s", aggregated_conf_matrix)
    # Save aggregated results
    save_evaluation_results(aggregated_metrics, "aggregated")
    if not (isinstance(loss, (int, float)) and isinstance(aggregated_metrics, dict)):
        log(INFO, "[ERROR] Output of evaluate_metrics_aggregation is not (loss, dict): %s, %s", type(loss), type(aggregated_metrics))
        raise TypeError("evaluate_metrics_aggregation must return (loss, dict)")
    return loss, aggregated_metrics

def save_predictions_to_csv(data, predictions, round_num: int, output_dir: str = None, true_labels=None, prediction_types=None):
    """
    Save dataset with predictions to CSV in the specified directory.
    
    Args:
        data: Original data
        predictions: Prediction labels (class indices)
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
    
    # Create predictions DataFrame
    predictions_dict = {
        'predicted_label': predictions,
    }
    
    # Add prediction types if provided
    if prediction_types is not None:
        predictions_dict['prediction_type'] = prediction_types
    else:
        # Default mapping for UNSW_NB15 multi-class predictions
        label_mapping = {
            0: 'Normal', 
            1: 'Reconnaissance', 
            2: 'Backdoor', 
            3: 'DoS', 
            4: 'Exploits', 
            5: 'Analysis', 
            6: 'Fuzzers', 
            7: 'Worms', 
            8: 'Shellcode', 
            9: 'Generic'
        }
        predictions_dict['prediction_type'] = [label_mapping.get(int(p), 'unknown') for p in predictions]
    
    # Add true labels if available
    if true_labels is not None:
        predictions_dict['true_label'] = true_labels
    
    predictions_df = pd.DataFrame(predictions_dict)
    
    # Save predictions
    output_path = os.path.join(output_dir, f"predictions_round_{round_num}.csv")
    predictions_df.to_csv(output_path, index=False)
    log(INFO, "Predictions saved to: %s", output_path)
    
    return output_path

def load_saved_model(model_path):
    """
    Load a saved XGBoost model from disk.
    
    Args:
        model_path (str): Path to the saved model file (.json or .bin)
        
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
            
            # If that fails too, try with params
            try:
                from utils import BST_PARAMS
                bst = xgb.Booster(params=BST_PARAMS)
                bst.load_model(model_path)
                log(INFO, "Model loaded successfully with params")
                return bst
            except Exception as e3:
                log(INFO, "All loading attempts failed")
                raise ValueError(f"Failed to load model: {str(e)}, {str(e2)}, {str(e3)}")

def predict_with_saved_model(model_path, dmatrix, output_path):
    # Load the model
    model = load_saved_model(model_path)
    
    # Make predictions
    raw_predictions = model.predict(dmatrix)
    
    # Log raw predictions
    log(INFO, "Raw predictions: %s", raw_predictions)
    
    # Log distribution of scores
    log(INFO, "Prediction score distribution - Min: %.4f, Max: %.4f, Mean: %.4f", 
        np.min(raw_predictions), np.max(raw_predictions), np.mean(raw_predictions))
    
    # Convert raw predictions to probabilities if necessary
    # (Assuming a binary classification with a threshold of 0.5)
    probabilities = 1 / (1 + np.exp(-raw_predictions))  # Example for sigmoid transformation
    predicted_labels = (probabilities >= 0.5).astype(int)
    
    # Log predicted class distribution
    unique, counts = np.unique(predicted_labels, return_counts=True)
    log(INFO, "Predicted class distribution: %s", dict(zip(unique, counts)))
    
    # Save predictions to CSV
    predictions_df = pd.DataFrame({
        'predicted_label': predicted_labels,
        'prediction_type': ['benign' if label == 0 else 'malicious' for label in predicted_labels],
        'prediction_score': probabilities
    })
    
    predictions_df.to_csv(output_path, index=False)
    log(INFO, "Predictions saved to: %s", output_path)
    
    return predictions

def get_evaluate_fn(test_data):
    """Return a function for centralised evaluation."""

    def evaluate_model(
        server_round: int, parameters: Parameters, config: Dict[str, Scalar]
    ):
        if server_round == 0:
            return 0, {}
        else:
            bst = xgb.Booster(params=BST_PARAMS)
            for para in parameters.tensors:
                para_b = bytearray(para)

            bst.load_model(para_b)
            
            # Predict on test data
            y_pred = bst.predict(test_data)
            y_pred_labels = y_pred.astype(int)
            
            # Get true labels
            y_true = test_data.get_label()
            
            # Save dataset with predictions to results directory
            output_path = save_predictions_to_csv(test_data, y_pred_labels, server_round, "results", y_true)
            
            # Evaluate
            predictions = bst.predict(test_data)
            pred_proba = bst.predict(test_data, output_margin=False) # Need probabilities for ROC/PR

            # Ensure pred_proba has the correct shape for multi-class
            if len(pred_proba.shape) == 1 or pred_proba.shape[1] == 1:
                 # If predict gives labels or single class proba, try predict_proba if available
                 try:
                     # Note: XGBoost predict() with multi:softmax directly gives labels.
                     # To get probabilities, the objective might need to be multi:softprob
                     log(WARNING, "Predict output seems 1D, attempting to handle for multi-class probability plots...")
                     if BST_PARAMS.get('objective') == 'multi:softmax':
                         # Create dummy probabilities centered around the predicted class
                         num_classes = BST_PARAMS.get('num_class', 10) # Default to 10 if not set
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
            elif pred_proba.shape[1] != BST_PARAMS.get('num_class', 10):
                 log(WARNING, f"Probability shape mismatch ({pred_proba.shape[1]} columns vs {BST_PARAMS.get('num_class', 10)} classes). Plots may fail.")
                 # Attempt to proceed, but plots requiring probabilities might error out

            # Calculate metrics
            # Ensure y_test is integer type for log_loss if using one-hot encoding
            y_test_int = y_true.astype(int)
            num_classes_actual = BST_PARAMS.get('num_class', 10)

            if pred_proba is not None and pred_proba.shape[1] == num_classes_actual:
                 try:
                     loss = log_loss(y_test_int, pred_proba, eps=1e-15, labels=range(num_classes_actual))
                 except ValueError as e:
                     log(WARNING, f"ValueError during log_loss calculation: {e}. Setting loss to high value.")
                     loss = 100.0 # Assign a high loss value
                     log(WARNING, f"y_test unique: {np.unique(y_test_int)}, shape: {y_test_int.shape}")
                     log(WARNING, f"pred_proba shape: {pred_proba.shape}")
                     log(WARNING, f"pred_proba sample: {pred_proba[:5]}")
            else:
                 log(WARNING, "Calculating log_loss using one-hot encoding due to missing/invalid probabilities.")
                 try:
                     loss = log_loss(y_test_int, np.eye(num_classes_actual)[predictions.astype(int)], eps=1e-15, labels=range(num_classes_actual))
                 except ValueError as e:
                     log(WARNING, f"ValueError during one-hot log_loss calculation: {e}. Setting loss to high value.")
                     loss = 100.0 # Assign a high loss value
                     log(WARNING, f"y_test unique: {np.unique(y_test_int)}, shape: {y_test_int.shape}")
                     log(WARNING, f"predictions unique: {np.unique(predictions.astype(int))}, shape: {predictions.shape}")


            accuracy = accuracy_score(y_true, predictions)
            precision = precision_score(y_true, predictions, average='weighted', zero_division=0)
            recall = recall_score(y_true, predictions, average='weighted', zero_division=0)
            f1 = f1_score(y_true, predictions, average='weighted', zero_division=0)
            cm = confusion_matrix(y_true, predictions, labels=range(num_classes_actual)) # Ensure labels match num_classes

            log(INFO, f"Centralized eval round {server_round} - Loss: {loss:.4f}, Accuracy: {accuracy:.4f}, F1: {f1:.4f}")

            # --- Generate and Save Plots ---
            output_dir = config.get("output_dir", "results") # Get output dir from config or default
            plots_dir = os.path.join(output_dir, "plots", f"round_{server_round}")
            os.makedirs(plots_dir, exist_ok=True)
            log(INFO, f"Saving evaluation plots to: {plots_dir}")

            class_names = ['Normal', 'Reconnaissance', 'Backdoor', 'DoS', 'Exploits', 'Analysis', 'Fuzzers', 'Worms', 'Shellcode', 'Generic'] # Make sure this matches your data

            # Plot Confusion Matrix
            cm_path = os.path.join(plots_dir, "confusion_matrix.png")
            plot_confusion_matrix(cm, class_names[:num_classes_actual], cm_path) # Use actual num_classes

            # Plot Class Distribution
            dist_path = os.path.join(plots_dir, "class_distribution.png")
            plot_class_distribution(y_test_int, predictions.astype(int), class_names[:num_classes_actual], dist_path)

            # Plot ROC and Precision-Recall Curves (only if probabilities are available and valid)
            if pred_proba is not None and pred_proba.shape[1] == num_classes_actual:
                roc_path = os.path.join(plots_dir, "roc_curves.png")
                plot_roc_curves(y_test_int, pred_proba, class_names[:num_classes_actual], roc_path)

                pr_path = os.path.join(plots_dir, "precision_recall_curves.png")
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
