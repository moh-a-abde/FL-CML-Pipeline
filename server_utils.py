from typing import Dict, List, Optional
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from logging import INFO
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
    aggregated_metrics["loss"] = loss  # Keep as "loss" for compatibility
    aggregated_metrics["mlogloss"] = loss  # Also store as mlogloss
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
        prediction_types (list, optional): List of prediction type strings (e.g., 'benign', 'dns_tunneling', etc.)
        
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
        # Default mapping for multi-class predictions
        label_mapping = {0: 'benign', 1: 'dns_tunneling', 2: 'icmp_tunneling'}
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

    def evaluate_fn(
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
            
            # Compute metrics
            precision = precision_score(y_true, y_pred_labels, average='weighted')
            recall = recall_score(y_true, y_pred_labels, average='weighted')
            f1 = f1_score(y_true, y_pred_labels, average='weighted')
            
            # Generate confusion matrix
            conf_matrix = confusion_matrix(y_true, y_pred_labels)
            
            # Create metrics dictionary
            metrics = {
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "true_negatives": int(conf_matrix[0][0]),
                "false_positives": int(conf_matrix[0][1]),
                "false_negatives": int(conf_matrix[1][0]),
                "true_positives": int(conf_matrix[1][1]),
                "predictions_file": output_path
            }

            log(INFO, f"Precision = {precision}, Recall = {recall}, F1 Score = {f1} at round {server_round}")
            log(INFO, f"Dataset with predictions saved to: {output_path}")

            return 0, metrics

    return evaluate_fn



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
