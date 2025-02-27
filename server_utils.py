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
    """Return aggregated metrics for evaluation."""
    total_num = sum([num for num, _ in eval_metrics])
    
    # Log the raw metrics received from clients
    log(INFO, f"Received metrics from {len(eval_metrics)} clients")
    for i, (num, metrics) in enumerate(eval_metrics):
        log(INFO, f"Client {i+1} metrics: {metrics.keys()}")
        if "loss" in metrics:
            log(INFO, f"Client {i+1} loss: {metrics['loss']}")
    
    # Check if we're in prediction mode or evaluation mode
    first_metrics = eval_metrics[0][1]
    is_prediction_mode = first_metrics.get("prediction_mode", False)
    
    # Add detailed logging about prediction mode
    log(INFO, f"PREDICTION MODE STATUS: {'ENABLED' if is_prediction_mode else 'DISABLED'}")
    
    # Log individual client prediction mode status
    prediction_modes = [metrics.get("prediction_mode", False) for _, metrics in eval_metrics]
    log(INFO, f"Client prediction modes: {prediction_modes}")
    
    if is_prediction_mode:
        # Aggregate prediction statistics
        total_predictions = sum([metrics.get("total_predictions", 0)for num, metrics in eval_metrics])
        malicious_predictions = sum([metrics.get("malicious_predictions", 0)for num, metrics in eval_metrics])
        benign_predictions = sum([metrics.get("benign_predictions", 0)for num, metrics in eval_metrics])
        
        # Calculate loss if available
        if all("loss" in metrics for _, metrics in eval_metrics):
            # Log individual client losses for analysis
            client_losses = [metrics["loss"] for _, metrics in eval_metrics]
            log(INFO, f"Individual client losses: {client_losses}")
            
            loss = sum([metrics["loss"] * num for num, metrics in eval_metrics]) / total_num
            log(INFO, f"PREDICTION MODE - Aggregated loss calculation: sum(loss*num)={sum([metrics['loss'] * num for num, metrics in eval_metrics])}, total_num={total_num}, result={loss}")
        else:
            loss = 0.0
            log(INFO, "PREDICTION MODE - Loss not available in all client metrics")
        
        # Create aggregated metrics dictionary
        aggregated_metrics = {
            "total_predictions": total_predictions,
            "malicious_predictions": malicious_predictions,
            "benign_predictions": benign_predictions,
            "prediction_mode": is_prediction_mode,
            "loss": loss
        }
        
        log(INFO, f"Aggregated prediction metrics: {aggregated_metrics}")
    else:
        # Aggregate evaluation metrics
        if all("precision" in metrics for _, metrics in eval_metrics):
            precision = sum([metrics["precision"] * num for num, metrics in eval_metrics]) / total_num
        else:
            precision = 0.0
            
        if all("recall" in metrics for _, metrics in eval_metrics):
            recall = sum([metrics["recall"] * num for num, metrics in eval_metrics]) / total_num
        else:
            recall = 0.0
            
        if all("f1" in metrics for _, metrics in eval_metrics):
            f1 = sum([metrics["f1"] * num for num, metrics in eval_metrics]) / total_num
        else:
            f1 = 0.0
            
        if all("loss" in metrics for _, metrics in eval_metrics):
            # Log individual client losses for analysis
            client_losses = [metrics["loss"] for _, metrics in eval_metrics]
            log(INFO, f"Individual client losses: {client_losses}")
            
            loss = sum([metrics["loss"] * num for num, metrics in eval_metrics]) / total_num
            log(INFO, f"EVALUATION MODE - Aggregated loss calculation: sum(loss*num)={sum([metrics['loss'] * num for num, metrics in eval_metrics])}, total_num={total_num}, result={loss}")
        else:
            loss = 0.0
            log(INFO, "EVALUATION MODE - Loss not available in all client metrics")
            
        # Aggregate confusion matrix elements
        tn = sum([metrics.get("true_negatives", 0) for _, metrics in eval_metrics])
        fp = sum([metrics.get("false_positives", 0) for _, metrics in eval_metrics])
        fn = sum([metrics.get("false_negatives", 0) for _, metrics in eval_metrics])
        tp = sum([metrics.get("true_positives", 0) for _, metrics in eval_metrics])
        
        # Create aggregated metrics dictionary
        aggregated_metrics = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "true_negatives": tn,
            "false_positives": fp,
            "false_negatives": fn,
            "true_positives": tp,
            "prediction_mode": is_prediction_mode,
            "loss": loss
        }
        
        log(INFO, f"Aggregated evaluation metrics: {aggregated_metrics}")
    
    # Save aggregated results
    save_evaluation_results(aggregated_metrics, "aggregated")
    
    return loss, aggregated_metrics

def save_predictions_to_csv(data, predictions, round_num: int, output_dir: str = None, true_labels=None):
    """
    Save dataset with predictions to CSV in the specified directory.
    
    Args:
        data: Original data (not used in this simplified version)
        predictions: Prediction labels
        round_num (int): Round number
        output_dir (str, optional): Directory to save results to. If None, uses the default results directory.
        true_labels (array, optional): True labels if available
        
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
        'prediction_type': ['malicious' if p == 1 else 'benign' for p in predictions]
    }
    
    # Add true labels if available and have the same length as predictions
    if true_labels is not None and len(true_labels) == len(predictions):
        log(INFO, "Including true labels in the output CSV (same length as predictions)")
        predictions_dict['true_label'] = true_labels
        predictions_dict['true_label_type'] = ['malicious' if t == 1 else 'benign' for t in true_labels]
    elif true_labels is not None:
        log(INFO, f"True labels available but length mismatch: predictions={len(predictions)}, true_labels={len(true_labels)}. Not including true labels in output.")
        
    predictions_df = pd.DataFrame(predictions_dict)
    
    # Save to CSV in the specified directory
    output_path = os.path.join(output_dir, f"predictions_round_{round_num}.csv")
    predictions_df.to_csv(output_path, index=False)
    log(INFO, "Predictions saved to: %s", output_path)
    
    return output_path

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
