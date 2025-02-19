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
from datetime import datetime
import numpy as np

def eval_config(rnd: int) -> Dict[str, str]:
    """Return a configuration with global epochs."""
    config = {
        "global_round": str(rnd),
    }
    return config

def save_evaluation_results(eval_metrics: Dict, round_num: int, output_dir: str = "results"):
    """
    Save evaluation results for each round.
    """
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
    
    log(INFO, f"Evaluation results saved to: {output_path}")

def fit_config(rnd: int) -> Dict[str, str]:
    """Return a configuration with global epochs."""
    config = {
        "global_round": str(rnd),
    }
    return config


def evaluate_metrics_aggregation(eval_metrics):
    """Return aggregated metrics for evaluation."""
    # Check if we're in prediction mode or evaluation mode
    first_metrics = eval_metrics[0][1]
    is_prediction_mode = (
        "total_predictions" in first_metrics or 
        "num_predictions" in first_metrics
    )
    
    if is_prediction_mode:
        # Aggregate prediction statistics without multiplying by num
        total_predictions = sum([metrics.get("total_predictions", 0) for num, metrics in eval_metrics])
        malicious_predictions = sum([metrics.get("malicious_predictions", 0) for num, metrics in eval_metrics])
        benign_predictions = sum([metrics.get("benign_predictions", 0) for num, metrics in eval_metrics])
        
        metrics_aggregated = {
            "total_predictions": total_predictions,
            "malicious_predictions": malicious_predictions,
            "benign_predictions": benign_predictions,
            "prediction_mode": True
        }
    else:
        total_num = sum([num for num, _ in eval_metrics])
        # Aggregate evaluation metrics
        precision_aggregated = sum([metrics["precision"] * num for num, metrics in eval_metrics]) / total_num
        recall_aggregated = sum([metrics["recall"] * num for num, metrics in eval_metrics]) / total_num
        f1_aggregated = sum([metrics["f1"] * num for num, metrics in eval_metrics]) / total_num
        
        # Aggregate confusion matrix metrics
        tn_aggregated = sum([metrics["true_negatives"] * num for num, metrics in eval_metrics]) / total_num
        fp_aggregated = sum([metrics["false_positives"] * num for num, metrics in eval_metrics]) / total_num
        fn_aggregated = sum([metrics["false_negatives"] * num for num, metrics in eval_metrics]) / total_num
        tp_aggregated = sum([metrics["true_positives"] * num for num, metrics in eval_metrics]) / total_num
        
        metrics_aggregated = {
            "precision": precision_aggregated,
            "recall": recall_aggregated,
            "f1": f1_aggregated,
            "true_negatives": tn_aggregated,
            "false_positives": fp_aggregated,
            "false_negatives": fn_aggregated,
            "true_positives": tp_aggregated,
            "prediction_mode": False
        }

    # Save aggregated results
    save_evaluation_results(metrics_aggregated, "aggregated")
    
    return metrics_aggregated
    
def save_predictions_to_csv(data, predictions, round_num: int, output_dir: str = "results"):
    """
    Save dataset with predictions to CSV in the results directory.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Get feature names from the original dataset
    feature_names = ['id.orig_h', 'id.orig_p', 'id.resp_h', 'id.resp_p', 'proto', 
                    'duration', 'orig_bytes', 'resp_bytes', 'conn_state', 'local_orig',
                    'local_resp', 'missed_bytes', 'history', 'orig_pkts', 'orig_ip_bytes',
                    'resp_pkts', 'resp_ip_bytes']
    
    # Convert DMatrix to DataFrame
    df = data.get_data()
    if isinstance(df, tuple):
        features, labels = df
        df = pd.DataFrame(features, columns=feature_names)  # Use feature names
        df['true_label'] = labels
    
    # Add predictions and label them as malicious/benign
    df['predicted_label'] = predictions
    df['traffic_type'] = df['predicted_label'].map({1: 'malicious', 0: 'benign'})
    
    # Save to CSV in the results directory
    output_path = os.path.join(output_dir, f"predictions_round_{round_num}.csv")
    df.to_csv(output_path, index=False)
    log(INFO, f"Predictions saved to: {output_path}")
    
    return output_path

def get_evaluate_fn(test_data, unlabeled_data=None):
    """Return a function for centralised evaluation."""

    def evaluate_fn(
        server_round: int, parameters: Parameters, config: Dict[str, Scalar]
    ):
        if server_round == 0:
            return 0, {}
        
        # Load model
        bst = xgb.Booster(params=BST_PARAMS)
        for para in parameters.tensors:
            para_b = bytearray(para)
        bst.load_model(para_b)
        
        # First evaluate on labeled test data
        log(INFO, "Evaluating on labeled dataset with %d samples", test_data.num_row())
        y_pred = bst.predict(test_data)
        y_pred_labels = y_pred.astype(int)
        y_true = test_data.get_label()
        
        # Compute metrics on labeled data
        precision = precision_score(y_true, y_pred_labels, average='weighted')
        recall = recall_score(y_true, y_pred_labels, average='weighted')
        f1 = f1_score(y_true, y_pred_labels, average='weighted')
        conf_matrix = confusion_matrix(y_true, y_pred_labels)
        
        # Create base metrics dictionary
        metrics = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "true_negatives": int(conf_matrix[0][0]),
            "false_positives": int(conf_matrix[0][1]),
            "false_negatives": int(conf_matrix[1][0]),
            "true_positives": int(conf_matrix[1][1])
        }
        
        # If unlabeled data is provided, make predictions on it
        if unlabeled_data is not None:
            log(INFO, "Making predictions on unlabeled data")
            unlabeled_pred = bst.predict(unlabeled_data)
            unlabeled_pred_labels = unlabeled_pred.astype(int)
            
            # Save predictions
            output_path = save_predictions_to_csv(unlabeled_data, unlabeled_pred_labels, server_round, "results")
            metrics["predictions_file"] = output_path
            
            # Add prediction statistics
            metrics.update({
                "total_predictions": len(unlabeled_pred_labels),
                "malicious_predictions": int(np.sum(unlabeled_pred_labels == 1)),
                "benign_predictions": int(np.sum(unlabeled_pred_labels == 0))
            })

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
