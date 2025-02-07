from typing import Dict, List, Optional
from sklearn.metrics import precision_score, recall_score, f1_score
from logging import INFO
import xgboost as xgb
from flwr.common.logger import log
from flwr.common import Parameters, Scalar
from flwr.server.client_manager import SimpleClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.criterion import Criterion
from utils import BST_PARAMS


def eval_config(rnd: int) -> Dict[str, str]:
    """Return a configuration with global epochs."""
    config = {
        "global_round": str(rnd),
    }
    return config


def fit_config(rnd: int) -> Dict[str, str]:
    """Return a configuration with global epochs."""
    config = {
        "global_round": str(rnd),
    }
    return config


def evaluate_metrics_aggregation(eval_metrics):
    """Return aggregated metrics for evaluation."""
    total_num = sum([num for num, _ in eval_metrics])
    
    # Weighted average across clients
    precision_aggregated = sum([metrics["precision"] * num for num, metrics in eval_metrics]) / total_num
    recall_aggregated = sum([metrics["recall"] * num for num, metrics in eval_metrics]) / total_num
    f1_aggregated = sum([metrics["f1"] * num for num, metrics in eval_metrics]) / total_num
    
    metrics_aggregated = {
        "precision": precision_aggregated,
        "recall": recall_aggregated,
        "f1": f1_aggregated,
    }
    return metrics_aggregated



def get_evaluate_fn(test_data):
    """Return a function for centralised evaluation."""

    def evaluate_fn(
        server_round: int, parameters: Parameters, config: Dict[str, Scalar]
    ):
        # If at the first round, skip the evaluation
        if server_round == 0:
            return 0, {}
        else:
            bst = xgb.Booster(params=BST_PARAMS)
            for para in parameters.tensors:
                para_b = bytearray(para)

            # Load global model
            bst.load_model(para_b)
            
            # Predict on test data
            y_pred = bst.predict(test_data)
            
            # For multi:softmax, y_pred contains class labels directly
            y_pred_labels = y_pred.astype(int)  # Ensure labels are integers
            
            # Get true labels
            y_true = test_data.get_label()
            
            # Compute precision, recall, and f1 score
            precision = precision_score(y_true, y_pred_labels, average='weighted')
            recall = recall_score(y_true, y_pred_labels, average='weighted')
            f1 = f1_score(y_true, y_pred_labels, average='weighted')

            log(INFO, f"Precision = {precision}, Recall = {recall}, F1 Score = {f1} at round {server_round}")

            return 0, {
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }

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
