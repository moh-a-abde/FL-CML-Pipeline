"""
client_utils.py

This module implements the XGBoost client functionality for Federated Learning using Flower framework.
It provides the core client-side operations including model training, evaluation, and parameter handling.

Key Components:
- XGBoost client implementation
- Model training and evaluation methods
- Parameter serialization and deserialization
- Metrics computation (precision, recall, F1)
"""

from logging import INFO
import xgboost as xgb
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, classification_report, accuracy_score
import flwr as fl
from flwr.common.logger import log
from flwr.common import (
    Code,
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    GetParametersIns,
    GetParametersRes,
    Parameters,
    Status,
)
from flwr.common.typing import Code
from flwr.common import Status
import numpy as np
import pandas as pd
import os
from server_utils import save_predictions_to_csv
import importlib.util
from sklearn.utils.class_weight import compute_sample_weight

# Default XGBoost parameters for multi-class classification
BST_PARAMS = {
    'objective': 'multi:softmax',  # Multi-class classification
    'num_class': 3,  # Classes: benign (0), dns_tunneling (1), icmp_tunneling (2)
    'eval_metric': ['mlogloss', 'merror'],  # Multi-class metrics
    'learning_rate': 0.05,  # Reduced to match utils.py and prevent overfitting
    'max_depth': 6,
    'min_child_weight': 1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'scale_pos_weight': [1.0, 2.0, 1.0]  # More moderate weight adjustment for dns_tunneling
}

# Try to import tuned parameters if available
try:
    # Check if tuned_params.py exists
    tuned_params_path = os.path.join(os.path.dirname(__file__), "tuned_params.py")
    if os.path.exists(tuned_params_path):
        # Dynamically import the tuned parameters
        spec = importlib.util.spec_from_file_location("tuned_params", tuned_params_path)
        tuned_params_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(tuned_params_module)
        
        # Use the tuned parameters
        TUNED_PARAMS = tuned_params_module.TUNED_PARAMS
        log(INFO, "Using tuned XGBoost parameters from Ray Tune optimization")
    else:
        TUNED_PARAMS = BST_PARAMS.copy()
except Exception as e:
    log(INFO, f"Could not load tuned parameters: {str(e)}")
    TUNED_PARAMS = BST_PARAMS.copy()

class XgbClient(fl.client.Client):
    """
    A Flower client implementing federated learning for XGBoost models.
    
    This class handles local model training, evaluation, and parameter exchange
    with the federated learning server.

    Attributes:
        train_dmatrix: Training data in XGBoost's DMatrix format
        valid_dmatrix: Validation data in XGBoost's DMatrix format
        num_train (int): Number of training samples
        num_val (int): Number of validation samples
        num_local_round (int): Number of local training rounds
        params (dict): XGBoost training parameters
        train_method (str): Training method ('bagging' or 'cyclic')
        is_prediction_only (bool): Flag indicating if the client is used for prediction only
        unlabeled_dmatrix: Unlabeled data in XGBoost's DMatrix format
    """

    def __init__(
        self,
        train_dmatrix,
        valid_dmatrix,
        num_train,
        num_val,
        num_local_round,
        params=None,
        train_method="cyclic",
        is_prediction_only=False,
        unlabeled_dmatrix=None,
        use_tuned_params=True
    ):
        """
        Initialize the XGBoost Flower client.

        Args:
            train_dmatrix: Training data in DMatrix format
            valid_dmatrix: Validation data in DMatrix format
            num_train (int): Number of training samples
            num_val (int): Number of validation samples
            num_local_round (int): Number of local training rounds
            params (dict): XGBoost parameters (defaults to BST_PARAMS if None)
            train_method (str): Training method ('bagging' or 'cyclic')
            is_prediction_only (bool): Flag indicating if the client is used for prediction only
            unlabeled_dmatrix: Unlabeled data in DMatrix format
            use_tuned_params (bool): Whether to use tuned parameters if available
        """
        self.train_dmatrix = train_dmatrix
        self.valid_dmatrix = valid_dmatrix
        self.num_train = num_train
        self.num_val = num_val
        self.num_local_round = num_local_round
        
        # Use tuned parameters if available and requested
        if params is not None:
            self.params = params
        elif use_tuned_params:
            self.params = TUNED_PARAMS.copy()
            log(INFO, "Using tuned parameters for XGBoost training")
        else:
            self.params = BST_PARAMS.copy()
            
        self.train_method = train_method
        self.is_prediction_only = is_prediction_only
        self.unlabeled_dmatrix = unlabeled_dmatrix

    def get_parameters(self, ins: GetParametersIns) -> GetParametersRes:
        """
        Return the current local model parameters.

        Args:
            ins (GetParametersIns): Input parameters from server

        Returns:
            GetParametersRes: Empty parameters (XGBoost doesn't use this method)
        """
        _ = (self, ins)
        return GetParametersRes(
            status=Status(
                code=Code.OK,
                message="OK",
            ),
            parameters=Parameters(tensor_type="", tensors=[]),
        )

    def _local_boost(self, bst_input):
        """
        Perform local boosting rounds on the input model.

        Args:
            bst_input: Input XGBoost model

        Returns:
            xgb.Booster: Updated model after local training
            
        Note:
            For bagging: returns only the last N trees
            For cyclic: returns the entire model
        """
        # Update trees based on local training data
        for i in range(self.num_local_round):
            bst_input.update(self.train_dmatrix, bst_input.num_boosted_rounds())

        # Handle model extraction based on training method
        bst = (
            bst_input[
                bst_input.num_boosted_rounds()
                - self.num_local_round : bst_input.num_boosted_rounds()
            ]
            if self.train_method == "bagging"
            else bst_input
        )

        return bst

    def fit(self, ins: FitIns) -> FitRes:
        """
        Perform local model training.
        """
        # --- PHASE 1: Aggressive Regularization (Overrides any loaded/tuned params) ---
        self.params.update({
            'max_depth': 3,
            'reg_lambda': 10.0,  # L2
            'reg_alpha': 2.0,    # L1
            'min_child_weight': 5,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'eta': 0.1
        })

        y_train = self.train_dmatrix.get_label()
        class_counts = np.bincount(y_train.astype(int))
        
        # Log class distribution for all three classes
        class_names = ['benign', 'dns_tunneling', 'icmp_tunneling']
        for i, count in enumerate(class_counts):
            class_name = class_names[i] if i < len(class_names) else f'unknown_{i}'
            log(INFO, f"Training data class {class_name}: {count}")
        
        # Compute sample weights for class imbalance
        sample_weights = compute_sample_weight('balanced', y_train)
        # Create a new DMatrix with weights for training
        dtrain_weighted = xgb.DMatrix(self.train_dmatrix.get_data(), label=y_train, weight=sample_weights, feature_names=self.train_dmatrix.feature_names)

        global_round = int(ins.config["global_round"])
        
        if global_round == 1:
            # First round: train from scratch with sample weights
            bst = xgb.train(
                self.params,
                dtrain_weighted,
                num_boost_round=self.num_local_round,
                evals=[(self.valid_dmatrix, "validate"), (dtrain_weighted, "train")],
                early_stopping_rounds=20,
                verbose_eval=True
            )
        else:
            # Subsequent rounds: update existing model
            bst = xgb.Booster(params=self.params)
            for item in ins.parameters.tensors:
                global_model = bytearray(item)

            # Load and update global model
            bst.load_model(global_model)
            bst = self._local_boost(bst)

        # Serialize model for transmission
        local_model = bst.save_raw("json")
        local_model_bytes = bytes(local_model)

        # Return with status
        return FitRes(
            status=Status(code=Code.OK, message="Success"),
            parameters=Parameters(tensor_type="", tensors=[local_model_bytes]),
            num_examples=self.num_train,
            metrics={}
        )
    
    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        """
        Evaluate the model on validation data and make predictions on unlabeled data.
        """
        # Load global model for evaluation
        bst = xgb.Booster(params=self.params)
        para_b = bytearray()
        for para in ins.parameters.tensors:
            para_b.extend(para)
        bst.load_model(para_b)

        # First evaluate on labeled validation data
        log(INFO, f"Evaluating on labeled dataset with {self.num_val} samples")
        
        # Generate predictions for multi-class classification
        y_pred_proba = bst.predict(self.valid_dmatrix, output_margin=True)  # Get raw predictions for mlogloss
        y_pred_labels = bst.predict(self.valid_dmatrix)  # Get class predictions
        
        # Get ground truth labels
        y_true = self.valid_dmatrix.get_label()
        
        # Log ground truth distribution
        true_counts = np.bincount(y_true.astype(int))
        class_names = ['benign', 'dns_tunneling', 'icmp_tunneling']
        for i, count in enumerate(true_counts):
            class_name = class_names[i] if i < len(class_names) else f'unknown_{i}'
            log(INFO, f"Ground truth {class_name}: {count}")
        
        # Compute multi-class metrics
        precision = precision_score(y_true, y_pred_labels, average='weighted')
        recall = recall_score(y_true, y_pred_labels, average='weighted')
        f1 = f1_score(y_true, y_pred_labels, average='weighted')
        accuracy = accuracy_score(y_true, y_pred_labels)
        
        # Calculate mlogloss manually
        epsilon = 1e-15  # Small constant to avoid log(0)
        y_pred_proba_softmax = np.exp(y_pred_proba) / np.sum(np.exp(y_pred_proba), axis=1, keepdims=True)
        y_true_one_hot = np.zeros_like(y_pred_proba_softmax)
        y_true_one_hot[np.arange(len(y_true)), y_true.astype(int)] = 1
        mlogloss = -np.mean(np.sum(y_true_one_hot * np.log(y_pred_proba_softmax + epsilon), axis=1))
        
        # Compute confusion matrix
        try:
            conf_matrix = confusion_matrix(y_true, y_pred_labels)
        except Exception:
            conf_matrix = np.zeros((3, 3), dtype=int)

        # Generate detailed classification report
        class_report = classification_report(y_true, y_pred_labels, target_names=class_names)
        
        # Log evaluation metrics
        log(INFO, f"Precision (weighted): {precision:.4f}")
        log(INFO, f"Recall (weighted): {recall:.4f}")
        log(INFO, f"F1 Score (weighted): {f1:.4f}")
        log(INFO, f"Accuracy: {accuracy:.4f}")
        log(INFO, f"Multi-class Log Loss: {mlogloss:.4f}")
        log(INFO, f"Confusion Matrix:\n{conf_matrix}")
        log(INFO, f"Classification Report:\n{class_report}")

        # Save predictions for this round
        global_round = int(ins.config["global_round"])
        from server_utils import save_predictions_to_csv
        save_predictions_to_csv(
            data=self.valid_dmatrix,
            predictions=y_pred_labels,
            round_num=global_round,
            output_dir=ins.config.get("output_dir", "results"),
            true_labels=y_true
        )

        # Format metrics in a way that Flower can handle
        metrics = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "accuracy": float(accuracy),
            "mlogloss": float(mlogloss),
            # Store confusion matrix elements as individual metrics
            "conf_00": int(conf_matrix[0][0]),  # benign correct
            "conf_01": int(conf_matrix[0][1]),  # benign misclassified as dns
            "conf_02": int(conf_matrix[0][2]),  # benign misclassified as icmp
            "conf_10": int(conf_matrix[1][0]),  # dns misclassified as benign
            "conf_11": int(conf_matrix[1][1]),  # dns correct
            "conf_12": int(conf_matrix[1][2]),  # dns misclassified as icmp
            "conf_20": int(conf_matrix[2][0]),  # icmp misclassified as benign
            "conf_21": int(conf_matrix[2][1]),  # icmp misclassified as dns
            "conf_22": int(conf_matrix[2][2]),  # icmp correct
            "confusion_matrix": conf_matrix.tolist() if conf_matrix is not None else None
        }
        
        return EvaluateRes(
            status=Status(code=Code.OK, message="Success"),
            loss=float(mlogloss),  # Use mlogloss as the primary loss metric
            num_examples=self.num_val,
            metrics=metrics
        )
