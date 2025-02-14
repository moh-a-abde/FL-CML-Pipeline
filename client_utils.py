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
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, classification_report
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
        params,
        train_method,
        is_prediction_only=False,
        unlabeled_dmatrix=None
    ):
        """
        Initialize the XGBoost Flower client.

        Args:
            train_dmatrix: Training data in DMatrix format
            valid_dmatrix: Validation data in DMatrix format
            num_train (int): Number of training samples
            num_val (int): Number of validation samples
            num_local_round (int): Number of local training rounds
            params (dict): XGBoost parameters
            train_method (str): Training method ('bagging' or 'cyclic')
            is_prediction_only (bool): Flag indicating if the client is used for prediction only
            unlabeled_dmatrix: Unlabeled data in DMatrix format
        """
        self.train_dmatrix = train_dmatrix
        self.valid_dmatrix = valid_dmatrix
        self.num_train = num_train
        self.num_val = num_val
        self.num_local_round = num_local_round
        self.params = params
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
        global_round = int(ins.config["global_round"])
        
        if global_round == 1:
            # First round: train from scratch
            bst = xgb.train(
                self.params,
                self.train_dmatrix,
                num_boost_round=self.num_local_round,
                evals=[(self.valid_dmatrix, "validate"), (self.train_dmatrix, "train")],
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
        
        # Generate predictions for validation data
        y_pred_proba = bst.predict(self.valid_dmatrix)
        y_pred_labels = y_pred_proba.astype(int)
        
        # Get ground truth labels
        y_true = self.valid_dmatrix.get_label()
        
        # Compute metrics for labeled data
        precision = precision_score(y_true, y_pred_labels, average='weighted')
        recall = recall_score(y_true, y_pred_labels, average='weighted')
        f1 = f1_score(y_true, y_pred_labels, average='weighted')
        error_rate = 1 - precision
        
        # Generate confusion matrix
        conf_matrix = confusion_matrix(y_true, y_pred_labels)
        tn, fp, fn, tp = conf_matrix.ravel()
        
        # Now make predictions on unlabeled data if available
        unlabeled_predictions = None
        if self.unlabeled_dmatrix is not None:
            log(INFO, "Making predictions on unlabeled data")
            unlabeled_pred = bst.predict(self.unlabeled_dmatrix)
            unlabeled_pred_labels = unlabeled_pred.astype(int)
            
            # Save predictions to file
            predictions_df = pd.DataFrame({
                'predicted_label': unlabeled_pred_labels,
                'confidence': unlabeled_pred,
                'prediction_type': ['malicious' if p == 1 else 'benign' for p in unlabeled_pred_labels]
            })
            
            # Save predictions
            output_dir = "results"
            os.makedirs(output_dir, exist_ok=True)
            round_num = ins.config.get("global_round", "final")
            output_path = os.path.join(output_dir, f"predictions_round_{round_num}.csv")
            predictions_df.to_csv(output_path, index=False)
            log(INFO, f"Predictions saved to: {output_path}")
            
            # Count predictions
            unlabeled_predictions = {
                "total_predictions": len(unlabeled_pred_labels),
                "malicious_predictions": int(np.sum(unlabeled_pred_labels == 1)),
                "benign_predictions": int(np.sum(unlabeled_pred_labels == 0)),
                "predictions_file": output_path
            }

        # Create metrics dictionary
        metrics = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp)
        }
        
        # Add unlabeled predictions if available
        if unlabeled_predictions:
            metrics.update(unlabeled_predictions)

        return EvaluateRes(
            status=Status(code=Code.OK, message="Success"),
            loss=float(error_rate),
            num_examples=self.num_val,
            metrics=metrics
        )

