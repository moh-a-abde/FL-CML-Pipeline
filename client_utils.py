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
        y_train = self.train_dmatrix.get_label()
        class_counts = np.bincount(y_train.astype(int))
        log(INFO, f"Training data class distribution: Benign={class_counts[0]}, Malicious={class_counts[1]}")
        
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
        
        # Generate predictions with custom threshold
        y_pred_proba = bst.predict(self.valid_dmatrix)
        
        # Log raw prediction probabilities for a sample of data points
        log(INFO, f"Raw prediction probabilities (first 10): {y_pred_proba[:10]}")
        log(INFO, f"Prediction probability histogram: {np.histogram(y_pred_proba, bins=10)[0]}")
        
        # Get ground truth labels before threshold selection
        y_true = self.valid_dmatrix.get_label()
        
        # Log ground truth distribution
        true_counts = np.bincount(y_true.astype(int))
        log(INFO, f"Ground truth distribution: Benign={true_counts[0]}, Malicious={true_counts[1]}")
        
        # Try different thresholds
        thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        best_threshold = 0.5
        best_balance = float('inf')
        
        for threshold in thresholds:
            temp_labels = (y_pred_proba > threshold).astype(int)
            temp_counts = np.bincount(temp_labels.astype(int))
            benign_count = temp_counts[0] if len(temp_counts) > 0 else 0
            malicious_count = temp_counts[1] if len(temp_counts) > 1 else 0
            
            # Calculate class balance ratio (closer to 1 is better)
            true_benign = true_counts[0]
            true_malicious = true_counts[1]
            
            # Calculate how well this threshold preserves the true class distribution
            balance_score = abs((benign_count/malicious_count) - (true_benign/true_malicious)) if malicious_count > 0 else float('inf')
            
            log(INFO, f"Threshold {threshold}: Benign={benign_count}, Malicious={malicious_count}, Balance Score={balance_score:.4f}")
            
            # Update best threshold if this one gives better class balance
            if balance_score < best_balance:
                best_balance = balance_score
                best_threshold = threshold
        
        log(INFO, f"Selected best threshold: {best_threshold} with balance score: {best_balance:.4f}")
        
        # Use the best threshold for actual predictions
        THRESHOLD = best_threshold
        y_pred_labels = (y_pred_proba > THRESHOLD).astype(int)
    
        # Log prediction distribution
        pred_counts = np.bincount(y_pred_labels.astype(int))
        log(INFO, f"Prediction distribution: Benign={pred_counts[0]}, Malicious={pred_counts[1]}")
        log(INFO, f"Prediction probabilities range: [{y_pred_proba.min():.3f}, {y_pred_proba.max():.3f}]")
        
        # Compute metrics for labeled data
        precision = precision_score(y_true, y_pred_labels, average='weighted')
        recall = recall_score(y_true, y_pred_labels, average='weighted')
        f1 = f1_score(y_true, y_pred_labels, average='weighted')
        loss = -np.mean(y_true * np.log(y_pred_proba + 1e-10) + (1 - y_true) * np.log(1 - y_pred_proba + 1e-10))
        
        # Log detailed metrics
        log(INFO, f"Evaluation metrics - Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, Loss: {loss:.4f}")
        
        # Generate confusion matrix
        conf_matrix = confusion_matrix(y_true, y_pred_labels)
        tn, fp, fn, tp = conf_matrix.ravel()
        log(INFO, f"Confusion matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")
        
        # Create base metrics dictionary
        metrics = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "loss": float(loss),  # Add loss to metrics dictionary
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp),
            "num_predictions": self.num_val
        }
        
        # Add unlabeled predictions if available
        if self.unlabeled_dmatrix is not None:
            # Get predictions
            unlabeled_pred_proba = bst.predict(self.unlabeled_dmatrix)
            
            # Log unlabeled prediction distribution
            log(INFO, f"Unlabeled prediction probabilities range: [{unlabeled_pred_proba.min():.3f}, {unlabeled_pred_proba.max():.3f}]")
            log(INFO, f"Unlabeled prediction probability histogram: {np.histogram(unlabeled_pred_proba, bins=10)[0]}")
            
            # Use the same threshold determined for validation data
            unlabeled_pred_labels = (unlabeled_pred_proba > THRESHOLD).astype(int)
            
            # Log prediction distribution
            unlabeled_counts = np.bincount(unlabeled_pred_labels.astype(int))
            benign_count = unlabeled_counts[0] if len(unlabeled_counts) > 0 else 0
            malicious_count = unlabeled_counts[1] if len(unlabeled_counts) > 1 else 0
            log(INFO, f"Unlabeled predictions with threshold {THRESHOLD}: Benign={benign_count}, Malicious={malicious_count}")
            
            # Save predictions using the server_utils function
            round_num = ins.config.get("global_round", "final")
            
            # Check if output directory is provided in config
            output_dir = ins.config.get("output_dir", "results")
            
            output_path = save_predictions_to_csv(
                None,  # We don't need the data anymore since we simplified save_predictions_to_csv
                unlabeled_pred_labels,
                round_num,
                output_dir=output_dir
            )
            
            # Add prediction metrics
            metrics.update({
                "total_predictions": len(unlabeled_pred_labels),
                "malicious_predictions": int(np.sum(unlabeled_pred_labels == 1)),
                "benign_predictions": int(np.sum(unlabeled_pred_labels == 0)),
                "predictions_file": output_path
            })

        return EvaluateRes(
            status=Status(code=Code.OK, message="Success"),
            loss=float(loss),
            num_examples=self.num_val,
            metrics=metrics
        )
