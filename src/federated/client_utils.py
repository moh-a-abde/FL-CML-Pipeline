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

from logging import INFO, ERROR, WARNING
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
from src.federated.utils import save_predictions_to_csv, get_class_names_list
import importlib.util
from sklearn.utils.class_weight import compute_sample_weight

# Import shared utilities for Phase 3 deduplication
from src.core.shared_utils import DMatrixFactory, XGBoostParamsBuilder


def get_default_model_params():
    """
    Get default XGBoost parameters for UNSW_NB15 multi-class classification.
    
    DEPRECATED: Use XGBoostParamsBuilder.build_params() instead.
    Kept for backward compatibility during Phase 3 migration.
    
    Returns:
        dict: Default XGBoost parameters
    """
    log(WARNING, "get_default_model_params() is deprecated. Use XGBoostParamsBuilder.build_params() instead.")
    return XGBoostParamsBuilder.build_params()


def load_tuned_params():
    """
    Try to load tuned parameters if available.
    
    DEPRECATED: Use XGBoostParamsBuilder.build_params(use_tuned=True) instead.
    Kept for backward compatibility during Phase 3 migration.
    
    Returns:
        dict: Tuned parameters if available, else default parameters
    """
    log(WARNING, "load_tuned_params() is deprecated. Use XGBoostParamsBuilder.build_params(use_tuned=True) instead.")
    return XGBoostParamsBuilder.build_params(use_tuned=True)


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
        cid: Client ID for logging purposes
    """

    def __init__(
        self,
        train_dmatrix,
        valid_dmatrix,
        num_train,
        num_val,
        num_local_round,
        cid,
        params=None,
        train_method="cyclic",
        is_prediction_only=False,
        unlabeled_dmatrix=None,
        use_tuned_params=True,
        config_manager=None
    ):
        """
        Initialize the XGBoost Flower client.

        Args:
            train_dmatrix: Training data in DMatrix format
            valid_dmatrix: Validation data in DMatrix format
            num_train (int): Number of training samples
            num_val (int): Number of validation samples
            num_local_round (int): Number of local training rounds
            cid: Client ID for logging purposes
            params (dict): XGBoost parameters (defaults to ConfigManager or fallback if None)
            train_method (str): Training method ('bagging' or 'cyclic')
            is_prediction_only (bool): Flag indicating if the client is used for prediction only
            unlabeled_dmatrix: Unlabeled data in DMatrix format
            use_tuned_params (bool): Whether to use tuned parameters if available
            config_manager (ConfigManager): ConfigManager instance for getting model parameters
        """
        self.train_dmatrix = train_dmatrix
        self.valid_dmatrix = valid_dmatrix
        self.num_train = num_train
        self.num_val = num_val
        self.num_local_round = num_local_round
        self.cid = cid
        
        # Set model parameters using centralized builder (Phase 3 migration)
        if params is not None:
            self.params = params
            log(INFO, "Using provided XGBoost parameters")
        else:
            # Use centralized parameter builder with proper priority handling
            self.params = XGBoostParamsBuilder.build_params(
                config_manager=config_manager,
                use_tuned=use_tuned_params
            )
            log(INFO, "Using XGBoost parameters from centralized builder")
            
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
        Perform local boosting rounds on the input model using centralized DMatrixFactory.

        Args:
            bst_input: Input XGBoost model

        Returns:
            xgb.Booster: Updated model after local training
            
        Note:
            For bagging: returns only the last N trees
            For cyclic: returns the entire model
        """
        # Get training data with weights
        y_train = self.train_dmatrix.get_label()
        y_train_int = y_train.astype(int)
        
        # Compute sample weights for class imbalance
        try:
            sample_weights = compute_sample_weight('balanced', y_train_int)
        except Exception as e:
            log(INFO, f"Error computing sample weights in _local_boost: {e}. Using uniform weights.")
            sample_weights = np.ones(len(y_train_int))
            
        # Create weighted DMatrix using centralized factory (Phase 3 migration)
        dtrain_weighted = DMatrixFactory.create_weighted_dmatrix(
            base_dmatrix=self.train_dmatrix,
            weights=sample_weights,
            feature_names=self.train_dmatrix.feature_names
        )
        
        # Use xgb.train with early stopping for better performance
        bst = xgb.train(
            self.params,
            dtrain_weighted,
            num_boost_round=self.num_local_round,
            xgb_model=bst_input,  # Continue training from existing model
            evals=[(self.valid_dmatrix, "validate"), (dtrain_weighted, "train")],
            early_stopping_rounds=10,  # Reduced for faster convergence in FL
            verbose_eval=False  # Reduced verbosity for performance
        )

        # Handle model extraction based on training method
        if self.train_method == "bagging":
            # For bagging, extract only the last N trees
            total_trees = bst.num_boosted_rounds()
            start_tree = max(0, total_trees - self.num_local_round)
            bst_extracted = bst[start_tree:total_trees]
            return bst_extracted
        else:
            # For cyclic, return the entire model
            return bst

    def fit(self, ins: FitIns) -> FitRes:
        """
        Perform local model training using centralized utilities.
        """
        # --- PHASE 1: Aggressive Regularization (Overrides any loaded/tuned params) --- REMOVED

        y_train = self.train_dmatrix.get_label()

        # --- Check if labels are empty ---
        if y_train.size == 0:
            log(ERROR, f"Client {self.cid}: Training DMatrix has no labels. Cannot proceed with fit.")
            return FitRes(
                status=Status(code=Code.FIT_NOT_IMPLEMENTED, message="Training data is missing labels."),
                parameters=Parameters(tensor_type="", tensors=[]), # Return empty params
                num_examples=0,
                metrics={}
            )
        # --- End Check ---
        
        # Ensure labels are integers for compute_sample_weight
        y_train_int = y_train.astype(int)

        class_counts = np.bincount(y_train_int)
        
        # Log class distribution for all classes - FIXED: Match server mapping order
        class_names = get_class_names_list()
        for i, count in enumerate(class_counts):
            if i < len(class_names):
                class_name = class_names[i]
            else:
                class_name = f'unknown_{i}'
            log(INFO, f"Training data class {class_name}: {count}")
        
        # --- Reduced Debugging for Performance ---
        log(INFO, f"Unique values in y_train_int: {np.unique(y_train_int)}")
        log(INFO, f"Min/Max values in y_train_int: {np.min(y_train_int)} / {np.max(y_train_int)}")
        # --- End Debugging ---

        # Compute sample weights for class imbalance
        try:
            sample_weights = compute_sample_weight('balanced', y_train_int) # Use integer labels
            log(INFO, f"Successfully computed sample weights. Shape: {sample_weights.shape}, dtype: {sample_weights.dtype}")
        except IndexError as e:
            log(INFO, f"IndexError during compute_sample_weight: {e}")
            log(INFO, f"Unique labels causing issue: {np.unique(y_train_int)}")
            # As a fallback, use uniform weights
            log(INFO, "Falling back to uniform sample weights.")
            sample_weights = np.ones(len(y_train_int))
        except Exception as e:
            log(INFO, f"Other error during compute_sample_weight: {e}")
            log(INFO, "Falling back to uniform sample weights due to unexpected error.")
            sample_weights = np.ones(len(y_train_int))
            
        # Create a new DMatrix with weights using centralized factory (Phase 3 migration)
        dtrain_weighted = DMatrixFactory.create_weighted_dmatrix(
            base_dmatrix=self.train_dmatrix,
            weights=sample_weights,
            feature_names=self.train_dmatrix.feature_names
        )

        global_round = int(ins.config["global_round"])
        
        if global_round == 1:
            # First round: train from scratch with sample weights - REDUCED early stopping for FL
            bst = xgb.train(
                self.params,
                dtrain_weighted,
                num_boost_round=self.num_local_round,
                evals=[(self.valid_dmatrix, "validate"), (dtrain_weighted, "train")],
                early_stopping_rounds=10,  # Reduced from 20 for faster FL
                verbose_eval=False  # Reduced verbosity for performance
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
        # Since objective is multi:softprob, predict() outputs probabilities
        y_pred_proba = bst.predict(self.valid_dmatrix)
        # Get class labels from probabilities
        y_pred_labels = np.argmax(y_pred_proba, axis=1)
        
        # Get ground truth labels
        y_true = self.valid_dmatrix.get_label()
        
        # Log ground truth distribution
        true_counts = np.bincount(y_true.astype(int))
        class_names = get_class_names_list()
        num_classes_actual = len(class_names) # Or get from self.params if needed
        for i, count in enumerate(true_counts):
            if i < len(class_names):
                class_name = class_names[i]
            else:
                class_name = f'unknown_{i}'
            log(INFO, f"Ground truth {class_name}: {count}")
        
        # Compute multi-class metrics using predicted labels
        # Add zero_division=0 to handle cases where a class might not be predicted
        precision = precision_score(y_true, y_pred_labels, average='weighted', zero_division=0)
        recall = recall_score(y_true, y_pred_labels, average='weighted', zero_division=0)
        f1 = f1_score(y_true, y_pred_labels, average='weighted', zero_division=0)
        accuracy = accuracy_score(y_true, y_pred_labels)
        
        # Calculate mlogloss using probabilities
        epsilon = 1e-15  # Small constant to avoid log(0)
        y_true_int = y_true.astype(int)
        # Ensure y_true_int does not contain labels outside the expected range [0, num_classes-1]
        valid_indices = (y_true_int >= 0) & (y_true_int < num_classes_actual)
        if not np.all(valid_indices):
            log(WARNING, f"Found {np.sum(~valid_indices)} labels outside expected range [0, {num_classes_actual-1}]. Clamping for mlogloss calculation.")
            y_true_int = np.clip(y_true_int, 0, num_classes_actual - 1)
            # Optionally filter data if clamping is not desired:
            # y_true_int = y_true_int[valid_indices]
            # y_pred_proba = y_pred_proba[valid_indices]
            
        y_true_one_hot = np.eye(num_classes_actual)[y_true_int]
        
        # Ensure y_pred_proba has the correct shape and handle potential issues
        if y_pred_proba.shape == (len(y_true_int), num_classes_actual):
            # Clip probabilities to avoid log(0)
            y_pred_proba_clipped = np.clip(y_pred_proba, epsilon, 1 - epsilon)
            mlogloss = -np.mean(np.sum(y_true_one_hot * np.log(y_pred_proba_clipped), axis=1))
        else:
            log(WARNING, f"Shape mismatch for mlogloss: y_pred_proba shape {y_pred_proba.shape}, expected ({len(y_true_int)}, {num_classes_actual}). Skipping mlogloss.")
            mlogloss = -1.0 # Indicate failure to calculate

        # Compute confusion matrix using predicted labels
        try:
            # Explicitly provide labels to ensure consistent matrix size
            conf_matrix = confusion_matrix(y_true, y_pred_labels, labels=range(num_classes_actual))
        except Exception as e:
            log(WARNING, f"Error computing confusion matrix: {str(e)}")
            # Create empty confusion matrix with the correct size
            conf_matrix = np.zeros((num_classes_actual, num_classes_actual), dtype=int)

        # Generate detailed classification report using predicted labels
        try:
            # Ensure target_names matches the actual number of classes
            unique_labels = np.unique(np.concatenate((y_true.astype(int), y_pred_labels))) # Get labels present in data
            target_names_filtered = [class_names[i] for i in range(num_classes_actual) if i in unique_labels]
            # Ensure we use labels consistent with target_names_filtered
            labels_for_report = [i for i in range(num_classes_actual) if i in unique_labels]

            class_report = classification_report(
                y_true, 
                y_pred_labels, 
                labels=labels_for_report, 
                target_names=target_names_filtered, 
                zero_division=0
            )
            log(INFO, f"Classification Report:\n{class_report}")
        except Exception as e:
            log(WARNING, f"Error generating classification report: {str(e)}")
        
        # Log evaluation metrics
        log(INFO, f"Precision (weighted): {precision:.4f}")
        log(INFO, f"Recall (weighted): {recall:.4f}")
        log(INFO, f"F1 Score (weighted): {f1:.4f}")
        log(INFO, f"Accuracy: {accuracy:.4f}")
        log(INFO, f"Multi-class Log Loss: {mlogloss:.4f}")
        log(INFO, f"Confusion Matrix shape: {conf_matrix.shape}")

        # Save predictions for this round
        global_round = int(ins.config["global_round"])
        from src.federated.utils import save_predictions_to_csv
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
            "mlogloss": float(mlogloss)
        }
        
        return EvaluateRes(
            status=Status(code=Code.OK, message="Success"),
            loss=float(mlogloss),  # Use mlogloss as the primary loss metric
            num_examples=self.num_val,
            metrics=metrics
        )


# Alias for backward compatibility
XGBClient = XgbClient 