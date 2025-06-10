#!/usr/bin/env python
"""
use_saved_model.py

This script demonstrates how to load and use a saved XGBoost model from the
federated learning process to make predictions on new data.

Usage:
    python use_saved_model.py --model_path <path_to_model> --data_path <path_to_data>
    --output_path <path_for_predictions>

Example:
    python use_saved_model.py --model_path outputs/2023-05-01/12-34-56/final_model.json
    --data_path data/test_data.csv --output_path predictions.csv
"""

import argparse
import os
from logging import INFO
import pandas as pd
import numpy as np
import xgboost as xgb
from flwr.common.logger import log
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

from src.federated.utils import load_saved_model
from src.core.dataset import transform_dataset_to_dmatrix, load_csv_data

# Import shared utilities for Phase 3 deduplication
from src.core.shared_utils import DMatrixFactory


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Use a saved XGBoost model to make predictions")

    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the saved model file (.json or .bin)",
    )

    parser.add_argument(
        "--data_path",
        type=str,
        default=None,
        help="Path to the data file (.csv)",
    )

    parser.add_argument(
        "--output_path",
        type=str,
        default="predictions.csv",
        help="Path to save the predictions (default: predictions.csv)",
    )

    parser.add_argument(
        "--has_labels",
        action="store_true",
        help="Specify if the data file contains labels (for evaluation)",
    )

    parser.add_argument(
        "--info_only",
        action="store_true",
        help="Only display model information without making predictions",
    )

    return parser.parse_args()


def display_model_info(model):
    """Display information about the loaded model."""
    log(INFO, "Model Information:")

    # Get number of trees
    num_trees = len(model.get_dump())
    log(INFO, "Number of trees: %d", num_trees)

    # Get feature names if available
    try:
        feature_names = model.feature_names
        if feature_names:
            log(INFO, "Feature names: %s", feature_names)
    except AttributeError:
        log(INFO, "Feature names not available in the model")

    # Get feature importance if available
    try:
        importance = model.get_score(importance_type='weight')
        log(INFO, "Feature importance (top 10):")
        sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]
        for feature, score in sorted_importance:
            log(INFO, "  %s: %.4f", feature, score)
    except (ValueError, KeyError) as error:
        log(INFO, "Could not get feature importance: %s", str(error))

    # Get model parameters
    try:
        params = model.get_params()
        log(INFO, "Model parameters: %s", params)
    except (ValueError, KeyError) as error:
        log(INFO, "Could not get model parameters: %s", str(error))


def clean_data_for_xgboost(data_frame):
    """
    Clean data for XGBoost by handling infinity values and extremely large numbers.

    Args:
        data_frame (pd.DataFrame): Input DataFrame

    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    # Create a copy to avoid modifying the original
    cleaned_df = data_frame.copy()

    # Replace infinity values with NaN
    cleaned_df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Cap extremely large values (adjust threshold as needed)
    numeric_cols = cleaned_df.select_dtypes(include=['float64', 'int64']).columns
    for col in numeric_cols:
        # Get the 99th percentile as a reference
        threshold = cleaned_df[col].quantile(0.99) * 10
        # Use max() to set minimum threshold
        threshold = max(threshold, 1e6)
        # Cap values and log the changes
        mask = cleaned_df[col] > threshold
        if mask.sum() > 0:
            log(INFO, "Capping %d extreme values in column '%s'", mask.sum(), col)
            cleaned_df.loc[mask, col] = np.nan

    return cleaned_df


def save_detailed_predictions(predictions, output_path):
    """
    Save detailed prediction information to CSV for multi-class classification.

    Args:
        predictions (np.ndarray): Raw predictions from the model
        output_path (str): Path to save the predictions
    """
    # Create a DataFrame to store predictions
    results_df = pd.DataFrame()

    # Check if predictions are multi-dimensional (one-hot encoded)
    if predictions.ndim > 1 and predictions.shape[1] > 1:
        # Store raw probabilities
        results_df['raw_probabilities'] = predictions.tolist()

        # Get predicted class (argmax)
        predicted_labels = np.argmax(predictions, axis=1)
        results_df['predicted_label'] = predicted_labels

        # Map numeric predictions to class names
        label_mapping = {0: 'benign', 1: 'dns_tunneling', 2: 'icmp_tunneling'}
        results_df['prediction_type'] = [
            label_mapping.get(int(p), 'unknown') for p in predicted_labels
        ]

        # Store confidence scores (probability of predicted class)
        results_df['prediction_score'] = predictions[
            np.arange(len(predicted_labels)),
            predicted_labels
        ]
    else:
        # For single class predictions
        results_df['predicted_label'] = predictions.astype(int)

        # Map numeric predictions to class names
        label_mapping = {0: 'benign', 1: 'dns_tunneling', 2: 'icmp_tunneling'}
        results_df['prediction_type'] = [
            label_mapping.get(int(p), 'unknown') for p in predictions
        ]

        # Default confidence score of 1.0 for direct class predictions
        results_df['prediction_score'] = 1.0

    # Save to CSV
    results_df.to_csv(output_path, index=False)
    log(INFO, "Saved %d predictions to %s", len(results_df), output_path)

    # Log prediction statistics
    label_counts = results_df['predicted_label'].value_counts()
    log(INFO, "Prediction counts by class:")
    for label, count in label_counts.items():
        class_name = label_mapping.get(int(label), f'unknown_{label}')
        log(INFO, "  %s: %d", class_name, count)

    if 'prediction_score' in results_df.columns:
        log(INFO, "Confidence score statistics: min=%.6f, max=%.6f, mean=%.6f",
            results_df['prediction_score'].min(),
            results_df['prediction_score'].max(),
            results_df['prediction_score'].mean())

    return results_df


def evaluate_labeled_data(model, dataset, output_path):
    """Handle evaluation of labeled data."""
    # Convert to DMatrix
    dmatrix = transform_dataset_to_dmatrix(dataset)

    # Get true labels for evaluation
    y_true = dmatrix.get_label()

    # Make predictions
    raw_predictions = model.predict(dmatrix)

    # Save detailed predictions
    _ = save_detailed_predictions(raw_predictions, output_path)

    # Evaluate if data has labels
    if raw_predictions.ndim > 1:
        y_pred_labels = np.argmax(raw_predictions, axis=1)
    else:
        y_pred_labels = raw_predictions.astype(int)

    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred_labels)
    precision = precision_score(y_true, y_pred_labels, average='weighted')
    recall = recall_score(y_true, y_pred_labels, average='weighted')
    f1_score_val = f1_score(y_true, y_pred_labels, average='weighted')

    # Generate confusion matrix
    conf_matrix = confusion_matrix(y_true, y_pred_labels)

    # Generate classification report
    class_names = ['benign', 'dns_tunneling', 'icmp_tunneling']
    report = classification_report(y_true, y_pred_labels, target_names=class_names)

    # Log evaluation results
    log(INFO, "Evaluation Results:")
    log(INFO, "  Accuracy: %.4f", accuracy)
    log(INFO, "  Precision (weighted): %.4f", precision)
    log(INFO, "  Recall (weighted): %.4f", recall)
    log(INFO, "  F1 Score (weighted): %.4f", f1_score_val)
    log(INFO, "Confusion Matrix:\n%s", conf_matrix)
    log(INFO, "Classification Report:\n%s", report)


def predict_unlabeled_data(model, data_path, output_path):
    """Handle prediction of unlabeled data using centralized DMatrixFactory."""
    # Load unlabeled data
    data = pd.read_csv(data_path)

    # Clean data
    data = clean_data_for_xgboost(data)

    # Convert to DMatrix using centralized factory (Phase 3 migration)
    dmatrix = DMatrixFactory.create_dmatrix(
        features=data,
        labels=None,  # No labels for prediction-only data
        handle_missing=True,
        validate=True,
        log_details=True
    )

    # Make predictions
    raw_predictions = model.predict(dmatrix)

    # Save predictions
    save_detailed_predictions(raw_predictions, output_path)


def main():
    """Main function to load model and make predictions."""
    args = parse_args()

    # Check if model file exists
    if not os.path.exists(args.model_path):
        log(INFO, "Error: Model file not found: %s", args.model_path)
        return

    try:
        # Load the model
        log(INFO, "Loading model from: %s", args.model_path)
        model = load_saved_model(args.model_path)

        # Display model information
        display_model_info(model)

        # If info_only flag is set, exit after displaying model info
        if args.info_only:
            log(INFO, "Info only mode - exiting without making predictions")
            return

        # Check if data path is provided
        if args.data_path is None:
            log(INFO, "No data path provided. Use --data_path to specify data for predictions.")
            return

        # Check if data file exists
        if not os.path.exists(args.data_path):
            log(INFO, "Error: Data file not found: %s", args.data_path)
            return

        log(INFO, "Loading data from: %s", args.data_path)

        # Process data based on whether it has labels
        if args.has_labels:
            try:
                dataset = load_csv_data(args.data_path)["test"]
                dataset.set_format("pandas")
                evaluate_labeled_data(model, dataset, args.output_path)
            except (ValueError, KeyError) as error:
                log(INFO, "Error during evaluation: %s", str(error))
                raise
        else:
            try:
                predict_unlabeled_data(model, args.data_path, args.output_path)
            except (ValueError, KeyError) as error:
                log(INFO, "Error during prediction: %s", str(error))
                raise

    except Exception as error:  # pylint: disable=broad-except
        log(INFO, "Error: %s", str(error))
        raise


if __name__ == "__main__":
    main() 