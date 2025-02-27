#!/usr/bin/env python
"""
use_saved_model.py

This script demonstrates how to load and use a saved XGBoost model from the federated learning process
to make predictions on new data.

Usage:
    python use_saved_model.py --model_path <path_to_model> --data_path <path_to_data> --output_path <path_for_predictions>

Example:
    python use_saved_model.py --model_path outputs/2023-05-01/12-34-56/final_model.json --data_path data/test_data.csv --output_path predictions.csv
"""

import argparse
import os
from logging import INFO
import pandas as pd
import numpy as np
import xgboost as xgb
from flwr.common.logger import log

from server_utils import load_saved_model, predict_with_saved_model
from dataset import transform_dataset_to_dmatrix, load_csv_data


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
        required=True,
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
    
    return parser.parse_args()


def main():
    """Main function to load model and make predictions."""
    args = parse_args()
    
    # Check if model file exists
    if not os.path.exists(args.model_path):
        log(INFO, "Error: Model file not found: %s", args.model_path)
        return
    
    # Check if data file exists
    if not os.path.exists(args.data_path):
        log(INFO, "Error: Data file not found: %s", args.data_path)
        return
    
    try:
        log(INFO, "Loading data from: %s", args.data_path)
        
        # Load data
        if args.has_labels:
            # Use the dataset loading function if data has labels
            dataset = load_csv_data(args.data_path)["test"]
            dataset.set_format("pandas")
            data = dataset.to_pandas()
            
            # Convert to DMatrix
            dmatrix = transform_dataset_to_dmatrix(dataset)
            
            # Get true labels for evaluation
            y_true = dmatrix.get_label()
            
            # Make predictions
            predictions = predict_with_saved_model(args.model_path, dmatrix, args.output_path)
            
            # Evaluate if data has labels
            y_pred_labels = predictions.astype(int)
            
            # Calculate accuracy
            accuracy = np.mean(y_pred_labels == y_true)
            log(INFO, "Accuracy: %.4f", accuracy)
            
            # Print confusion matrix
            from sklearn.metrics import confusion_matrix, classification_report
            cm = confusion_matrix(y_true, y_pred_labels)
            log(INFO, "Confusion Matrix:\n%s", cm)
            
            # Print classification report
            report = classification_report(y_true, y_pred_labels)
            log(INFO, "Classification Report:\n%s", report)
            
        else:
            # If data doesn't have labels, load as pandas DataFrame
            data = pd.read_csv(args.data_path)
            
            # Check if data has expected features
            log(INFO, "Data columns: %s", data.columns.tolist())
            
            # Convert to DMatrix (without label)
            dmatrix = xgb.DMatrix(data)
            
            # Make predictions
            predictions = predict_with_saved_model(args.model_path, dmatrix, args.output_path)
            
        log(INFO, "Predictions saved to: %s", args.output_path)
        
    except (FileNotFoundError, ValueError, TypeError, xgb.core.XGBoostError) as e:
        log(INFO, "Error: %s", str(e))
    except Exception as e:
        log(INFO, "Unexpected error: %s", str(e))
        raise


if __name__ == "__main__":
    main() 