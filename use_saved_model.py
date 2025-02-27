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
    except Exception as e:
        log(INFO, "Could not get feature importance: %s", str(e))
    
    # Get model parameters
    try:
        params = model.get_params()
        log(INFO, "Model parameters: %s", params)
    except Exception as e:
        log(INFO, "Could not get model parameters: %s", str(e))


def clean_data_for_xgboost(df):
    """
    Clean data for XGBoost by handling infinity values and extremely large numbers.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    # Create a copy to avoid modifying the original
    cleaned_df = df.copy()
    
    # Replace infinity values with NaN
    cleaned_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # Cap extremely large values (adjust threshold as needed)
    numeric_cols = cleaned_df.select_dtypes(include=['float64', 'int64']).columns
    for col in numeric_cols:
        # Get the 99th percentile as a reference
        threshold = cleaned_df[col].quantile(0.99) * 10
        # If threshold is too small, use a default large value
        if threshold < 1e6:
            threshold = 1e6
        # Cap values and log the changes
        mask = cleaned_df[col] > threshold
        if mask.sum() > 0:
            log(INFO, "Capping %d extreme values in column '%s'", mask.sum(), col)
            cleaned_df.loc[mask, col] = np.nan
    
    return cleaned_df


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
        
        # Load data
        if args.has_labels:
            try:
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
                
            except Exception as e:
                log(INFO, "Error processing labeled data: %s", str(e))
                log(INFO, "Falling back to unlabeled data processing")
                
                # Fall back to unlabeled data processing
                data = pd.read_csv(args.data_path)
                log(INFO, "Data columns: %s", data.columns.tolist())
                
                # Clean data for XGBoost
                data = clean_data_for_xgboost(data)
                
                # Convert to DMatrix (without label)
                dmatrix = xgb.DMatrix(data, missing=np.nan)
                
                # Make predictions
                predictions = predict_with_saved_model(args.model_path, dmatrix, args.output_path)
                
        else:
            # If data doesn't have labels, load as pandas DataFrame
            data = pd.read_csv(args.data_path)
            
            # Check if data has expected features
            log(INFO, "Data columns: %s", data.columns.tolist())
            
            # Handle Timestamp column if present
            if 'Timestamp' in data.columns:
                log(INFO, "Dropping Timestamp column as it's not needed for prediction")
                data = data.drop(columns=['Timestamp'])
            
            # Clean data for XGBoost
            data = clean_data_for_xgboost(data)
            
            # Handle categorical columns
            categorical_cols = []
            for col in data.columns:
                if data[col].dtype == 'object' or data[col].dtype.name == 'category':
                    if col not in ['Label', 'Timestamp']:  # Skip label and timestamp
                        categorical_cols.append(col)
                        data[col] = data[col].astype('category').cat.codes
            
            if categorical_cols:
                log(INFO, "Converted categorical columns to codes: %s", categorical_cols)
            
            # Convert to DMatrix with enable_categorical=True if there are categorical features
            if categorical_cols:
                dmatrix = xgb.DMatrix(data, enable_categorical=True, missing=np.nan)
            else:
                dmatrix = xgb.DMatrix(data, missing=np.nan)
            
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