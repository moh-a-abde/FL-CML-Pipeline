"""
ray_tune_xgboost.py

This script implements Ray Tune for hyperparameter optimization of XGBoost models in the
federated learning pipeline. It leverages the existing data processing pipeline while
adding a tuning layer to find optimal hyperparameters.

Key Components:
- Ray Tune integration for hyperparameter search
- XGBoost parameter space definition
- Multi-class evaluation metrics (precision, recall, F1)
- Optimal model selection and persistence
"""

import os
import argparse
import json
import xgboost as xgb
import pandas as pd
import numpy as np
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.hyperopt import HyperOptSearch
from hyperopt import hp
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, log_loss
import logging
from ray.air.config import RunConfig

# Import existing data processing code
from dataset import load_csv_data, transform_dataset_to_dmatrix
from utils import BST_PARAMS

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def train_xgboost(config, train_df: pd.DataFrame, test_df: pd.DataFrame):
    """
    Training function for XGBoost that can be used with Ray Tune.
    Args:
        config (dict): Hyperparameters to use for training
        train_df (pd.DataFrame): Training data as DataFrame
        test_df (pd.DataFrame): Test data as DataFrame
    """
    logger.info(f"Starting trial with config: {config}")
    # Instantiate FeatureProcessor inside the trial
    from dataset import FeatureProcessor
    processor = FeatureProcessor()
    processor.fit(train_df)
    train_processed = processor.transform(train_df, is_training=True)
    test_processed = processor.transform(test_df, is_training=False)
    train_features = train_processed.drop(columns=['label'])
    train_labels = train_processed['label'].astype(int)
    test_features = test_processed.drop(columns=['label'])
    test_labels = test_processed['label'].astype(int)
    
    # Create DMatrix objects inside the function to avoid pickling issues
    train_data = xgb.DMatrix(train_features, label=train_labels, missing=np.nan)
    test_data = xgb.DMatrix(test_features, label=test_labels, missing=np.nan)
    
    # Prepare the XGBoost parameters
    params = {
        # Fixed parameters
        'objective': 'multi:softmax',
        'num_class': 3,  # benign (0), dns_tunneling (1), icmp_tunneling (2)
        'eval_metric': ['mlogloss', 'merror'],
        
        # Tunable parameters from config - convert float values to integers where needed
        'max_depth': int(config['max_depth']),
        'min_child_weight': int(config['min_child_weight']),
        'eta': config['eta'],
        'subsample': config['subsample'],
        'colsample_bytree': config['colsample_bytree'],
        'reg_alpha': config['reg_alpha'],
        'reg_lambda': config['reg_lambda'],
        
        # Fixed parameters for reproducibility
        'seed': 42
    }
    
    # Optional GPU support if available
    if config.get('tree_method') == 'gpu_hist':
        params['tree_method'] = 'gpu_hist'
    
    # Store evaluation results
    results = {}
    
    # Train the model
    bst = xgb.train(
        params,
        train_data,
        num_boost_round=int(config['num_boost_round']),
        evals=[(test_data, 'eval'), (train_data, 'train')],
        evals_result=results,
        verbose_eval=False
    )
    
    # Get the final evaluation metrics
    final_iteration = len(results['eval']['mlogloss']) - 1
    eval_mlogloss = results['eval']['mlogloss'][final_iteration]
    eval_merror = results['eval']['merror'][final_iteration]
    
    # Make predictions for more detailed metrics
    y_pred = bst.predict(test_data)
    y_true = test_data.get_label()
    
    # Compute multi-class metrics
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    accuracy = accuracy_score(y_true, y_pred)
    
    # Return metrics to Ray Tune instead of using tune.report
    return {
        "mlogloss": eval_mlogloss,
        "merror": eval_merror,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy
    }

def tune_xgboost(train_file=None, test_file=None, data_file=None, num_samples=100, cpus_per_trial=1, gpu_fraction=None, output_dir="./tune_results"):
    """
    Run hyperparameter tuning for XGBoost using Ray Tune.
    
    Args:
        train_file (str): Path to the training CSV data file (required for HPO)
        test_file (str): Path to the testing CSV data file (ignored for HPO validation split)
        data_file (str): Path to a single CSV data file (fallback if train_file not provided, will be split)
        num_samples (int): Number of hyperparameter combinations to try (default: 100)
        cpus_per_trial (int): CPUs to allocate per trial
        gpu_fraction (float): Fraction of GPU to use per trial (if None, no GPU is used)
        output_dir (str): Directory to save results
        
    Returns:
        dict: Best hyperparameters found
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load and prepare data
    if train_file and test_file:
        logger.info(f"Loading training data from {train_file}")
        train_data = load_csv_data(train_file)["train"].to_pandas()
        
        logger.info(f"Loading testing data from {test_file}")
        test_data = load_csv_data(test_file)["train"].to_pandas()
        
        # Ensure label column is correctly handled - case insensitive check
        for df in [train_data]:
            if 'label' not in df.columns and 'Label' in df.columns:
                df['label'] = df['Label']
                
        # Check if test data has a label column
        if 'label' not in test_data.columns and 'Label' in test_data.columns:
            test_data['label'] = test_data['Label']
            
        # If test data doesn't have a label column, create a dummy one
        if 'label' not in test_data.columns:
            logger.warning("Test data doesn't have a label column. Creating a dummy label column with zeros.")
            test_data['label'] = 0
                
        # Create feature processor and fit it on training data
        from dataset import FeatureProcessor
        
        # Create and fit the feature processor
        processor = FeatureProcessor()
        processor.fit(train_data)
        
        # Process the data, but don't create DMatrix yet (to avoid pickling issues)
        train_processed = processor.transform(train_data, is_training=True)
        test_processed = processor.transform(test_data, is_training=False)
        
        # Extract features and labels
        train_features = train_processed.drop(columns=['label'])
        train_labels = train_processed['label'].astype(int)
        
        test_features = test_processed.drop(columns=['label'])
        test_labels = test_processed['label'].astype(int)
        
        logger.info(f"Training data size: {len(train_features)}")
        logger.info(f"Validation data size: {len(test_features)}")
    else:
        # Fall back to original behavior with single file and splitting
        logger.info(f"Loading data from {data_file}")
        data = load_csv_data(data_file)["train"].to_pandas()
        
        # Ensure label column is correctly handled - case insensitive check
        if 'label' not in data.columns and 'Label' in data.columns:
            data['label'] = data['Label']
            
        # Import this function only if needed
        from dataset import train_test_split
        
        # Split the data, but use pandas directly
        from sklearn.model_selection import train_test_split as sklearn_split
        train_processed, test_processed = sklearn_split(data, test_size=0.2, random_state=42)
        
        # Process with feature processor
        processor = FeatureProcessor()
        processor.fit(train_processed)
        
        train_processed = processor.transform(train_processed, is_training=True)
        test_processed = processor.transform(test_processed, is_training=False)
        
        # Extract features and labels
        train_features = train_processed.drop(columns=['label'])
        train_labels = train_processed['label'].astype(int)
        
        test_features = test_processed.drop(columns=['label'])
        test_labels = test_processed['label'].astype(int)
        
        logger.info(f"Training data size: {len(train_features)}")
        logger.info(f"Validation data size: {len(test_features)}")
    
    # Define the search space using hyperopt's hp module
    search_space = {
        "max_depth": hp.quniform("max_depth", 3, 10, 1),
        "min_child_weight": hp.quniform("min_child_weight", 1, 20, 1),
        "reg_alpha": hp.loguniform("reg_alpha", np.log(1e-3), np.log(10.0)),
        "reg_lambda": hp.loguniform("reg_lambda", np.log(1e-3), np.log(10.0)),
        "eta": hp.loguniform("eta", np.log(1e-3), np.log(0.3)),
        "subsample": hp.uniform("subsample", 0.5, 1.0),
        "colsample_bytree": hp.uniform("colsample_bytree", 0.5, 1.0),
        "num_boost_round": hp.quniform("num_boost_round", 50, 200, 1)
    }
    if gpu_fraction is not None and gpu_fraction > 0:
        search_space["tree_method"] = hp.choice("tree_method", ["gpu_hist"])
    
    # Create a wrapper function that includes the data DataFrames
    def _train_with_data_wrapper(config):
        return train_xgboost(config, train_data.copy(), test_data.copy()) # Pass copies of FULL DataFrames with labels

    # Set up HyperOptSearch
    algo = HyperOptSearch(
        search_space,
        metric="mlogloss",
        mode="min"
    )
    scheduler = ASHAScheduler(
        max_t=200,
        grace_period=10,
        reduction_factor=2,
        metric="mlogloss",
        mode="min"
    )
    
    # Initialize the tuner
    logger.info("Starting hyperparameter tuning")
    
    # Run the tuning with updated API
    tuner = tune.Tuner(
        _train_with_data_wrapper,
        tune_config=tune.TuneConfig(
            scheduler=scheduler,
            num_samples=num_samples,
            search_alg=algo
        ),
        param_space={},  # search space is handled by HyperOptSearch
        run_config=RunConfig(
            local_dir=output_dir,
            name="xgboost_tune"
        )
    )
    
    # Execute the hyperparameter search
    results = tuner.fit()
    
    # Get the best trial
    best_result = results.get_best_result(metric="mlogloss", mode="min")
    best_config = best_result.config
    best_metrics = best_result.metrics
    
    # Log the best configuration and metrics
    logger.info("Best hyperparameters found:")
    logger.info(json.dumps(best_config, indent=2))
    logger.info("Best metrics:")
    logger.info(f"  mlogloss: {best_metrics['mlogloss']:.4f}")
    logger.info(f"  merror: {best_metrics['merror']:.4f}")
    logger.info(f"  precision: {best_metrics['precision']:.4f}")
    logger.info(f"  recall: {best_metrics['recall']:.4f}")
    logger.info(f"  f1: {best_metrics['f1']:.4f}")
    logger.info(f"  accuracy: {best_metrics['accuracy']:.4f}")
    
    # Save the best hyperparameters to a file
    best_params_file = os.path.join(output_dir, "best_params.json")
    with open(best_params_file, 'w') as f:
        json.dump(best_config, f, indent=2)
    logger.info(f"Best parameters saved to {best_params_file}")
    
    # Train a final model with the best parameters
    train_final_model(best_config, train_features, train_labels, test_features, test_labels, output_dir)
    
    return best_config

def train_final_model(config, train_features, train_labels, test_features, test_labels, output_dir):
    """
    Train a final model using the best hyperparameters found.
    
    Args:
        config (dict): Best hyperparameters
        train_features (pd.DataFrame): Training features
        train_labels (pd.Series): Training labels
        test_features (pd.DataFrame): Test features
        test_labels (pd.Series): Test labels
        output_dir (str): Directory to save the model
    """
    # Create DMatrix objects
    train_data = xgb.DMatrix(train_features, label=train_labels, missing=np.nan)
    test_data = xgb.DMatrix(test_features, label=test_labels, missing=np.nan)
    
    # Prepare the XGBoost parameters
    params = {
        # Fixed parameters
        'objective': 'multi:softmax',
        'num_class': 3,
        'eval_metric': ['mlogloss', 'merror'],
        
        # Best parameters from tuning - convert float values to integers where needed
        'max_depth': int(config['max_depth']),
        'min_child_weight': int(config['min_child_weight']),
        'eta': config['eta'],
        'subsample': config['subsample'],
        'colsample_bytree': config['colsample_bytree'],
        'reg_alpha': config['reg_alpha'],
        'reg_lambda': config['reg_lambda'],
        
        # Set the seed for reproducibility
        'seed': 42
    }
    
    # Optional GPU support if available
    if config.get('tree_method') == 'gpu_hist':
        params['tree_method'] = 'gpu_hist'
    
    # Train the final model
    logger.info("Training final model with best parameters")
    final_model = xgb.train(
        params,
        train_data,
        num_boost_round=int(config['num_boost_round']),
        evals=[(test_data, 'eval'), (train_data, 'train')],
        verbose_eval=True
    )
    
    # Save the model
    model_path = os.path.join(output_dir, "best_model.json")
    final_model.save_model(model_path)
    logger.info(f"Final model saved to {model_path}")
    
    # Evaluate the model
    y_pred = final_model.predict(test_data)
    y_true = test_data.get_label()
    
    # Generate performance metrics
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    accuracy = accuracy_score(y_true, y_pred)
    
    # Log final performance
    logger.info("Final model performance:")
    logger.info(f"  Precision: {precision:.4f}")
    logger.info(f"  Recall: {recall:.4f}")
    logger.info(f"  F1 Score: {f1:.4f}")
    logger.info(f"  Accuracy: {accuracy:.4f}")
    
    return final_model

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Ray Tune for XGBoost hyperparameter optimization")
    parser.add_argument("--data-file", type=str, help="Path to single CSV data file (optional if train and test files are provided)")
    parser.add_argument("--train-file", type=str, help="Path to training CSV data file")
    parser.add_argument("--test-file", type=str, help="Path to testing CSV data file")
    parser.add_argument("--num-samples", type=int, default=10, help="Number of hyperparameter combinations to try")
    parser.add_argument("--cpus-per-trial", type=int, default=1, help="CPUs per trial")
    parser.add_argument("--gpu-fraction", type=float, default=None, help="GP