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
from ray.tune.search.basic_variant import BasicVariantGenerator
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, log_loss
import logging

# Import existing data processing code
from dataset import load_csv_data, train_test_split, transform_dataset_to_dmatrix
from utils import BST_PARAMS

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def train_xgboost(config, train_data, test_data):
    """
    Training function for XGBoost that can be used with Ray Tune.
    
    Args:
        config (dict): Hyperparameters to use for training
        train_data (xgb.DMatrix): Training data in XGBoost DMatrix format
        test_data (xgb.DMatrix): Test data in XGBoost DMatrix format
    """
    # Prepare the XGBoost parameters
    params = {
        # Fixed parameters
        'objective': 'multi:softmax',
        'num_class': 3,  # benign (0), dns_tunneling (1), icmp_tunneling (2)
        'eval_metric': ['mlogloss', 'merror'],
        
        # Tunable parameters from config
        'max_depth': config['max_depth'],
        'min_child_weight': config['min_child_weight'],
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
        num_boost_round=config['num_boost_round'],
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
    
    # Report metrics to Ray Tune
    tune.report(
        mlogloss=eval_mlogloss,
        merror=eval_merror,
        precision=precision,
        recall=recall,
        f1=f1,
        accuracy=accuracy
    )
    
    return bst

def tune_xgboost(data_file, num_samples=10, cpus_per_trial=1, gpu_fraction=None, output_dir="./tune_results"):
    """
    Run hyperparameter tuning for XGBoost using Ray Tune.
    
    Args:
        data_file (str): Path to the CSV data file
        num_samples (int): Number of hyperparameter combinations to try
        cpus_per_trial (int): CPUs to allocate per trial
        gpu_fraction (float): Fraction of GPU to use per trial (if None, no GPU is used)
        output_dir (str): Directory to save results
        
    Returns:
        dict: Best hyperparameters found
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load and prepare data
    logger.info(f"Loading data from {data_file}")
    data = load_csv_data(data_file)["train"].to_pandas()
    
    # Split the data and create DMatrix objects
    train_dmatrix, valid_dmatrix, _ = train_test_split(data, test_fraction=0.2)
    
    logger.info(f"Training data size: {train_dmatrix.num_row()}")
    logger.info(f"Validation data size: {valid_dmatrix.num_row()}")
    
    # Define the search space
    search_space = {
        # Tree structure parameters
        "max_depth": tune.randint(3, 10),
        "min_child_weight": tune.randint(1, 20),
        
        # Regularization parameters
        "reg_alpha": tune.loguniform(1e-3, 10.0),
        "reg_lambda": tune.loguniform(1e-3, 10.0),
        
        # Learning parameters
        "eta": tune.loguniform(1e-3, 0.3),
        "subsample": tune.uniform(0.5, 1.0),
        "colsample_bytree": tune.uniform(0.5, 1.0),
        
        # Number of rounds
        "num_boost_round": tune.randint(50, 200)
    }
    
    # Add GPU-specific parameter if GPU fraction is specified
    if gpu_fraction is not None and gpu_fraction > 0:
        search_space["tree_method"] = "gpu_hist"
    
    # Create the scheduler for early stopping
    scheduler = ASHAScheduler(
        max_t=200,  # Maximum number of training iterations
        grace_period=10,  # Minimum iterations before pruning
        reduction_factor=2,
        metric="mlogloss",
        mode="min"
    )
    
    # Set resources per trial
    resources_per_trial = {"cpu": cpus_per_trial}
    if gpu_fraction is not None and gpu_fraction > 0:
        resources_per_trial["gpu"] = gpu_fraction
    
    # Initialize the tuner
    logger.info("Starting hyperparameter tuning")
    
    # Create a wrapper function that includes the data
    def _train_with_data(config):
        return train_xgboost(config, train_dmatrix, valid_dmatrix)
    
    # Run the tuning
    tuner = tune.Tuner(
        _train_with_data,
        tune_config=tune.TuneConfig(
            metric="mlogloss",
            mode="min",
            scheduler=scheduler,
            num_samples=num_samples,
            search_alg=BasicVariantGenerator(max_concurrent=cpus_per_trial)
        ),
        param_space=search_space,
        run_config=tune.RunConfig(
            resources_per_trial=resources_per_trial,
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
    train_final_model(best_config, train_dmatrix, valid_dmatrix, output_dir)
    
    return best_config

def train_final_model(config, train_data, test_data, output_dir):
    """
    Train a final model using the best hyperparameters found.
    
    Args:
        config (dict): Best hyperparameters
        train_data (xgb.DMatrix): Training data
        test_data (xgb.DMatrix): Test data
        output_dir (str): Directory to save the model
    """
    # Prepare the XGBoost parameters
    params = {
        # Fixed parameters
        'objective': 'multi:softmax',
        'num_class': 3,
        'eval_metric': ['mlogloss', 'merror'],
        
        # Best parameters from tuning
        'max_depth': config['max_depth'],
        'min_child_weight': config['min_child_weight'],
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
        num_boost_round=config['num_boost_round'],
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
    parser.add_argument("--data-file", type=str, required=True, help="Path to CSV data file")
    parser.add_argument("--num-samples", type=int, default=10, help="Number of hyperparameter combinations to try")
    parser.add_argument("--cpus-per-trial", type=int, default=1, help="CPUs per trial")
    parser.add_argument("--gpu-fraction", type=float, default=None, help="GPU fraction per trial (0.1 for 10%)")
    parser.add_argument("--output-dir", type=str, default="./tune_results", help="Output directory for results")
    args = parser.parse_args()
    
    # Run the hyperparameter tuning
    tune_xgboost(
        data_file=args.data_file,
        num_samples=args.num_samples,
        cpus_per_trial=args.cpus_per_trial,
        gpu_fraction=args.gpu_fraction,
        output_dir=args.output_dir
    )

if __name__ == "__main__":
    main() 