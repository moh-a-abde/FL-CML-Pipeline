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
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import logging
from ray.air.config import RunConfig

# Import existing data processing code
from dataset import load_csv_data, preprocess_data, FeatureProcessor

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
    logger.info("Starting trial with config: %s", config)
    
    # Use preprocess_data to get features and encoded labels
    processor = FeatureProcessor()
    train_features, train_labels = preprocess_data(train_df, processor=processor, is_training=True)
    test_features, test_labels = preprocess_data(test_df, processor=processor, is_training=False)
    
    # Handle cases where test data might be unlabeled
    if test_labels is None:
        logger.warning("Test data has no labels. Creating dummy labels for DMatrix.")
        test_labels = np.zeros(len(test_features))
    else:
        test_labels = test_labels.astype(int) # Ensure labels are integers
        
    # Ensure train labels are integers
    train_labels = train_labels.astype(int)
    
    # Create DMatrix objects inside the function to avoid pickling issues
    # FeatureProcessor already handles feature types and drops unnecessary columns
    train_data = xgb.DMatrix(train_features, label=train_labels, missing=np.nan)
    test_data = xgb.DMatrix(test_features, label=test_labels, missing=np.nan)
    
    # Prepare the XGBoost parameters
    params = {
        # Fixed parameters
        'objective': 'multi:softmax',
        'num_class': 10,  # UNSW_NB15 has 10 classes
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

def tune_xgboost(train_file: str, test_file: str, num_samples: int = 100, gpu_fraction: float = None, output_dir: str = "./tune_results"):
    """
    Run hyperparameter tuning for XGBoost using Ray Tune.
    
    Args:
        train_file (str): Path to the training CSV data file.
        test_file (str): Path to the testing CSV data file.
        num_samples (int): Number of hyperparameter combinations to try.
        gpu_fraction (float): Fraction of GPU to use per trial (if None, no GPU is used).
        output_dir (str): Directory to save results.
        
    Returns:
        dict: Best hyperparameters found.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load and prepare data *once* before tuning starts
    logger.info("Loading training data from %s", train_file)
    train_df = load_csv_data(train_file)["train"].to_pandas()
    
    logger.info("Loading testing data from %s", test_file)
    test_df = load_csv_data(test_file)["train"].to_pandas() # Use train split of test file for validation

    # Preprocess data *once* to get features/labels for the final model training
    # and to fit the processor
    logger.info("Preprocessing data for final model training and fitting processor...")
    processor = FeatureProcessor()
    # We need the original dataframes (train_df, test_df) to pass to the trial wrapper
    # But we also need the processed features/labels for the final model training step
    final_train_features, final_train_labels = preprocess_data(train_df, processor=processor, is_training=True)
    final_test_features, final_test_labels = preprocess_data(test_df, processor=processor, is_training=False)
    
    # Handle potential missing labels in the test set for final model evaluation
    if final_test_labels is None:
        logger.warning("Test data has no labels. Using zeros for final model evaluation.")
        final_test_labels = np.zeros(len(final_test_features))
    else:
        final_test_labels = final_test_labels.astype(int) # Ensure labels are integers
    final_train_labels = final_train_labels.astype(int)

    logger.info("Training data size for final model: %d", len(final_train_features))
    logger.info("Validation data size for final model: %d", len(final_test_features))
    
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
        # NOTE: Ray Tune handles GPU allocation via resources_per_trial typically
        # Setting tree_method here might be sufficient, but review Ray Tune docs if issues persist
        search_space["tree_method"] = hp.choice("tree_method", ["gpu_hist"])
    
    # Create a wrapper function that includes the *original* data DataFrames
    # The train_xgboost function inside the trial will handle preprocessing
    def _train_with_data_wrapper(config):
        # Pass copies of the original dataframes to the training function
        return train_xgboost(config, train_df.copy(), test_df.copy()) 

    # Set up HyperOptSearch
    algo = HyperOptSearch(
        search_space,
        metric="mlogloss",
        mode="min"
    )
    scheduler = ASHAScheduler(
        max_t=200, # Corresponds roughly to num_boost_round max
        grace_period=10,
        reduction_factor=2,
        metric="mlogloss",
        mode="min"
    )
    
    # Initialize the tuner
    logger.info("Starting hyperparameter tuning")
    
    # Define resources per trial if GPU is requested
    resources_per_trial = None
    if gpu_fraction is not None and gpu_fraction > 0:
        # Allocate a fraction of GPU resources per trial
        resources_per_trial = {"gpu": gpu_fraction}
        # Ensure Ray is initialized with GPU visibility if needed (usually automatic)
        # ray.init(num_gpus=...) # Might be needed depending on setup

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
            name="xgboost_tune",
            # resources_per_trial=resources_per_trial # Pass resource requests here - causing issues, let Ray manage?
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
    logger.info("  mlogloss: %.4f", best_metrics['mlogloss'])
    logger.info("  merror: %.4f", best_metrics['merror'])
    logger.info("  precision: %.4f", best_metrics['precision'])
    logger.info("  recall: %.4f", best_metrics['recall'])
    logger.info("  f1: %.4f", best_metrics['f1'])
    logger.info("  accuracy: %.4f", best_metrics['accuracy'])
    
    # Save the best hyperparameters to a file
    best_params_file = os.path.join(output_dir, "best_params.json")
    # Use utf-8 encoding when writing JSON
    with open(best_params_file, 'w', encoding='utf-8') as f:
        json.dump(best_config, f, indent=2)
    logger.info("Best parameters saved to %s", best_params_file)
    
    # Train a final model with the best parameters using the preprocessed data
    train_final_model(best_config, final_train_features, final_train_labels, final_test_features, final_test_labels, output_dir)
    
    return best_config

def train_final_model(config: dict, 
                      train_features: pd.DataFrame, train_labels: pd.Series, 
                      test_features: pd.DataFrame, test_labels: pd.Series, 
                      output_dir: str):
    """
    Train a final model using the best hyperparameters found.
    
    Args:
        config (dict): Best hyperparameters
        train_features (pd.DataFrame): Training features
        train_labels (pd.Series): Training labels (encoded)
        test_features (pd.DataFrame): Test features
        test_labels (pd.Series): Test labels (encoded)
        output_dir (str): Directory to save the model
    """
    # Create DMatrix objects
    train_data = xgb.DMatrix(train_features, label=train_labels, missing=np.nan)
    test_data = xgb.DMatrix(test_features, label=test_labels, missing=np.nan)
    
    # Prepare the XGBoost parameters
    params = {
        # Fixed parameters
        'objective': 'multi:softmax',
        'num_class': 10,  # UNSW_NB15 has 10 classes
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
        verbose_eval=True # Show progress for final model
    )
    
    # Save the model
    model_path = os.path.join(output_dir, "best_model.json")
    final_model.save_model(model_path)
    logger.info("Final model saved to %s", model_path)
    
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
    logger.info("  Precision: %.4f", precision)
    logger.info("  Recall: %.4f", recall)
    logger.info("  F1 Score: %.4f", f1)
    logger.info("  Accuracy: %.4f", accuracy)
    
    return final_model

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Ray Tune for XGBoost hyperparameter optimization")
    # Removed --data-file as it complicates logic, require train/test
    parser.add_argument("--train-file", type=str, required=True, help="Path to training CSV data file")
    parser.add_argument("--test-file", type=str, required=True, help="Path to testing CSV data file")
    parser.add_argument("--num-samples", type=int, default=50, help="Number of hyperparameter combinations to try") # Increased default based on workflow
    # Removed --cpus-per-trial, let Ray manage CPU allocation
    parser.add_argument("--gpu-fraction", type=float, default=None, help="GPU fraction per trial (e.g., 0.25 for 25%%)")
    parser.add_argument("--output-dir", type=str, default="./tune_results", help="Output directory for results")
    args = parser.parse_args()
    
    # Validate arguments (simplified)
    # ... (train/test file existence checks could be added) ...
    
    # Run the hyperparameter tuning
    try:
        tune_xgboost(
            train_file=args.train_file,
            test_file=args.test_file,
            num_samples=args.num_samples,
            gpu_fraction=args.gpu_fraction,
            output_dir=args.output_dir
        )
        logger.info("===== Hyperparameter tuning completed successfully ====")
    except Exception as e:
        logger.error("===== Hyperparameter tuning failed =====", exc_info=True)
        # Potentially re-raise or exit differently depending on CI requirements
        raise e 

if __name__ == "__main__":
    main() 