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

Recent Fixes:
- Fixed Ray Tune worker file access issues by passing data directly to trial functions
  instead of trying to reload files from worker environments
- Ensured consistent preprocessing across hyperparameter tuning and federated learning phases
- Fixed processor path issues by using absolute paths for Ray Tune workers
- Added robust label column handling to work with different feature processor behaviors
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
from dataset import load_csv_data, transform_dataset_to_dmatrix, create_global_feature_processor, load_global_feature_processor
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
    
    # Load the global feature processor instead of creating a new one
    processor_path = config.get('global_processor_path', 'outputs/global_feature_processor.pkl')
    try:
        processor = load_global_feature_processor(processor_path)
        logger.info("Using global feature processor for consistent preprocessing")
    except FileNotFoundError:
        logger.warning(f"Global processor not found at {processor_path}, creating new one")
        from dataset import FeatureProcessor
        processor = FeatureProcessor()
        processor.fit(train_df)
    
    # Transform data using the global processor
    train_processed = processor.transform(train_df, is_training=True)
    test_processed = processor.transform(test_df, is_training=False)
    
    # Debug: Check what columns are available after processing
    logger.info(f"Train processed columns: {list(train_processed.columns)}")
    logger.info(f"Test processed columns: {list(test_processed.columns)}")
    
    # Handle label column extraction more robustly
    if 'label' in train_processed.columns:
        train_features = train_processed.drop(columns=['label'])
        train_labels = train_processed['label'].astype(int)
    else:
        # If label column doesn't exist after processing, use original data
        logger.warning("Label column not found in processed data, using original labels")
        train_features = train_processed
        train_labels = train_df['label'].astype(int) if 'label' in train_df.columns else train_df['Label'].astype(int)
    
    if 'label' in test_processed.columns:
        test_features = test_processed.drop(columns=['label'])
        test_labels = test_processed['label'].astype(int)
    else:
        # If label column doesn't exist after processing, use original data
        logger.warning("Label column not found in processed test data, using original labels")
        test_features = test_processed
        test_labels = test_df['label'].astype(int) if 'label' in test_df.columns else test_df['Label'].astype(int)
    
    # Create DMatrix objects inside the function to avoid pickling issues
    train_data = xgb.DMatrix(train_features, label=train_labels, missing=np.nan)
    test_data = xgb.DMatrix(test_features, label=test_labels, missing=np.nan)
    
    # Prepare the XGBoost parameters
    params = {
        # Fixed parameters
        'objective': 'multi:softprob',
        'num_class': 11,  # Fixed to match dataset - has classes 0-10 (11 classes total)
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
    
    # Train the model with early stopping
    bst = xgb.train(
        params,
        train_data,
        num_boost_round=int(config['num_boost_round']),
        evals=[(train_data, 'train'), (test_data, 'eval')],
        evals_result=results,
        early_stopping_rounds=30,  # Stop if no improvement for 30 rounds
        verbose_eval=False
    )
    
    # Get the best iteration metrics (early stopping may have stopped before num_boost_round)
    best_iteration = getattr(bst, 'best_iteration', len(results['eval']['mlogloss']) - 1)
    eval_mlogloss = results['eval']['mlogloss'][best_iteration]
    eval_merror = results['eval']['merror'][best_iteration]
    
    # Make predictions for more detailed metrics
    y_pred_proba = bst.predict(test_data)  # Get probabilities from multi:softprob
    y_pred_labels = np.argmax(y_pred_proba, axis=1)  # Convert probabilities to predicted labels
    y_true = test_data.get_label()
    
    # Compute multi-class metrics using predicted labels
    precision = precision_score(y_true, y_pred_labels, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred_labels, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred_labels, average='weighted', zero_division=0)
    accuracy = accuracy_score(y_true, y_pred_labels)
    
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
    Run hyperparameter tuning for XGBoost using Ray Tune with consistent preprocessing.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Step 1: Create global feature processor for consistent preprocessing
    logger.info("Creating global feature processor for consistent preprocessing...")
    if data_file:
        processor_path = create_global_feature_processor(data_file, output_dir)
    elif train_file:
        processor_path = create_global_feature_processor(train_file, output_dir)
    else:
        raise ValueError("Either data_file or train_file must be provided to create global processor")
    
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
                
        # Load the global feature processor and process data
        processor = load_global_feature_processor(processor_path)
        
        # Process the data using the global processor
        train_processed = processor.transform(train_data, is_training=True)
        test_processed = processor.transform(test_data, is_training=False)
        
        # Extract features and labels
        if 'label' in train_processed.columns:
            train_features = train_processed.drop(columns=['label'])
            train_labels = train_processed['label'].astype(int)
        else:
            logger.warning("Label column not found in processed training data, using original labels")
            train_features = train_processed
            train_labels = train_data['label'].astype(int) if 'label' in train_data.columns else train_data['Label'].astype(int)
        
        if 'label' in test_processed.columns:
            test_features = test_processed.drop(columns=['label'])
            test_labels = test_processed['label'].astype(int)
        else:
            logger.warning("Label column not found in processed test data, using original labels")
            test_features = test_processed
            test_labels = test_data['label'].astype(int) if 'label' in test_data.columns else test_data['Label'].astype(int)
        
        logger.info(f"Training data size: {len(train_features)}")
        logger.info(f"Validation data size: {len(test_features)}")
    else:
        # Replace the old temporal splitting logic with consistent data loading
        # Use the same data loading logic as the Ray Tune trials for consistency
        from dataset import load_csv_data
        
        logger.info("Loading data using consistent splitting logic...")
        dataset = load_csv_data(data_file)
        
        # Extract the processed data and convert to pandas for compatibility
        train_split_orig = dataset['train'].to_pandas()
        test_split_orig = dataset['test'].to_pandas()
        
        # Load the global processor and transform data
        processor = load_global_feature_processor(processor_path)
        
        train_processed = processor.transform(train_split_orig, is_training=True)
        test_processed = processor.transform(test_split_orig, is_training=False)
        
        # Extract features and labels
        if 'label' in train_processed.columns:
            train_features = train_processed.drop(columns=['label'])
            train_labels = train_processed['label'].astype(int)
        else:
            logger.warning("Label column not found in processed training data, using original labels from split")
            train_features = train_processed
            # Get labels from the original split data before processing
            train_labels = train_split_orig['label'].astype(int)
        
        if 'label' in test_processed.columns:
            test_features = test_processed.drop(columns=['label'])
            test_labels = test_processed['label'].astype(int)
        else:
            logger.warning("Label column not found in processed test data, using original labels from split")
            test_features = test_processed
            # Get labels from the original split data before processing
            test_labels = test_split_orig['label'].astype(int)
        
        logger.info(f"Training data size: {len(train_features)}")
        logger.info(f"Validation data size: {len(test_features)}")

    # Define the search space using hyperopt's hp module
    search_space = {
        "max_depth": hp.quniform("max_depth", 4, 12, 1),            # Expanded range for deeper trees
        "min_child_weight": hp.quniform("min_child_weight", 1, 10, 1),  # Reduced upper bound
        "reg_alpha": hp.loguniform("reg_alpha", np.log(0.01), np.log(10.0)),
        "reg_lambda": hp.loguniform("reg_lambda", np.log(0.01), np.log(10.0)),
        "eta": hp.uniform("eta", 0.01, 0.3),                        # More practical learning rate range
        "subsample": hp.uniform("subsample", 0.6, 1.0),             # Improved lower bound
        "colsample_bytree": hp.uniform("colsample_bytree", 0.6, 1.0), # Improved lower bound
        "num_boost_round": hp.quniform("num_boost_round", 50, 200, 10)  # CRITICAL FIX: Realistic range!
    }

    if gpu_fraction is not None and gpu_fraction > 0:
        search_space["tree_method"] = hp.choice("tree_method", ["gpu_hist"])

    # Create a wrapper function that includes the original data DataFrames and processor path
    def _train_with_data_wrapper(config):
        # Add processor path to config - ensure it's absolute for Ray Tune workers
        config['global_processor_path'] = os.path.abspath(processor_path)
        # Pass copies of the original dataframes to the training function
        if train_file and test_file:
            return train_xgboost(config, train_data.copy(), test_data.copy())
        else:
            # For single data file mode, pass the original loaded data directly
            # instead of trying to reload it in the worker
            return train_xgboost(config, data.copy(), data.copy())

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
            storage_path=os.path.abspath(output_dir),
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
        'objective': 'multi:softprob',
        'num_class': 11,  # Fixed to match dataset - has classes 0-10 (11 classes total)
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
    
    # Train the final model with early stopping
    logger.info("Training final model with best parameters")
    eval_results = {}
    final_model = xgb.train(
        params,
        train_data,
        num_boost_round=int(config['num_boost_round']),
        evals=[(train_data, 'train'), (test_data, 'eval')],
        evals_result=eval_results,
        early_stopping_rounds=30,  # Stop if no improvement for 30 rounds
        verbose_eval=True
    )
    
    # Save the model
    model_path = os.path.join(output_dir, "best_model.json")
    final_model.save_model(model_path)
    logger.info(f"Final model saved to {model_path}")
    
    # Evaluate the model
    y_pred_proba = final_model.predict(test_data)  # Get probabilities from multi:softprob
    y_pred_labels = np.argmax(y_pred_proba, axis=1)  # Convert probabilities to predicted labels
    y_true = test_data.get_label()
    
    # Generate performance metrics
    precision = precision_score(y_true, y_pred_labels, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred_labels, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred_labels, average='weighted', zero_division=0)
    accuracy = accuracy_score(y_true, y_pred_labels)
    
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
    parser.add_argument("--gpu-fraction", type=float, default=None, help="GPU fraction per trial (0.1 for 10%)")
    parser.add_argument("--output-dir", type=str, default="./tune_results", help="Output directory for results")
    args = parser.parse_args()
    
    # Validate arguments
    if not args.data_file and not (args.train_file and args.test_file):
        parser.error("Either --data-file or both --train-file and --test-file must be provided")
    
    # Run the hyperparameter tuning
    tune_xgboost(
        train_file=args.train_file,
        test_file=args.test_file,
        data_file=args.data_file,
        num_samples=args.num_samples,
        cpus_per_trial=args.cpus_per_trial,
        gpu_fraction=args.gpu_fraction,
        output_dir=args.output_dir
    )

if __name__ == "__main__":
    main() 