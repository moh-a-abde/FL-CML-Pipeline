"""
Ray Tune hyperparameter optimization for Random Forest models.

This module provides hyperparameter tuning capabilities for Random Forest
classifiers using Ray Tune with advanced scheduling and early stopping.
"""

import os
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
from functools import partial

import ray
from ray import tune, train
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from ray.tune.stopper import TrialPlateauStopper
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import train_test_split

from src.core.dataset import load_csv_data, preprocess_data
from src.models.random_forest_model import RandomForestModel

logger = logging.getLogger(__name__)

# Global constants for Random Forest hyperparameter search
RF_PARAM_SPACE = {
    # Number of trees in the forest
    'n_estimators': tune.randint(50, 500),  # Wide range for ensemble size
    
    # Maximum depth of trees
    'max_depth': tune.choice([None, 5, 10, 15, 20, 30, 50]),  # Include unlimited depth
    
    # Minimum samples required to split a node
    'min_samples_split': tune.randint(2, 20),
    
    # Minimum samples required at a leaf node
    'min_samples_leaf': tune.randint(1, 10),
    
    # Number of features to consider when looking for the best split
    'max_features': tune.choice(['sqrt', 'log2', 0.3, 0.5, 0.7, 1.0]),
    
    # Criterion for measuring quality of splits
    'criterion': tune.choice(['gini', 'entropy']),
    
    # Whether bootstrap samples are used when building trees
    'bootstrap': tune.choice([True, False]),
    
    # Class weighting strategy
    'class_weight': tune.choice([None, 'balanced', 'balanced_subsample']),
    
    # Fraction of samples to draw for training each tree (if bootstrap=True)
    'max_samples': tune.choice([None, 0.5, 0.7, 0.9]),
    
    # Minimum weighted fraction of samples required at a leaf
    'min_weight_fraction_leaf': tune.uniform(0.0, 0.1),
    
    # Maximum number of leaf nodes
    'max_leaf_nodes': tune.choice([None, 50, 100, 200, 500]),
    
    # Minimum impurity decrease required for a split
    'min_impurity_decrease': tune.uniform(0.0, 0.01),
}


def train_with_config(config: Dict[str, Any], 
                     train_data: Tuple[np.ndarray, np.ndarray],
                     val_data: Tuple[np.ndarray, np.ndarray]) -> None:
    """
    Train Random Forest with given hyperparameters for Ray Tune.
    
    Args:
        config: Hyperparameter configuration from Ray Tune
        train_data: Tuple of (X_train, y_train)
        val_data: Tuple of (X_val, y_val)
    """
    X_train, y_train = train_data
    X_val, y_val = val_data
    
    try:
        # Create Random Forest model with the given configuration
        rf_model = RandomForestModel(config)
        
        # Train the model
        rf_model.fit(X_train, y_train, X_val, y_val)
        
        # Make predictions on validation set
        val_pred = rf_model.predict(X_val)
        val_pred_proba = rf_model.predict_proba(X_val)
        
        # Calculate metrics
        accuracy = accuracy_score(y_val, val_pred)
        f1_weighted = f1_score(y_val, val_pred, average='weighted')
        f1_macro = f1_score(y_val, val_pred, average='macro')
        
        # Calculate training metrics for comparison
        train_pred = rf_model.predict(X_train)
        train_accuracy = accuracy_score(y_train, train_pred)
        train_f1 = f1_score(y_train, train_pred, average='weighted')
        
        # Get model info
        model_info = rf_model.get_model_info()
        
        # Report metrics to Ray Tune
        train.report({
            "accuracy": accuracy,
            "f1_weighted": f1_weighted,
            "f1_macro": f1_macro,
            "train_accuracy": train_accuracy,
            "train_f1": train_f1,
            "n_estimators": model_info.get('n_estimators', config.get('n_estimators')),
            "oob_score": model_info.get('oob_score'),
            "done": True
        })
        
    except Exception as e:
        logger.error(f"Training failed with config {config}: {e}")
        # Report failure to Ray Tune
        train.report({
            "accuracy": 0.0,
            "f1_weighted": 0.0,
            "f1_macro": 0.0,
            "train_accuracy": 0.0,
            "train_f1": 0.0,
            "done": True
        })


class EnhancedRandomForestTrainer:
    """Enhanced Random Forest trainer with Ray Tune optimization."""
    
    def __init__(self, data_file: str, output_dir: str = "./tune_results_rf"):
        """
        Initialize the trainer.
        
        Args:
            data_file: Path to the dataset CSV file
            output_dir: Directory to save tuning results
        """
        self.data_file = data_file
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load and prepare data
        self.X_train, self.X_val, self.X_test = None, None, None
        self.y_train, self.y_val, self.y_test = None, None, None
        self._prepare_data()
        
    def _prepare_data(self) -> None:
        """Load and split the dataset."""
        logger.info(f"Loading data from {self.data_file}")
        
        # Load data using the existing CSV loader
        dataset_dict = load_csv_data(self.data_file)
        
        # Convert to pandas for sklearn processing
        full_dataset = dataset_dict["train"].to_pandas()
        
        # Prepare features and target
        if 'label' in full_dataset.columns:
            X = full_dataset.drop(columns=['label'])
            y = full_dataset['label']
        elif 'attack_cat' in full_dataset.columns:
            X = full_dataset.drop(columns=['attack_cat'])
            y = full_dataset['attack_cat']
        else:
            raise ValueError("No suitable target column found (expected 'label' or 'attack_cat')")
        
        # Create train/validation/test splits
        X_temp, self.X_test, y_temp, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y if len(np.unique(y)) > 1 else None
        )
        
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp if len(np.unique(y_temp)) > 1 else None
        )
        
        # Convert to numpy arrays
        self.X_train = self.X_train.values
        self.X_val = self.X_val.values
        self.X_test = self.X_test.values
        self.y_train = self.y_train.values
        self.y_val = self.y_val.values
        self.y_test = self.y_test.values
        
        logger.info(f"Data loaded: Train={self.X_train.shape}, Val={self.X_val.shape}, Test={self.X_test.shape}")
        
    def tune_hyperparameters(self, 
                           num_samples: int = 50,
                           max_concurrent_trials: int = 4,
                           cpus_per_trial: int = 2) -> Dict[str, Any]:
        """
        Perform hyperparameter tuning using Ray Tune.
        
        Args:
            num_samples: Number of hyperparameter configurations to try
            max_concurrent_trials: Maximum number of concurrent trials
            cpus_per_trial: CPUs to allocate per trial
            
        Returns:
            Best hyperparameter configuration
        """
        logger.info(f"Starting hyperparameter tuning with {num_samples} samples")
        
        # Initialize Ray if not already initialized
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)
        
        # Configure the scheduler
        scheduler = ASHAScheduler(
            max_t=200,  # Maximum iterations per trial
            grace_period=20,  # Minimum iterations before elimination
            reduction_factor=2,
            metric="f1_weighted",
            mode="max"
        )
        
        # Configure early stopping
        stopper = TrialPlateauStopper(
            metric="f1_weighted",
            std=0.001,
            num_results=10,
            grace_period=20,
            mode="max"
        )
        
        # Configure reporting
        reporter = CLIReporter(
            metric_columns=["accuracy", "f1_weighted", "f1_macro", "train_accuracy", "n_estimators"],
            max_report_frequency=30
        )
        
        # Prepare data for the trainable function
        train_data = (self.X_train, self.y_train)
        val_data = (self.X_val, self.y_val)
        
        # Create the trainable function with data
        trainable = partial(train_with_config, train_data=train_data, val_data=val_data)
        
        # Run the tuning with updated Ray Tune API (storage_path requires absolute path)
        analysis = tune.run(
            trainable,
            config=RF_PARAM_SPACE,
            num_samples=num_samples,
            scheduler=scheduler,
            stop=stopper,
            progress_reporter=reporter,
            storage_path=str(self.output_dir.absolute()),
            name="rf_tune",
            resources_per_trial={"cpu": cpus_per_trial},
            max_concurrent_trials=max_concurrent_trials,
            raise_on_failed_trial=False,
            resume="AUTO"
        )
        
        # Get the best configuration
        best_config = analysis.get_best_config(metric="f1_weighted", mode="max")
        best_result = analysis.get_best_trial(metric="f1_weighted", mode="max").last_result
        
        logger.info(f"Best configuration found: {best_config}")
        logger.info(f"Best F1 Score: {best_result['f1_weighted']:.4f}")
        logger.info(f"Best Accuracy: {best_result['accuracy']:.4f}")
        
        # Save the best configuration
        best_config_file = self.output_dir / "best_config_rf.json"
        with open(best_config_file, 'w') as f:
            json.dump(best_config, f, indent=2)
        
        # Save detailed results
        results_df = analysis.results_df
        results_file = self.output_dir / "all_results_rf.csv"
        results_df.to_csv(results_file, index=False)
        
        return best_config
    
    def train_final_model(self, best_config: Dict[str, Any]) -> RandomForestModel:
        """
        Train the final model with the best hyperparameters.
        
        Args:
            best_config: Best hyperparameter configuration from tuning
            
        Returns:
            Trained Random Forest model
        """
        logger.info("Training final model with best configuration")
        
        # Combine train and validation data for final training
        X_final = np.vstack([self.X_train, self.X_val])
        y_final = np.hstack([self.y_train, self.y_val])
        
        # Create and train the final model
        final_model = RandomForestModel(best_config)
        final_model.fit(X_final, y_final)
        
        # Evaluate on test set
        test_pred = final_model.predict(self.X_test)
        test_pred_proba = final_model.predict_proba(self.X_test)
        
        test_accuracy = accuracy_score(self.y_test, test_pred)
        test_f1 = f1_score(self.y_test, test_pred, average='weighted')
        
        logger.info(f"Final model performance - Accuracy: {test_accuracy:.4f}, F1: {test_f1:.4f}")
        
        # Save the final model
        model_path = self.output_dir / "final_model_rf.joblib"
        final_model.save_model(str(model_path))
        
        # Save test results
        test_results = {
            'test_accuracy': test_accuracy,
            'test_f1_weighted': test_f1,
            'best_config': best_config,
            'model_info': final_model.get_model_info()
        }
        
        results_file = self.output_dir / "final_results_rf.json"
        with open(results_file, 'w') as f:
            json.dump(test_results, f, indent=2, default=str)
        
        return final_model


def main():
    """Main function for running Random Forest hyperparameter tuning."""
    parser = argparse.ArgumentParser(description="Random Forest Hyperparameter Tuning with Ray Tune")
    
    parser.add_argument("--data-file", type=str, default="data/received/final_dataset.csv",
                       help="Path to the dataset CSV file")
    parser.add_argument("--num-samples", type=int, default=5,
                       help="Number of hyperparameter configurations to try")
    parser.add_argument("--cpus-per-trial", type=int, default=2,
                       help="Number of CPUs to allocate per trial")
    parser.add_argument("--max-concurrent-trials", type=int, default=2,
                       help="Maximum number of concurrent trials")
    parser.add_argument("--output-dir", type=str, default="./tune_results_rf",
                       help="Output directory for results")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try:
        # Create trainer and run tuning
        trainer = EnhancedRandomForestTrainer(
            data_file=args.data_file,
            output_dir=args.output_dir
        )
        
        # Tune hyperparameters
        best_config = trainer.tune_hyperparameters(
            num_samples=args.num_samples,
            max_concurrent_trials=args.max_concurrent_trials,
            cpus_per_trial=args.cpus_per_trial
        )
        
        # Train final model
        final_model = trainer.train_final_model(best_config)
        
        logger.info("Random Forest hyperparameter tuning completed successfully!")
        logger.info(f"Results saved to: {args.output_dir}")
        
    except Exception as e:
        logger.error(f"Tuning failed: {e}")
        raise
    finally:
        # Cleanup Ray
        if ray.is_initialized():
            ray.shutdown()


if __name__ == "__main__":
    main() 