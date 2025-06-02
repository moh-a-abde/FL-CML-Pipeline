"""
Ray Tune XGBoost Training with Improved Hyperparameter Optimization

Enhanced implementation with expanded search spaces, better early stopping,
and robust error handling for federated learning environments.
"""

import os
import sys
import warnings
import json
import pickle
import time
from typing import Dict, Any, Optional, Tuple, List
import numpy as np
import pandas as pd
import xgboost as xgb
from functools import partial

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Ray Tune imports
import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.stopper import TrialPlateauStopper

# Scikit-learn imports  
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight

# Add project root directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))  # Go up two levels to project root
sys.path.insert(0, project_root)

# Import local modules
from src.core.dataset import load_csv_data, transform_dataset_to_dmatrix, create_global_feature_processor, load_global_feature_processor

# Ray setup - reduce verbosity and resource warnings
ray.init(
    ignore_reinit_error=True,
    log_to_driver=False,
    configure_logging=False,
    local_mode=False  # Set to True for debugging
)

# Global constants for the enhanced hyperparameter search
ENHANCED_PARAM_SPACE = {
    # Learning rate - expanded range for better exploration
    'eta': tune.loguniform(0.01, 0.5),
    
    # Tree depth - wider range for complex patterns
    'max_depth': tune.randint(3, 15),
    
    # Minimum child weight - helps with overfitting
    'min_child_weight': tune.randint(1, 15),
    
    # Subsample ratios - prevent overfitting
    'colsample_bytree': tune.uniform(0.5, 1.0),
    'colsample_bylevel': tune.uniform(0.5, 1.0),
    'colsample_bynode': tune.uniform(0.5, 1.0),
    
    # Regularization - L1 and L2
    'reg_alpha': tune.loguniform(1e-8, 100),  # L1 regularization
    'reg_lambda': tune.loguniform(1e-8, 100),  # L2 regularization
    
    # Gamma - minimum loss reduction required to make split
    'gamma': tune.loguniform(1e-8, 1.0),
    
    # Number of boosting rounds
    'num_boost_round': tune.randint(50, 500)
}

# Import FeatureProcessor for consistent preprocessing
from src.core.dataset import FeatureProcessor

class EnhancedXGBoostTrainer:
    """Enhanced XGBoost trainer with improved hyperparameter optimization."""
    
    def __init__(self, data_file: str, test_data_file: str = None, 
                 num_samples: int = 100, max_concurrent_trials: int = 4,
                 use_global_processor: bool = True):
        """
        Initialize the enhanced XGBoost trainer.
        
        Args:
            data_file (str): Path to training data CSV file
            test_data_file (str): Path to test data CSV file (optional)
            num_samples (int): Number of hyperparameter configurations to try
            max_concurrent_trials (int): Maximum concurrent Ray Tune trials
            use_global_processor (bool): Whether to use global feature processor
        """
        self.data_file = data_file
        self.test_data_file = test_data_file
        self.num_samples = num_samples
        self.max_concurrent_trials = max_concurrent_trials
        self.use_global_processor = use_global_processor
        
        # Initialize data structures
        self.train_data = None
        self.test_data = None
        self.global_processor = None
        self.best_params = None
        self.best_score = None
        self.results_history = []
        
        print(f"Initialized Enhanced XGBoost Trainer")
        print(f"Training data: {data_file}")
        print(f"Test data: {test_data_file if test_data_file else 'Using train/test split'}")
        print(f"Hyperparameter samples: {num_samples}")
        print(f"Max concurrent trials: {max_concurrent_trials}")
        print(f"Using global processor: {use_global_processor}")
    
    def load_and_prepare_data(self):
        """Load and prepare training and test data."""
        print("\n" + "="*60)
        print("LOADING AND PREPARING DATA")
        print("="*60)
        
        # Create or load global feature processor
        if self.use_global_processor:
            processor_path = "outputs/global_feature_processor.pkl"
            if os.path.exists(processor_path):
                print(f"Loading existing global feature processor from {processor_path}")
                self.global_processor = load_global_feature_processor(processor_path)
            else:
                print(f"Creating new global feature processor")
                processor_path = create_global_feature_processor(self.data_file, "outputs")
                self.global_processor = load_global_feature_processor(processor_path)
        
        # Load training data
        print(f"\nLoading training data from: {self.data_file}")
        dataset_dict = load_csv_data(self.data_file)
        
        # Convert to DMatrix format
        train_dataset = dataset_dict["train"]
        self.train_data = transform_dataset_to_dmatrix(
            train_dataset, 
            processor=self.global_processor,
            is_training=True
        )
        
        print(f"Training data: {self.train_data.num_row()} samples, {self.train_data.num_col()} features")
        
        # Load test data
        if self.test_data_file and os.path.exists(self.test_data_file):
            print(f"\nLoading separate test data from: {self.test_data_file}")
            # Load test data using the same processor
            test_dataset_dict = load_csv_data(self.test_data_file)
            test_dataset = test_dataset_dict["test"]  # Use test split from test file
            self.test_data = transform_dataset_to_dmatrix(
                test_dataset,
                processor=self.global_processor,
                is_training=False
            )
        else:
            print(f"\nUsing train/test split from main dataset")
            test_dataset = dataset_dict["test"]
            self.test_data = transform_dataset_to_dmatrix(
                test_dataset,
                processor=self.global_processor,
                is_training=False
            )
        
        print(f"Test data: {self.test_data.num_row()} samples, {self.test_data.num_col()} features")
        
        # Verify data integrity
        train_labels = self.train_data.get_label()
        test_labels = self.test_data.get_label()
        
        print(f"\nData integrity check:")
        print(f"Training labels - min: {train_labels.min()}, max: {train_labels.max()}, unique: {len(np.unique(train_labels))}")
        print(f"Test labels - min: {test_labels.min()}, max: {test_labels.max()}, unique: {len(np.unique(test_labels))}")
        
        # Check for class distribution
        train_unique, train_counts = np.unique(train_labels, return_counts=True)
        test_unique, test_counts = np.unique(test_labels, return_counts=True)
        
        print(f"\nClass distribution:")
        print(f"Training: {dict(zip(train_unique, train_counts))}")
        print(f"Test: {dict(zip(test_unique, test_counts))}")

    def _train_with_data_wrapper(self, config: Dict[str, Any], train_data: xgb.DMatrix, test_data: xgb.DMatrix):
        """
        Wrapper function to train XGBoost with given hyperparameters.
        This function is compatible with Ray Tune's training interface.
        """
        from src.core.dataset import load_csv_data
        
        try:
            # Extract hyperparameters from config
            params = {
                'objective': 'multi:softprob',  # Multi-class classification
                'eval_metric': 'mlogloss',      # Multi-class log loss
                'eta': config['eta'],
                'max_depth': int(config['max_depth']),
                'min_child_weight': int(config['min_child_weight']),
                'colsample_bytree': config['colsample_bytree'],
                'colsample_bylevel': config.get('colsample_bylevel', 1.0),
                'colsample_bynode': config.get('colsample_bynode', 1.0),
                'reg_alpha': config.get('reg_alpha', 0),
                'reg_lambda': config.get('reg_lambda', 1),
                'gamma': config.get('gamma', 0),
                'seed': 42,
                'verbosity': 0  # Reduce XGBoost verbosity
            }
            
            num_boost_round = int(config['num_boost_round'])
            
            # Determine number of classes for multi-class setup
            train_labels = train_data.get_label()
            num_classes = len(np.unique(train_labels))
            params['num_class'] = num_classes
            
            # Early stopping configuration
            early_stopping_rounds = max(10, num_boost_round // 10)
            
            # Create evaluation list for monitoring
            evallist = [(train_data, 'train'), (test_data, 'eval')]
            
            # Train the model with early stopping
            evals_result = {}
            model = xgb.train(
                params,
                train_data,
                num_boost_round=num_boost_round,
                evals=evallist,
                evals_result=evals_result,
                early_stopping_rounds=early_stopping_rounds,
                verbose_eval=False  # Disable verbose evaluation
            )
            
            # Make predictions on test set
            test_pred_proba = model.predict(test_data)
            
            # For multi-class, convert probabilities to class predictions
            if len(test_pred_proba.shape) > 1:
                test_pred = np.argmax(test_pred_proba, axis=1)
            else:
                test_pred = (test_pred_proba > 0.5).astype(int)
            
            # Get true labels
            test_true = test_data.get_label().astype(int)
            
            # Calculate metrics
            accuracy = accuracy_score(test_true, test_pred)
            
            # Calculate per-class metrics for multi-class
            if num_classes > 2:
                # Get weighted average F1 score for multi-class
                from sklearn.metrics import f1_score
                f1 = f1_score(test_true, test_pred, average='weighted')
                
                # Use F1 score as the main metric for multi-class problems
                main_metric = f1
                metric_name = 'f1_weighted'
            else:
                # For binary classification, use accuracy
                main_metric = accuracy
                metric_name = 'accuracy'
            
            # Get the final evaluation loss
            final_train_loss = evals_result['train']['mlogloss'][-1]
            final_eval_loss = evals_result['eval']['mlogloss'][-1]
            
            # Report metrics to Ray Tune
            tune.report(
                accuracy=accuracy,
                **{metric_name: main_metric},
                train_loss=final_train_loss,
                eval_loss=final_eval_loss,
                num_boost_round_used=model.best_iteration + 1 if hasattr(model, 'best_iteration') else num_boost_round
            )
            
        except Exception as e:
            print(f"Training failed with error: {e}")
            # Report poor performance for failed trials
            tune.report(accuracy=0.0, f1_weighted=0.0, train_loss=float('inf'), eval_loss=float('inf'))

    def tune_hyperparameters(self):
        """Run hyperparameter tuning using Ray Tune."""
        print("\n" + "="*60)
        print("HYPERPARAMETER TUNING")
        print("="*60)
        
        if self.train_data is None:
            self.load_and_prepare_data()
        
        # Create partial function with data
        train_func = partial(
            self._train_with_data_wrapper,
            train_data=self.train_data,
            test_data=self.test_data
        )
        
        # Configure ASHA scheduler for early stopping
        scheduler = ASHAScheduler(
            metric="f1_weighted",  # Use F1 score as main metric
            mode="max",
            max_t=500,  # Maximum number of boosting rounds
            grace_period=50,  # Minimum rounds before stopping
            reduction_factor=2
        )
        
        # Configure trial stopping criteria
        stopper = TrialPlateauStopper(
            metric="f1_weighted",
            mode="max",
            patience=10
        )
        
        print(f"Starting hyperparameter tuning with {self.num_samples} trials...")
        print(f"Search space: {ENHANCED_PARAM_SPACE}")
        
        # Run hyperparameter tuning
        analysis = tune.run(
            train_func,
            config=ENHANCED_PARAM_SPACE,
            num_samples=self.num_samples,
            scheduler=scheduler,
            stop=stopper,
            resources_per_trial={"cpu": 1},
            max_concurrent_trials=self.max_concurrent_trials,
            verbose=1,
            raise_on_failed_trial=False  # Continue even if some trials fail
        )
        
        # Get best parameters
        best_trial = analysis.get_best_trial("f1_weighted", "max", "last")
        self.best_params = best_trial.config
        self.best_score = best_trial.last_result["f1_weighted"]
        
        print(f"\n" + "="*60)
        print("BEST HYPERPARAMETERS FOUND")
        print("="*60)
        print(f"Best F1 Score: {self.best_score:.4f}")
        print(f"Best Parameters:")
        for param, value in self.best_params.items():
            print(f"  {param}: {value}")
        
        # Store results for analysis
        self.results_history = analysis.results_df
        
        return analysis

    def train_final_model(self):
        """Train final model with best hyperparameters."""
        print("\n" + "="*60)
        print("TRAINING FINAL MODEL")
        print("="*60)
        
        if self.best_params is None:
            raise ValueError("No best parameters found. Run tune_hyperparameters() first.")
        
        # Prepare final parameters
        final_params = {
            'objective': 'multi:softprob',
            'eval_metric': 'mlogloss',
            'eta': self.best_params['eta'],
            'max_depth': int(self.best_params['max_depth']),
            'min_child_weight': int(self.best_params['min_child_weight']),
            'colsample_bytree': self.best_params['colsample_bytree'],
            'colsample_bylevel': self.best_params.get('colsample_bylevel', 1.0),
            'colsample_bynode': self.best_params.get('colsample_bynode', 1.0),
            'reg_alpha': self.best_params.get('reg_alpha', 0),
            'reg_lambda': self.best_params.get('reg_lambda', 1),
            'gamma': self.best_params.get('gamma', 0),
            'seed': 42,
            'verbosity': 1
        }
        
        # Determine number of classes
        train_labels = self.train_data.get_label()
        num_classes = len(np.unique(train_labels))
        final_params['num_class'] = num_classes
        
        num_boost_round = int(self.best_params['num_boost_round'])
        
        print(f"Training final model with {num_boost_round} boosting rounds...")
        print(f"Number of classes: {num_classes}")
        
        # Train final model
        evallist = [(self.train_data, 'train'), (self.test_data, 'eval')]
        evals_result = {}
        
        final_model = xgb.train(
            final_params,
            self.train_data,
            num_boost_round=num_boost_round,
            evals=evallist,
            evals_result=evals_result,
            verbose_eval=50
        )
        
        # Evaluate final model
        test_pred_proba = final_model.predict(self.test_data)
        
        if len(test_pred_proba.shape) > 1:
            test_pred = np.argmax(test_pred_proba, axis=1)
        else:
            test_pred = (test_pred_proba > 0.5).astype(int)
        
        test_true = self.test_data.get_label().astype(int)
        
        # Calculate comprehensive metrics
        accuracy = accuracy_score(test_true, test_pred)
        
        print(f"\n" + "="*60)
        print("FINAL MODEL PERFORMANCE")
        print("="*60)
        print(f"Test Accuracy: {accuracy:.4f}")
        
        # Detailed classification report
        print("\nClassification Report:")
        print(classification_report(test_true, test_pred))
        
        # Save final model and parameters
        os.makedirs("outputs", exist_ok=True)
        
        model_path = "outputs/final_xgboost_model.json"
        final_model.save_model(model_path)
        print(f"\nFinal model saved to: {model_path}")
        
        params_path = "outputs/final_hyperparameters.json"
        with open(params_path, 'w') as f:
            json.dump(self.best_params, f, indent=2)
        print(f"Best hyperparameters saved to: {params_path}")
        
        return final_model, final_params

    def run_complete_tuning(self):
        """Run complete hyperparameter tuning and training pipeline."""
        print("Starting Enhanced XGBoost Hyperparameter Tuning Pipeline")
        print(f"Data file: {self.data_file}")
        
        start_time = time.time()
        
        try:
            # Step 1: Load and prepare data
            self.load_and_prepare_data()
            
            # Step 2: Tune hyperparameters
            analysis = self.tune_hyperparameters()
            
            # Step 3: Train final model
            final_model, final_params = self.train_final_model()
            
            total_time = time.time() - start_time
            
            print(f"\n" + "="*60)
            print("PIPELINE COMPLETED SUCCESSFULLY")
            print("="*60)
            print(f"Total time: {total_time/60:.2f} minutes")
            print(f"Best F1 Score: {self.best_score:.4f}")
            
            return {
                'model': final_model,
                'params': final_params,
                'best_score': self.best_score,
                'analysis': analysis,
                'total_time': total_time
            }
            
        except Exception as e:
            print(f"Pipeline failed with error: {e}")
            raise
        finally:
            # Shutdown Ray
            ray.shutdown()

def main():
    """Main function to run hyperparameter tuning."""
    
    # Configuration
    data_file = "data/UNSW-NB15_1.csv"  # Update with your data path
    test_data_file = None  # Set to path if separate test file exists
    num_samples = 50  # Number of hyperparameter configurations to try
    max_concurrent_trials = 2  # Adjust based on your system resources
    
    print("Enhanced XGBoost Hyperparameter Tuning")
    print("="*60)
    
    # Check if data file exists
    if not os.path.exists(data_file):
        print(f"Error: Data file not found at {data_file}")
        print("Please update the data_file path in the main() function")
        return
    
    # Initialize trainer
    trainer = EnhancedXGBoostTrainer(
        data_file=data_file,
        test_data_file=test_data_file,
        num_samples=num_samples,
        max_concurrent_trials=max_concurrent_trials,
        use_global_processor=True
    )
    
    # Run complete tuning pipeline
    results = trainer.run_complete_tuning()
    
    print(f"\nTuning completed successfully!")
    print(f"Best model saved in outputs/")

if __name__ == "__main__":
    main() 