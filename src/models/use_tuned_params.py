"""
use_tuned_params.py

This script loads the optimized hyperparameters found by Ray Tune and integrates them
into the existing federated learning system. It replaces the default XGBoost parameters
in both client_utils.py and utils.py with the optimized ones.

Usage:
    python use_tuned_params.py --params-file ./tune_results/best_params.json
"""

import os
import sys
import json
import argparse
import logging

# Add project root directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))  # Go up two levels to project root
sys.path.insert(0, project_root)

from src.config.config_manager import ConfigManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def get_default_model_params():
    """
    Get default model parameters using ConfigManager or fallback values.
    
    Returns:
        dict: Default XGBoost parameters
    """
    try:
        # Try to get parameters from ConfigManager
        config_manager = ConfigManager()
        config_manager.load_config()  # Load the configuration first
        return config_manager.get_model_params_dict()
    except (ImportError, AttributeError, ValueError, KeyError, RuntimeError) as e:
        logger.warning("Could not load parameters from ConfigManager: %s", e)
        # Fallback to hardcoded defaults
        return {
            "objective": "multi:softprob",
            "num_class": 11,
            "eta": 0.05,
            "max_depth": 8,
            "min_child_weight": 5,
            "gamma": 0.5,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "colsample_bylevel": 0.8,
            "nthread": 16,
            "tree_method": "hist",
            "eval_metric": "mlogloss",
            "max_delta_step": 1,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "base_score": 0.5,
            "scale_pos_weight": 1.0,
            "grow_policy": "depthwise",
            "normalize_type": "tree",
            "random_state": 42
        }

def load_tuned_params(params_file):
    """
    Load the optimized hyperparameters from a JSON file.
    
    Args:
        params_file (str): Path to the JSON file containing the optimized parameters
        
    Returns:
        dict: Optimized hyperparameters
    """
    if not os.path.exists(params_file):
        if params_file == "./tune_results/best_params.json":
            logger.error("Default parameters file not found: %s", params_file)
            logger.error("This usually means Ray Tune hasn't been run yet or completed successfully.")
            logger.error("Please run the Ray Tune optimization first or specify a different --params-file")
        raise FileNotFoundError("Parameters file not found: %s" % params_file)
    
    logger.info("Loading optimized parameters from %s", params_file)
    with open(params_file, 'r', encoding='utf-8') as f:
        params = json.load(f)
    
    return params

def create_xgboost_params(tuned_params):
    """
    Create XGBoost parameters dictionary from the tuned parameters.
    
    Args:
        tuned_params (dict): Optimized hyperparameters from Ray Tune
        
    Returns:
        dict: XGBoost parameters dictionary for use in the existing system
    """
    # Start with the base parameters from ConfigManager or defaults
    xgb_params = get_default_model_params()
    
    # Update with tuned parameters - convert float values to ints for integer parameters
    # Use .get() with defaults to handle missing parameters gracefully
    xgb_params.update({
        'max_depth': int(tuned_params.get('max_depth', 6)),
        'min_child_weight': int(tuned_params.get('min_child_weight', 1)),
        'eta': tuned_params.get('eta', 0.1),  # Use eta instead of learning_rate
        'subsample': tuned_params.get('subsample', 0.8),
        'colsample_bytree': tuned_params.get('colsample_bytree', 0.8),
        'reg_alpha': tuned_params.get('reg_alpha', 0.1),
        'reg_lambda': tuned_params.get('reg_lambda', 1.0)
    })
    
    # Remove learning_rate if eta is present to avoid conflicts
    if 'eta' in xgb_params and 'learning_rate' in xgb_params:
        del xgb_params['learning_rate']
    
    # Add new hyperparameters if they exist in tuned_params
    if 'gamma' in tuned_params:
        xgb_params['gamma'] = tuned_params['gamma']
    if 'scale_pos_weight' in tuned_params:
        xgb_params['scale_pos_weight'] = tuned_params['scale_pos_weight']
    if 'max_delta_step' in tuned_params:
        xgb_params['max_delta_step'] = int(tuned_params['max_delta_step'])
    if 'colsample_bylevel' in tuned_params:
        xgb_params['colsample_bylevel'] = tuned_params['colsample_bylevel']
    if 'colsample_bynode' in tuned_params:
        xgb_params['colsample_bynode'] = tuned_params['colsample_bynode']
    
    # Add num_boost_round if it exists in tuned_params
    if 'num_boost_round' in tuned_params:
        xgb_params['num_boost_round'] = int(tuned_params['num_boost_round'])
    
    # Add GPU support if specified in tuned parameters
    if 'tree_method' in tuned_params:
        if isinstance(tuned_params['tree_method'], list) and len(tuned_params['tree_method']) > 0:
            # If it's from hp.choice, it will be a list
            xgb_params['tree_method'] = tuned_params['tree_method'][0]
        else:
            xgb_params['tree_method'] = tuned_params['tree_method']
    
    return xgb_params

def save_updated_params(params, output_file):
    """
    Save updated parameters to a Python file that can be imported by client_utils.py
    
    Args:
        params (dict): Updated XGBoost parameters
        output_file (str): Path to save the updated parameters
    """
    # Extract num_boost_round if present to use as NUM_LOCAL_ROUND
    num_local_round = None
    if 'num_boost_round' in params:
        num_local_round = int(params['num_boost_round'])
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("# This file is generated automatically by use_tuned_params.py\n")
        f.write("# It contains optimized XGBoost parameters found by Ray Tune\n\n")
        
        # Add NUM_LOCAL_ROUND if it was extracted from num_boost_round
        if num_local_round is not None:
            f.write(f"NUM_LOCAL_ROUND = {num_local_round}\n\n")
        
        f.write("TUNED_PARAMS = {\n")
        for key, value in params.items():
            if isinstance(value, str):
                f.write(f"    '{key}': '{value}',\n")
            elif isinstance(value, list):
                f.write(f"    '{key}': {value},\n")
            else:
                f.write(f"    '{key}': {value},\n")
        f.write("}\n")
    logger.info("Updated XGBoost parameters saved to %s", output_file)
    if num_local_round is not None:
        logger.info("NUM_LOCAL_ROUND set to %d based on tuned num_boost_round", num_local_round)

def backup_original_params():
    """
    Backup the original parameters to a JSON file for reference.
    
    Returns:
        str: Path to the backup file
    """
    backup_file = "original_bst_params.json"
    original_params = get_default_model_params()
    with open(backup_file, 'w', encoding='utf-8') as f:
        json.dump(original_params, f, indent=2)
    
    logger.info("Original parameters backed up to %s", backup_file)
    return backup_file

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Use tuned hyperparameters with the existing XGBoost client")
    parser.add_argument(
        "--params-file", 
        type=str, 
        default="./tune_results/best_params.json",
        help="Path to the tuned parameters JSON file (default: ./tune_results/best_params.json)"
    )
    parser.add_argument("--output-file", type=str, default="tuned_params.py", help="Output Python file for updated parameters")
    args = parser.parse_args()
    
    # Log which parameters file is being used
    if args.params_file == "./tune_results/best_params.json":
        logger.info("Using default parameters file: %s", args.params_file)
    else:
        logger.info("Using specified parameters file: %s", args.params_file)
    
    # Backup original parameters
    backup_file = backup_original_params()
    
    # Load tuned parameters
    tuned_params = load_tuned_params(args.params_file)
    logger.info("Loaded the following optimized parameters:")
    for key, value in tuned_params.items():
        logger.info("  %s: %s", key, value)
    
    # Create updated XGBoost parameters
    updated_params = create_xgboost_params(tuned_params)
    
    # Save updated parameters
    save_updated_params(updated_params, args.output_file)
    
    # Success message
    logger.info("Optimized parameters saved to %s", args.output_file)
    logger.info("Original parameters backed up to %s", backup_file)
    logger.info("These parameters will be automatically used by the XGBoost client")

def get_tuned_params():
    """
    Load tuned parameters for use by other modules.
    This function is called by XGBoostParamsBuilder.
    
    Returns:
        dict: Tuned XGBoost parameters or None if not available
    """
    try:
        # Try multiple locations for tuned parameters
        possible_files = [
            "./tune_results/best_params.json",
            "tune_results/best_params.json",
            "best_params.json"
        ]
        
        params_file = None
        for file_path in possible_files:
            if os.path.exists(file_path):
                params_file = file_path
                break
        
        if params_file is None:
            logger.warning("No tuned parameters file found in: %s", possible_files)
            return None
            
        logger.info("Loading tuned parameters from: %s", params_file)
        with open(params_file, 'r', encoding='utf-8') as f:
            tuned_params = json.load(f)
            
        # Convert to XGBoost format
        xgb_params = create_xgboost_params(tuned_params)
        
        logger.info("Successfully loaded tuned parameters with %d parameters", len(xgb_params))
        return xgb_params
        
    except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
        logger.warning("Error loading tuned parameters: %s", e)
        return None

if __name__ == "__main__":
    main() 