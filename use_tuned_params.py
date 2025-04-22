"""
use_tuned_params.py

This script loads the optimized hyperparameters found by Ray Tune and integrates them
into the existing federated learning system. It replaces the default XGBoost parameters
in both client_utils.py and utils.py with the optimized ones.

Usage:
    python use_tuned_params.py --params-file ./tune_results/best_params.json
"""

import os
import json
import argparse
import logging
from utils import BST_PARAMS

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_tuned_params(params_file):
    """
    Load the optimized hyperparameters from a JSON file.
    
    Args:
        params_file (str): Path to the JSON file containing the optimized parameters
        
    Returns:
        dict: Optimized hyperparameters
    """
    if not os.path.exists(params_file):
        raise FileNotFoundError(f"Parameters file not found: {params_file}")
    
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
    # Start with the base parameters
    xgb_params = BST_PARAMS.copy()
    
    # Update with tuned parameters - convert float values to ints for integer parameters
    xgb_params.update({
        'max_depth': int(tuned_params['max_depth']),  # HyperOpt returns float, convert to int
        'min_child_weight': int(tuned_params['min_child_weight']),  # HyperOpt returns float, convert to int
        'eta': tuned_params['eta'],  # Learning rate
        'subsample': tuned_params['subsample'],
        'colsample_bytree': tuned_params['colsample_bytree'],
        'reg_alpha': tuned_params['reg_alpha'],
        'reg_lambda': tuned_params['reg_lambda']
    })
    
    # Add num_boost_round if it exists in tuned_params
    if 'num_boost_round' in tuned_params:
        xgb_params['num_boost_round'] = int(tuned_params['num_boost_round'])  # Convert to int
    
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
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("# This file is generated automatically by use_tuned_params.py\n")
        f.write("# It contains optimized XGBoost parameters found by Ray Tune\n\n")
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

def backup_original_params():
    """
    Backup the original parameters to a Python file for reference.
    
    Returns:
        str: Path to the backup file
    """
    backup_file = "original_bst_params.json"
    with open(backup_file, 'w', encoding='utf-8') as f:
        json.dump(BST_PARAMS, f, indent=2)
    
    logger.info("Original parameters backed up to %s", backup_file)
    return backup_file

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Use tuned hyperparameters with the existing XGBoost client")
    parser.add_argument("--params-file", type=str, required=True, help="Path to the tuned parameters JSON file")
    parser.add_argument("--output-file", type=str, default="tuned_params.py", help="Output Python file for updated parameters")
    args = parser.parse_args()
    
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

if __name__ == "__main__":
    main() 