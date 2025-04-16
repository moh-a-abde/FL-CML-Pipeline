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
    
    logger.info(f"Loading optimized parameters from {params_file}")
    with open(params_file, 'r') as f:
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
    
    # Update with tuned parameters
    xgb_params.update({
        'max_depth': tuned_params['max_depth'],
        'min_child_weight': tuned_params['min_child_weight'],
        'eta': tuned_params['eta'],  # Learning rate
        'subsample': tuned_params['subsample'],
        'colsample_bytree': tuned_params['colsample_bytree'],
        'reg_alpha': tuned_params['reg_alpha'],
        'reg_lambda': tuned_params['reg_lambda']
    })
    
    # Add GPU support if specified in tuned parameters
    if 'tree_method' in tuned_params and tuned_params['tree_method'] == 'gpu_hist':
        xgb_params['tree_method'] = 'gpu_hist'
    
    return xgb_params

def save_updated_params(params, output_file):
    """
    Save updated parameters to a Python file that can be imported by client_utils.py
    
    Args:
        params (dict): Updated XGBoost parameters
        output_file (str): Path to save the updated parameters
    """
    with open(output_file, 'w') as f:
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
    
    logger.info(f"Updated XGBoost parameters saved to {output_file}")

def backup_original_params():
    """
    Backup the original parameters in BST_PARAMS from utils.py
    
    Returns:
        str: Path to the backup file
    """
    backup_file = "original_bst_params.json"
    with open(backup_file, 'w') as f:
        json.dump(BST_PARAMS, f, indent=2)
    
    logger.info(f"Original parameters backed up to {backup_file}")
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
        logger.info(f"  {key}: {value}")
    
    # Create updated XGBoost parameters
    updated_params = create_xgboost_params(tuned_params)
    
    # Save updated parameters
    save_updated_params(updated_params, args.output_file)
    
    # Print instructions for using the tuned parameters
    logger.info("\nTo use these optimized parameters in your federated learning system:")
    logger.info("1. Import the parameters in client_utils.py:")
    logger.info("   from tuned_params import TUNED_PARAMS")
    logger.info("2. Replace BST_PARAMS with TUNED_PARAMS in the XgbClient class:")
    logger.info("   self.params = params if params is not None else TUNED_PARAMS.copy()")
    logger.info(f"\nOriginal parameters are backed up to {backup_file}")

if __name__ == "__main__":
    main() 