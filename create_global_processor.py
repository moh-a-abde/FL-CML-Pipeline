#!/usr/bin/env python3
"""
create_global_processor.py

Script to create a global feature processor that ensures consistent preprocessing 
across Ray Tune hyperparameter optimization and Federated Learning phases.

This addresses the disconnection between individual client training and federated
learning by ensuring all phases use the same preprocessing statistics.
"""

import argparse
import os
import sys
from dataset import create_global_feature_processor, load_global_feature_processor
from flwr.common.logger import log
from logging import INFO

def main():
    """Create global feature processor for consistent preprocessing."""
    parser = argparse.ArgumentParser(
        description="Create global feature processor for consistent preprocessing"
    )
    parser.add_argument(
        "--data-file",
        type=str,
        default="data/received/final_dataset.csv",
        help="Path to the dataset file (default: data/received/final_dataset.csv)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Output directory for the processor (default: outputs)"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force recreation of processor even if it already exists"
    )
    
    args = parser.parse_args()
    
    # Check if data file exists
    if not os.path.exists(args.data_file):
        log(INFO, "Error: Data file not found: %s", args.data_file)
        sys.exit(1)
    
    # Check if processor already exists
    processor_path = os.path.join(args.output_dir, "global_feature_processor.pkl")
    if os.path.exists(processor_path) and not args.force:
        log(INFO, "Global feature processor already exists at: %s", processor_path)
        log(INFO, "Use --force to recreate it")
        
        # Load and display info about existing processor
        try:
            processor = load_global_feature_processor(processor_path)
            log(INFO, "Existing processor details:")
            log(INFO, "  Dataset type: %s", getattr(processor, 'dataset_type', 'unknown'))
            log(INFO, "  Categorical features: %d", len(processor.categorical_features))
            log(INFO, "  Numerical features: %d", len(processor.numerical_features))
            log(INFO, "  Is fitted: %s", processor.is_fitted)
        except (FileNotFoundError, ImportError, AttributeError) as e:
            log(INFO, "Error loading existing processor: %s", str(e))
            log(INFO, "Consider using --force to recreate it")
        
        sys.exit(0)
    
    # Create the global processor
    log(INFO, "Creating global feature processor...")
    try:
        processor_path = create_global_feature_processor(args.data_file, args.output_dir)
        log(INFO, "Successfully created global feature processor at: %s", processor_path)
        
        # Verify the processor
        processor = load_global_feature_processor(processor_path)
        log(INFO, "Verification successful!")
        log(INFO, "  Dataset type: %s", getattr(processor, 'dataset_type', 'unknown'))
        log(INFO, "  Categorical features: %d", len(processor.categorical_features))
        log(INFO, "  Numerical features: %d", len(processor.numerical_features))
        log(INFO, "  Is fitted: %s", processor.is_fitted)
        
        if hasattr(processor, 'unique_labels'):
            log(INFO, "  Unique labels: %s", processor.unique_labels)
        
    except (FileNotFoundError, ImportError, AttributeError, ValueError) as e:
        log(INFO, "Error creating global feature processor: %s", str(e))
        sys.exit(1)

if __name__ == "__main__":
    main() 