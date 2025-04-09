"""
smote_processor.py

This module implements SMOTE (Synthetic Minority Over-sampling Technique) specifically for
the benign class in the network traffic classification task. It provides a wrapper that
can be used without modifying the existing codebase.

Key Components:
- SMOTE application to benign class
- Integration with XGBoost DMatrix
- Utility functions for train/test dataset modification
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from logging import INFO
from flwr.common.logger import log
from imblearn.over_sampling import SMOTE
from typing import Tuple, Optional, Any
import warnings

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)

def apply_smote_to_benign(
    train_dmatrix: xgb.DMatrix,
    valid_dmatrix: xgb.DMatrix,
    processor: Any,
    random_state: int = 42
) -> Tuple[xgb.DMatrix, xgb.DMatrix, Any]:
    """
    Apply SMOTE specifically to the benign class (label 0) in the training data.
    
    Args:
        train_dmatrix (xgb.DMatrix): Training data in DMatrix format
        valid_dmatrix (xgb.DMatrix): Validation data in DMatrix format
        processor (Any): Feature processor instance for consistent preprocessing
        random_state (int): Random seed for reproducibility
        
    Returns:
        Tuple[xgb.DMatrix, xgb.DMatrix, Any]: 
            - Training DMatrix with SMOTE applied to benign class
            - Original validation DMatrix (unchanged)
            - Original processor (unchanged)
    """
    log(INFO, "Starting SMOTE processing for benign class (label 0)...")
    
    try:
        # Get feature names if they exist (important for preserving model compatibility)
        feature_names = None
        feature_types = None
        
        try:
            feature_names = train_dmatrix.feature_names
            feature_types = train_dmatrix.feature_types
        except AttributeError:
            log(INFO, "No feature names found in DMatrix")
        
        # Extract features and labels from training DMatrix
        X_train = train_dmatrix.get_data()
        try:
            # Convert to dense array if it's a sparse matrix
            X_train = X_train.toarray()
        except:
            # Already dense or other format, proceed
            pass
        
        y_train = train_dmatrix.get_label()
        
        # Check class distribution before SMOTE
        class_counts_before = np.bincount(y_train.astype(int))
        log(INFO, "Class distribution before SMOTE:")
        class_names = ['benign', 'dns_tunneling', 'icmp_tunneling']
        for i, count in enumerate(class_counts_before):
            class_name = class_names[i] if i < len(class_names) else f'unknown_{i}'
            log(INFO, f"  Class {class_name}: {count}")
        
        # We want to oversample only the benign class (label 0)
        # First, we'll find the size of the largest class
        target_class_size = max(class_counts_before)
        
        # Only proceed with SMOTE if benign is not already the largest class
        if class_counts_before[0] < target_class_size:
            log(INFO, f"Applying SMOTE to increase benign class from {class_counts_before[0]} to {target_class_size}")
            
            # Prepare the sampling strategy dictionary
            # We only want to oversample class 0 (benign) to match the largest class
            sampling_strategy = {0: target_class_size}
            
            # Configure SMOTE for the benign class only
            smote = SMOTE(
                sampling_strategy=sampling_strategy,
                random_state=random_state,
                k_neighbors=min(5, class_counts_before[0] - 1),  # Ensure k is suitable for the class size
                n_jobs=-1  # Use all available cores
            )
            
            # Apply SMOTE
            X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
            
            # Check class distribution after SMOTE
            class_counts_after = np.bincount(y_resampled.astype(int))
            log(INFO, "Class distribution after SMOTE:")
            for i, count in enumerate(class_counts_after):
                class_name = class_names[i] if i < len(class_names) else f'unknown_{i}'
                log(INFO, f"  Class {class_name}: {count}")
            
            # Create new DMatrix with the resampled data, preserving feature names
            kwargs = {'missing': np.nan}
            if feature_names is not None:
                kwargs['feature_names'] = feature_names
                log(INFO, f"Preserved {len(feature_names)} feature names from original DMatrix")
            if feature_types is not None:
                kwargs['feature_types'] = feature_types
            
            # Create a new DMatrix with preserved metadata
            train_dmatrix_resampled = xgb.DMatrix(
                data=X_resampled, 
                label=y_resampled,
                **kwargs
            )
            
            # Additional compatibility check (debug information)
            if feature_names is not None:
                log(INFO, "Original DMatrix feature count: %d, Resampled DMatrix feature count: %d", 
                    len(feature_names), train_dmatrix_resampled.num_col())
            
            return train_dmatrix_resampled, valid_dmatrix, processor
        else:
            log(INFO, "Benign class is already the largest class. No need to apply SMOTE.")
            return train_dmatrix, valid_dmatrix, processor
    except Exception as e:
        log(INFO, f"Error in SMOTE processing: {str(e)}")
        log(INFO, "Continuing with original training data")
        return train_dmatrix, valid_dmatrix, processor

def apply_smote_wrapper(
    train_test_split_result: Tuple[xgb.DMatrix, xgb.DMatrix, Any],
    random_state: int = 42
) -> Tuple[xgb.DMatrix, xgb.DMatrix, Any]:
    """
    Wrapper function that takes the result of train_test_split and applies SMOTE
    to the benign class.
    
    Args:
        train_test_split_result (Tuple): Result from the original train_test_split function
            (train_dmatrix, valid_dmatrix, processor)
        random_state (int): Random seed for reproducibility
        
    Returns:
        Tuple: Same format as input but with SMOTE applied to train_dmatrix
    """
    train_dmatrix, valid_dmatrix, processor = train_test_split_result
    
    # Apply SMOTE
    return apply_smote_to_benign(
        train_dmatrix=train_dmatrix,
        valid_dmatrix=valid_dmatrix,
        processor=processor,
        random_state=random_state
    ) 