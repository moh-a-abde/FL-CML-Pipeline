"""
client.py

This module implements the Federated Learning client functionality for distributed XGBoost training.
It handles data loading, preprocessing, and client-side model training operations.

Key Components:
- Data loading and partitioning
- Client initialization
- Model training configuration
- Connection to FL server
"""

import warnings
from logging import INFO, WARNING, ERROR
import os
import pandas as pd
import xgboost as xgb
import numpy as np

import flwr as fl
from flwr.common.logger import log

from dataset import (
    load_csv_data,
    instantiate_partitioner,
    train_test_split,
    FeatureProcessor,
    preprocess_data,
    load_global_feature_processor,
    create_global_feature_processor
)
from utils import client_args_parser, BST_PARAMS

# Try to import NUM_LOCAL_ROUND from tuned_params if available, otherwise from utils
try:
    from tuned_params import NUM_LOCAL_ROUND
    import logging
    logging.getLogger(__name__).info("Using NUM_LOCAL_ROUND from tuned_params.py")
except ImportError:
    from utils import NUM_LOCAL_ROUND
    import logging
    logging.getLogger(__name__).info("Using default NUM_LOCAL_ROUND from utils.py")

from client_utils import XgbClient

warnings.filterwarnings("ignore", category=UserWarning)

def get_latest_csv(directory: str) -> str:
    """
    Retrieves the most recently modified CSV file from the specified directory.

    Args:
        directory (str): Path to the directory containing CSV files

    Returns:
        str: Full path to the most recent CSV file

    Example:
        latest_file = get_latest_csv("/path/to/data/directory")
    """
    csv_files = [f for f in os.listdir(directory) if f.endswith('.csv')]
    latest_file = max(csv_files, key=lambda x: os.path.getctime(os.path.join(directory, x)))
    return os.path.join(directory, latest_file)

if __name__ == "__main__":
    # Parse command line arguments for experimental settings
    args = client_args_parser()

    data_directory = "data/received"
    
    # Ensure data/received/ directory exists
    os.makedirs("data/received", exist_ok=True)
    
    # Load labeled data for training - using the original final dataset with proper temporal splitting
    labeled_csv_path = "data/received/final_dataset.csv"
    log(INFO, "Using original final dataset with temporal splitting: %s", labeled_csv_path)

    # Check for global feature processor first
    global_processor_path = "outputs/global_feature_processor.pkl"
    if os.path.exists(global_processor_path):
        log(INFO, "Loading global feature processor for consistent preprocessing")
        global_processor = load_global_feature_processor(global_processor_path)
    else:
        log(WARNING, "Global feature processor not found, creating one from the dataset")
        global_processor_path = create_global_feature_processor(labeled_csv_path, "outputs")
        global_processor = load_global_feature_processor(global_processor_path)

    labeled_dataset = load_csv_data(labeled_csv_path)
    
    # Load unlabeled data for prediction if available
    # For now, we'll use a portion of the labeled data as unlabeled
    # This should be updated once an unlabeled version of the engineered dataset is available
    unlabeled_csv_path = labeled_csv_path  # Using the same file for now
    unlabeled_dataset = labeled_dataset    # Using the same dataset for now
    
    # Initialize data partitioner based on specified strategy
    partitioner = instantiate_partitioner(
        partitioner_type=args.partitioner_type,
        num_partitions=args.num_partitions
    )
    
    # Load the specific partition for training based on partition_id
    log(INFO, "Loading training partition for client with partition_id=%d...", args.partition_id)
    
    # Get the entire dataset first
    full_train_data = labeled_dataset["train"] 
    full_train_data.set_format("numpy")
    
    # Apply the partitioner to get client-specific data partition
    # The ExponentialPartitioner doesn't have get_indices, but provides partition method
    # which returns the partition directly rather than just indices
    try:
        # First try to use get_partition method which returns the partition subset directly
        train_partition = partitioner.get_partition(full_train_data, args.partition_id)
    except Exception as e:
        log(INFO, f"get_partition failed ({e}), using fallback partitioning")
        # Fallback to simple index-based partitioning if get_partition is not available
        total_samples = len(full_train_data)
        samples_per_partition = total_samples // args.num_partitions
        start_idx = args.partition_id * samples_per_partition
        end_idx = (args.partition_id + 1) * samples_per_partition if args.partition_id < args.num_partitions - 1 else total_samples
        
        log(INFO, f"Used fallback partitioning. Partition {args.partition_id}: samples {start_idx} to {end_idx}")
        log(INFO, f"Partition size: {end_idx - start_idx} samples (out of {total_samples} total)")
        
        # Get the partition slice
        train_partition = full_train_data.select(range(start_idx, end_idx))
    
    # Convert to pandas for easier manipulation
    train_partition_df = train_partition.to_pandas()
    
    # Perform local train/test split using client-specific random seed
    client_seed = args.seed + args.partition_id * 1000  # Make each client's seed unique
    log(INFO, f"Using client-specific random seed for train/test split: {client_seed}")
    
    # Log data shape before splitting
    log(INFO, f"Original data shape before splitting: {train_partition_df.shape}")
    
    # Check class distribution before splitting
    if 'label' in train_partition_df.columns:
        class_dist = train_partition_df['label'].value_counts().to_dict()
        log(INFO, f"Class distribution in original data: {class_dist}")
    
    # Use different random states for train/validation split to ensure no overlap
    train_random_state = client_seed
    val_random_state = client_seed + 5000  # Different seed for validation
    log(INFO, f"Using different random states for train/validation split: {train_random_state}/{val_random_state}")
    
    # Perform the split
    from sklearn.model_selection import train_test_split as sklearn_split
    train_df, valid_df = sklearn_split(
        train_partition_df, 
        test_size=args.test_fraction, 
        random_state=train_random_state,
        stratify=train_partition_df['label'] if 'label' in train_partition_df.columns else None
    )
    
    log(INFO, f"Train data shape: {train_df.shape}, Test data shape: {valid_df.shape}")
    
    # Log class distributions after splitting
    if 'label' in train_df.columns:
        train_class_dist = train_df['label'].value_counts().to_dict()
        valid_class_dist = valid_df['label'].value_counts().to_dict()
        log(INFO, f"Class distribution in train data: {train_class_dist}")
        log(INFO, f"Class distribution in test data: {valid_class_dist}")
    
    # Use the global processor to transform the data
    log(INFO, "Transforming data using global feature processor")
    
    try:
        # Transform using the global processor
        train_processed = global_processor.transform(train_df, is_training=True)
        valid_processed = global_processor.transform(valid_df, is_training=False)
        
        # Convert to DMatrix
        train_dmatrix = transform_dataset_to_dmatrix(train_processed, processor=global_processor, is_training=True)
        valid_dmatrix = transform_dataset_to_dmatrix(valid_processed, processor=global_processor, is_training=False)
        
        num_train = len(train_processed)
        num_val = len(valid_processed)
        
        log(INFO, f"Train DMatrix has {train_dmatrix.num_row()} rows, Test DMatrix has {valid_dmatrix.num_row()} rows")
        log(INFO, f"Local split: {num_train} train samples, {num_val} validation samples")
        
    except Exception as e:
        log(ERROR, f"Error in data transformation: {str(e)}")
        log(INFO, "Falling back to original train_test_split method")
        
        # Fallback to the original method if global processor fails
        train_dmatrix, valid_dmatrix, _ = train_test_split(
            train_partition, test_fraction=args.test_fraction, seed=args.seed
        )
        num_train = train_dmatrix.num_row()
        num_val = valid_dmatrix.num_row()
    
    # Transform unlabeled data for prediction using the processor from train_test_split
    log(INFO, "Reformatting unlabeled data...")
    unlabeled_data = unlabeled_dataset["train"]
    # Convert unlabeled data to pandas DataFrame first
    if not isinstance(unlabeled_data, pd.DataFrame):
        unlabeled_data = unlabeled_data.to_pandas()
    
    # Preprocess unlabeled data using the fitted processor from train/test split
    try:
        # Use the processor returned by train_test_split
        unlabeled_features, _ = preprocess_data(unlabeled_data, processor=global_processor, is_training=False)
        unlabeled_dmatrix = xgb.DMatrix(unlabeled_features, missing=np.nan)
        log(INFO, "Successfully preprocessed unlabeled data.")
    except Exception as e:
        log(ERROR, "Failed to preprocess unlabeled data or create DMatrix: %s", e)
        # Handle error appropriately, e.g., skip prediction for this client or use an empty DMatrix
        unlabeled_dmatrix = xgb.DMatrix(np.empty((0,0))) # Create an empty DMatrix as fallback

    
    # Configure training parameters
    num_local_round = NUM_LOCAL_ROUND
    params = BST_PARAMS
    
    # Adjust learning rate for bagging method if specified
    if args.train_method == "bagging" and args.scaled_lr:
        new_lr = params["eta"] / args.num_partitions
        params.update({"eta": new_lr})
    
    # Create client with both training and prediction data
    client = XgbClient(
        train_dmatrix=train_dmatrix,
        valid_dmatrix=valid_dmatrix,
        num_train=num_train,
        num_val=num_val,
        num_local_round=num_local_round,
        cid=args.partition_id,
        params=params,
        train_method=args.train_method,
        is_prediction_only=False,  # Set to False for training
        unlabeled_dmatrix=unlabeled_dmatrix  # Add unlabeled data for prediction
    )
    
    # Initialize and start Flower client
    fl.client.start_client(
        server_address="127.0.0.1:8080",
        client=client,
    )
