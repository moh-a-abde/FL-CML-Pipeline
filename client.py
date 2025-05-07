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
    preprocess_data
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
    #latest_csv_path = get_latest_csv(data_directory)
    #labeled_dataset = load_csv_data(latest_csv_path)
    #unlabeled_dataset = load_csv_data(latest_csv_path)
    
    # Ensure data/received/ directory exists
    os.makedirs("data/received", exist_ok=True)
    
    # Load labeled data for training
    labeled_csv_path = "data/received/UNSW_NB15_training-set.csv"
    labeled_dataset = load_csv_data(labeled_csv_path)
    
    # Load unlabeled data for prediction
    unlabeled_csv_path = "data/received/UNSW_NB15_testing-set.csv"
    unlabeled_dataset = load_csv_data(unlabeled_csv_path)
    
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
    except AttributeError:
        # If that fails, try using the partition method (used in newer versions)
        try:
            # Newer versions use a different API
            train_partition = partitioner.partition(full_train_data)[args.partition_id]
        except (AttributeError, TypeError, IndexError):
            # As a fallback, if all partition methods fail, use a simple numerical partition
            # by getting evenly spaced indices based on partition ID
            total_samples = len(full_train_data)
            samples_per_partition = total_samples // args.num_partitions
            start_idx = args.partition_id * samples_per_partition
            end_idx = start_idx + samples_per_partition if args.partition_id < args.num_partitions - 1 else total_samples
            partition_indices = list(range(start_idx, end_idx))
            train_partition = full_train_data.select(partition_indices)
            log(INFO, "Used fallback partitioning. Partition %d: samples %d to %d", 
                args.partition_id, start_idx, end_idx)
    
    log(INFO, "Partition size: %d samples (out of %d total)", 
        len(train_partition), len(full_train_data))
    
    # Handle data splitting based on evaluation strategy
    if args.centralised_eval:
        # Use centralized test set for evaluation
        train_data = train_partition
        valid_data = labeled_dataset["test"]
        valid_data.set_format("numpy")
        num_train = train_data.shape[0]
        num_val = valid_data.shape[0]
    else:
        # Perform local train/test split using the updated function
        # This now returns the fitted processor as well
        log(INFO, "Performing local train/test split...")
        
        # Generate a unique seed for each client based on partition_id to ensure
        # different clients get different train/test splits
        client_specific_seed = args.seed + (args.partition_id * 1000)
        log(INFO, "Using client-specific random seed for train/test split: %d", client_specific_seed)
        
        train_dmatrix, valid_dmatrix, processor = train_test_split( # Capture processor
            train_partition,
            test_fraction=args.test_fraction,
            random_state=client_specific_seed  # Use client-specific seed
        )
        # Get counts from the DMatrix objects
        num_train = train_dmatrix.num_row()
        num_val = valid_dmatrix.num_row()
        log(INFO, "Local split: %d train samples, %d validation samples", num_train, num_val)
    
    # Transform unlabeled data for prediction using the processor from train_test_split
    log(INFO, "Reformatting unlabeled data...")
    unlabeled_data = unlabeled_dataset["train"]
    # Convert unlabeled data to pandas DataFrame first
    if not isinstance(unlabeled_data, pd.DataFrame):
        unlabeled_data = unlabeled_data.to_pandas()
    
    # Preprocess unlabeled data using the fitted processor from train/test split
    try:
        # Use the processor returned by train_test_split
        unlabeled_features, _ = preprocess_data(unlabeled_data, processor=processor, is_training=False)
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
