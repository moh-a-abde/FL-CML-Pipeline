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
from logging import INFO
import os

import flwr as fl
from flwr.common.logger import log

from dataset import (
    load_csv_data,
    instantiate_partitioner,
    train_test_split,
    transform_dataset_to_dmatrix,
    resplit,
)
from utils import client_args_parser, BST_PARAMS, NUM_LOCAL_ROUND
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
    
    # Load labeled data for training
    labeled_csv_path = "data/static_data.csv"
    labeled_dataset = load_csv_data(labeled_csv_path)
    
    # Load unlabeled data for prediction
    unlabeled_csv_path = "data/testing_data.csv"
    unlabeled_dataset = load_csv_data(unlabeled_csv_path)
    
    # Initialize data partitioner based on specified strategy
    partitioner = instantiate_partitioner(
        partitioner_type=args.partitioner_type,
        num_partitions=args.num_partitions
    )
    
    # Load the specific partition for training
    log(INFO, "Loading training partition...")
    train_partition = labeled_dataset["train"]
    train_partition.set_format("numpy")
    
    # Handle data splitting based on evaluation strategy
    if args.centralised_eval:
        # Use centralized test set for evaluation
        train_data = train_partition
        valid_data = labeled_dataset["test"]
        valid_data.set_format("numpy")
        num_train = train_data.shape[0]
        num_val = valid_data.shape[0]
    else:
        # Perform local train/test split
        train_data, valid_data, num_train, num_val = train_test_split(
            train_partition,
            test_fraction=args.test_fraction,
            seed=args.seed
        )
    
    # Transform training data into XGBoost's DMatrix format
    log(INFO, "Reformatting training data...")
    train_dmatrix = transform_dataset_to_dmatrix(train_data)
    valid_dmatrix = transform_dataset_to_dmatrix(valid_data)
    
    # Transform unlabeled data for prediction
    log(INFO, "Reformatting unlabeled data...")
    unlabeled_data = unlabeled_dataset["train"]
    unlabeled_data.set_format("numpy")
    unlabeled_dmatrix = transform_dataset_to_dmatrix(unlabeled_data)
    
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
