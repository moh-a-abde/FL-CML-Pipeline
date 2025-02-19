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
import time

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

def main():
    args = client_args_parser()
    
    # Load and preprocess data
    labeled_csv_path = "data/combined_labelled.csv"
    unlabeled_csv_path = "data/combined_unlabelled.csv"
    labeled_dataset = load_csv_data(labeled_csv_path)
    unlabeled_dataset = load_csv_data(unlabeled_csv_path)
    
    # Initialize client with retry mechanism
    max_retries = 5
    retry_delay = 10  # seconds
    
    for attempt in range(max_retries):
        try:
            # Create client
            client = XgbClient(
                partition_id=args.partition_id,
                num_partitions=args.num_partitions,
                labeled_dataset=labeled_dataset,
                unlabeled_dataset=unlabeled_dataset,
                partitioner_type=args.partitioner_type,
                test_fraction=args.test_fraction,
                seed=args.seed,
                params=BST_PARAMS,
                train_method=args.train_method
            )
            
            # Start client
            fl.client.start_client(
                server_address="127.0.0.1:8080",
                client=client,
            )
            break
        except Exception as e:
            if attempt < max_retries - 1:
                log(INFO, f"Connection attempt {attempt + 1} failed, retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                raise e

if __name__ == "__main__":
    main()
