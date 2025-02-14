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

# Parse command line arguments for experimental settings
args = client_args_parser()

# Set training method (either 'bagging' or 'cyclic')
train_method = args.train_method

# Load and prepare dataset
csv_file_path = "data/unlabeled.csv"
#csv_file_path = get_latest_csv("/home/mohamed/Desktop/test_repo/data")
dataset = load_csv_data(csv_file_path)

# Initialize data partitioner based on specified strategy
partitioner = instantiate_partitioner(
    partitioner_type=args.partitioner_type,  # Type of partitioning strategy
    num_partitions=args.num_partitions       # Number of total partitions
)
fds = dataset

# Load the specific partition for this client
log(INFO, "Loading partition...")
partition = fds["train"]
partition.set_format("numpy")

# Handle data splitting based on evaluation strategy
if args.centralised_eval:
    # Use centralized test set for evaluation
    train_data = partition
    valid_data = fds["test"]
    valid_data.set_format("numpy")
    num_train = train_data.shape[0]
    num_val = valid_data.shape[0]
else:
    # Perform local train/test split
    train_data, valid_data, num_train, num_val = train_test_split(
        partition,
        test_fraction=args.test_fraction,
        seed=args.seed
    )

# Transform data into XGBoost's DMatrix format
log(INFO, "Reformatting data...")
train_dmatrix = transform_dataset_to_dmatrix(train_data)
valid_dmatrix = transform_dataset_to_dmatrix(valid_data)

# Configure training parameters
num_local_round = NUM_LOCAL_ROUND  # Number of local training rounds
params = BST_PARAMS                # XGBoost parameters

# Adjust learning rate for bagging method if specified
if args.train_method == "bagging" and args.scaled_lr:
    new_lr = params["eta"] / args.num_partitions
    params.update({"eta": new_lr})

# Initialize and start Flower client
fl.client.start_client(
    server_address="127.0.0.1:8080",  # FL server address
    client=XgbClient(
        train_dmatrix,      # Training data
        valid_dmatrix,      # Validation data
        num_train,          # Number of training samples
        num_val,            # Number of validation samples
        num_local_round,    # Number of local training rounds
        params,             # XGBoost parameters
        train_method,       # Training method (bagging/cyclic)
    ),
)
