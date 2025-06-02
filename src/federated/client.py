"""
Client implementation for XGBoost federated learning.

This module implements the Flower client for federated XGBoost training,
including data loading, local training, and parameter exchange with the server.
"""

import os
import sys
import json
import numpy as np
import xgboost as xgb
from typing import Dict, List, Tuple, Union, Optional
import flwr as fl
from flwr.common.logger import log
from logging import INFO, WARNING, ERROR
from flwr.common import (
    NDArrays,
    Parameters,
    FitIns,
    FitRes,
    EvaluateIns,
    EvaluateRes,
)

# Import dataset and utility functions
from src.core.dataset import (
    load_csv_data,
    transform_dataset_to_dmatrix,
    FeatureProcessor,
    create_global_feature_processor,
    load_global_feature_processor,
    resplit,
    instantiate_partitioner,
)
from datasets import Dataset
from src.config.legacy_constants import client_args_parser, BST_PARAMS
from .client_utils import XGBClient

def load_data(client_id: int, data_file: str, partition_type: str = "uniform", 
              num_clients: int = 2, global_processor_path: str = None) -> Tuple[xgb.DMatrix, xgb.DMatrix, FeatureProcessor]:
    """
    Load and partition data for a specific client.
    
    Args:
        client_id (int): The ID of the client (0-indexed)
        data_file (str): Path to the data file
        partition_type (str): Type of data partitioning ("uniform", "linear", "square", "exponential")
        num_clients (int): Total number of clients
        global_processor_path (str): Path to the global feature processor
        
    Returns:
        Tuple[xgb.DMatrix, xgb.DMatrix, FeatureProcessor]: Training data, test data, and processor
    """
    log(INFO, f"Loading data for client {client_id}")
    
    # Load global feature processor if available
    processor = None
    if global_processor_path and os.path.exists(global_processor_path):
        log(INFO, f"Loading global feature processor from {global_processor_path}")
        processor = load_global_feature_processor(global_processor_path)
    else:
        log(WARNING, f"Global processor not found at {global_processor_path}, will create new one")
    
    # Load the dataset
    dataset = load_csv_data(data_file)
    
    # Create partitioner
    partitioner = instantiate_partitioner(partition_type, num_clients)
    
    # Apply the partitioner to the dataset
    partitioner.dataset = dataset
    
    # Get the partition for this client
    client_dataset = partitioner.load_partition(client_id, "train")  # Use train split for partitioning
    
    # Convert to DMatrix for training
    train_data = transform_dataset_to_dmatrix(client_dataset, processor=processor, is_training=True)
    
    # Use the test split from the main dataset for evaluation (not partitioned)
    test_dataset = dataset["test"]
    test_data = transform_dataset_to_dmatrix(test_dataset, processor=processor, is_training=False)
    
    log(INFO, f"Client {client_id} - Training samples: {train_data.num_row()}, Test samples: {test_data.num_row()}")
    log(INFO, f"Client {client_id} - Features: {train_data.num_col()}")
    
    return train_data, test_data, processor

def start_client(
    server_address: str,
    client_id: int,
    data_file: str,
    partition_type: str = "uniform",
    num_clients: int = 2,
    global_processor_path: str = None,
    use_https: bool = False
):
    """
    Start a Flower client for federated XGBoost learning.
    
    Args:
        server_address (str): Address of the Flower server
        client_id (int): Unique identifier for this client
        data_file (str): Path to the training data file
        partition_type (str): Type of data partitioning
        num_clients (int): Total number of clients in the federation
        global_processor_path (str): Path to the global feature processor
        use_https (bool): Whether to use HTTPS for server communication
    """
    log(INFO, f"Starting client {client_id}")
    log(INFO, f"Server address: {server_address}")
    log(INFO, f"Data file: {data_file}")
    log(INFO, f"Partition type: {partition_type}")
    log(INFO, f"Number of clients: {num_clients}")
    log(INFO, f"Global processor: {global_processor_path}")
    
    # Load client data
    train_data, test_data, processor = load_data(
        client_id=client_id,
        data_file=data_file,
        partition_type=partition_type,
        num_clients=num_clients,
        global_processor_path=global_processor_path
    )
    
    # Create XGBoost client
    client = XGBClient(
        train_data=train_data,
        test_data=test_data,
        processor=processor,
        client_id=client_id
    )
    
    # Connect to server
    if use_https:
        fl.client.start_client(
            server_address=server_address,
            client=client.to_client(),
            transport="grpc-bidi"
        )
    else:
        fl.client.start_client(
            server_address=server_address,
            client=client.to_client()
        )

def main():
    """Main function to start the federated learning client."""
    
    # Parse command line arguments
    args = client_args_parser()
    
    log(INFO, f"Starting Federated XGBoost Client {args.client_id}")
    log(INFO, f"Arguments: {vars(args)}")
    
    # Validate data file exists
    if not os.path.exists(args.data_file):
        log(ERROR, f"Data file not found: {args.data_file}")
        return
    
    # Create global processor if it doesn't exist
    if args.global_processor_path:
        processor_dir = os.path.dirname(args.global_processor_path)
        if not os.path.exists(processor_dir):
            os.makedirs(processor_dir, exist_ok=True)
        
        if not os.path.exists(args.global_processor_path):
            log(INFO, f"Creating global feature processor at {args.global_processor_path}")
            create_global_feature_processor(args.data_file, processor_dir)
    
    # Start the client
    try:
        start_client(
            server_address=args.server_address,
            client_id=args.client_id,
            data_file=args.data_file,
            partition_type=args.partition_type,
            num_clients=args.num_clients,
            global_processor_path=args.global_processor_path,
            use_https=args.use_https
        )
    except KeyboardInterrupt:
        log(INFO, "Client stopped by user")
    except Exception as e:
        log(ERROR, f"Client failed with error: {e}")
        raise

if __name__ == "__main__":
    main() 