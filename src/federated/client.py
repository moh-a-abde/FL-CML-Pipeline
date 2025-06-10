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
from src.config.config_manager import get_config_manager, load_config
from .client_utils import XGBClient

def load_data(client_id: int, config, global_processor_path: str = None) -> Tuple[xgb.DMatrix, xgb.DMatrix, FeatureProcessor]:
    """
    Load and partition data for a specific client.
    
    Args:
        client_id (int): The ID of the client (0-indexed)
        config: Configuration object from ConfigManager
        global_processor_path (str): Path to the global feature processor
        
    Returns:
        Tuple[xgb.DMatrix, xgb.DMatrix, FeatureProcessor]: Training data, test data, and processor
    """
    log(INFO, f"Loading data for client {client_id}")
    
    # Get data file path from config
    data_file = os.path.join(config.data.path, config.data.filename)
    
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
    partitioner = instantiate_partitioner(
        config.federated.partitioner_type, 
        config.federated.num_partitions
    )
    
    # Apply the partitioner to the train dataset (not the full DatasetDict)
    partitioner.dataset = dataset["train"]
    
    # Get the partition for this client
    client_dataset = partitioner.load_partition(client_id)  # Load client's partition
    
    # Convert to DMatrix for training
    train_data = transform_dataset_to_dmatrix(client_dataset, processor=processor, is_training=True)
    
    # Use the test split from the main dataset for evaluation (not partitioned)
    test_dataset = dataset["test"]
    test_data = transform_dataset_to_dmatrix(test_dataset, processor=processor, is_training=False)
    
    log(INFO, f"Client {client_id} - Training samples: {train_data.num_row()}, Test samples: {test_data.num_row()}")
    log(INFO, f"Client {client_id} - Features: {train_data.num_col()}")
    
    return train_data, test_data, processor

def start_client(config, client_id: int, global_processor_path: str = None, use_https: bool = False):
    """
    Start a Flower client for federated XGBoost learning.
    
    Args:
        config: Configuration object from ConfigManager
        client_id (int): Unique identifier for this client
        global_processor_path (str): Path to the global feature processor
        use_https (bool): Whether to use HTTPS for server communication
    """
    log(INFO, f"Starting client {client_id}")
    log(INFO, f"Server address: 0.0.0.0:8080")  # Default server address
    log(INFO, f"Data path: {config.data.path}")
    log(INFO, f"Data filename: {config.data.filename}")
    log(INFO, f"Partition type: {config.federated.partitioner_type}")
    log(INFO, f"Number of partitions: {config.federated.num_partitions}")
    log(INFO, f"Global processor: {global_processor_path}")
    
    # Load client data
    train_data, test_data, processor = load_data(
        client_id=client_id,
        config=config,
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
    server_address = "0.0.0.0:8080"  # Default server address
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
    
    # Load configuration using ConfigManager
    log(INFO, "Loading configuration for federated client...")
    config = load_config()  # Load base configuration
    
    log(INFO, "Configuration loaded successfully:")
    log(INFO, "Data path: %s", config.data.path)
    log(INFO, "Data filename: %s", config.data.filename)
    log(INFO, "Partition type: %s", config.federated.partitioner_type)
    log(INFO, "Number of partitions: %d", config.federated.num_partitions)
    
    # For demonstration, start client 0 (in real deployment, this would be passed as argument)
    client_id = 0  # This could be passed as environment variable or command line arg
    
    # Validate data file exists
    data_file = os.path.join(config.data.path, config.data.filename)
    if not os.path.exists(data_file):
        log(ERROR, f"Data file not found: {data_file}")
        return
    
    # Set up global processor path
    global_processor_path = os.path.join(config.outputs.base_dir, "global_feature_processor.pkl")
    
    # Create global processor if it doesn't exist
    processor_dir = os.path.dirname(global_processor_path)
    if not os.path.exists(processor_dir):
        os.makedirs(processor_dir, exist_ok=True)
    
    if not os.path.exists(global_processor_path):
        log(INFO, f"Creating global feature processor at {global_processor_path}")
        create_global_feature_processor(data_file, processor_dir)
    
    # Start the client
    try:
        start_client(
            config=config,
            client_id=client_id,
            global_processor_path=global_processor_path,
            use_https=False
        )
    except KeyboardInterrupt:
        log(INFO, "Client stopped by user")
    except Exception as e:
        log(ERROR, f"Client failed with error: {e}")
        raise

if __name__ == "__main__":
    main() 