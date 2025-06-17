import warnings
import os
import sys
from logging import INFO
import xgboost as xgb
from tqdm import tqdm
import numpy as np
import pandas as pd

# Add project root directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))  # Go up two levels to project root
sys.path.insert(0, project_root)

import flwr as fl
from flwr.common.logger import log
from flwr.server.strategy import FedXgbBagging, FedXgbCyclic

from src.core.dataset import (
    instantiate_partitioner,
    train_test_split,
    transform_dataset_to_dmatrix,
    separate_xy,
    resplit,
    load_csv_data,
    FeatureProcessor,
    create_global_feature_processor,
    load_global_feature_processor,
)
from src.config.config_manager import get_config_manager, load_config
from src.utils.enhanced_logging import get_enhanced_logger
from src.federated.generic_client import GenericFederatedClient
from src.federated.strategies.random_forest_strategy import RandomForestBagging, RandomForestFedAvg

# Try to import NUM_LOCAL_ROUND from tuned_params if available, otherwise from utils
try:
    from src.config.tuned_params import NUM_LOCAL_ROUND
    import logging
    logging.getLogger(__name__).info("Using NUM_LOCAL_ROUND from tuned_params.py")
except ImportError:
    # We'll use the value from ConfigManager instead
    import logging
    logging.getLogger(__name__).info("Using NUM_LOCAL_ROUND from ConfigManager")

from src.federated.utils import (
    setup_output_directory,
    eval_config,
    fit_config,
    evaluate_metrics_aggregation,
    get_evaluate_fn,
    CyclicClientManager
)
from src.federated.client_utils import XgbClient

warnings.filterwarnings("ignore", category=UserWarning)

def get_latest_csv(directory: str) -> str:
    csv_files = [f for f in os.listdir(directory) if f.endswith('.csv')]
    latest_file = max(csv_files, key=lambda x: os.path.getctime(os.path.join(directory, x)))
    return os.path.join(directory, latest_file)
    
def get_client_fn(
    train_data_list, valid_data_list, train_method, params, num_local_round, config_manager, model_type
):
    """Return a function to construct a client.

    The VirtualClientEngine will execute this function whenever a client is sampled by
    the strategy to participate.
    """

    def client_fn(cid: str) -> fl.client.Client:
        """Construct a FlowerClient with its own dataset partition."""
        client_id = int(cid)
        
        if model_type == "xgboost":
            # XGBoost client creation (legacy approach)
            x_train, y_train = train_data_list[client_id][0]
            x_valid, y_valid = valid_data_list[client_id][0]

            # Reformat data to DMatrix
            train_dmatrix = xgb.DMatrix(x_train, label=y_train)
            valid_dmatrix = xgb.DMatrix(x_valid, label=y_valid)

            # Fetch the number of examples
            num_train = train_data_list[client_id][1]
            num_val = valid_data_list[client_id][1]

            # Create and return XGBoost client
            return XgbClient(
                train_dmatrix,
                valid_dmatrix,
                num_train,
                num_val,
                num_local_round,
                cid,
                params,
                train_method,
            )
        
        elif model_type == "random_forest":
            # Random Forest client creation using generic client
            x_train, y_train = train_data_list[client_id][0]
            x_valid, y_valid = valid_data_list[client_id][0]
            
            # Create generic client for Random Forest
            generic_client = GenericFederatedClient(
                client_id=client_id,
                config_manager=config_manager,
                global_processor_path="outputs/global_feature_processor.pkl"
            )
            
            # Manually set the data for the generic client
            generic_client.train_data = {
                'X': x_train,
                'y': y_train
            }
            generic_client.test_data = {
                'X': x_valid,
                'y': y_valid
            }
            
            # Return the Flower client
            return generic_client.get_flower_client()
        
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

    return client_fn

def main():
    # Get enhanced logger instance
    enhanced_logger = get_enhanced_logger()
    
    # Load configuration using ConfigManager
    enhanced_logger.logger.info("Loading configuration for federated simulation...")
    
    # Try to load the current configuration saved by run.py, fallback to base config
    try:
        config_manager = get_config_manager()
        
        # Check if there's a saved current config from run.py
        current_config_path = "outputs/current_config.yaml"
        if os.path.exists(current_config_path):
            enhanced_logger.logger.info("Loading current pipeline configuration from: %s", current_config_path)
            # Load the exact configuration used by run.py
            from omegaconf import OmegaConf
            raw_config = OmegaConf.load(current_config_path)
            config_manager._raw_config = raw_config
            config = config_manager._convert_to_structured_config(raw_config)
        else:
            enhanced_logger.logger.info("No current config found, loading base configuration...")
            config = load_config()  # Load base configuration
    except Exception as e:
        enhanced_logger.logger.warning("Failed to load current config: %s. Using base config.", str(e))
        config = load_config()  # Fallback to base configuration
    
    enhanced_logger.logger.info("Configuration loaded successfully:")
    enhanced_logger.logger.info("Training method: %s", config.federated.train_method)
    enhanced_logger.logger.info("Model type: %s", config.model.type)
    enhanced_logger.logger.info("Pool size: %d", config.federated.pool_size)
    enhanced_logger.logger.info("Number of rounds: %d", config.federated.num_rounds)
    enhanced_logger.logger.info("Clients per round: %d", config.federated.num_clients_per_round)
    enhanced_logger.logger.info("Centralized evaluation: %s", config.federated.centralised_eval)
    enhanced_logger.logger.info("Partitioner type: %s", config.federated.partitioner_type)
    
    # Get model type for strategy selection
    model_type = config.model.type.lower()
    
    # Get data file path
    csv_file_path = os.path.join(config.data.path, config.data.filename)
    enhanced_logger.logger.info("Loading dataset from: %s", csv_file_path)
    
    # Load CSV dataset
    dataset = load_csv_data(csv_file_path)

    # Log dataset statistics with proper error handling
    try:
        # Calculate total samples more robustly
        if hasattr(dataset, 'num_rows'):
            total_samples = dataset.num_rows
        else:
            train_samples = len(dataset['train']) if 'train' in dataset else 0
            test_samples = len(dataset['test']) if 'test' in dataset else 0
            total_samples = train_samples + test_samples
        
        # Extract features for logging
        if 'train' in dataset and len(dataset['train']) > 0:
            sample_data = dataset['train'][0]
            features = list(sample_data.keys()) if hasattr(sample_data, 'keys') else []
        else:
            features = []
        
        # Build data statistics safely
        data_stats = {
            'total_samples': int(total_samples),  # Ensure it's an integer
            'features': features,
            'train_samples': int(len(dataset['train']) if 'train' in dataset else 0),
            'test_samples': int(len(dataset['test']) if 'test' in dataset else 0),
        }
        
        enhanced_logger.log_data_statistics(data_stats)
        
    except Exception as e:
        enhanced_logger.logger.warning("Could not extract detailed dataset statistics: %s", str(e))
        enhanced_logger.logger.info("Dataset loaded successfully from: %s", csv_file_path)

    # Conduct partitioning
    partitioner = instantiate_partitioner(
        partitioner_type=config.federated.partitioner_type, 
        num_partitions=config.federated.pool_size
    )
    fds = dataset
    
    # Apply the partitioner to the train dataset (not the full DatasetDict)
    partitioner.dataset = fds["train"]

    # Load centralised test set based on model type
    test_data_prepared = None
    if config.federated.centralised_eval:
        enhanced_logger.logger.info("Loading centralised test set...")
        test_data = fds["test"]
        test_data.set_format("numpy")
        num_test = test_data.shape[0]
        
        if model_type == "xgboost":
            test_data_prepared = transform_dataset_to_dmatrix(test_data)
        elif model_type == "random_forest":
            # For Random Forest, keep as numpy arrays
            test_data_prepared = test_data
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

    # Load partitions and reformat data appropriately
    enhanced_logger.logger.info("Loading client local partitions...")
    train_data_list = []
    valid_data_list = []

    # Load and process all client partitions. This upfront cost is amortized soon
    # after the simulation begins since clients wont need to preprocess their partition.
    for partition_id in tqdm(range(config.federated.pool_size), desc="Extracting client partition"):
        # Extract partition for client with partition_id
        partition = partitioner.load_partition(partition_id)
        partition.set_format("numpy")

        if config.federated.centralised_eval:
            # Use centralised test set for evaluation
            train_data = partition
            num_train = train_data.shape[0]
            x_test, y_test = separate_xy(test_data)
            valid_data_list.append(((x_test, y_test), num_test))
        else:
            # Train/test splitting
            train_data, valid_data, num_train, num_val = train_test_split(
                partition, test_fraction=config.federated.test_fraction, seed=config.data.seed
            )
            x_valid, y_valid = separate_xy(valid_data)
            valid_data_list.append(((x_valid, y_valid), num_val))

        x_train, y_train = separate_xy(train_data)
        train_data_list.append(((x_train, y_train), num_train))

    # Define strategy based on model type and training method
    enhanced_logger.logger.info("Setting up strategy for model type: %s, training method: %s", model_type, config.federated.train_method)
    
    if model_type == "xgboost":
        # XGBoost strategies
        if config.federated.train_method == "bagging":
            strategy = FedXgbBagging(
                evaluate_function=(
                    get_evaluate_fn(test_data_prepared) if config.federated.centralised_eval else None
                ),
                fraction_fit=(float(config.federated.num_clients_per_round) / config.federated.pool_size),
                min_fit_clients=config.federated.num_clients_per_round,
                min_available_clients=config.federated.pool_size,
                min_evaluate_clients=(
                    config.federated.num_evaluate_clients if not config.federated.centralised_eval else 0
                ),
                fraction_evaluate=1.0 if not config.federated.centralised_eval else 0.0,
                on_evaluate_config_fn=eval_config,
                on_fit_config_fn=fit_config,
                evaluate_metrics_aggregation_fn=(
                    evaluate_metrics_aggregation if not config.federated.centralised_eval else None
                ),
            )
        else:
            # Cyclic training
            strategy = FedXgbCyclic(
                fraction_fit=1.0,
                min_available_clients=config.federated.pool_size,
                fraction_evaluate=1.0,
                evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation,
                on_evaluate_config_fn=eval_config,
                on_fit_config_fn=fit_config,
            )
    
    elif model_type == "random_forest":
        # Random Forest strategies
        if config.federated.train_method == "bagging":
            strategy = RandomForestBagging(
                fraction_fit=(float(config.federated.num_clients_per_round) / config.federated.pool_size),
                min_fit_clients=config.federated.num_clients_per_round,
                min_available_clients=config.federated.pool_size,
                min_evaluate_clients=(
                    config.federated.num_evaluate_clients if not config.federated.centralised_eval else 0
                ),
                fraction_evaluate=1.0 if not config.federated.centralised_eval else 0.0,
                evaluate_fn=get_evaluate_fn(test_data_prepared) if config.federated.centralised_eval else None
            )
        else:
            # FedAvg for Random Forest
            strategy = RandomForestFedAvg(
                fraction_fit=(float(config.federated.num_clients_per_round) / config.federated.pool_size),
                min_fit_clients=config.federated.num_clients_per_round,
                min_available_clients=config.federated.pool_size,
                min_evaluate_clients=(
                    config.federated.num_evaluate_clients if not config.federated.centralised_eval else 0
                ),
                fraction_evaluate=1.0 if not config.federated.centralised_eval else 0.0,
                evaluate_fn=get_evaluate_fn(test_data_prepared) if config.federated.centralised_eval else None
            )
    
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    # Resources to be assigned to each virtual client
    # In this example we use CPU by default
    client_resources = {
        "num_cpus": config.federated.num_cpus_per_client,
        "num_gpus": 0.0,
    }

    # Hyper-parameters for training
    num_local_round = config.model.num_local_rounds
    
    # Get model parameters from ConfigManager
    config_manager = get_config_manager()
    config_manager._config = config  # Set the config in manager
    params = config_manager.get_model_params_dict()

    # Setup learning rate scaling only for XGBoost
    if model_type == "xgboost" and config.federated.train_method == "bagging" and config.federated.scaled_lr:
        new_lr = params["eta"] / config.federated.pool_size
        params.update({"eta": new_lr})
        enhanced_logger.logger.info("Scaled learning rate applied: %f", new_lr)

    enhanced_logger.logger.info("🚀 Starting simulation with %d rounds...", config.federated.num_rounds)
    
    # Initialize metrics history
    metrics_history = []
    
    # Run simulation
    history = fl.simulation.start_simulation(
        client_fn=get_client_fn(
            train_data_list,
            valid_data_list,
            config.federated.train_method,
            params,
            num_local_round,
            config_manager,
            model_type
        ),
        num_clients=config.federated.pool_size,
        client_resources=client_resources,
        config=fl.server.ServerConfig(num_rounds=config.federated.num_rounds),
        strategy=strategy,
        client_manager=CyclicClientManager() if config.federated.train_method == "cyclic" else None,
    )
    
    # Print metrics for each round
    enhanced_logger.logger.info("\n📊 Federated Learning Metrics Summary:")
    enhanced_logger.logger.info("=" * 50)
    enhanced_logger.logger.info("Round | Accuracy | Precision | Recall | F1 Score")
    enhanced_logger.logger.info("-" * 50)
    
    for round_num, metrics in enumerate(history.metrics_distributed, 1):
        if metrics:
            # Get the latest metrics for this round
            latest_metrics = metrics[-1][1]
            accuracy = latest_metrics.get("accuracy", 0.0)
            precision = latest_metrics.get("precision", 0.0)
            recall = latest_metrics.get("recall", 0.0)
            f1 = latest_metrics.get("f1", 0.0)
            
            enhanced_logger.logger.info(
                f"{round_num:5d} | {accuracy:.4f} | {precision:.4f} | {recall:.4f} | {f1:.4f}"
            )
    
    enhanced_logger.logger.info("=" * 50)
    
    enhanced_logger.logger.info("✅ Simulation completed successfully!")

if __name__ == "__main__":
    main()

