"""
Generic federated learning client implementation.

This module implements a unified Flower client that can work with different model types
(XGBoost, Random Forest) based on configuration, providing a single interface for
federated learning regardless of the underlying model.
"""

import os
import logging
import numpy as np
from typing import Dict, Tuple, Optional, Any
import flwr as fl
from flwr.common import NDArrays

from src.core.dataset import (
    load_csv_data,
    transform_dataset_to_dmatrix,
    load_global_feature_processor,
    instantiate_partitioner,
)
from src.config.config_manager import ConfigManager
from src.models.model_factory import ModelFactory
from src.federated.client_utils import XgbClient

logger = logging.getLogger(__name__)


class GenericFederatedClient:
    """
    Generic federated learning client that supports multiple model types.
    
    This client automatically adapts to the configured model type (XGBoost, Random Forest)
    and provides a unified interface for federated learning operations.
    """
    
    def __init__(self, 
                 client_id: int,
                 config_manager: ConfigManager,
                 global_processor_path: Optional[str] = None):
        """
        Initialize the generic federated client.
        
        Args:
            client_id: Unique identifier for this client
            config_manager: ConfigManager instance with loaded configuration
            global_processor_path: Path to the global feature processor
        """
        self.client_id = client_id
        self.config_manager = config_manager
        self.config = config_manager.config
        self.model_type = config_manager.get_model_type()
        self.global_processor_path = global_processor_path
        
        # Initialize data containers
        self.train_data = None
        self.test_data = None
        self.processor = None
        self.model = None
        
        logger.info("Initialized GenericFederatedClient for client %d with model type: %s", 
                   client_id, self.model_type)
    
    def load_data(self) -> None:
        """Load and partition data for this client."""
        logger.info("Loading data for client %d", self.client_id)
        
        # Get data file path from config
        data_file = os.path.join(self.config.data.path, self.config.data.filename)
        
        if not os.path.exists(data_file):
            raise FileNotFoundError(f"Data file not found: {data_file}")
        
        # Load global feature processor if available
        processor = None
        if self.global_processor_path and os.path.exists(self.global_processor_path):
            logger.info("Loading global feature processor from %s", self.global_processor_path)
            processor = load_global_feature_processor(self.global_processor_path)
        else:
            logger.warning("Global processor not found at %s", self.global_processor_path)
        
        self.processor = processor
        
        if self.model_type == "xgboost":
            self._load_xgboost_data(data_file)
        elif self.model_type == "random_forest":
            self._load_random_forest_data(data_file)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
    
    def _load_xgboost_data(self, data_file: str) -> None:
        """Load data for XGBoost training."""
        # Load the dataset
        dataset = load_csv_data(data_file)
        
        # Create partitioner
        partitioner = instantiate_partitioner(
            self.config.federated.partitioner_type, 
            self.config.federated.num_partitions
        )
        
        # Apply the partitioner to the train dataset
        partitioner.dataset = dataset["train"]
        
        # Get the partition for this client
        client_dataset = partitioner.load_partition(self.client_id)
        
        # Convert to DMatrix for training
        self.train_data = transform_dataset_to_dmatrix(
            client_dataset, processor=self.processor, is_training=True
        )
        
        # Use the test split from the main dataset for evaluation
        test_dataset = dataset["test"]
        self.test_data = transform_dataset_to_dmatrix(
            test_dataset, processor=self.processor, is_training=False
        )
        
        logger.info("Client %d - XGBoost data loaded: Train=%d, Test=%d, Features=%d", 
                   self.client_id, self.train_data.num_row(), 
                   self.test_data.num_row(), self.train_data.num_col())
    
    def _load_random_forest_data(self, data_file: str) -> None:
        """Load data for Random Forest training."""
        # Load and preprocess data using the existing pipeline
        # First load the CSV data
        dataset_dict = load_csv_data(data_file)
        
        # Convert to pandas for sklearn processing
        full_dataset = dataset_dict["train"].to_pandas()
        
        # Simple train/test split using sklearn
        from sklearn.model_selection import train_test_split
        target_col = 'label' if 'label' in full_dataset.columns else 'attack_cat'
        feature_cols = [col for col in full_dataset.columns if col != target_col]
        
        X = full_dataset[feature_cols]
        y = full_dataset[target_col]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=self.config.federated.test_fraction,
            random_state=self.config.data.seed,
            stratify=y if len(np.unique(y)) > 1 else None
        )
        
        # Create partitioner for train data
        # Simple partitioning for Random Forest (could be improved)
        partition_size = len(X_train) // self.config.federated.num_partitions
        start_idx = self.client_id * partition_size
        if self.client_id == self.config.federated.num_partitions - 1:
            # Last client gets remaining data
            end_idx = len(X_train)
        else:
            end_idx = start_idx + partition_size
        
        self.train_data = {
            'X': X_train.iloc[start_idx:end_idx].values,
            'y': y_train.iloc[start_idx:end_idx].values
        }
        
        # Use common test data for all clients
        self.test_data = {
            'X': X_test.values,
            'y': y_test.values
        }
        
        logger.info("Client %d - Random Forest data loaded: Train=%d, Test=%d, Features=%d", 
                   self.client_id, len(self.train_data['X']), 
                   len(self.test_data['X']), self.train_data['X'].shape[1])
    
    def create_model(self) -> None:
        """Create the appropriate model based on configuration."""
        logger.info("Creating %s model for client %d", self.model_type, self.client_id)
        
        # Get model parameters from config
        model_params = self.config_manager.get_model_params_dict()
        
        # Create model using the factory
        self.model = ModelFactory.create_model(
            model_type=self.model_type,
            params=model_params
        )
        
        logger.info("Model created successfully for client %d", self.client_id)
    
    def get_flower_client(self) -> fl.client.Client:
        """
        Get the appropriate Flower client based on model type.
        
        Returns:
            fl.client.Client: Flower client instance
        """
        if self.model_type == "xgboost":
            return self._get_xgboost_flower_client()
        if self.model_type == "random_forest":
            return self._get_random_forest_flower_client()
        raise ValueError(f"Unsupported model type: {self.model_type}")
    
    def _get_xgboost_flower_client(self) -> fl.client.Client:
        """Get XGBoost Flower client."""
        # Use the existing XgbClient
        return XgbClient(
            train_dmatrix=self.train_data,
            valid_dmatrix=self.test_data,
            num_train=self.train_data.num_row(),
            num_val=self.test_data.num_row(),
            num_local_round=self.config.model.num_local_rounds,
            cid=self.client_id,
            params=self.config_manager.get_model_params_dict(),
            train_method=self.config.federated.train_method,
            config_manager=self.config_manager
        )
    
    def _get_random_forest_flower_client(self) -> fl.client.Client:
        """Get Random Forest Flower client."""
        return RandomForestFlowerClient(
            train_data=self.train_data,
            test_data=self.test_data,
            client_id=self.client_id,
            model_params=self.config_manager.get_model_params_dict(),
            num_local_rounds=self.config.model.num_local_rounds
        )


class RandomForestFlowerClient(fl.client.NumPyClient):
    """Flower client implementation for Random Forest models."""
    
    def __init__(self, 
                 train_data: Dict[str, np.ndarray],
                 test_data: Dict[str, np.ndarray],
                 client_id: int,
                 model_params: Dict[str, Any],
                 num_local_rounds: int = 1):
        """
        Initialize Random Forest Flower client.
        
        Args:
            train_data: Dictionary with 'X' and 'y' training data
            test_data: Dictionary with 'X' and 'y' test data  
            client_id: Client identifier
            model_params: Random Forest parameters
            num_local_rounds: Number of local training rounds
        """
        self.train_data = train_data
        self.test_data = test_data
        self.client_id = client_id
        self.model_params = model_params
        self.num_local_rounds = num_local_rounds
        self.model = None
        
        logger.info("Initialized RandomForestFlowerClient for client %d", client_id)
    
    def get_parameters(self, config: Dict[str, Any]) -> NDArrays:
        """Return model parameters as a list of NumPy ndarrays."""
        if self.model is None:
            # Return empty parameters if model not trained yet
            return []
        
        # For Random Forest, we can return tree parameters or feature importances
        # This is a simplified implementation - in practice, you might want to
        # serialize the entire model or use other aggregation strategies
        try:
            feature_importances = self.model.feature_importances_
            return [feature_importances]
        except AttributeError:
            return []
    
    def set_parameters(self, parameters: NDArrays) -> None:
        """Update model parameters from a list of NumPy ndarrays."""
        # For Random Forest, this is more complex since we can't directly
        # set parameters like in XGBoost. In practice, you might use
        # model averaging or other ensemble techniques.
        # This is a placeholder implementation.
        if parameters and len(parameters) > 0:
            logger.info("Received parameters for client %d", self.client_id)
    
    def fit(self, parameters: NDArrays, config: Dict[str, Any]) -> Tuple[NDArrays, int, Dict[str, Any]]:
        """Train the model on the locally held training set."""
        logger.info("Training Random Forest model for client %d", self.client_id)
        
        # Set parameters if provided
        self.set_parameters(parameters)
        
        # Create and train Random Forest model
        from sklearn.ensemble import RandomForestClassifier
        
        self.model = RandomForestClassifier(**self.model_params)
        
        # Train the model
        self.model.fit(self.train_data['X'], self.train_data['y'])
        
        # Return updated parameters
        new_parameters = self.get_parameters({})
        
        # Calculate training metrics
        train_accuracy = self.model.score(self.train_data['X'], self.train_data['y'])
        
        metrics = {
            "train_accuracy": train_accuracy,
            "train_samples": len(self.train_data['X'])
        }
        
        logger.info("Client %d training completed. Accuracy: %.4f", 
                   self.client_id, train_accuracy)
        
        return new_parameters, len(self.train_data['X']), metrics
    
    def evaluate(self, parameters: NDArrays, config: Dict[str, Any]) -> Tuple[float, int, Dict[str, Any]]:
        """Evaluate the model on the locally held test set."""
        logger.info("Evaluating Random Forest model for client %d", self.client_id)
        
        if self.model is None:
            logger.warning("Model not trained yet for client %d", self.client_id)
            return 0.0, 0, {}
        
        # Set parameters if provided
        self.set_parameters(parameters)
        
        # Evaluate on test data
        test_predictions = self.model.predict(self.test_data['X'])
        test_accuracy = self.model.score(self.test_data['X'], self.test_data['y'])
        
        # Calculate additional metrics
        from sklearn.metrics import f1_score, precision_score, recall_score
        
        try:
            f1 = f1_score(self.test_data['y'], test_predictions, average='weighted')
            precision = precision_score(self.test_data['y'], test_predictions, average='weighted')
            recall = recall_score(self.test_data['y'], test_predictions, average='weighted')
        except ValueError as e:
            logger.warning("Could not calculate additional metrics: %s", e)
            f1 = precision = recall = 0.0
        
        metrics = {
            "accuracy": test_accuracy,
            "f1_score": f1,
            "precision": precision,
            "recall": recall,
            "test_samples": len(self.test_data['X'])
        }
        
        logger.info("Client %d evaluation completed. Accuracy: %.4f, F1: %.4f", 
                   self.client_id, test_accuracy, f1)
        
        # Return loss (1 - accuracy), number of samples, and metrics
        loss = 1.0 - test_accuracy
        return loss, len(self.test_data['X']), metrics


def create_generic_client(client_id: int, 
                         config_manager: ConfigManager,
                         global_processor_path: Optional[str] = None) -> fl.client.Client:
    """
    Factory function to create a generic federated client.
    
    Args:
        client_id: Unique identifier for the client
        config_manager: ConfigManager instance with loaded configuration
        global_processor_path: Path to the global feature processor
        
    Returns:
        fl.client.Client: Flower client instance
    """
    # Create generic client
    generic_client = GenericFederatedClient(
        client_id=client_id,
        config_manager=config_manager,
        global_processor_path=global_processor_path
    )
    
    # Load data and create model
    generic_client.load_data()
    generic_client.create_model()
    
    # Return the appropriate Flower client
    return generic_client.get_flower_client()


def start_generic_client(client_id: int,
                        config_manager: ConfigManager,
                        global_processor_path: Optional[str] = None,
                        server_address: str = "0.0.0.0:8080",
                        use_https: bool = False) -> None:
    """
    Start a generic federated learning client.
    
    Args:
        client_id: Unique identifier for the client
        config_manager: ConfigManager instance with loaded configuration
        global_processor_path: Path to the global feature processor
        server_address: Server address to connect to
        use_https: Whether to use HTTPS for server communication
    """
    logger.info("Starting generic federated client %d for model type: %s", 
               client_id, config_manager.get_model_type())
    
    # Create the client
    client = create_generic_client(
        client_id=client_id,
        config_manager=config_manager,
        global_processor_path=global_processor_path
    )
    
    # Connect to server
    try:
        if use_https:
            fl.client.start_client(
                server_address=server_address,
                client=client,
                transport="grpc-bidi"
            )
        else:
            fl.client.start_client(
                server_address=server_address,
                client=client
            )
    except Exception as e:
        logger.error("Failed to start client %d: %s", client_id, e)
        raise 