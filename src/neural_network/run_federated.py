import os
import flwr as fl
import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import logging
from .model import NeuralNetwork
from .federated_client import NeuralNetworkClient

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_and_preprocess_data(data_path: str, num_clients: int = 3):
    """
    Load and preprocess data for federated learning.
    
    Args:
        data_path: Path to the data file
        num_clients: Number of clients to split data among
        
    Returns:
        tuple: (client_data, test_data)
    """
    # Load data
    data = pd.read_csv(data_path)
    
    # Separate features and labels
    X = data.drop('label', axis=1)
    y = data['label']
    
    # Split into train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Split training data among clients
    client_data = []
    for i in range(num_clients):
        # Create non-IID split by sorting labels and splitting
        indices = np.argsort(y_train)
        client_indices = indices[i::num_clients]
        
        client_X = X_train[client_indices]
        client_y = y_train.iloc[client_indices]
        
        # Split into train and validation
        client_X_train, client_X_val, client_y_train, client_y_val = train_test_split(
            client_X, client_y, test_size=0.2, random_state=42
        )
        
        client_data.append({
            'train': (client_X_train, client_y_train),
            'val': (client_X_val, client_y_val)
        })
    
    test_data = (X_test, y_test)
    
    return client_data, test_data

def main():
    """Run federated learning with neural network."""
    # Load and preprocess data
    data_path = "data/received/final_dataset.csv"
    num_clients = 3
    client_data, test_data = load_and_preprocess_data(data_path, num_clients)
    
    # Create output directory
    os.makedirs("outputs/neural_network", exist_ok=True)
    
    # Initialize clients
    clients = []
    for i, data in enumerate(client_data):
        # Create model
        input_size = client_data[0]['train'][0].shape[1]
        model = NeuralNetwork(
            input_size=input_size,
            hidden_sizes=[256, 128, 64],
            num_classes=11,  # Update based on your number of classes
            dropout_rate=0.3
        )
        
        # Create client
        client = NeuralNetworkClient(
            model=model,
            train_features=data['train'][0],
            train_labels=data['train'][1],
            val_features=data['val'][0],
            val_labels=data['val'][1],
            cid=f"neural_network_client_{i}"
        )
        clients.append(client)
    
    # Define strategy
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,
        min_fit_clients=num_clients,
        min_available_clients=num_clients,
        eval_fn=None,  # We'll evaluate on each client
        initial_parameters=None,
    )
    
    # Start server
    fl.server.start_server(
        server_address="[::]:8080",
        config=fl.server.ServerConfig(num_rounds=10),
        strategy=strategy
    )
    
    # Start clients
    for client in clients:
        fl.client.start_numpy_client(
            server_address="[::]:8080",
            client=client
        )
    
    logger.info("Federated learning completed")

if __name__ == "__main__":
    main() 