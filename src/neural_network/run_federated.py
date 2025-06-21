import os
import flwr as fl
import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
import logging
from .model import NeuralNetwork
from .federated_client import NeuralNetworkClient
from .evaluation import save_evaluation_results, plot_training_history
import multiprocessing
import time

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Configure flwr logger to not propagate to root logger
fl_logger = logging.getLogger('flwr')
fl_logger.propagate = False

# Set up our logger
logger = logging.getLogger(__name__)

def create_stratified_client_data(X_train, y_train, num_clients=3, random_state=42):
    """
    Create stratified client data ensuring balanced class distribution.
    Each client gets representative samples from all classes.
    
    Args:
        X_train: Training features (numpy array or pandas DataFrame)
        y_train: Training labels (numpy array or pandas Series)  
        num_clients: Number of clients to create data for
        random_state: Random seed for reproducibility
        
    Returns:
        List of client data dictionaries with 'train' and 'val' keys
    """
    logger.info(f"Creating stratified data for {num_clients} clients...")
    
    # Convert to numpy arrays if needed
    if hasattr(X_train, 'values'):
        X_train_np = X_train.values if hasattr(X_train, 'values') else X_train
    else:
        X_train_np = X_train
        
    if hasattr(y_train, 'values'):
        y_train_np = y_train.values if hasattr(y_train, 'values') else y_train
    else:
        y_train_np = y_train
    
    # Log initial class distribution
    unique_classes, class_counts = np.unique(y_train_np, return_counts=True)
    logger.info(f"Original class distribution: {dict(zip(unique_classes, class_counts))}")
    
    client_data = []
    remaining_X, remaining_y = X_train_np.copy(), y_train_np.copy()
    
    # Calculate client sizes (slightly different sizes for realism)
    total_samples = len(X_train_np)
    base_size = total_samples // num_clients
    client_sizes = [base_size] * (num_clients - 1) + [total_samples - base_size * (num_clients - 1)]
    
    logger.info(f"Client sizes: {client_sizes}")
    
    for i, client_size in enumerate(client_sizes[:-1]):
        test_size = client_size / len(remaining_X)
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state + i)
        
        for train_idx, client_idx in splitter.split(remaining_X, remaining_y):
            client_X = remaining_X[client_idx]
            client_y = remaining_y[client_idx]
            
            # Update remaining data
            remaining_X = remaining_X[train_idx]
            remaining_y = remaining_y[train_idx]
            
            # Split client data into train and validation
            client_X_train, client_X_val, client_y_train, client_y_val = train_test_split(
                client_X, client_y, test_size=0.2, random_state=random_state + i, stratify=client_y
            )
            
            client_data.append({
                'train': (client_X_train, client_y_train),
                'val': (client_X_val, client_y_val)
            })
            
            # Log client's class distribution
            unique_client, counts_client = np.unique(client_y, return_counts=True)
            logger.info(f"Client {i} class distribution: {dict(zip(unique_client, counts_client))}")
            break
    
    # Last client gets remaining data
    client_X_train, client_X_val, client_y_train, client_y_val = train_test_split(
        remaining_X, remaining_y, test_size=0.2, random_state=random_state + num_clients - 1, stratify=remaining_y
    )
    
    client_data.append({
        'train': (client_X_train, client_y_train),
        'val': (client_X_val, client_y_val)
    })
    
    # Log last client's class distribution
    unique_client, counts_client = np.unique(remaining_y, return_counts=True)
    logger.info(f"Client {num_clients-1} class distribution: {dict(zip(unique_client, counts_client))}")
    
    # Verify total data distribution
    total_train_samples = sum(len(client['train'][1]) for client in client_data)
    total_val_samples = sum(len(client['val'][1]) for client in client_data)
    logger.info(f"Total training samples distributed: {total_train_samples}")
    logger.info(f"Total validation samples distributed: {total_val_samples}")
    logger.info(f"Original total samples: {len(y_train_np)}")
    
    return client_data

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
    
    # Split training data among clients using stratified distribution
    client_data = create_stratified_client_data(X_train, y_train, num_clients)
    
    test_data = (X_test, y_test)
    
    return client_data, test_data

def start_client(client):
    """Start a single client."""
    fl.client.start_client(
        server_address="[::]:8080",
        client=client.to_client()
    )

def start_server(strategy):
    """Start the Flower server."""
    fl.server.start_server(
        server_address="[::]:8080",
        config=fl.server.ServerConfig(num_rounds=200),
        strategy=strategy
    )

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
        initial_parameters=None,
    )
    
    # Start server in a separate process
    server_process = multiprocessing.Process(
        target=start_server,
        args=(strategy,)
    )
    server_process.start()
    
    # Give the server time to start
    time.sleep(5)
    
    # Start clients in separate processes
    client_processes = []
    for client in clients:
        process = multiprocessing.Process(target=start_client, args=(client,))
        process.start()
        client_processes.append(process)
    
    # Wait for all processes to complete
    server_process.join()
    for process in client_processes:
        process.join()
    
    logger.info("Federated learning completed")
    
    # Evaluate final model on test data
    X_test, y_test = test_data
    final_model = clients[0].model  # Use the first client's model as the final model
    final_model.eval()
    
    # Make predictions
    with torch.no_grad():
        X_test_tensor = torch.FloatTensor(X_test)
        outputs = final_model(X_test_tensor)
        predictions = torch.argmax(outputs, dim=1).numpy()
    
    # Get evaluation metrics
    eval_loader = clients[0].trainer.prepare_data(X_test, y_test)
    metrics = clients[0].trainer.evaluate(eval_loader)
    
    # Save evaluation results and create visualizations
    save_evaluation_results(
        metrics=metrics,
        predictions=predictions,
        true_labels=y_test,
        output_dir="outputs/neural_network"
    )
    
    # Save the final model
    final_model.save("outputs/neural_network/final_model.pt")
    logger.info("Final model saved to outputs/neural_network/final_model.pt")

if __name__ == "__main__":
    # Set the start method for multiprocessing
    multiprocessing.set_start_method('spawn')
    main() 