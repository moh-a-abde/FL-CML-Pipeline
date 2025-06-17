import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import logging
from src.neural_network.model import NeuralNetwork
from src.neural_network.trainer import NeuralNetworkTrainer

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_and_preprocess_data(data_path: str):
    """
    Load and preprocess the data.
    
    Args:
        data_path: Path to the data file
        
    Returns:
        tuple: (X_train, X_val, y_train, y_val, feature_names)
    """
    # Load data
    data = pd.read_csv(data_path)
    
    # Separate features and labels
    X = data.drop('label', axis=1)
    y = data['label']
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    
    return X_train, X_val, y_train, y_val, X.columns.tolist()

def main():
    """Main function to demonstrate neural network usage."""
    # Load and preprocess data
    data_path = "data/received/final_dataset.csv"  # Update with your data path
    X_train, X_val, y_train, y_val, feature_names = load_and_preprocess_data(data_path)
    
    # Create model
    input_size = len(feature_names)
    model = NeuralNetwork(
        input_size=input_size,
        hidden_sizes=[256, 128, 64],
        num_classes=11,  # Update based on your number of classes
        dropout_rate=0.3
    )
    
    # Create trainer
    trainer = NeuralNetworkTrainer(
        model=model,
        learning_rate=0.001,
        weight_decay=1e-5
    )
    
    # Train model
    trainer.train(
        train_features=X_train,
        train_labels=y_train,
        val_features=X_val,
        val_labels=y_val,
        num_epochs=100,
        batch_size=32,
        early_stopping_patience=10
    )
    
    # Evaluate final model
    final_metrics = trainer.evaluate(
        trainer.prepare_data(X_val, y_val)
    )
    
    logger.info("Final evaluation metrics:")
    for metric, value in final_metrics.items():
        logger.info("%s: %.4f", metric, value)
    
    # Save model
    model.save("models/neural_network_model.pt")
    logger.info("Model saved to models/neural_network_model.pt")

if __name__ == "__main__":
    main() 