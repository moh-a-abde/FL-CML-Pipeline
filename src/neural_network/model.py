import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import logging
import numpy as np

logger = logging.getLogger(__name__)

class NeuralNetwork(nn.Module):
    """
    A flexible neural network model for classification tasks.
    """
    def __init__(
        self,
        input_size: int,
        hidden_sizes: List[int] = [256, 128, 64],
        num_classes: int = 11,
        dropout_rate: float = 0.3,
        activation: str = 'relu'
    ):
        """
        Initialize the neural network.
        
        Args:
            input_size (int): Number of input features
            hidden_sizes (List[int]): List of hidden layer sizes
            num_classes (int): Number of output classes
            dropout_rate (float): Dropout rate for regularization
            activation (str): Activation function ('relu', 'leaky_relu', or 'elu')
        """
        super().__init__()
        
        # Store configuration
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        
        # Set activation function
        self.activation_fn = self._get_activation_fn(activation)
        
        # Build layers
        self.layers = nn.ModuleList()
        
        # Input layer
        self.layers.append(nn.Linear(input_size, hidden_sizes[0]))
        
        # Hidden layers
        for i in range(len(hidden_sizes) - 1):
            self.layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
        
        # Output layer
        self.layers.append(nn.Linear(hidden_sizes[-1], num_classes))
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout_rate)
        
        # Initialize weights
        self._initialize_weights()
        
    def _get_activation_fn(self, activation: str) -> nn.Module:
        """Get the specified activation function."""
        if activation == 'relu':
            return nn.ReLU()
        elif activation == 'leaky_relu':
            return nn.LeakyReLU(0.1)
        elif activation == 'elu':
            return nn.ELU()
        else:
            logger.warning(f"Unknown activation function {activation}, using ReLU")
            return nn.ReLU()
    
    def _initialize_weights(self):
        """Initialize weights using Kaiming initialization."""
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_size)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_classes)
        """
        for i, layer in enumerate(self.layers[:-1]):
            x = self.activation_fn(layer(x))
            x = self.dropout(x)
        
        # Output layer (no activation, will be handled by loss function)
        x = self.layers[-1](x)
        return x
    
    def get_parameters(self) -> Dict[str, np.ndarray]:
        """Get model parameters."""
        return {name: param.data.clone().cpu().numpy() for name, param in self.named_parameters()}
    
    def set_parameters(self, parameters: Dict[str, np.ndarray]):
        """Set model parameters."""
        for name, param in self.named_parameters():
            if name in parameters:
                # Make the numpy array writable and ensure correct shape
                param_data = np.array(parameters[name], copy=True)
                if param_data.shape != param.data.shape:
                    raise ValueError(f"Parameter shape mismatch for {name}: expected {param.data.shape}, got {param_data.shape}")
                param.data.copy_(torch.from_numpy(param_data))
    
    def save(self, path: str):
        """Save model to disk."""
        torch.save({
            'model_state_dict': self.state_dict(),
            'input_size': self.input_size,
            'hidden_sizes': self.hidden_sizes,
            'num_classes': self.num_classes,
            'dropout_rate': self.dropout_rate
        }, path)
    
    @classmethod
    def load(cls, path: str) -> 'NeuralNetwork':
        """Load model from disk."""
        checkpoint = torch.load(path)
        model = cls(
            input_size=checkpoint['input_size'],
            hidden_sizes=checkpoint['hidden_sizes'],
            num_classes=checkpoint['num_classes'],
            dropout_rate=checkpoint['dropout_rate']
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        return model 