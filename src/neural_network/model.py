import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import logging
import numpy as np

logger = logging.getLogger(__name__)

def calculate_optimal_depth(input_size: int, num_classes: int, complexity_factor: float = 2.5) -> int:
    """
    Calculate optimal network depth based on problem complexity.
    
    Args:
        input_size: Number of input features
        num_classes: Number of output classes
        complexity_factor: Multiplier for base complexity calculation
        
    Returns:
        int: Optimal number of hidden layers (minimum 4, maximum 8)
    """
    # Base calculation: log2 of classes + complexity adjustment
    base_layers = int(np.log2(num_classes)) + 2
    
    # Adjust based on input dimensionality (more features = need more depth)
    input_factor = max(1.0, np.log10(input_size / 100))
    
    optimal_depth = int(base_layers * complexity_factor * input_factor)
    
    # Constrain to reasonable bounds for cyber-attack classification
    return max(4, min(8, optimal_depth))


def get_cyber_attack_architecture(input_size: int, num_classes: int = 11, strategy: str = 'adaptive') -> List[int]:
    """
    Get optimal hidden layer sizes for cyber-attack classification.
    
    Args:
        input_size: Number of input features
        num_classes: Number of classes (default 11 for cyber-attacks)
        strategy: Architecture strategy ('adaptive', 'deep', 'bottleneck', 'wide')
        
    Returns:
        List[int]: Hidden layer sizes
    """
    if strategy == 'adaptive':
        # Adaptive architecture based on problem complexity
        depth = calculate_optimal_depth(input_size, num_classes)
        
        # Start wide, gradually narrow towards output
        if depth == 4:
            return [768, 512, 256, 128]
        elif depth == 5:
            return [768, 512, 384, 256, 128]
        elif depth == 6:
            return [768, 512, 384, 256, 128, 64]
        elif depth == 7:
            return [1024, 768, 512, 384, 256, 128, 64]
        else:  # depth >= 8
            return [1024, 768, 512, 384, 256, 192, 128, 64]
    
    elif strategy == 'deep':
        # Deep architecture with 8 layers
        return [1024, 768, 512, 384, 256, 192, 128, 64]
    
    elif strategy == 'bottleneck':
        # Bottleneck pattern: expand-compress-expand-compress
        return [512, 256, 128, 256, 128, 64]
    
    elif strategy == 'wide':
        # Wider layers with moderate depth
        return [1024, 768, 512, 256]
    
    elif strategy == 'balanced':
        # Balanced depth and width for good performance/efficiency trade-off
        return [768, 512, 384, 256, 128, 64]
    
    else:
        # Default to balanced architecture
        return [768, 512, 384, 256, 128, 64]

class NeuralNetwork(nn.Module):
    """
    A flexible neural network model for classification tasks with residual connections and batch normalization.
    """
    def __init__(
        self,
        input_size: int,
        hidden_sizes: Optional[List[int]] = None,
        num_classes: int = 11,
        dropout_rate: float = 0.3,
        activation: str = 'relu',
        architecture_strategy: str = 'balanced'
    ):
        """
        Initialize the neural network.
        
        Args:
            input_size (int): Number of input features
            hidden_sizes (List[int], optional): List of hidden layer sizes. If None, uses optimal architecture
            num_classes (int): Number of output classes
            dropout_rate (float): Dropout rate for regularization
            activation (str): Activation function ('relu', 'leaky_relu', or 'elu')
            architecture_strategy (str): Strategy for automatic architecture selection when hidden_sizes is None
        """
        super().__init__()
        
        # Use optimal architecture if hidden_sizes not provided
        if hidden_sizes is None:
            self.hidden_sizes = get_cyber_attack_architecture(
                input_size, num_classes, architecture_strategy
            )
            logger.info(f"Using optimal architecture strategy '{architecture_strategy}': {self.hidden_sizes}")
        else:
            self.hidden_sizes = hidden_sizes
            logger.info(f"Using provided architecture: {self.hidden_sizes}")
        
        # Store configuration
        self.input_size = input_size
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        
        # Set activation function
        self.activation_fn = self._get_activation_fn(activation)
        
        # Build layers
        self.layers = nn.ModuleList()
        self.batchnorms = nn.ModuleList()
        self.residual_indices = []  # Store pairs of (from, to) for residuals
        
        # Input layer
        self.layers.append(nn.Linear(input_size, self.hidden_sizes[0]))
        self.batchnorms.append(nn.BatchNorm1d(self.hidden_sizes[0]))
        
        # Hidden layers
        for i in range(len(self.hidden_sizes) - 1):
            self.layers.append(nn.Linear(self.hidden_sizes[i], self.hidden_sizes[i + 1]))
            self.batchnorms.append(nn.BatchNorm1d(self.hidden_sizes[i + 1]))
            # Add residual connection if dimensions match (every two layers)
            if i >= 1 and self.hidden_sizes[i - 1] == self.hidden_sizes[i + 1]:
                self.residual_indices.append((i - 1, i + 1))
        
        # Output layer (no batchnorm or activation)
        self.layers.append(nn.Linear(self.hidden_sizes[-1], num_classes))
        
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
        Forward pass through the network with batch normalization, dropout, and residual connections.
        """
        residuals = {}
        for i, layer in enumerate(self.layers[:-1]):
            x_in = x
            x = layer(x)
            x = self.batchnorms[i](x)
            x = self.activation_fn(x)
            x = self.dropout(x)
            # Store for possible residual connection
            residuals[i] = x
            # Apply residual connection if defined for this layer
            for from_idx, to_idx in self.residual_indices:
                if to_idx == i and x.shape == residuals[from_idx].shape:
                    x = x + residuals[from_idx]
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
    
    @classmethod
    def for_cyber_attack_classification(
        cls,
        input_size: int,
        num_classes: int = 11,
        architecture_strategy: str = 'adaptive',
        dropout_rate: float = 0.4,
        activation: str = 'leaky_relu'
    ) -> 'NeuralNetwork':
        """
        Create a neural network optimized for cyber-attack classification.
        
        This method implements the solution for Issue 3, Problem 1: Insufficient Depth.
        It automatically calculates optimal architecture depth and width based on the
        complexity of cyber-attack classification tasks.
        
        Args:
            input_size (int): Number of input features
            num_classes (int): Number of classes (default 11 for cyber-attacks)
            architecture_strategy (str): Architecture optimization strategy:
                - 'adaptive': Automatically calculate optimal depth (recommended)
                - 'deep': Use deep 8-layer architecture for maximum capacity
                - 'balanced': Balanced 6-layer architecture for good performance/efficiency
                - 'bottleneck': Bottleneck pattern for parameter efficiency
                - 'wide': Wider layers with moderate depth
            dropout_rate (float): Dropout rate optimized for cyber-attack data
            activation (str): Activation function optimized for deep networks
        
        Returns:
            NeuralNetwork: Optimized neural network for cyber-attack classification
            
        Example:
            >>> # Create adaptive architecture (recommended)
            >>> model = NeuralNetwork.for_cyber_attack_classification(
            ...     input_size=74, 
            ...     architecture_strategy='adaptive'
            ... )
            >>> 
            >>> # Create deep architecture for maximum performance
            >>> model = NeuralNetwork.for_cyber_attack_classification(
            ...     input_size=74,
            ...     architecture_strategy='deep'
            ... )
        """
        logger.info(f"Creating cyber-attack classification model with strategy: {architecture_strategy}")
        
        model = cls(
            input_size=input_size,
            hidden_sizes=None,  # Will use optimal architecture
            num_classes=num_classes,
            dropout_rate=dropout_rate,
            activation=activation,
            architecture_strategy=architecture_strategy
        )
        
        # Log architecture details
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Created model with {len(model.hidden_sizes)} hidden layers")
        logger.info(f"Architecture: {input_size} -> {' -> '.join(map(str, model.hidden_sizes))} -> {num_classes}")
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Estimated capacity improvement over [256,128,64]: {total_params / 130000:.1f}x")
        
        return model 