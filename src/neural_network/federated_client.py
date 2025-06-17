import torch
import flwr as fl
from flwr.common import (
    Code,
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    GetParametersIns,
    GetParametersRes,
    Parameters,
    Status,
)
from flwr.common.logger import log
import numpy as np
from typing import Dict, List, Optional, Tuple
from .model import NeuralNetwork
from .trainer import NeuralNetworkTrainer

class NeuralNetworkClient(fl.client.Client):
    """
    Flower client for neural network federated learning.
    """
    def __init__(
        self,
        model: NeuralNetwork,
        train_features: np.ndarray,
        train_labels: np.ndarray,
        val_features: Optional[np.ndarray] = None,
        val_labels: Optional[np.ndarray] = None,
        cid: str = "neural_network_client",
        learning_rate: float = 0.001,
        weight_decay: float = 1e-5,
        batch_size: int = 32
    ):
        """
        Initialize the neural network client.
        
        Args:
            model: Neural network model
            train_features: Training features
            train_labels: Training labels
            val_features: Validation features (optional)
            val_labels: Validation labels (optional)
            cid: Client ID
            learning_rate: Learning rate for optimization
            weight_decay: Weight decay for regularization
            batch_size: Batch size for training
        """
        self.model = model
        self.train_features = train_features
        self.train_labels = train_labels
        self.val_features = val_features
        self.val_labels = val_labels
        self.cid = cid
        self.batch_size = batch_size
        
        # Initialize trainer
        self.trainer = NeuralNetworkTrainer(
            model=model,
            learning_rate=learning_rate,
            weight_decay=weight_decay
        )
        
    def get_parameters(self, ins: GetParametersIns) -> GetParametersRes:
        """
        Return the current model parameters.
        
        Args:
            ins: GetParametersIns instance
            
        Returns:
            GetParametersRes: Model parameters
        """
        parameters = self.model.get_parameters()
        tensors = []
        shapes = []
        for param in parameters.values():
            tensors.append(param.tobytes())
            shapes.append(param.shape)
        
        return GetParametersRes(
            status=Status(code=Code.OK, message="OK"),
            parameters=Parameters(
                tensor_type="",
                tensors=tensors,
                tensor_shapes=shapes  # Add shapes to the parameters
            )
        )
    
    def fit(self, ins: FitIns) -> FitRes:
        """
        Train the model using the provided parameters.
        
        Args:
            ins: FitIns instance containing parameters
            
        Returns:
            FitRes: Training results
        """
        # Update model parameters
        parameters = ins.parameters
        if parameters.tensors:
            # Convert parameters to numpy arrays with correct shapes
            param_dict = {}
            for i, (name, _) in enumerate(self.model.named_parameters()):
                param_bytes = parameters.tensors[i]
                param_shape = parameters.tensor_shapes[i]  # Get the shape
                param_array = np.frombuffer(param_bytes, dtype=np.float32).reshape(param_shape)
                param_dict[name] = param_array
            
            # Update model parameters
            self.model.set_parameters(param_dict)
        
        # Train for one round
        metrics = self.trainer.train_epoch(
            self.trainer.prepare_data(self.train_features, self.train_labels, self.batch_size),
            epoch=0  # We don't track epochs in federated learning
        )
        
        # Get updated parameters
        parameters = self.get_parameters(GetParametersIns({}))
        
        return FitRes(
            status=Status(code=Code.OK, message="OK"),
            parameters=parameters.parameters,
            num_examples=len(self.train_labels),
            metrics=metrics
        )
    
    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        """
        Evaluate the model using the provided parameters.
        
        Args:
            ins: EvaluateIns instance containing parameters
            
        Returns:
            EvaluateRes: Evaluation results
        """
        # Update model parameters
        parameters = ins.parameters
        if parameters.tensors:
            # Convert parameters to tensors
            param_tensors = [torch.from_numpy(np.frombuffer(t, dtype=np.float32)) 
                           for t in parameters.tensors]
            
            # Update model parameters
            self.model.set_parameters(dict(zip(self.model.state_dict().keys(), param_tensors)))
        
        # Evaluate model
        if self.val_features is not None and self.val_labels is not None:
            metrics = self.trainer.evaluate(
                self.trainer.prepare_data(self.val_features, self.val_labels, self.batch_size)
            )
            num_examples = len(self.val_labels)
        else:
            metrics = self.trainer.evaluate(
                self.trainer.prepare_data(self.train_features, self.train_labels, self.batch_size)
            )
            num_examples = len(self.train_labels)
        
        return EvaluateRes(
            status=Status(code=Code.OK, message="OK"),
            loss=metrics["loss"],
            num_examples=num_examples,
            metrics=metrics
        ) 