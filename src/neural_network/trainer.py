import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm
import logging
from .model import NeuralNetwork
from .losses import FocalLoss, LabelSmoothingCrossEntropy, get_recommended_focal_params, analyze_class_distribution

logger = logging.getLogger(__name__)

class NeuralNetworkTrainer:
    """
    Trainer class for the neural network model.
    """
    def __init__(
        self,
        model: NeuralNetwork,
        learning_rate: float = 0.001,
        weight_decay: float = 1e-5,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        loss_type: str = 'focal',
        focal_gamma: Optional[float] = None,
        focal_alpha: Optional[Union[torch.Tensor, float]] = None,
        auto_loss_params: bool = True,
        label_smoothing: float = 0.0
    ):
        """
        Initialize the trainer with enhanced loss function support.
        
        Args:
            model (NeuralNetwork): The neural network model
            learning_rate (float): Learning rate for optimization
            weight_decay (float): Weight decay for regularization
            device (str): Device to use for training ('cuda' or 'cpu')
            loss_type (str): Type of loss function ('focal', 'crossentropy', 'label_smoothing')
            focal_gamma (float, optional): Gamma parameter for Focal Loss. If None and auto_loss_params=True,
                                         will be determined automatically based on data
            focal_alpha (torch.Tensor or float, optional): Alpha parameter for Focal Loss class weighting
            auto_loss_params (bool): Whether to automatically determine loss parameters from data
            label_smoothing (float): Label smoothing parameter (0.0 = no smoothing)
        """
        self.model = model
        self.device = device
        self.model.to(device)
        self.loss_type = loss_type
        self.focal_gamma = focal_gamma
        self.focal_alpha = focal_alpha
        self.auto_loss_params = auto_loss_params
        self.label_smoothing = label_smoothing
        
        # Initialize optimizer
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Loss function will be initialized in _initialize_loss_function()
        # after we have access to training data
        self.criterion = None
        self._loss_initialized = False
    
    def _initialize_loss_function(self, train_labels: Union[np.ndarray, pd.Series]) -> None:
        """
        Initialize the loss function based on training data characteristics.
        
        Args:
            train_labels: Training labels for analyzing class distribution
        """
        if self._loss_initialized:
            return
            
        logger.info(f"Initializing loss function: {self.loss_type}")
        
        # Convert labels to numpy if needed
        if isinstance(train_labels, pd.Series):
            labels_array = train_labels.values
        else:
            labels_array = train_labels
        
        # Analyze class distribution
        class_analysis = analyze_class_distribution(labels_array)
        logger.info(f"Class distribution analysis: {class_analysis}")
        
        if self.loss_type == 'focal':
            # Get recommended parameters if auto mode is enabled
            if self.auto_loss_params:
                recommendations = get_recommended_focal_params(labels_array)
                gamma = self.focal_gamma if self.focal_gamma is not None else recommendations['gamma']
                alpha = self.focal_alpha if self.focal_alpha is not None else recommendations['alpha']
                
                logger.info(f"Auto-selected Focal Loss parameters: gamma={gamma}, alpha shape={alpha.shape}")
                logger.info(f"Reasoning: {recommendations['reasoning']}")
            else:
                gamma = self.focal_gamma if self.focal_gamma is not None else 2.0
                alpha = self.focal_alpha
                
                logger.info(f"Manual Focal Loss parameters: gamma={gamma}, alpha={alpha}")
            
            self.criterion = FocalLoss(
                alpha=alpha,
                gamma=gamma,
                reduction='mean',
                device=self.device
            )
            
        elif self.loss_type == 'label_smoothing':
            self.criterion = LabelSmoothingCrossEntropy(
                epsilon=self.label_smoothing,
                reduction='mean'
            )
            logger.info(f"Label Smoothing Cross Entropy initialized with epsilon={self.label_smoothing}")
            
        elif self.loss_type == 'crossentropy':
            self.criterion = nn.CrossEntropyLoss()
            logger.info("Standard Cross Entropy Loss initialized")
            
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
        
        self._loss_initialized = True
    
    def get_loss_info(self) -> Dict[str, any]:
        """
        Get information about the current loss function configuration.
        
        Returns:
            Dict containing loss function information
        """
        info = {
            'loss_type': self.loss_type,
            'loss_initialized': self._loss_initialized,
            'criterion_class': self.criterion.__class__.__name__ if self.criterion else None
        }
        
        if self.loss_type == 'focal' and isinstance(self.criterion, FocalLoss):
            info.update({
                'gamma': self.criterion.gamma,
                'alpha': self.criterion.alpha.tolist() if self.criterion.alpha is not None else None,
                'reduction': self.criterion.reduction
            })
        elif self.loss_type == 'label_smoothing' and isinstance(self.criterion, LabelSmoothingCrossEntropy):
            info.update({
                'epsilon': self.criterion.epsilon,
                'reduction': self.criterion.reduction
            })
        
        return info
        
    def prepare_data(
        self,
        features: Union[np.ndarray, pd.DataFrame],
        labels: Optional[Union[np.ndarray, pd.Series]] = None,
        batch_size: int = 32
    ) -> DataLoader:
        """
        Prepare data for training or evaluation.
        
        Args:
            features: Input features
            labels: Target labels (optional)
            batch_size: Batch size for the DataLoader
            
        Returns:
            DataLoader: DataLoader for the dataset
        """
        # Convert features to tensor
        if isinstance(features, pd.DataFrame):
            features = features.values
        features = torch.FloatTensor(features)
        
        # Create dataset
        if labels is not None:
            if isinstance(labels, pd.Series):
                labels = labels.values
            labels = torch.LongTensor(labels)
            dataset = TensorDataset(features, labels)
        else:
            dataset = TensorDataset(features)
            
        return DataLoader(dataset, batch_size=batch_size, shuffle=(labels is not None))
    
    def train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int
    ) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            train_loader: DataLoader for training data
            epoch: Current epoch number
            
        Returns:
            Dict[str, float]: Training metrics
        """
        # Ensure loss function is initialized
        if not self._loss_initialized:
            # For federated learning compatibility, use default CrossEntropy if no labels are provided
            # We'll extract labels from the train_loader for initialization
            all_labels = []
            for batch in train_loader:
                _, labels = batch
                all_labels.extend(labels.cpu().numpy())
            
            self._initialize_loss_function(np.array(all_labels))
            logger.info(f"Loss function auto-initialized in train_epoch with {len(all_labels)} samples")
        
        self.model.train()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch}')
        for batch in progress_bar:
            features, labels = [b.to(self.device) for b in batch]
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(features)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Update progress bar
            progress_bar.set_postfix({'loss': loss.item()})
        
        # Calculate metrics
        metrics = {
            'loss': total_loss / len(train_loader),
            'accuracy': accuracy_score(all_labels, all_preds),
            'precision': precision_score(all_labels, all_preds, average='weighted'),
            'recall': recall_score(all_labels, all_preds, average='weighted'),
            'f1': f1_score(all_labels, all_preds, average='weighted')
        }
        
        return metrics
    
    def evaluate(
        self,
        eval_loader: DataLoader
    ) -> Dict[str, float]:
        """
        Evaluate the model.
        
        Args:
            eval_loader: DataLoader for evaluation data
            
        Returns:
            Dict[str, float]: Evaluation metrics
        """
        # Ensure loss function is initialized
        if not self._loss_initialized:
            # For federated learning compatibility, extract labels from eval_loader for initialization
            all_labels = []
            for batch in eval_loader:
                _, labels = batch
                all_labels.extend(labels.cpu().numpy())
            
            self._initialize_loss_function(np.array(all_labels))
            logger.info(f"Loss function auto-initialized in evaluate with {len(all_labels)} samples")
        
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in eval_loader:
                features, labels = [b.to(self.device) for b in batch]
                
                # Forward pass
                outputs = self.model(features)
                loss = self.criterion(outputs, labels)
                
                # Update metrics
                total_loss += loss.item()
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        metrics = {
            'loss': total_loss / len(eval_loader),
            'accuracy': accuracy_score(all_labels, all_preds),
            'precision': precision_score(all_labels, all_preds, average='weighted'),
            'recall': recall_score(all_labels, all_preds, average='weighted'),
            'f1': f1_score(all_labels, all_preds, average='weighted')
        }
        
        return metrics
    
    def train(
        self,
        train_features: Union[np.ndarray, pd.DataFrame],
        train_labels: Union[np.ndarray, pd.Series],
        val_features: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        val_labels: Optional[Union[np.ndarray, pd.Series]] = None,
        num_epochs: int = 100,
        batch_size: int = 32,
        early_stopping_patience: int = 10
    ) -> Dict[str, List[float]]:
        """
        Train the model with enhanced loss function support.
        
        Args:
            train_features: Training features
            train_labels: Training labels
            val_features: Validation features (optional)
            val_labels: Validation labels (optional)
            num_epochs: Number of epochs to train
            batch_size: Batch size
            early_stopping_patience: Number of epochs to wait for improvement
            
        Returns:
            Dict[str, List[float]]: Training history
        """
        # Initialize loss function based on training data
        self._initialize_loss_function(train_labels)
        
        # Prepare data
        train_loader = self.prepare_data(train_features, train_labels, batch_size)
        val_loader = None
        if val_features is not None and val_labels is not None:
            val_loader = self.prepare_data(val_features, val_labels, batch_size)
        
        # Initialize history
        history = {
            'train_loss': [],
            'train_accuracy': [],
            'val_loss': [],
            'val_accuracy': []
        }
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(num_epochs):
            # Train
            train_metrics = self.train_epoch(train_loader, epoch)
            history['train_loss'].append(train_metrics['loss'])
            history['train_accuracy'].append(train_metrics['accuracy'])
            
            # Validate
            if val_loader is not None:
                val_metrics = self.evaluate(val_loader)
                history['val_loss'].append(val_metrics['loss'])
                history['val_accuracy'].append(val_metrics['accuracy'])
                
                # Early stopping
                if val_metrics['loss'] < best_val_loss:
                    best_val_loss = val_metrics['loss']
                    patience_counter = 0
                else:
                    patience_counter += 1
                    
                if patience_counter >= early_stopping_patience:
                    logger.info(f'Early stopping at epoch {epoch}')
                    break
            
            # Log progress
            logger.info(
                f'Epoch {epoch}: '
                f'Train Loss: {train_metrics["loss"]:.4f}, '
                f'Train Acc: {train_metrics["accuracy"]:.4f}'
                + (f', Val Loss: {val_metrics["loss"]:.4f}, Val Acc: {val_metrics["accuracy"]:.4f}'
                   if val_loader is not None else '')
            )
        
        return history
    
    def predict(
        self,
        features: Union[np.ndarray, pd.DataFrame],
        batch_size: int = 32
    ) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            features: Input features
            batch_size: Batch size
            
        Returns:
            np.ndarray: Predicted labels
        """
        self.model.eval()
        test_loader = self.prepare_data(features, batch_size=batch_size)
        all_preds = []
        
        with torch.no_grad():
            for batch in test_loader:
                features = batch[0].to(self.device)
                outputs = self.model(features)
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
        
        return np.array(all_preds) 