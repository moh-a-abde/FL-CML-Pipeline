import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union
import numpy as np
from sklearn.utils.class_weight import compute_class_weight


class FocalLoss(nn.Module):
    """
    Focal Loss implementation for addressing hard sample learning.
    Paper: "Focal Loss for Dense Object Detection" (Lin et al., 2017)
    
    This loss is particularly effective for:
    - Handling class imbalance (via alpha parameter)
    - Focusing on hard-to-classify samples (via gamma parameter)
    - Reducing the impact of easy samples during training
    """
    
    def __init__(
        self,
        alpha: Optional[Union[torch.Tensor, float]] = None,
        gamma: float = 2.0,
        reduction: str = 'mean',
        device: Optional[str] = None
    ):
        """
        Initialize Focal Loss.
        
        Args:
            alpha: Class weights tensor or single weight for all classes.
                  If None, no class weighting is applied.
            gamma: Focusing parameter. Higher values focus more on hard samples.
                  gamma=0 reduces to standard CrossEntropy loss.
            reduction: Specifies the reduction to apply to the output:
                      'none' | 'mean' | 'sum'
            device: Device to place tensors on
        """
        super(FocalLoss, self).__init__()
        
        self.gamma = gamma
        self.reduction = reduction
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Handle alpha (class weights)
        if alpha is not None:
            if isinstance(alpha, (float, int)):
                # Single weight for all classes
                self.alpha = torch.tensor([alpha], device=self.device)
            elif isinstance(alpha, (list, np.ndarray)):
                # Different weights for each class
                self.alpha = torch.tensor(alpha, dtype=torch.float32, device=self.device)
            elif isinstance(alpha, torch.Tensor):
                self.alpha = alpha.to(self.device)
            else:
                raise TypeError(f"alpha must be float, list, np.ndarray, or torch.Tensor, got {type(alpha)}")
        else:
            self.alpha = None
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of Focal Loss.
        
        Args:
            inputs: Predictions tensor with shape (N, C) where N is batch size
                   and C is number of classes
            targets: Ground truth labels with shape (N,)
            
        Returns:
            torch.Tensor: Computed focal loss
        """
        # Compute standard cross-entropy loss
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        
        # Calculate pt (probability of true class)
        pt = torch.exp(-ce_loss)
        
        # Apply focal term: (1-pt)^gamma
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        # Apply class weights if provided
        if self.alpha is not None:
            # Ensure alpha is on the same device as inputs
            if self.alpha.device != inputs.device:
                self.alpha = self.alpha.to(inputs.device)
            
            # Handle different alpha shapes
            if self.alpha.size(0) == 1:
                # Single weight for all classes
                alpha_t = self.alpha.expand(targets.size(0))
            else:
                # Different weights for each class
                alpha_t = self.alpha.gather(0, targets.data.view(-1))
            
            focal_loss = alpha_t * focal_loss
        
        # Apply reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:  # 'none'
            return focal_loss


class LabelSmoothingCrossEntropy(nn.Module):
    """
    Label Smoothing Cross Entropy Loss.
    
    Label smoothing prevents the model from becoming overconfident by
    distributing some probability mass to incorrect classes.
    """
    
    def __init__(self, epsilon: float = 0.1, reduction: str = 'mean'):
        """
        Initialize Label Smoothing Cross Entropy.
        
        Args:
            epsilon: Smoothing parameter. Higher values = more smoothing
            reduction: Specifies the reduction to apply to the output
        """
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.epsilon = epsilon
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Forward pass of Label Smoothing Cross Entropy."""
        num_classes = inputs.size(-1)
        log_preds = F.log_softmax(inputs, dim=-1)
        
        # Create smoothed targets
        targets_one_hot = torch.zeros_like(log_preds).scatter_(1, targets.unsqueeze(1), 1)
        targets_smooth = (1 - self.epsilon) * targets_one_hot + self.epsilon / num_classes
        
        # Compute loss
        loss = -targets_smooth * log_preds
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss.sum(dim=-1)


def compute_adaptive_class_weights(
    labels: Union[np.ndarray, torch.Tensor],
    method: str = 'balanced',
    power: float = 0.5
) -> torch.Tensor:
    """
    Compute adaptive class weights for imbalanced datasets.
    
    Args:
        labels: Ground truth labels
        method: Method for computing weights:
               'balanced' - sklearn's balanced weights
               'inverse' - 1 / class_frequency
               'sqrt_inverse' - 1 / sqrt(class_frequency)
               'log_inverse' - 1 / log(class_frequency + 1)
               'effective_number' - Based on effective number of samples
        power: Power to apply to the weights (for smoothing)
        
    Returns:
        torch.Tensor: Class weights tensor
    """
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()
    
    unique_classes = np.unique(labels)
    num_classes = len(unique_classes)
    
    if method == 'balanced':
        # Use sklearn's balanced weight computation
        weights = compute_class_weight('balanced', classes=unique_classes, y=labels)
    
    elif method == 'inverse':
        # Inverse frequency weighting
        class_counts = np.bincount(labels)
        weights = 1.0 / (class_counts + 1e-8)  # Add small epsilon to avoid division by zero
    
    elif method == 'sqrt_inverse':
        # Square root inverse frequency weighting
        class_counts = np.bincount(labels)
        weights = 1.0 / np.sqrt(class_counts + 1e-8)
    
    elif method == 'log_inverse':
        # Logarithmic inverse frequency weighting
        class_counts = np.bincount(labels)
        weights = 1.0 / np.log(class_counts + 1.0)
    
    elif method == 'effective_number':
        # Effective number of samples weighting
        # Paper: "Class-Balanced Loss Based on Effective Number of Samples"
        beta = 0.9999
        class_counts = np.bincount(labels)
        effective_num = 1.0 - np.power(beta, class_counts)
        weights = (1.0 - beta) / effective_num
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Apply power transformation for smoothing
    if power != 1.0:
        weights = np.power(weights, power)
    
    # Normalize weights so they sum to num_classes
    weights = weights * num_classes / np.sum(weights)
    
    return torch.tensor(weights, dtype=torch.float32)


def analyze_class_distribution(labels: Union[np.ndarray, torch.Tensor]) -> dict:
    """
    Analyze class distribution in the dataset.
    
    Args:
        labels: Ground truth labels
        
    Returns:
        dict: Dictionary containing class distribution analysis
    """
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()
    
    unique_classes, counts = np.unique(labels, return_counts=True)
    total_samples = len(labels)
    
    analysis = {
        'total_samples': total_samples,
        'num_classes': len(unique_classes),
        'class_counts': dict(zip(unique_classes, counts)),
        'class_frequencies': dict(zip(unique_classes, counts / total_samples)),
        'imbalance_ratio': max(counts) / min(counts),
        'entropy': -np.sum((counts / total_samples) * np.log(counts / total_samples + 1e-8))
    }
    
    return analysis


def get_recommended_focal_params(labels: Union[np.ndarray, torch.Tensor]) -> dict:
    """
    Get recommended Focal Loss parameters based on dataset characteristics.
    
    Args:
        labels: Ground truth labels
        
    Returns:
        dict: Recommended parameters for Focal Loss
    """
    analysis = analyze_class_distribution(labels)
    
    # Recommend gamma based on imbalance ratio
    imbalance_ratio = analysis['imbalance_ratio']
    if imbalance_ratio < 2:
        gamma = 1.0  # Low imbalance
    elif imbalance_ratio < 5:
        gamma = 2.0  # Moderate imbalance  
    elif imbalance_ratio < 10:
        gamma = 3.0  # High imbalance
    else:
        gamma = 4.0  # Very high imbalance
    
    # Compute class weights
    alpha = compute_adaptive_class_weights(labels, method='balanced')
    
    recommendations = {
        'gamma': gamma,
        'alpha': alpha,
        'reasoning': {
            'imbalance_ratio': imbalance_ratio,
            'num_classes': analysis['num_classes'],
            'total_samples': analysis['total_samples']
        }
    }
    
    return recommendations 