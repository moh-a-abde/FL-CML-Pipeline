import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
    auc
)
import seaborn as sns
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)

def save_evaluation_results(
    metrics: Dict[str, float],
    predictions: np.ndarray,
    true_labels: np.ndarray,
    output_dir: str = "outputs/neural_network"
) -> None:
    """
    Save evaluation results and create visualizations.
    
    Args:
        metrics: Dictionary of evaluation metrics
        predictions: Model predictions
        true_labels: True labels
        output_dir: Directory to save results
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save metrics
    with open(os.path.join(output_dir, "evaluation_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)
    
    # Save predictions
    predictions_df = pd.DataFrame({
        'true_label': true_labels,
        'predicted_label': predictions
    })
    predictions_df.to_csv(os.path.join(output_dir, "predictions.csv"), index=False)
    
    # Create and save confusion matrix
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(true_labels, predictions)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"))
    plt.close()
    
    # Create and save ROC curves
    plt.figure(figsize=(10, 8))
    n_classes = len(np.unique(true_labels))
    
    # Convert labels to one-hot encoding
    y_true_bin = np.zeros((len(true_labels), n_classes))
    y_true_bin[np.arange(len(true_labels)), true_labels] = 1
    
    # Calculate ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], predictions == i)
        roc_auc[i] = auc(fpr[i], tpr[i])
        plt.plot(fpr[i], tpr[i], label=f'Class {i} (AUC = {roc_auc[i]:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(output_dir, "roc_curves.png"))
    plt.close()
    
    # Create and save precision-recall curves
    plt.figure(figsize=(10, 8))
    
    for i in range(n_classes):
        precision, recall, _ = precision_recall_curve(y_true_bin[:, i], predictions == i)
        plt.plot(recall, precision, label=f'Class {i}')
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curves')
    plt.legend(loc="lower left")
    plt.savefig(os.path.join(output_dir, "precision_recall_curves.png"))
    plt.close()
    
    logger.info(f"Evaluation results saved to {output_dir}")

def plot_training_history(
    history: Dict[str, List[float]],
    output_dir: str = "outputs/neural_network"
) -> None:
    """
    Plot training history and save the plots.
    
    Args:
        history: Dictionary containing training history
        output_dir: Directory to save plots
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot loss
    plt.figure(figsize=(10, 6))
    plt.plot(history['train_loss'], label='Training Loss')
    if 'val_loss' in history:
        plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig(os.path.join(output_dir, "loss_history.png"))
    plt.close()
    
    # Plot accuracy
    plt.figure(figsize=(10, 6))
    plt.plot(history['train_accuracy'], label='Training Accuracy')
    if 'val_accuracy' in history:
        plt.plot(history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.savefig(os.path.join(output_dir, "accuracy_history.png"))
    plt.close()
    
    logger.info(f"Training history plots saved to {output_dir}") 