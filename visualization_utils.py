import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from sklearn.preprocessing import label_binarize
from itertools import cycle
from flwr.common.logger import log
from logging import INFO, WARNING

def plot_confusion_matrix(conf_matrix, class_names, output_path):
    """Generates and saves a heatmap visualization of the confusion matrix."""
    try:
        plt.figure(figsize=(10, 8))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        log(INFO, f"Confusion matrix plot saved to {output_path}")
    except Exception as e:
        log(WARNING, f"Could not generate confusion matrix plot: {e}")

def plot_roc_curves(y_true, y_pred_proba, class_names, output_path):
    """Generates and saves ROC curves for multi-class classification (One-vs-Rest)."""
    try:
        n_classes = len(class_names)
        y_true_binarized = label_binarize(y_true, classes=range(n_classes))

        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_true_binarized[:, i], y_pred_proba[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # Plot all ROC curves
        plt.figure(figsize=(10, 8))
        colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive'])
        for i, color in zip(range(n_classes), colors):
            plt.plot(fpr[i], tpr[i], color=color, lw=2,
                     label='ROC curve of class {0} (area = {1:0.2f})'
                     ''.format(class_names[i], roc_auc[i]))

        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) - One-vs-Rest')
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        log(INFO, f"ROC curve plot saved to {output_path}")
    except Exception as e:
        log(WARNING, f"Could not generate ROC curve plot: {e}")

def plot_precision_recall_curves(y_true, y_pred_proba, class_names, output_path):
    """Generates and saves Precision-Recall curves for multi-class classification (One-vs-Rest)."""
    try:
        n_classes = len(class_names)
        y_true_binarized = label_binarize(y_true, classes=range(n_classes))

        # Compute Precision-Recall curve and area for each class
        precision = dict()
        recall = dict()
        average_precision = dict()
        for i in range(n_classes):
            precision[i], recall[i], _ = precision_recall_curve(y_true_binarized[:, i], y_pred_proba[:, i])
            average_precision[i] = auc(recall[i], precision[i])

        # Plot all Precision-Recall curves
        plt.figure(figsize=(10, 8))
        colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive'])
        for i, color in zip(range(n_classes), colors):
            plt.plot(recall[i], precision[i], color=color, lw=2,
                     label='PR curve of class {0} (AP = {1:0.2f})'
                     ''.format(class_names[i], average_precision[i]))

        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.title('Precision-Recall Curve - One-vs-Rest')
        plt.legend(loc="lower left")
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        log(INFO, f"Precision-Recall curve plot saved to {output_path}")
    except Exception as e:
        log(WARNING, f"Could not generate Precision-Recall curve plot: {e}")


def plot_class_distribution(y_true, y_pred, class_names, output_path):
    """Generates and saves bar plots comparing true and predicted class distributions."""
    try:
        n_classes = len(class_names)
        true_counts = np.bincount(y_true.astype(int), minlength=n_classes)
        pred_counts = np.bincount(y_pred.astype(int), minlength=n_classes)

        x = np.arange(n_classes)
        width = 0.35

        fig, ax = plt.subplots(figsize=(12, 6))
        rects1 = ax.bar(x - width/2, true_counts, width, label='True Labels')
        rects2 = ax.bar(x + width/2, pred_counts, width, label='Predicted Labels')

        ax.set_ylabel('Count')
        ax.set_title('True vs Predicted Class Distribution')
        ax.set_xticks(x)
        ax.set_xticklabels(class_names, rotation=45, ha='right')
        ax.legend()

        # Add counts above bars
        ax.bar_label(rects1, padding=3)
        ax.bar_label(rects2, padding=3)

        fig.tight_layout()
        plt.savefig(output_path)
        plt.close()
        log(INFO, f"Class distribution plot saved to {output_path}")
    except Exception as e:
        log(WARNING, f"Could not generate class distribution plot: {e}") 