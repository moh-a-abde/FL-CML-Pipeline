import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import pickle
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from sklearn.preprocessing import label_binarize
from itertools import cycle
from flwr.common.logger import log
from logging import INFO, WARNING

def plot_confusion_matrix(conf_matrix, class_names, output_path):
    """Generates and saves a heatmap visualization of the confusion matrix."""
    try:
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=class_names, yticklabels=class_names, ax=ax)
        ax.set_title('Confusion Matrix')
        ax.set_ylabel('True Label')
        ax.set_xlabel('Predicted Label')
        fig.tight_layout()
        fig.savefig(output_path)
        plt.close(fig)
        log(INFO, "Confusion matrix plot saved to %s", output_path)
    except Exception as e:
        log(WARNING, "Could not generate confusion matrix plot: %s", e)

def plot_roc_curves(y_true, y_pred_proba, class_names, output_path):
    """Generates and saves ROC curves for multi-class classification (One-vs-Rest)."""
    try:
        n_classes = len(class_names)
        y_true_binarized = label_binarize(y_true, classes=range(n_classes))

        fpr = {}
        tpr = {}
        roc_auc = {}
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_true_binarized[:, i], y_pred_proba[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        fig, ax = plt.subplots(figsize=(10, 8))
        colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive'])
        for i, color in zip(range(n_classes), colors):
            ax.plot(fpr[i], tpr[i], color=color, lw=2,
                     label='ROC curve of class {0} (area = {1:0.2f})'
                     ''.format(class_names[i], roc_auc[i]))

        ax.plot([0, 1], [0, 1], 'k--', lw=2)
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('Receiver Operating Characteristic (ROC) - One-vs-Rest')
        ax.legend(loc="lower right")
        fig.tight_layout()
        fig.savefig(output_path)
        plt.close(fig)
        log(INFO, "ROC curve plot saved to %s", output_path)
    except Exception as e:
        log(WARNING, "Could not generate ROC curve plot: %s", e)

def plot_precision_recall_curves(y_true, y_pred_proba, class_names, output_path):
    """Generates and saves Precision-Recall curves for multi-class classification (One-vs-Rest)."""
    try:
        n_classes = len(class_names)
        y_true_binarized = label_binarize(y_true, classes=range(n_classes))

        precision = {}
        recall = {}
        average_precision = {}
        for i in range(n_classes):
            precision[i], recall[i], _ = precision_recall_curve(y_true_binarized[:, i], y_pred_proba[:, i])
            average_precision[i] = auc(recall[i], precision[i])

        fig, ax = plt.subplots(figsize=(10, 8))
        colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive'])
        for i, color in zip(range(n_classes), colors):
            ax.plot(recall[i], precision[i], color=color, lw=2,
                     label='PR curve of class {0} (AP = {1:0.2f})'
                     ''.format(class_names[i], average_precision[i]))

        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_ylim([0.0, 1.05])
        ax.set_xlim([0.0, 1.0])
        ax.set_title('Precision-Recall Curve - One-vs-Rest')
        ax.legend(loc="lower left")
        fig.tight_layout()
        fig.savefig(output_path)
        plt.close(fig)
        log(INFO, "Precision-Recall curve plot saved to %s", output_path)
    except Exception as e:
        log(WARNING, "Could not generate Precision-Recall curve plot: %s", e)


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

        ax.bar_label(rects1, padding=3)
        ax.bar_label(rects2, padding=3)

        fig.tight_layout()
        fig.savefig(output_path)
        plt.close(fig)
        log(INFO, "Class distribution plot saved to %s", output_path)
    except Exception as e:
        log(WARNING, "Could not generate class distribution plot: %s", e)

def plot_per_class_metrics(y_true, y_pred, class_names, output_path):
    """Generates and saves a bar chart of per-class precision, recall, and F1-score."""
    try:
        from sklearn.metrics import classification_report
        report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0)
        
        metrics_to_plot = ['precision', 'recall', 'f1-score']
        class_metrics = {class_name: [] for class_name in class_names}
        
        for class_name in class_names:
            if class_name in report:
                for metric in metrics_to_plot:
                    class_metrics[class_name].append(report[class_name][metric])
            else: 
                for _ in metrics_to_plot:
                    class_metrics[class_name].append(0)

        x = np.arange(len(class_names)) 
        width = 0.2  
        multiplier = 0

        fig, ax = plt.subplots(figsize=(max(12, len(class_names) * 1.5), 6))

        for i, metric_name in enumerate(metrics_to_plot):
            metric_values = [class_metrics[cn][i] for cn in class_names]
            offset = width * multiplier
            rects = ax.bar(x + offset, metric_values, width, label=metric_name.capitalize())
            ax.bar_label(rects, padding=3, fmt='%.2f')
            multiplier += 1

        ax.set_ylabel('Score')
        ax.set_title('Per-Class Precision, Recall, and F1-Score')
        ax.set_xticks(x + width, class_names, rotation=45, ha="right")
        ax.legend(loc='lower right', ncols=len(metrics_to_plot))
        ax.set_ylim(0, 1.1) 

        fig.tight_layout()
        fig.savefig(output_path)
        plt.close(fig)
        log(INFO, "Per-class metrics plot saved to %s", output_path)
    except Exception as e:
        log(WARNING, "Could not generate per-class metrics plot: %s", e)

def _plot_single_metric_curve(ax, history_attr, metric_key, label_prefix, marker):
    """Helper function to plot a single metric curve on a given axis."""
    plot_successful = False
    if hasattr(history_attr, '__contains__') and metric_key in history_attr: # Flower 1.x format
        if isinstance(history_attr[metric_key], list) and history_attr[metric_key]:
            rounds, values = zip(*history_attr[metric_key])
            ax.plot(rounds, values, label=f'{label_prefix} {metric_key.capitalize()}', marker=marker)
            plot_successful = True
    # Flower 2.x format: history_attr is a list of (round, {dict_of_metrics})
    elif isinstance(history_attr, list) and history_attr:
        if all(isinstance(item, tuple) and len(item) == 2 and isinstance(item[1], dict) for item in history_attr):
            rounds = [item[0] for item in history_attr if metric_key in item[1]]
            values = [item[1][metric_key] for item in history_attr if metric_key in item[1]]
            if rounds and values:
                ax.plot(rounds, values, label=f'{label_prefix} {metric_key.capitalize()}', marker=marker)
                plot_successful = True
    return plot_successful

def plot_learning_curves(results_pkl_path, metrics_to_plot, output_dir):
    """Generates and saves learning curve plots (loss and specified metrics vs. round)
       from a pickled Flower history object.

    Args:
        results_pkl_path (str): Path to the results.pkl file.
        metrics_to_plot (list): A list of metric keys (str) to plot from the history object.
        output_dir (str): Directory to save the plots.
    """
    try:
        if not os.path.exists(results_pkl_path):
            log(WARNING, "Results file not found: %s", results_pkl_path)
            return

        with open(results_pkl_path, 'rb') as f:
            history_data = pickle.load(f) # history_data is now a dict

        # Plot Loss
        fig_loss, ax_loss = plt.subplots(figsize=(12, 6))
        loss_plotted = False
        # Access as dictionary keys
        if "losses_distributed" in history_data and history_data["losses_distributed"]:
            rounds_dist, losses_dist = zip(*history_data["losses_distributed"])
            ax_loss.plot(rounds_dist, losses_dist, label='Distributed Loss', marker='o')
            loss_plotted = True
        
        if "losses_centralized" in history_data and history_data["losses_centralized"]:
            rounds_cent, losses_cent = zip(*history_data["losses_centralized"])
            ax_loss.plot(rounds_cent, losses_cent, label='Centralized Loss', marker='x')
            loss_plotted = True
        
        if loss_plotted:
            ax_loss.set_title('Loss Over Federated Learning Rounds')
            ax_loss.set_xlabel('Server Round')
            ax_loss.set_ylabel('Loss')
            ax_loss.legend()
            ax_loss.grid(True)
            fig_loss.tight_layout()
            loss_plot_path = os.path.join(output_dir, "learning_curve_loss.png")
            fig_loss.savefig(loss_plot_path)
            log(INFO, "Loss learning curve plot saved to %s", loss_plot_path)
        else:
            log(INFO, "No loss data found to plot.")
        plt.close(fig_loss)

        # Plot Specified Metrics
        if not metrics_to_plot:
            log(INFO, "No metrics specified for plotting learning curves.")
            return

        num_metrics = len(metrics_to_plot)
        fig_metrics, axes_metrics = plt.subplots(num_metrics, 1, figsize=(12, 6 * num_metrics), sharex=True)
        if num_metrics == 1:
            axes_metrics = [axes_metrics] 

        any_metric_plotted = False
        for i, metric_key in enumerate(metrics_to_plot):
            ax = axes_metrics[i]
            dist_plotted = False
            cent_plotted = False

            # Access as dictionary keys
            if "metrics_distributed" in history_data and history_data["metrics_distributed"]:
                dist_plotted = _plot_single_metric_curve(ax, history_data["metrics_distributed"], metric_key, 'Distributed', 'o')
            
            if "metrics_centralized" in history_data and history_data["metrics_centralized"]:
                cent_plotted = _plot_single_metric_curve(ax, history_data["metrics_centralized"], metric_key, 'Centralized', 'x')
            
            if dist_plotted or cent_plotted:
                ax.set_title(f'{metric_key.replace("_", " ").capitalize()} Over Federated Learning Rounds')
                ax.set_ylabel(metric_key.capitalize())
                ax.legend()
                ax.grid(True)
                any_metric_plotted = True
            else:
                log(WARNING, "Could not plot metric '%s' from history. Data not found or in unexpected format.", metric_key)
                ax.text(0.5, 0.5, f"Data for '{metric_key}' not found \nor in unexpected format.", 
                        horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
                ax.set_title(f'{metric_key.replace("_", " ").capitalize()} (Data Unavailable)')

        if any_metric_plotted:
            axes_metrics[-1].set_xlabel('Server Round') # Set x-label only for the bottom-most plot
            fig_metrics.tight_layout()
            metrics_plot_path = os.path.join(output_dir, "learning_curves_metrics.png")
            fig_metrics.savefig(metrics_plot_path)
            log(INFO, "Metrics learning curves plot saved to %s", metrics_plot_path)
        else:
            log(INFO, "No data found for any of the specified metrics to plot.")
        plt.close(fig_metrics)

    except FileNotFoundError:
        log(WARNING, "Results file not found: %s", results_pkl_path)
    except pickle.UnpicklingError:
        log(WARNING, "Error unpickling results file: %s", results_pkl_path)
    except Exception as e:
        log(WARNING, "Could not generate learning curve plots: %s", e)

def plot_prediction_probability_distributions(y_true, y_pred_proba, class_names, output_dir, bins=50):
    """Generates and saves histograms of prediction probabilities for each true class.

    For each class, this plot shows the distribution of the predicted probabilities 
    assigned to that class, for samples that actually belong to that class.
    High probabilities bunched towards 1.0 are desirable.

    Args:
        y_true (np.array): Array of true labels (integers).
        y_pred_proba (np.array): Array of predicted probabilities, shape (n_samples, n_classes).
        class_names (list): List of class names (strings).
        output_dir (str): Directory to save the plot.
        bins (int): Number of bins for the histograms.
    """
    try:
        n_classes = len(class_names)
        if y_pred_proba.shape[1] != n_classes:
            log(WARNING, "Number of classes in y_pred_proba does not match len(class_names).")
            return

        # Determine the number of rows and columns for subplots
        n_cols = 3 
        n_rows = (n_classes + n_cols - 1) // n_cols # Ceiling division

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), sharex=True, sharey=False)
        axes = axes.flatten() # Flatten to easily iterate regardless of shape

        for i in range(n_classes):
            ax = axes[i]
            # Get probabilities for the current class where the true label is this class
            true_class_indices = np.where(y_true == i)[0]
            if len(true_class_indices) == 0:
                ax.text(0.5, 0.5, "No true samples for this class", horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
                ax.set_title(f'{class_names[i]} (No true samples)')
                ax.set_xlabel('Predicted Probability')
                ax.set_ylabel('Frequency')
                continue
                
            class_probabilities = y_pred_proba[true_class_indices, i]
            
            ax.hist(class_probabilities, bins=bins, range=(0,1), edgecolor='black', alpha=0.7)
            ax.set_title(f'Class: {class_names[i]}')
            ax.set_xlabel('Predicted Probability for this Class')
            ax.set_ylabel('Frequency')
            ax.grid(True, axis='y', linestyle='--', alpha=0.7)
            mean_proba = np.mean(class_probabilities)
            ax.axvline(mean_proba, color='r', linestyle='dashed', linewidth=2, label=f'Mean: {mean_proba:.2f}')
            ax.legend(fontsize='small')

        # Hide any unused subplots
        for j in range(n_classes, n_rows * n_cols):
            fig.delaxes(axes[j])

        fig.suptitle('Distribution of Predicted Probabilities for True Classes', fontsize=16, y=1.02)
        fig.tight_layout(rect=[0, 0, 1, 0.98]) # Adjust layout to make space for suptitle
        
        plot_path = os.path.join(output_dir, "prediction_probability_distributions.png")
        fig.savefig(plot_path)
        plt.close(fig)
        log(INFO, "Prediction probability distribution plot saved to %s", plot_path)

    except Exception as e:
        log(WARNING, "Could not generate prediction probability distribution plot: %s", e) 