"""Evaluation metrics for vulnerability detection."""

import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, roc_curve, precision_recall_curve,
    confusion_matrix, matthews_corrcoef
)
from typing import Dict, List, Tuple, Optional
import logging


class MetricsComputer:
    """Computes comprehensive evaluation metrics for vulnerability detection."""
    
    def __init__(self):
        """Initialize metrics computer."""
        self.logger = logging.getLogger(__name__)
    
    def compute_metrics(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        probabilities: Optional[np.ndarray] = None,
        compute_auc: bool = True
    ) -> Dict[str, float]:
        """Compute comprehensive metrics.
        
        Args:
            predictions: Predicted labels (0 or 1)
            targets: Ground truth labels
            probabilities: Predicted probabilities (optional)
            compute_auc: Whether to compute ROC-AUC and PR-AUC
            
        Returns:
            Dict with computed metrics
        """
        metrics = {}
        
        # Basic classification metrics
        metrics['accuracy'] = accuracy_score(targets, predictions)
        metrics['precision'] = precision_score(targets, predictions, zero_division=0)
        metrics['recall'] = recall_score(targets, predictions, zero_division=0)
        metrics['f1'] = f1_score(targets, predictions, zero_division=0)
        
        # Matthews Correlation Coefficient
        metrics['mcc'] = matthews_corrcoef(targets, predictions)
        
        # Confusion matrix metrics
        tn, fp, fn, tp = confusion_matrix(targets, predictions).ravel()
        
        metrics['true_positive_rate'] = tp / (tp + fn) if (tp + fn) > 0 else 0  # Same as recall
        metrics['false_positive_rate'] = fp / (fp + tn) if (fp + tn) > 0 else 0
        metrics['true_negative_rate'] = tn / (tn + fp) if (tn + fp) > 0 else 0  # Specificity
        metrics['false_negative_rate'] = fn / (fn + tp) if (fn + tp) > 0 else 0
        
        # AUC scores (if probabilities provided)
        if probabilities is not None:
            try:
                if compute_auc:
                    metrics['auc_roc'] = roc_auc_score(targets, probabilities[:, 1])
                    metrics['auc_pr'] = average_precision_score(targets, probabilities[:, 1])
            except Exception as e:
                self.logger.warning(f"Could not compute AUC scores: {e}")
        
        return metrics
    
    def compute_curve_points(
        self,
        targets: np.ndarray,
        probabilities: np.ndarray
    ) -> Dict:
        """Compute ROC and PR curve points.
        
        Args:
            targets: Ground truth labels
            probabilities: Predicted probabilities for positive class
            
        Returns:
            Dict with curve points for plotting
        """
        roc_fpr, roc_tpr, roc_thresholds = roc_curve(targets, probabilities)
        pr_precisions, pr_recalls, pr_thresholds = precision_recall_curve(
            targets, probabilities
        )
        
        return {
            'roc': {
                'fpr': roc_fpr,
                'tpr': roc_tpr,
                'thresholds': roc_thresholds
            },
            'pr': {
                'precision': pr_precisions,
                'recall': pr_recalls,
                'thresholds': pr_thresholds
            }
        }
    
    def compute_threshold_metrics(
        self,
        targets: np.ndarray,
        probabilities: np.ndarray,
        thresholds: List[float]
    ) -> Dict[float, Dict]:
        """Compute metrics at different decision thresholds.
        
        Args:
            targets: Ground truth labels
            probabilities: Predicted probabilities
            thresholds: Thresholds to evaluate at
            
        Returns:
            Dict mapping threshold to metrics
        """
        results = {}
        
        for threshold in thresholds:
            predictions = (probabilities[:, 1] >= threshold).astype(int)
            metrics = self.compute_metrics(predictions, targets)
            results[threshold] = metrics
        
        return results
    
    def compute_per_class_metrics(
        self,
        predictions: np.ndarray,
        targets: np.ndarray
    ) -> Dict[int, Dict]:
        """Compute per-class metrics.
        
        Args:
            predictions: Predicted labels
            targets: Ground truth labels
            
        Returns:
            Dict with metrics for each class
        """
        from sklearn.metrics import precision_recall_fscore_support
        
        precision, recall, f1, support = precision_recall_fscore_support(
            targets, predictions, zero_division=0
        )
        
        classes = np.unique(targets)
        results = {}
        
        for class_idx, class_label in enumerate(classes):
            results[class_label] = {
                'precision': precision[class_idx],
                'recall': recall[class_idx],
                'f1': f1[class_idx],
                'support': support[class_idx]
            }
        
        return results


class ConfusionMatrixAnalysis:
    """Analyzes confusion matrix for error patterns."""
    
    @staticmethod
    def analyze(targets: np.ndarray, predictions: np.ndarray) -> Dict:
        """Analyze confusion matrix.
        
        Args:
            targets: Ground truth labels
            predictions: Predicted labels
            
        Returns:
            Dict with confusion matrix analysis
        """
        tn, fp, fn, tp = confusion_matrix(targets, predictions).ravel()
        
        return {
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'true_positives': int(tp),
            'tpr': tp / (tp + fn) if (tp + fn) > 0 else 0,  # Sensitivity
            'tnr': tn / (tn + fp) if (tn + fp) > 0 else 0,  # Specificity
            'fpr': fp / (fp + tn) if (fp + tn) > 0 else 0,  # False positive rate
            'fnr': fn / (fn + tp) if (fn + tp) > 0 else 0,  # False negative rate
        }
