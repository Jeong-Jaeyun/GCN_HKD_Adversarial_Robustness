
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

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def compute_metrics(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        probabilities: Optional[np.ndarray] = None,
        compute_auc: bool = True
    ) -> Dict[str, float]:
        metrics = {}

        predictions = np.asarray(predictions)
        targets = np.asarray(targets)


        metrics['accuracy'] = accuracy_score(targets, predictions)
        metrics['precision'] = precision_score(targets, predictions, zero_division=0)
        metrics['recall'] = recall_score(targets, predictions, zero_division=0)
        metrics['f1'] = f1_score(targets, predictions, zero_division=0)


        metrics['mcc'] = matthews_corrcoef(targets, predictions)


        tn, fp, fn, tp = confusion_matrix(targets, predictions, labels=[0, 1]).ravel()

        metrics['true_positive_rate'] = tp / (tp + fn) if (tp + fn) > 0 else 0
        metrics['false_positive_rate'] = fp / (fp + tn) if (fp + tn) > 0 else 0
        metrics['true_negative_rate'] = tn / (tn + fp) if (tn + fp) > 0 else 0
        metrics['false_negative_rate'] = fn / (fn + tp) if (fn + tp) > 0 else 0


        if probabilities is not None:
            try:
                if compute_auc:
                    pos_probs = self._extract_positive_probs(probabilities)
                    metrics['auc_roc'] = roc_auc_score(targets, pos_probs)
                    metrics['auc_pr'] = average_precision_score(targets, pos_probs)
            except Exception as e:
                self.logger.warning(f"Could not compute AUC scores: {e}")

        return metrics

    def compute_curve_points(
        self,
        targets: np.ndarray,
        probabilities: np.ndarray
    ) -> Dict:
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
        results = {}

        for threshold in thresholds:
            pos_probs = self._extract_positive_probs(probabilities)
            predictions = (pos_probs >= threshold).astype(int)
            metrics = self.compute_metrics(predictions, targets)
            results[threshold] = metrics

        return results

    def _extract_positive_probs(self, probabilities: np.ndarray) -> np.ndarray:
        probs = np.asarray(probabilities)
        if probs.ndim == 1:
            return probs
        if probs.ndim == 2:
            if probs.shape[1] == 1:
                return probs[:, 0]
            return probs[:, 1]
        raise ValueError(
            f"probabilities must be 1D or 2D, got shape {probs.shape}"
        )

    def compute_per_class_metrics(
        self,
        predictions: np.ndarray,
        targets: np.ndarray
    ) -> Dict[int, Dict]:
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

    @staticmethod
    def analyze(targets: np.ndarray, predictions: np.ndarray) -> Dict:
        tn, fp, fn, tp = confusion_matrix(targets, predictions, labels=[0, 1]).ravel()

        return {
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'true_positives': int(tp),
            'tpr': tp / (tp + fn) if (tp + fn) > 0 else 0,
            'tnr': tn / (tn + fp) if (tn + fp) > 0 else 0,
            'fpr': fp / (fp + tn) if (fp + tn) > 0 else 0,
            'fnr': fn / (fn + tp) if (fn + tp) > 0 else 0,
        }
