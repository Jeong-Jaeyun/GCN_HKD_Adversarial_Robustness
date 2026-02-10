
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Optional
import logging


class Validator:

    def __init__(self, device: str = 'cuda'):
        self.device = device
        self.logger = logging.getLogger(__name__)

    def validate(
        self,
        model: nn.Module,
        val_loader: DataLoader,
        loss_fn: Optional[nn.Module] = None
    ) -> Dict[str, float]:
        model.eval()

        total_loss = 0.0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):

                batch = {
                    k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }


                logits, probs = model(
                    batch['graph_x'],
                    batch['graph_edge_index'],
                    batch['graph_edge_attr']
                )


                if loss_fn is not None:
                    loss = loss_fn(logits, batch['label'])
                    total_loss += loss.item()


                preds = logits.argmax(dim=-1)
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(batch['label'].cpu().numpy())


        metrics = self._compute_metrics(all_preds, all_targets)

        if loss_fn is not None:
            metrics['val_loss'] = total_loss / len(val_loader)

        return metrics

    def _compute_metrics(self, preds: list, targets: list) -> Dict[str, float]:
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, f1_score,
            roc_auc_score, average_precision_score, confusion_matrix
        )

        metrics = {
            'accuracy': accuracy_score(targets, preds),
            'precision': precision_score(targets, preds, zero_division=0),
            'recall': recall_score(targets, preds, zero_division=0),
            'f1': f1_score(targets, preds, zero_division=0),
        }


        tn, fp, fn, tp = confusion_matrix(targets, preds).ravel()
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0

        return metrics


class AdversarialValidator:

    def __init__(self, device: str = 'cuda'):
        self.device = device
        self.logger = logging.getLogger(__name__)

    def evaluate_robustness(
        self,
        model: nn.Module,
        test_loader: DataLoader,
        perturbation_fn,
        perturbation_budgets: list = [0.05, 0.1, 0.15, 0.2]
    ) -> Dict:
        model.eval()
        results = {}

        for budget in perturbation_budgets:
            correct_clean = 0
            correct_perturbed = 0
            total = 0

            with torch.no_grad():
                for batch in test_loader:

                    batch = {
                        k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                        for k, v in batch.items()
                    }


                    logits_clean, _ = model(
                        batch['graph_x'],
                        batch['graph_edge_index'],
                        batch['graph_edge_attr']
                    )
                    preds_clean = logits_clean.argmax(dim=-1)


                    x_pert, edge_index_pert, edge_attr_pert = perturbation_fn(
                        batch['graph_x'],
                        batch['graph_edge_index'],
                        batch['graph_edge_attr'],
                        budget
                    )

                    logits_pert, _ = model(
                        x_pert,
                        edge_index_pert,
                        edge_attr_pert
                    )
                    preds_pert = logits_pert.argmax(dim=-1)


                    targets = batch['label']
                    correct_clean += (preds_clean == targets).sum().item()
                    correct_perturbed += (preds_pert == targets).sum().item()
                    total += targets.size(0)

            results[budget] = {
                'clean_accuracy': correct_clean / total,
                'robust_accuracy': correct_perturbed / total,
                'robustness_drop': (correct_clean - correct_perturbed) / total
            }

            self.logger.info(
                f"Budget {budget}: Clean={results[budget]['clean_accuracy']:.4f} "
                f"Robust={results[budget]['robust_accuracy']:.4f} "
                f"Drop={results[budget]['robustness_drop']:.4f}"
            )

        return results
