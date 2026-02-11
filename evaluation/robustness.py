
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple, Optional
import logging
from tqdm import tqdm

from adversarial.perturbations import AdversarialAttackGenerator
from .metrics import MetricsComputer


class RobustnessEvaluator:

    def __init__(
        self,
        device: str = 'cuda',
        perturbation_types: Optional[List[str]] = None
    ):
        self.device = device
        self.logger = logging.getLogger(__name__)

        self.attack_generator = AdversarialAttackGenerator(
            perturbation_types=perturbation_types
        )

        self.metrics_computer = MetricsComputer()

    def evaluate_robustness(
        self,
        model: nn.Module,
        test_loader: DataLoader,
        perturbation_budgets: List[float] = [0.05, 0.1, 0.15, 0.2, 0.3],
        attack_types: Optional[List[str]] = None,
        return_predictions: bool = False
    ) -> Dict:
        model.eval()

        if attack_types is None:
            attack_types = ['node_feature', 'edge', 'structural']

        results = {
            'clean': self._evaluate_clean(model, test_loader),
            'adversarial': {}
        }

        all_predictions = {}

        for budget in perturbation_budgets:
            budget_results = {}
            budget_predictions = {}

            for attack_type in attack_types:
                attack_results, preds = self._evaluate_attack(
                    model,
                    test_loader,
                    attack_type,
                    budget,
                    return_predictions=True
                )

                budget_results[attack_type] = attack_results
                budget_predictions[attack_type] = preds

                self.logger.info(
                    f"Budget {budget:.2f} Attack {attack_type}: "
                    f"Acc={attack_results['accuracy_perturbed']:.4f} "
                    f"Robust={attack_results['robustness_gap']:.4f}"
                )

            results['adversarial'][budget] = budget_results

            if return_predictions:
                all_predictions[budget] = budget_predictions


        results['summary'] = self._compute_robustness_summary(results)

        if return_predictions:
            results['predictions'] = all_predictions

        return results

    def _evaluate_clean(
        self,
        model: nn.Module,
        test_loader: DataLoader
    ) -> Dict[str, float]:
        all_preds = []
        all_targets = []
        all_probs = []

        with torch.no_grad():
            for batch in test_loader:

                batch = self._move_batch_to_device(batch)


                logits, probs = model(
                    batch['graph_x'],
                    batch['graph_edge_index'],
                    batch['graph_edge_attr'],
                    batch.get('graph_batch')
                )

                preds = logits.argmax(dim=-1)
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(batch['label'].cpu().numpy())
                all_probs.extend(self._extract_positive_probs(probs).cpu().numpy())

        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)
        all_probs = np.array(all_probs)

        metrics = self.metrics_computer.compute_metrics(
            all_preds, all_targets, all_probs
        )

        return metrics

    def _evaluate_attack(
        self,
        model: nn.Module,
        test_loader: DataLoader,
        attack_type: str,
        perturbation_budget: float,
        return_predictions: bool = False
    ) -> Tuple[Dict, Optional[Dict]]:
        all_preds_clean = []
        all_preds_pert = []
        all_targets = []

        with torch.no_grad():
            for batch in tqdm(test_loader, desc=f"Eval {attack_type} budget={perturbation_budget}"):

                batch = self._move_batch_to_device(batch)


                logits_clean, _ = model(
                    batch['graph_x'],
                    batch['graph_edge_index'],
                    batch['graph_edge_attr'],
                    batch.get('graph_batch')
                )
                preds_clean = logits_clean.argmax(dim=-1)


                x_pert, edge_index_pert, edge_attr_pert, batch_pert = self._generate_perturbed_batch(
                    batch=batch,
                    attack_type=attack_type,
                    perturbation_budget=perturbation_budget
                )


                logits_pert, _ = model(
                    x_pert,
                    edge_index_pert,
                    edge_attr_pert,
                    batch_pert
                )
                preds_pert = logits_pert.argmax(dim=-1)

                all_preds_clean.extend(preds_clean.cpu().numpy())
                all_preds_pert.extend(preds_pert.cpu().numpy())
                all_targets.extend(batch['label'].cpu().numpy())

        all_preds_clean = np.array(all_preds_clean)
        all_preds_pert = np.array(all_preds_pert)
        all_targets = np.array(all_targets)


        metrics_clean = self.metrics_computer.compute_metrics(
            all_preds_clean, all_targets
        )
        metrics_pert = self.metrics_computer.compute_metrics(
            all_preds_pert, all_targets
        )


        robustness_gap = metrics_clean['accuracy'] - metrics_pert['accuracy']

        metrics = {
            'accuracy_clean': metrics_clean['accuracy'],
            'accuracy_perturbed': metrics_pert['accuracy'],
            'robustness_gap': robustness_gap,
            'agreement': (all_preds_clean == all_preds_pert).mean(),
            'prediction_stability': (all_preds_clean == all_preds_pert).mean()
        }

        predictions = None
        if return_predictions:
            predictions = {
                'clean': all_preds_clean,
                'perturbed': all_preds_pert,
                'targets': all_targets
            }

        return metrics, predictions

    def _move_batch_to_device(self, batch: Dict) -> Dict:
        return {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

    def _compute_robustness_summary(self, results: Dict) -> Dict:
        clean_acc = results['clean']['accuracy']


        avg_acc_pert = 0.0
        avg_gap = 0.0
        num_evals = 0

        for budget, budget_results in results['adversarial'].items():
            for attack_type, metrics in budget_results.items():
                avg_acc_pert += metrics['accuracy_perturbed']
                avg_gap += metrics['robustness_gap']
                num_evals += 1

        return {
            'clean_accuracy': clean_acc,
            'avg_robustness_accuracy': avg_acc_pert / num_evals if num_evals > 0 else 0,
            'avg_robustness_gap': avg_gap / num_evals if num_evals > 0 else 0,
            'total_evaluations': num_evals
        }

    @staticmethod
    def _extract_positive_probs(probs: torch.Tensor) -> torch.Tensor:
        if probs.dim() == 1:
            return probs
        if probs.dim() == 2 and probs.size(1) >= 2:
            return probs[:, 1]
        if probs.dim() == 2 and probs.size(1) == 1:
            return probs[:, 0]
        raise ValueError(f"Unexpected probability tensor shape: {tuple(probs.shape)}")

    def evaluate_certified_robustness(
        self,
        model: nn.Module,
        test_loader: DataLoader,
        radius: float = 0.1,
        num_samples: int = 100
    ) -> Dict:
        model.eval()

        certified_correct = 0
        total = 0

        for batch in test_loader:
            batch = self._move_batch_to_device(batch)


            with torch.no_grad():
                logits, _ = model(
                    batch['graph_x'],
                    batch['graph_edge_index'],
                    batch['graph_edge_attr'],
                    batch.get('graph_batch')
                )
                pred_clean = logits.argmax(dim=-1)


            num_consistent = 0
            for _ in range(num_samples):

                x_pert, edge_index_pert, edge_attr_pert, _ = self.attack_generator.generate_attack(
                    batch['graph_x'],
                    batch['graph_edge_index'],
                    batch['graph_edge_attr'],
                    attack_type='node_feature',
                    perturbation_budget=radius
                )

                with torch.no_grad():
                    logits_pert, _ = model(
                        x_pert,
                        edge_index_pert,
                        edge_attr_pert,
                        batch.get('graph_batch')
                    )
                    pred_pert = logits_pert.argmax(dim=-1)


                if (pred_pert == pred_clean).all():
                    num_consistent += 1


            if num_consistent >= num_samples * 0.5:
                certified_correct += 1

            total += 1

        return {
            'certified_accuracy': certified_correct / total if total > 0 else 0,
            'radius': radius,
            'num_samples': num_samples
        }

    def _generate_perturbed_batch(
        self,
        batch: Dict,
        attack_type: str,
        perturbation_budget: float
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        graph_batch = batch.get("graph_batch")
        if graph_batch is None or graph_batch.numel() == 0:
            x_pert, edge_index_pert, edge_attr_pert, _ = self.attack_generator.generate_attack(
                batch["graph_x"],
                batch["graph_edge_index"],
                batch.get("graph_edge_attr"),
                attack_type=attack_type,
                perturbation_budget=perturbation_budget
            )
            return x_pert, edge_index_pert, edge_attr_pert, graph_batch

        x = batch["graph_x"]
        edge_index = batch["graph_edge_index"]
        edge_attr = batch.get("graph_edge_attr")
        src = edge_index[0]
        dst = edge_index[1]
        num_graphs = int(graph_batch.max().item()) + 1

        x_parts = []
        edge_index_parts = []
        edge_attr_parts = []
        graph_batch_parts = []
        edge_attr_missing = False
        node_offset = 0

        for graph_id in range(num_graphs):
            node_ids = torch.nonzero(graph_batch == graph_id, as_tuple=False).view(-1)
            if node_ids.numel() == 0:
                continue

            node_features = x.index_select(0, node_ids)
            global_to_local = torch.full(
                (x.size(0),),
                -1,
                dtype=torch.long,
                device=x.device
            )
            global_to_local[node_ids] = torch.arange(
                node_ids.numel(),
                dtype=torch.long,
                device=x.device
            )

            edge_mask = (graph_batch[src] == graph_id) & (graph_batch[dst] == graph_id)
            local_edge_index = torch.stack(
                [
                    global_to_local[src[edge_mask]],
                    global_to_local[dst[edge_mask]]
                ],
                dim=0
            )
            local_edge_attr = None
            if edge_attr is not None and edge_attr.size(0) == edge_index.size(1):
                local_edge_attr = edge_attr[edge_mask]

            pert_x, pert_edge_index, pert_edge_attr, _ = self.attack_generator.generate_attack(
                node_features,
                local_edge_index,
                local_edge_attr,
                attack_type=attack_type,
                perturbation_budget=perturbation_budget
            )

            x_parts.append(pert_x)
            graph_batch_parts.append(
                torch.full(
                    (pert_x.size(0),),
                    graph_id,
                    dtype=graph_batch.dtype,
                    device=graph_batch.device
                )
            )
            edge_index_parts.append(pert_edge_index + node_offset)
            node_offset += pert_x.size(0)

            if pert_edge_attr is not None:
                edge_attr_parts.append(pert_edge_attr)
            else:
                edge_attr_missing = True

        if not x_parts:
            return (
                x.clone(),
                edge_index.clone(),
                edge_attr.clone() if edge_attr is not None else None,
                graph_batch.clone()
            )

        x_pert = torch.cat(x_parts, dim=0)
        edge_index_pert = (
            torch.cat(edge_index_parts, dim=1)
            if edge_index_parts
            else torch.zeros((2, 0), dtype=edge_index.dtype, device=edge_index.device)
        )
        graph_batch_pert = torch.cat(graph_batch_parts, dim=0)

        if edge_attr_parts and not edge_attr_missing:
            edge_attr_pert = torch.cat(edge_attr_parts, dim=0)
        elif edge_attr is not None:
            attr_dim = edge_attr.size(1) if edge_attr.dim() > 1 else 1
            edge_attr_pert = torch.zeros(
                (edge_index_pert.size(1), attr_dim),
                dtype=edge_attr.dtype,
                device=edge_attr.device
            )
        else:
            edge_attr_pert = None

        return x_pert, edge_index_pert, edge_attr_pert, graph_batch_pert
