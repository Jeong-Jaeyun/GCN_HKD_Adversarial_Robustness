"""Robustness evaluation against adversarial perturbations."""

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple, Optional, Callable
import logging
from tqdm import tqdm

from adversarial.perturbations import AdversarialAttackGenerator, PerturbationFactory
from .metrics import MetricsComputer


class RobustnessEvaluator:
    """Evaluates model robustness against adversarial attacks.
    
    Implements comprehensive robustness evaluation including:
    - Graph perturbation attacks
    - Robustness curves
    - Adversarial training analysis
    """
    
    def __init__(
        self,
        device: str = 'cuda',
        perturbation_types: Optional[List[str]] = None
    ):
        """Initialize robustness evaluator.
        
        Args:
            device: Computing device
            perturbation_types: Types of perturbations to test
        """
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
        """Evaluate model robustness across perturbation budgets.
        
        Args:
            model: Model to evaluate
            test_loader: Test data loader
            perturbation_budgets: Perturbation strength levels
            attack_types: Types of attacks to test
            return_predictions: Whether to return predictions
            
        Returns:
            Dict with robustness metrics
        """
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
                    f"Acc={attack_results['accuracy']:.4f} "
                    f"Robust={attack_results['robustness_gap']:.4f}"
                )
            
            results['adversarial'][budget] = budget_results
            
            if return_predictions:
                all_predictions[budget] = budget_predictions
        
        # Compute robustness summary
        results['summary'] = self._compute_robustness_summary(results)
        
        if return_predictions:
            results['predictions'] = all_predictions
        
        return results
    
    def _evaluate_clean(
        self,
        model: nn.Module,
        test_loader: DataLoader
    ) -> Dict[str, float]:
        """Evaluate model on clean (non-perturbed) data.
        
        Args:
            model: Model to evaluate
            test_loader: Test data loader
            
        Returns:
            Dict with clean accuracy metrics
        """
        all_preds = []
        all_targets = []
        all_probs = []
        
        with torch.no_grad():
            for batch in test_loader:
                # Move to device
                batch = self._move_batch_to_device(batch)
                
                # Forward pass
                logits, probs = model(
                    batch['graph_x'],
                    batch['graph_edge_index'],
                    batch['graph_edge_attr']
                )
                
                preds = logits.argmax(dim=-1)
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(batch['label'].cpu().numpy())
                all_probs.extend(probs[:, 1].cpu().numpy())  # Prob of class 1
        
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
        """Evaluate model on adversarial examples.
        
        Args:
            model: Model to evaluate
            test_loader: Test data loader
            attack_type: Type of attack
            perturbation_budget: Perturbation strength
            return_predictions: Whether to return predictions
            
        Returns:
            Tuple of (metrics, predictions)
        """
        all_preds_clean = []
        all_preds_pert = []
        all_targets = []
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc=f"Eval {attack_type} budget={perturbation_budget}"):
                # Move to device
                batch = self._move_batch_to_device(batch)
                
                # Clean prediction
                logits_clean, _ = model(
                    batch['graph_x'],
                    batch['graph_edge_index'],
                    batch['graph_edge_attr']
                )
                preds_clean = logits_clean.argmax(dim=-1)
                
                # Generate adversarial example
                x_pert, edge_index_pert, edge_attr_pert, _ = self.attack_generator.generate_attack(
                    batch['graph_x'],
                    batch['graph_edge_index'],
                    batch['graph_edge_attr'],
                    attack_type=attack_type,
                    perturbation_budget=perturbation_budget
                )
                
                # Perturbed prediction
                logits_pert, _ = model(
                    x_pert,
                    edge_index_pert,
                    edge_attr_pert
                )
                preds_pert = logits_pert.argmax(dim=-1)
                
                all_preds_clean.extend(preds_clean.cpu().numpy())
                all_preds_pert.extend(preds_pert.cpu().numpy())
                all_targets.extend(batch['label'].cpu().numpy())
        
        all_preds_clean = np.array(all_preds_clean)
        all_preds_pert = np.array(all_preds_pert)
        all_targets = np.array(all_targets)
        
        # Compute metrics
        metrics_clean = self.metrics_computer.compute_metrics(
            all_preds_clean, all_targets
        )
        metrics_pert = self.metrics_computer.compute_metrics(
            all_preds_pert, all_targets
        )
        
        # Robustness gap
        robustness_gap = metrics_clean['accuracy'] - metrics_pert['accuracy']
        
        metrics = {
            'accuracy_clean': metrics_clean['accuracy'],
            'accuracy_perturbed': metrics_pert['accuracy'],
            'robustness_gap': robustness_gap,
            'agreement': (all_preds_clean == all_preds_pert).mean()
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
        """Move batch to device."""
        return {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }
    
    def _compute_robustness_summary(self, results: Dict) -> Dict:
        """Compute summary statistics of robustness.
        
        Args:
            results: Full robustness evaluation results
            
        Returns:
            Dict with summary statistics
        """
        clean_acc = results['clean']['accuracy']
        
        # Average accuracy and gap across all perturbations
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
    
    def evaluate_certified_robustness(
        self,
        model: nn.Module,
        test_loader: DataLoader,
        radius: float = 0.1,
        num_samples: int = 100
    ) -> Dict:
        """Evaluate certified robustness (abstract certification).
        
        Note: This is a simplified approach. For production, use more
        sophisticated certification methods (randomized smoothing, etc.)
        
        Args:
            model: Model to evaluate
            test_loader: Test data loader
            radius: Radius of perturbation
            num_samples: Number of samples for Monte Carlo estimation
            
        Returns:
            Dict with certification results
        """
        model.eval()
        
        certified_correct = 0
        total = 0
        
        for batch in test_loader:
            batch = self._move_batch_to_device(batch)
            
            # Get ground truth prediction
            with torch.no_grad():
                logits, _ = model(
                    batch['graph_x'],
                    batch['graph_edge_index'],
                    batch['graph_edge_attr']
                )
                pred_clean = logits.argmax(dim=-1)
            
            # Sample perturbations
            num_consistent = 0
            for _ in range(num_samples):
                # Generate random perturbation within radius
                x_pert, _, _, _ = self.attack_generator.generate_attack(
                    batch['graph_x'],
                    batch['graph_edge_index'],
                    batch['graph_edge_attr'],
                    attack_type='node_feature',
                    perturbation_budget=radius
                )
                
                with torch.no_grad():
                    logits_pert, _ = model(
                        x_pert,
                        batch['graph_edge_index'],
                        batch['graph_edge_attr']
                    )
                    pred_pert = logits_pert.argmax(dim=-1)
                
                # Check if prediction is consistent
                if (pred_pert == pred_clean).all():
                    num_consistent += 1
            
            # Certified if majority of samples are consistent
            if num_consistent >= num_samples * 0.5:
                certified_correct += 1
            
            total += 1
        
        return {
            'certified_accuracy': certified_correct / total if total > 0 else 0,
            'radius': radius,
            'num_samples': num_samples
        }
