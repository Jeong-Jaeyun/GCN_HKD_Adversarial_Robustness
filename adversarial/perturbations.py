"""Graph-level adversarial perturbations for robustness testing.

Implements graph-level attacks and perturbations to evaluate
the model's robustness against adversarial graph modifications.
"""

import torch
import numpy as np
import networkx as nx
from typing import Tuple, Dict, List, Optional
from abc import ABC, abstractmethod


class GraphPerturbation(ABC):
    """Abstract base class for graph perturbations."""
    
    @abstractmethod
    def perturb(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        perturbation_budget: float = 0.1
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perturb graph structure.
        
        Args:
            x: Node features
            edge_index: Edge indices
            edge_attr: Edge attributes
            perturbation_budget: Budget for perturbations
            
        Returns:
            Tuple of perturbed (x, edge_index, edge_attr)
        """
        pass


class NodeFeaturePerturbation(GraphPerturbation):
    """Perturb node features with Gaussian noise."""
    
    def __init__(self, noise_scale: float = 0.1):
        """Initialize node feature perturbation.
        
        Args:
            noise_scale: Standard deviation of Gaussian noise
        """
        self.noise_scale = noise_scale
    
    def perturb(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        perturbation_budget: float = 0.1
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Add Gaussian noise to node features."""
        # Add noise proportional to perturbation budget
        noise = torch.randn_like(x) * self.noise_scale * perturbation_budget
        perturbed_x = x + noise
        
        return perturbed_x, edge_index, edge_attr


class EdgePerturbation(GraphPerturbation):
    """Add or remove edges with noise."""
    
    def perturb(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        perturbation_budget: float = 0.1
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perturb edge structure by adding/removing edges."""
        num_nodes = x.shape[0]
        num_existing_edges = edge_index.shape[1]
        
        # Number of edges to add/remove based on budget
        num_perturbations = int(num_existing_edges * perturbation_budget)
        
        perturbed_edge_index = edge_index.clone()
        
        # Remove edges
        num_to_remove = num_perturbations // 2
        if num_to_remove > 0 and num_existing_edges > 0:
            remove_indices = np.random.choice(
                num_existing_edges,
                size=min(num_to_remove, num_existing_edges),
                replace=False
            )
            mask = torch.ones(num_existing_edges, dtype=torch.bool)
            mask[remove_indices] = False
            perturbed_edge_index = perturbed_edge_index[:, mask]
        
        # Add edges
        num_to_add = num_perturbations - num_to_remove
        if num_to_add > 0:
            new_edges = torch.randint(
                0, num_nodes,
                size=(2, num_to_add)
            )
            perturbed_edge_index = torch.cat(
                [perturbed_edge_index, new_edges],
                dim=1
            )
        
        # Perturb edge attributes if they exist
        if edge_attr is not None:
            perturbed_edge_attr = edge_attr.clone()
            if edge_attr.shape[0] > 0:
                noise = torch.randn_like(edge_attr) * 0.1 * perturbation_budget
                perturbed_edge_attr = edge_attr + noise
        else:
            perturbed_edge_attr = None
        
        return x, perturbed_edge_index, perturbed_edge_attr


class StructuralPerturbation(GraphPerturbation):
    """Perturb graph structure (add/remove edges and nodes)."""
    
    def __init__(self, add_node_prob: float = 0.05, remove_node_prob: float = 0.02):
        """Initialize structural perturbation.
        
        Args:
            add_node_prob: Probability of adding a node
            remove_node_prob: Probability of removing a node
        """
        self.add_node_prob = add_node_prob
        self.remove_node_prob = remove_node_prob
    
    def perturb(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        perturbation_budget: float = 0.1
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perturb graph structure by adding/removing nodes."""
        # Add nodes
        num_to_add = int(x.shape[0] * self.add_node_prob * perturbation_budget)
        if num_to_add > 0:
            new_features = torch.randn(num_to_add, x.shape[1]) * 0.1
            x = torch.cat([x, new_features], dim=0)
            
            # Add edges to new nodes
            num_nodes = x.shape[0]
            for new_node_idx in range(num_nodes - num_to_add, num_nodes):
                # Connect to random existing nodes
                targets = torch.randint(0, num_nodes - num_to_add, (2,))
                new_edges = torch.tensor(
                    [[new_node_idx, targets[0].item()],
                     [targets[1].item(), new_node_idx]],
                    dtype=torch.long
                )
                edge_index = torch.cat([edge_index, new_edges.T], dim=1)
        
        return x, edge_index, edge_attr


class AdversarialAttackGenerator:
    """Generates adversarial examples using various attack methods."""
    
    def __init__(self, perturbation_types: Optional[List[str]] = None):
        """Initialize attack generator.
        
        Args:
            perturbation_types: Types of perturbations to use
        """
        self.perturbations = {
            'node_feature': NodeFeaturePerturbation(),
            'edge': EdgePerturbation(),
            'structural': StructuralPerturbation()
        }
        
        self.perturbation_types = perturbation_types or list(self.perturbations.keys())
    
    def generate_attack(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        attack_type: Optional[str] = None,
        perturbation_budget: float = 0.1,
        num_steps: int = 1
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        """Generate adversarial example.
        
        Args:
            x: Node features
            edge_index: Edge indices
            edge_attr: Edge attributes
            attack_type: Type of attack ('node_feature', 'edge', 'structural')
            perturbation_budget: Budget for perturbations
            num_steps: Number of perturbation steps
            
        Returns:
            Tuple of (perturbed_x, perturbed_edge_index, perturbed_edge_attr, info)
        """
        if attack_type is None:
            import random
            attack_type = random.choice(self.perturbation_types)
        
        if attack_type not in self.perturbations:
            raise ValueError(f"Unknown attack type: {attack_type}")
        
        perturbation = self.perturbations[attack_type]
        
        # Apply perturbation multiple times
        perturbed_x = x.clone()
        perturbed_edge_index = edge_index.clone()
        perturbed_edge_attr = edge_attr.clone() if edge_attr is not None else None
        
        for _ in range(num_steps):
            perturbed_x, perturbed_edge_index, perturbed_edge_attr = \
                perturbation.perturb(
                    perturbed_x,
                    perturbed_edge_index,
                    perturbed_edge_attr,
                    perturbation_budget / num_steps
                )
        
        return perturbed_x, perturbed_edge_index, perturbed_edge_attr, {
            'attack_type': attack_type,
            'perturbation_budget': perturbation_budget,
            'num_steps': num_steps
        }


class PerturbationFactory:
    """Factory for creating perturbations."""
    
    _perturbations = {
        'node_feature': NodeFeaturePerturbation,
        'edge': EdgePerturbation,
        'structural': StructuralPerturbation,
    }
    
    @classmethod
    def create(cls, perturbation_type: str, **kwargs) -> GraphPerturbation:
        """Create perturbation by type.
        
        Args:
            perturbation_type: Type of perturbation
            **kwargs: Arguments for perturbation class
            
        Returns:
            GraphPerturbation instance
        """
        if perturbation_type not in cls._perturbations:
            raise ValueError(
                f"Unknown perturbation type: {perturbation_type}. "
                f"Available: {list(cls._perturbations.keys())}"
            )
        
        perturbation_class = cls._perturbations[perturbation_type]
        return perturbation_class(**kwargs)
    
    @classmethod
    def register(cls, name: str, perturbation_class: type) -> None:
        """Register a custom perturbation."""
        cls._perturbations[name] = perturbation_class
