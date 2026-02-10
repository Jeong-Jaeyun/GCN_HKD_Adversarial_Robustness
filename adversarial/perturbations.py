
import random
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

import torch


class GraphPerturbation(ABC):

    @staticmethod
    def _validate_budget(perturbation_budget: float) -> None:
        if perturbation_budget < 0:
            raise ValueError("perturbation_budget must be >= 0")

    @abstractmethod
    def perturb(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        perturbation_budget: float = 0.1
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        pass


class NodeFeaturePerturbation(GraphPerturbation):

    def __init__(self, noise_scale: float = 0.1, clamp_range: Optional[Tuple[float, float]] = None):
        self.noise_scale = noise_scale
        self.clamp_range = clamp_range

    def perturb(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        perturbation_budget: float = 0.1
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        self._validate_budget(perturbation_budget)
        if perturbation_budget == 0:
            return x, edge_index, edge_attr


        noise = torch.randn_like(x) * self.noise_scale * perturbation_budget
        perturbed_x = x + noise

        if self.clamp_range is not None:
            min_val, max_val = self.clamp_range
            perturbed_x = torch.clamp(perturbed_x, min=min_val, max=max_val)

        return perturbed_x, edge_index, edge_attr


class EdgePerturbation(GraphPerturbation):

    def __init__(
        self,
        edge_attr_noise_scale: float = 0.1,
        allow_self_loops: bool = False
    ):
        self.edge_attr_noise_scale = edge_attr_noise_scale
        self.allow_self_loops = allow_self_loops

    def perturb(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        perturbation_budget: float = 0.1
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        self._validate_budget(perturbation_budget)
        if perturbation_budget == 0:
            return x, edge_index, edge_attr

        if edge_index.dim() != 2 or edge_index.size(0) != 2:
            raise ValueError("edge_index must have shape (2, num_edges)")

        num_nodes = x.shape[0]
        num_existing_edges = edge_index.size(1)
        device = edge_index.device


        reference_edges = max(num_existing_edges, 1)
        num_perturbations = max(1, int(round(reference_edges * perturbation_budget)))

        perturbed_edge_index = edge_index.clone()
        perturbed_edge_attr: Optional[torch.Tensor] = edge_attr.clone() if edge_attr is not None else None


        num_to_remove = num_perturbations // 2
        if num_to_remove > 0 and num_existing_edges > 0:
            remove_count = min(num_to_remove, num_existing_edges)
            remove_indices = torch.randperm(num_existing_edges, device=device)[:remove_count]
            mask = torch.ones(num_existing_edges, dtype=torch.bool, device=device)
            mask[remove_indices] = False
            perturbed_edge_index = perturbed_edge_index[:, mask]
            if perturbed_edge_attr is not None and perturbed_edge_attr.size(0) == num_existing_edges:
                perturbed_edge_attr = perturbed_edge_attr[mask]


        num_to_add = num_perturbations - num_to_remove
        if num_to_add > 0 and num_nodes > 0:
            new_edges = torch.randint(
                0, num_nodes,
                size=(2, num_to_add),
                device=device
            )

            if not self.allow_self_loops and num_nodes > 1:
                self_loop_mask = new_edges[0] == new_edges[1]
                while self_loop_mask.any():
                    new_edges[1, self_loop_mask] = torch.randint(
                        0,
                        num_nodes,
                        size=(int(self_loop_mask.sum().item()),),
                        device=device
                    )
                    self_loop_mask = new_edges[0] == new_edges[1]

            perturbed_edge_index = torch.cat(
                [perturbed_edge_index, new_edges],
                dim=1
            )

            if perturbed_edge_attr is not None:
                if perturbed_edge_attr.dim() == 1:
                    perturbed_edge_attr = perturbed_edge_attr.unsqueeze(1)
                attr_dim = perturbed_edge_attr.size(1)
                new_edge_attr = torch.zeros(
                    (num_to_add, attr_dim),
                    device=perturbed_edge_attr.device,
                    dtype=perturbed_edge_attr.dtype
                )
                perturbed_edge_attr = torch.cat([perturbed_edge_attr, new_edge_attr], dim=0)


        if perturbed_edge_attr is not None and perturbed_edge_attr.numel() > 0:
            noise = torch.randn_like(perturbed_edge_attr) * self.edge_attr_noise_scale * perturbation_budget
            perturbed_edge_attr = perturbed_edge_attr + noise

        return x, perturbed_edge_index, perturbed_edge_attr


class StructuralPerturbation(GraphPerturbation):

    def __init__(
        self,
        add_node_prob: float = 0.05,
        connections_per_new_node: int = 2,
        feature_noise_scale: float = 0.1
    ):
        self.add_node_prob = add_node_prob
        self.connections_per_new_node = connections_per_new_node
        self.feature_noise_scale = feature_noise_scale

    def perturb(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        perturbation_budget: float = 0.1
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        self._validate_budget(perturbation_budget)
        if perturbation_budget == 0:
            return x, edge_index, edge_attr

        if edge_index.dim() != 2 or edge_index.size(0) != 2:
            raise ValueError("edge_index must have shape (2, num_edges)")

        num_original_nodes = x.size(0)
        if num_original_nodes == 0:
            return x, edge_index, edge_attr


        num_to_add = int(num_original_nodes * self.add_node_prob * perturbation_budget)
        if num_to_add > 0:
            new_features = (
                torch.randn(
                    num_to_add,
                    x.size(1),
                    device=x.device,
                    dtype=x.dtype
                ) * self.feature_noise_scale
            )
            x = torch.cat([x, new_features], dim=0)


            device = edge_index.device
            new_edges_list = []

            for new_node_idx in range(num_original_nodes, num_original_nodes + num_to_add):
                num_connections = min(self.connections_per_new_node, num_original_nodes)
                if num_connections == 0:
                    continue

                targets = torch.randperm(num_original_nodes, device=device)[:num_connections]
                src = torch.full((num_connections,), new_node_idx, device=device, dtype=torch.long)

                outgoing = torch.stack([src, targets], dim=0)
                incoming = torch.stack([targets, src], dim=0)
                new_edges_list.extend([outgoing, incoming])

            if new_edges_list:
                new_edges = torch.cat(new_edges_list, dim=1)
                edge_index = torch.cat([edge_index, new_edges], dim=1)

                if edge_attr is not None:
                    if edge_attr.dim() == 1:
                        edge_attr = edge_attr.unsqueeze(1)
                    attr_dim = edge_attr.size(1)
                    new_attr = torch.zeros(
                        (new_edges.size(1), attr_dim),
                        device=edge_attr.device,
                        dtype=edge_attr.dtype
                    )
                    edge_attr = torch.cat([edge_attr, new_attr], dim=0)

        return x, edge_index, edge_attr


class AdversarialAttackGenerator:

    def __init__(
        self,
        perturbation_types: Optional[List[str]] = None,
        seed: Optional[int] = None
    ):
        self.perturbations = {
            'node_feature': NodeFeaturePerturbation(),
            'edge': EdgePerturbation(),
            'structural': StructuralPerturbation()
        }

        self.perturbation_types = perturbation_types or list(self.perturbations.keys())
        self._rng = random.Random(seed)

    def generate_attack(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        attack_type: Optional[str] = None,
        perturbation_budget: float = 0.1,
        num_steps: int = 1
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Dict]:
        if num_steps <= 0:
            raise ValueError("num_steps must be >= 1")
        if perturbation_budget < 0:
            raise ValueError("perturbation_budget must be >= 0")

        if attack_type is None:
            attack_type = self._rng.choice(self.perturbation_types)

        if attack_type not in self.perturbations:
            raise ValueError(f"Unknown attack type: {attack_type}")

        perturbation = self.perturbations[attack_type]


        perturbed_x = x.clone()
        perturbed_edge_index = edge_index.clone()
        perturbed_edge_attr = edge_attr.clone() if edge_attr is not None else None

        for _ in range(num_steps):
            perturbed_x, perturbed_edge_index, perturbed_edge_attr =\
                perturbation.perturb(
                    perturbed_x,
                    perturbed_edge_index,
                    perturbed_edge_attr,
                    perturbation_budget / num_steps
                )

        return perturbed_x, perturbed_edge_index, perturbed_edge_attr, {
            'attack_type': attack_type,
            'perturbation_budget': perturbation_budget,
            'num_steps': num_steps,
            'num_nodes_before': int(x.size(0)),
            'num_nodes_after': int(perturbed_x.size(0)),
            'num_edges_before': int(edge_index.size(1)),
            'num_edges_after': int(perturbed_edge_index.size(1))
        }


class PerturbationFactory:

    _perturbations = {
        'node_feature': NodeFeaturePerturbation,
        'edge': EdgePerturbation,
        'structural': StructuralPerturbation,
    }

    @classmethod
    def create(cls, perturbation_type: str, **kwargs) -> GraphPerturbation:
        if perturbation_type not in cls._perturbations:
            raise ValueError(
                f"Unknown perturbation type: {perturbation_type}. "
                f"Available: {list(cls._perturbations.keys())}"
            )

        perturbation_class = cls._perturbations[perturbation_type]
        return perturbation_class(**kwargs)

    @classmethod
    def register(cls, name: str, perturbation_class: type) -> None:
        cls._perturbations[name] = perturbation_class
