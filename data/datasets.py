"""Dataset classes for PyTorch DataLoader integration."""

import torch
from torch.utils.data import Dataset
from typing import Dict, Tuple, Optional, List
import pandas as pd
import numpy as np
from .loaders import DataLoaderFactory
from .preprocessors import HybridGraphConstructor
from utils.graph_utils import convert_networkx_to_tensor


class VulnerabilityDataset(Dataset):
    """PyTorch Dataset for vulnerability detection.
    
    Handles loading, preprocessing, and batching of code samples
    with their CFG+DFG graphs.
    """
    
    def __init__(
        self,
        dataset_name: str,
        split: str = 'train',
        max_nodes: int = 10000,
        node_feature_dim: int = 768,
        transform=None,
        use_cached: bool = True
    ):
        """Initialize vulnerability dataset.
        
        Args:
            dataset_name: Name of dataset ('SARD', 'BigVul', 'D-Sieve')
            split: Data split ('train', 'val', 'test')
            max_nodes: Maximum number of nodes in graph
            node_feature_dim: Feature dimension for nodes
            transform: Optional data transformation
            use_cached: Whether to use cached graphs
        """
        self.dataset_name = dataset_name
        self.split = split
        self.max_nodes = max_nodes
        self.node_feature_dim = node_feature_dim
        self.transform = transform
        self.use_cached = use_cached
        
        # Load dataset metadata
        self.loader = DataLoaderFactory.create(dataset_name)
        self.data = self.loader.load()
        
        # Filter by split (assumes split column in data)
        if 'split' in self.data.columns:
            self.data = self.data[self.data['split'] == split]
        
        self.samples = self.data.to_dict('records')
        
        # Initialize graph constructor
        self.graph_constructor = HybridGraphConstructor(
            max_nodes=max_nodes,
            feature_dim=node_feature_dim
        )
    
    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict:
        """Get a single sample.
        
        Args:
            idx: Index of sample
            
        Returns:
            Dict containing:
                - 'graph_x': Node feature matrix (N, feature_dim)
                - 'graph_edge_index': Edge indices (2, num_edges)
                - 'graph_edge_attr': Edge attributes (num_edges, edge_feature_dim)
                - 'label': Vulnerability label (0 or 1)
                - 'metadata': Additional information (CWE, CVE, etc.)
        """
        sample = self.samples[idx]
        
        # Parse sample to get code
        code, label, metadata = self.loader.parse_sample(sample)
        
        # Construct graph
        try:
            graph = self.graph_constructor.construct_from_source(code)
        except Exception as e:
            # Return dummy graph on error
            print(f"Warning: Failed to construct graph for sample {idx}: {e}")
            return self._get_dummy_sample(label, metadata)
        
        # Convert NetworkX graph to tensor format
        graph_x, edge_index, edge_attr = convert_networkx_to_tensor(
            graph,
            max_nodes=self.max_nodes,
            node_feature_dim=self.node_feature_dim
        )
        
        # Apply optional transformations
        if self.transform:
            graph_x, edge_index, edge_attr = self.transform(
                graph_x, edge_index, edge_attr
            )
        
        return {
            'graph_x': torch.FloatTensor(graph_x),
            'graph_edge_index': torch.LongTensor(edge_index),
            'graph_edge_attr': torch.FloatTensor(edge_attr),
            'label': torch.tensor(label, dtype=torch.long),
            'metadata': metadata,
            'sample_idx': idx
        }
    
    def _get_dummy_sample(self, label: int, metadata: Dict) -> Dict:
        """Return a dummy sample for error cases."""
        return {
            'graph_x': torch.zeros((self.max_nodes, self.node_feature_dim)),
            'graph_edge_index': torch.zeros((2, 0), dtype=torch.long),
            'graph_edge_attr': torch.zeros((0, 5)),
            'label': torch.tensor(label, dtype=torch.long),
            'metadata': metadata,
            'sample_idx': -1
        }
    
    def get_class_distribution(self) -> Dict[int, int]:
        """Get distribution of vulnerability labels."""
        labels = [s.get('is_vulnerable', 0) for s in self.samples]
        return {
            0: labels.count(0),
            1: labels.count(1)
        }


class AdversarialVulnerabilityDataset(Dataset):
    """Dataset with adversarially perturbed samples.
    
    Creates pairs of (clean, perturbed) code samples for
    learning adversarial robustness.
    """
    
    def __init__(
        self,
        base_dataset: VulnerabilityDataset,
        adversarial_transform,
        num_perturbations: int = 3
    ):
        """Initialize adversarial dataset.
        
        Args:
            base_dataset: Base vulnerability dataset
            adversarial_transform: Function to apply adversarial transformations
            num_perturbations: Number of perturbations per sample
        """
        self.base_dataset = base_dataset
        self.adversarial_transform = adversarial_transform
        self.num_perturbations = num_perturbations
    
    def __len__(self) -> int:
        """Return total number of (clean, perturbed) pairs."""
        return len(self.base_dataset) * self.num_perturbations
    
    def __getitem__(self, idx: int) -> Dict:
        """Get a (clean, perturbed) pair.
        
        Returns:
            Dict containing:
                - 'clean_sample': Original sample
                - 'perturbed_sample': Adversarially transformed sample
                - 'perturbation_type': Type of perturbation applied
        """
        base_idx = idx // self.num_perturbations
        perturbation_idx = idx % self.num_perturbations
        
        clean_sample = self.base_dataset[base_idx]
        
        # Apply adversarial transformation
        perturbed_sample, perturbation_type = self.adversarial_transform(
            clean_sample,
            perturbation_idx
        )
        
        return {
            'clean_sample': clean_sample,
            'perturbed_sample': perturbed_sample,
            'perturbation_type': perturbation_type
        }


class MultiDatasetVulnerabilityDataset(Dataset):
    """Dataset combining multiple vulnerability datasets.
    
    Useful for training on diverse data sources (SARD, BigVul, D-Sieve).
    """
    
    def __init__(
        self,
        dataset_names: List[str],
        split: str = 'train',
        max_nodes: int = 10000,
        node_feature_dim: int = 768,
        balance_datasets: bool = True
    ):
        """Initialize multi-dataset.
        
        Args:
            dataset_names: List of dataset names
            split: Data split
            max_nodes: Maximum nodes per graph
            node_feature_dim: Node feature dimension
            balance_datasets: Whether to balance dataset contributions
        """
        self.datasets = [
            VulnerabilityDataset(
                dataset_name,
                split=split,
                max_nodes=max_nodes,
                node_feature_dim=node_feature_dim
            )
            for dataset_name in dataset_names
        ]
        
        self.dataset_names = dataset_names
        self.balance_datasets = balance_datasets
        
        # Create mapping of global index to (dataset_idx, sample_idx)
        self._create_index_mapping()
    
    def _create_index_mapping(self) -> None:
        """Create mapping from global index to dataset/sample indices."""
        self.index_mapping = []
        
        if self.balance_datasets:
            # Find minimum dataset size for balanced sampling
            min_size = min(len(ds) for ds in self.datasets)
            
            for dataset_idx, dataset in enumerate(self.datasets):
                for sample_idx in range(min_size):
                    self.index_mapping.append((dataset_idx, sample_idx))
        else:
            # Sequential mapping
            for dataset_idx, dataset in enumerate(self.datasets):
                for sample_idx in range(len(dataset)):
                    self.index_mapping.append((dataset_idx, sample_idx))
    
    def __len__(self) -> int:
        """Return total number of samples."""
        return len(self.index_mapping)
    
    def __getitem__(self, idx: int) -> Dict:
        """Get sample from appropriate dataset."""
        dataset_idx, sample_idx = self.index_mapping[idx]
        sample = self.datasets[dataset_idx][sample_idx]
        
        # Add source information
        sample['source_dataset'] = self.dataset_names[dataset_idx]
        
        return sample
