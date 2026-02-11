import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Optional

from .loaders import DataLoaderFactory
from .preprocessors import HybridGraphConstructor
from utils.graph_utils import convert_networkx_to_tensor


class VulnerabilityDataset(Dataset):
    def __init__(
        self,
        dataset_name: str,
        split: str = "train",
        max_nodes: int = 10000,
        node_feature_dim: int = 768,
        transform=None,
        use_cached: bool = True,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        split_seed: int = 42,
        chronological_split: bool = False,
        time_column: Optional[str] = None,
    ):
        self.dataset_name = dataset_name
        self.split = split
        self.max_nodes = max_nodes
        self.node_feature_dim = node_feature_dim
        self.transform = transform
        self.use_cached = use_cached
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.split_seed = split_seed
        self.chronological_split = chronological_split
        self.time_column = time_column

        self.loader = DataLoaderFactory.create(dataset_name)
        self.data = self.loader.load()
        if "split" not in self.data.columns:
            self.data = self._assign_split_column(self.data)
        if "split" in self.data.columns:
            self.data = self.data[self.data["split"] == split]
        self.samples = self.data.to_dict("records")

        self.graph_constructor = HybridGraphConstructor(
            max_nodes=max_nodes,
            feature_dim=node_feature_dim,
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]
        code, label, metadata = self.loader.parse_sample(sample)

        try:
            graph = self.graph_constructor.construct_from_source(code)
        except Exception as exc:
            print(f"Warning: Failed to construct graph for sample {idx}: {exc}")
            return self._get_dummy_sample(label, metadata)

        graph_x, edge_index, edge_attr = convert_networkx_to_tensor(
            graph,
            max_nodes=self.max_nodes,
            node_feature_dim=self.node_feature_dim,
        )
        if self.transform:
            graph_x, edge_index, edge_attr = self.transform(graph_x, edge_index, edge_attr)

        return {
            "graph_x": torch.FloatTensor(graph_x),
            "graph_edge_index": torch.LongTensor(edge_index),
            "graph_edge_attr": torch.FloatTensor(edge_attr),
            "label": torch.tensor(label, dtype=torch.long),
            "metadata": metadata,
            "sample_idx": idx,
        }

    def _get_dummy_sample(self, label: int, metadata: Dict) -> Dict:
        return {
            "graph_x": torch.zeros((self.max_nodes, self.node_feature_dim)),
            "graph_edge_index": torch.zeros((2, 0), dtype=torch.long),
            "graph_edge_attr": torch.zeros((0, 5)),
            "label": torch.tensor(label, dtype=torch.long),
            "metadata": metadata,
            "sample_idx": -1,
        }

    def _assign_split_column(self, data: pd.DataFrame) -> pd.DataFrame:
        data = data.copy()
        if data.empty:
            data["split"] = []
            return data

        total_ratio = self.train_ratio + self.val_ratio + self.test_ratio
        if total_ratio <= 0:
            raise ValueError("train_ratio + val_ratio + test_ratio must be greater than 0")

        train_ratio = self.train_ratio / total_ratio
        val_ratio = self.val_ratio / total_ratio

        if self.chronological_split:
            time_col = self._resolve_time_column(data, self.time_column)
            if time_col is not None:
                parsed = pd.to_datetime(data[time_col], errors="coerce")
                if parsed.isna().all() and time_col == "year":
                    parsed = pd.to_datetime(
                        data[time_col].astype(str) + "-01-01",
                        errors="coerce",
                    )
                if not parsed.isna().all():
                    data["_parsed_time"] = parsed
                    data = data.sort_values("_parsed_time", kind="stable").drop(
                        columns=["_parsed_time"]
                    )
                else:
                    data = data.sample(frac=1.0, random_state=self.split_seed)
            else:
                data = data.sample(frac=1.0, random_state=self.split_seed)
        else:
            data = data.sample(frac=1.0, random_state=self.split_seed)

        data = data.reset_index(drop=True)
        n_total = len(data)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)

        split_values = np.full(n_total, "test", dtype=object)
        split_values[:n_train] = "train"
        split_values[n_train : n_train + n_val] = "val"
        data["split"] = split_values
        return data

    @staticmethod
    def _resolve_time_column(data: pd.DataFrame, time_column: Optional[str]) -> Optional[str]:
        if time_column is not None and time_column in data.columns:
            return time_column
        for candidate in ["published_date", "commit_date", "date", "year", "timestamp"]:
            if candidate in data.columns:
                return candidate
        return None

    def get_class_distribution(self) -> Dict[int, int]:
        labels = [sample.get("is_vulnerable", 0) for sample in self.samples]
        return {0: labels.count(0), 1: labels.count(1)}


class AdversarialVulnerabilityDataset(Dataset):
    def __init__(
        self,
        base_dataset: VulnerabilityDataset,
        adversarial_transform,
        num_perturbations: int = 3,
    ):
        self.base_dataset = base_dataset
        self.adversarial_transform = adversarial_transform
        self.num_perturbations = num_perturbations

    def __len__(self) -> int:
        return len(self.base_dataset) * self.num_perturbations

    def __getitem__(self, idx: int) -> Dict:
        base_idx = idx // self.num_perturbations
        perturbation_idx = idx % self.num_perturbations
        clean_sample = self.base_dataset[base_idx]
        perturbed_sample, perturbation_type = self.adversarial_transform(clean_sample, perturbation_idx)
        return {
            "clean_sample": clean_sample,
            "perturbed_sample": perturbed_sample,
            "perturbation_type": perturbation_type,
        }


class MultiDatasetVulnerabilityDataset(Dataset):
    def __init__(
        self,
        dataset_names: List[str],
        split: str = "train",
        max_nodes: int = 10000,
        node_feature_dim: int = 768,
        balance_datasets: bool = True,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        split_seed: int = 42,
        chronological_split: bool = False,
        time_column: Optional[str] = None,
    ):
        self.datasets = [
            VulnerabilityDataset(
                dataset_name=dataset_name,
                split=split,
                max_nodes=max_nodes,
                node_feature_dim=node_feature_dim,
                train_ratio=train_ratio,
                val_ratio=val_ratio,
                test_ratio=test_ratio,
                split_seed=split_seed,
                chronological_split=chronological_split,
                time_column=time_column,
            )
            for dataset_name in dataset_names
        ]
        self.dataset_names = dataset_names
        self.balance_datasets = balance_datasets
        self._create_index_mapping()

    def _create_index_mapping(self) -> None:
        self.index_mapping = []
        if not self.datasets:
            return
        if self.balance_datasets:
            min_size = min(len(dataset) for dataset in self.datasets)
            for dataset_idx, _dataset in enumerate(self.datasets):
                for sample_idx in range(min_size):
                    self.index_mapping.append((dataset_idx, sample_idx))
        else:
            for dataset_idx, dataset in enumerate(self.datasets):
                for sample_idx in range(len(dataset)):
                    self.index_mapping.append((dataset_idx, sample_idx))

    def __len__(self) -> int:
        return len(self.index_mapping)

    def __getitem__(self, idx: int) -> Dict:
        dataset_idx, sample_idx = self.index_mapping[idx]
        sample = self.datasets[dataset_idx][sample_idx]
        sample["source_dataset"] = self.dataset_names[dataset_idx]
        return sample


def graph_batch_collate(samples: List[Dict]) -> Dict:
    if len(samples) == 0:
        return {}

    graph_x_parts = []
    edge_index_parts = []
    edge_attr_parts = []
    graph_batch_parts = []
    labels = []
    metadata = []
    sample_indices = []
    source_datasets = []
    node_offset = 0

    for graph_id, sample in enumerate(samples):
        graph_x = sample["graph_x"]
        edge_index = sample["graph_edge_index"]
        edge_attr = sample.get("graph_edge_attr")

        if edge_index.dim() != 2 or edge_index.size(0) != 2:
            raise ValueError("graph_edge_index must have shape (2, num_edges)")

        graph_x_parts.append(graph_x)
        graph_batch_parts.append(
            torch.full(
                (graph_x.size(0),),
                graph_id,
                dtype=torch.long,
                device=graph_x.device,
            )
        )

        if edge_index.numel() > 0:
            edge_index_parts.append(edge_index + node_offset)
        else:
            edge_index_parts.append(torch.zeros((2, 0), dtype=torch.long, device=edge_index.device))

        if edge_attr is not None:
            if edge_attr.dim() == 1:
                edge_attr = edge_attr.unsqueeze(1)
            edge_attr_parts.append(edge_attr)

        labels.append(sample["label"])
        metadata.append(sample.get("metadata"))
        sample_indices.append(int(sample.get("sample_idx", -1)))
        source_datasets.append(sample.get("source_dataset"))
        node_offset += graph_x.size(0)

    batched_graph_x = torch.cat(graph_x_parts, dim=0)
    batched_graph_batch = torch.cat(graph_batch_parts, dim=0)
    batched_edge_index = (
        torch.cat(edge_index_parts, dim=1)
        if edge_index_parts
        else torch.zeros((2, 0), dtype=torch.long, device=batched_graph_x.device)
    )

    if edge_attr_parts:
        batched_edge_attr = torch.cat(edge_attr_parts, dim=0)
    else:
        batched_edge_attr = torch.zeros(
            (batched_edge_index.size(1), 0),
            dtype=batched_graph_x.dtype,
            device=batched_graph_x.device,
        )

    batched_labels = torch.stack(labels, dim=0).view(-1)
    return {
        "graph_x": batched_graph_x,
        "graph_edge_index": batched_edge_index,
        "graph_edge_attr": batched_edge_attr,
        "graph_batch": batched_graph_batch,
        "label": batched_labels,
        "metadata": metadata,
        "sample_idx": sample_indices,
        "source_dataset": source_datasets,
    }
