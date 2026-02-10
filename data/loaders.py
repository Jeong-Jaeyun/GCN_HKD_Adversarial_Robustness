
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import json
import pandas as pd
from abc import ABC, abstractmethod
from config.paths import PathConfig


class BaseDataLoader(ABC):

    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        self.data = None

    @abstractmethod
    def load(self) -> pd.DataFrame:
        pass

    @abstractmethod
    def parse_sample(self, sample: Dict) -> Tuple[str, int]:
        pass


class SARDDataLoader(BaseDataLoader):

    def load(self) -> pd.DataFrame:
        if self.data is not None:
            return self.data


        metadata_path = self.data_dir / "metadata.json"
        labels_path = self.data_dir / "labels.csv"

        if not metadata_path.exists() or not labels_path.exists():
            raise FileNotFoundError(
                f"SARD dataset not found at {self.data_dir}. "
                "Expected: metadata.json, labels.csv"
            )


        self.data = pd.read_csv(labels_path)


        with open(metadata_path, 'r') as f:
            cwe_info = json.load(f)

        self.data['cwe_id'] = self.data['cwe_id'].apply(
            lambda x: cwe_info.get(str(x), {})
        )

        return self.data

    def parse_sample(self, sample: Dict) -> Tuple[str, int, Dict]:
        file_path = self.data_dir / "source_code" / sample['filename']

        if not file_path.exists():
            raise FileNotFoundError(f"Source file not found: {file_path}")

        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            code = f.read()


        label = int(sample['is_vulnerable'])
        cwe_info = sample.get('cwe_id', {})

        return code, label, cwe_info

    def get_cwe_categories(self) -> Dict[int, str]:
        with open(self.data_dir / "metadata.json", 'r') as f:
            metadata = json.load(f)
        return metadata


class BigVulDataLoader(BaseDataLoader):

    def load(self) -> pd.DataFrame:
        if self.data is not None:
            return self.data

        commits_path = self.data_dir / "commits.json"
        labels_path = self.data_dir / "labels.csv"

        if not commits_path.exists() or not labels_path.exists():
            raise FileNotFoundError(
                f"BigVul dataset not found at {self.data_dir}. "
                "Expected: commits.json, labels.csv"
            )


        with open(commits_path, 'r') as f:
            commits = json.load(f)


        self.data = pd.read_csv(labels_path)


        commits_df = pd.DataFrame(commits)
        self.data = self.data.merge(commits_df, on='commit_hash', how='left')

        return self.data

    def parse_sample(self, sample: Dict) -> Tuple[str, int, Dict]:
        file_path = self.data_dir / "source" / sample['filename']

        if not file_path.exists():
            raise FileNotFoundError(f"Source file not found: {file_path}")

        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            code = f.read()

        label = int(sample['is_vulnerable'])
        cve_info = {
            'cve_id': sample.get('cve_id'),
            'commit_hash': sample.get('commit_hash'),
            'project': sample.get('project')
        }

        return code, label, cve_info

    def get_projects(self) -> List[str]:
        commits_path = self.data_dir / "commits.json"
        with open(commits_path, 'r') as f:
            commits = json.load(f)
        return list(set(c.get('project') for c in commits))


class DSieveDataLoader(BaseDataLoader):

    def load(self) -> pd.DataFrame:
        if self.data is not None:
            return self.data

        labels_path = self.data_dir / "labels.csv"

        if not labels_path.exists():
            raise FileNotFoundError(
                f"D-Sieve dataset not found at {self.data_dir}. "
                "Expected: labels.csv"
            )

        self.data = pd.read_csv(labels_path)
        return self.data

    def parse_sample(self, sample: Dict) -> Tuple[str, int, Dict]:
        binary_path = self.data_dir / "binaries" / sample['binary_filename']

        if not binary_path.exists():
            raise FileNotFoundError(f"Binary file not found: {binary_path}")


        with open(binary_path, 'rb') as f:
            binary_content = f.read()

        label = int(sample['is_vulnerable'])
        binary_info = {
            'optimization_level': sample.get('optimization_level'),
            'obfuscation': sample.get('obfuscation_type'),
            'compiler': sample.get('compiler')
        }

        return binary_content, label, binary_info

    def get_optimization_levels(self) -> List[str]:
        if self.data is None:
            self.load()
        return list(self.data['optimization_level'].unique())


class DataLoaderFactory:

    _loaders = {
        'SARD': SARDDataLoader,
        'BigVul': BigVulDataLoader,
        'D-Sieve': DSieveDataLoader,
    }

    @classmethod
    def create(cls, dataset_name: str, data_dir: Optional[Path] = None) -> BaseDataLoader:
        if dataset_name not in cls._loaders:
            raise ValueError(
                f"Unknown dataset: {dataset_name}. "
                f"Available: {list(cls._loaders.keys())}"
            )

        if data_dir is None:
            paths = PathConfig()
            data_dir = getattr(paths, f"{dataset_name.lower()}_dir")

        loader_class = cls._loaders[dataset_name]
        return loader_class(data_dir)

    @classmethod
    def register(cls, name: str, loader_class: type) -> None:
        cls._loaders[name] = loader_class


def load_multimodal_dataset(
    dataset_names: List[str],
    split_ratio: Tuple[float, float, float] = (0.7, 0.15, 0.15)
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    all_data = []

    for dataset_name in dataset_names:
        loader = DataLoaderFactory.create(dataset_name)
        data = loader.load()
        data['dataset_source'] = dataset_name
        all_data.append(data)

    combined_data = pd.concat(all_data, ignore_index=True)


    n_total = len(combined_data)
    n_train = int(n_total * split_ratio[0])
    n_val = int(n_total * split_ratio[1])

    train_df = combined_data[:n_train]
    val_df = combined_data[n_train:n_train+n_val]
    test_df = combined_data[n_train+n_val:]

    return train_df, val_df, test_df
