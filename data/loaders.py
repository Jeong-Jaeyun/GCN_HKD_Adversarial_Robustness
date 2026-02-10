"""Data loader for various vulnerability datasets (SARD, BigVul, D-Sieve)."""

from typing import List, Dict, Tuple, Optional
from pathlib import Path
import json
import pandas as pd
from abc import ABC, abstractmethod
from config.paths import PathConfig


class BaseDataLoader(ABC):
    """Abstract base class for data loading."""
    
    def __init__(self, data_dir: Path):
        """Initialize data loader.
        
        Args:
            data_dir: Directory containing dataset files
        """
        self.data_dir = Path(data_dir)
        self.data = None
    
    @abstractmethod
    def load(self) -> pd.DataFrame:
        """Load dataset."""
        pass
    
    @abstractmethod
    def parse_sample(self, sample: Dict) -> Tuple[str, int]:
        """Parse a single sample into (code, label).
        
        Returns:
            Tuple of (code_str, vulnerability_label: 1/0)
        """
        pass


class SARDDataLoader(BaseDataLoader):
    """Loader for NIST SARD (Software Assurance Reference Dataset).
    
    SARD dataset contains diverse CWE categories with structured metadata.
    Suitable for learning CWE hierarchical structure.
    
    Format expected:
    - metadata.json: CWE info
    - source_code/: C/C++ source files
    - labels.csv: vulnerability labels per file
    """
    
    def load(self) -> pd.DataFrame:
        """Load SARD dataset."""
        if self.data is not None:
            return self.data
        
        # Load metadata
        metadata_path = self.data_dir / "metadata.json"
        labels_path = self.data_dir / "labels.csv"
        
        if not metadata_path.exists() or not labels_path.exists():
            raise FileNotFoundError(
                f"SARD dataset not found at {self.data_dir}. "
                "Expected: metadata.json, labels.csv"
            )
        
        # Load labels
        self.data = pd.read_csv(labels_path)
        
        # Enrich with CWE information
        with open(metadata_path, 'r') as f:
            cwe_info = json.load(f)
        
        self.data['cwe_id'] = self.data['cwe_id'].apply(
            lambda x: cwe_info.get(str(x), {})
        )
        
        return self.data
    
    def parse_sample(self, sample: Dict) -> Tuple[str, int, Dict]:
        """Parse SARD sample.
        
        Returns:
            Tuple of (code_str, vulnerability_label, cwe_info)
        """
        file_path = self.data_dir / "source_code" / sample['filename']
        
        if not file_path.exists():
            raise FileNotFoundError(f"Source file not found: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            code = f.read()
        
        # vulnerability_label: 1 for vulnerable, 0 for safe
        label = int(sample['is_vulnerable'])
        cwe_info = sample.get('cwe_id', {})
        
        return code, label, cwe_info
    
    def get_cwe_categories(self) -> Dict[int, str]:
        """Get mapping of CWE ID to description."""
        with open(self.data_dir / "metadata.json", 'r') as f:
            metadata = json.load(f)
        return metadata


class BigVulDataLoader(BaseDataLoader):
    """Loader for BigVul dataset from real CVEs.
    
    BigVul contains real vulnerabilities extracted from open-source projects
    (C/C++ code). Each sample includes commit hash and CVE ID.
    
    Format expected:
    - commits.json: List of commits with CVE info
    - source/: C/C++ source files
    - labels.csv: Vulnerability labels
    """
    
    def load(self) -> pd.DataFrame:
        """Load BigVul dataset."""
        if self.data is not None:
            return self.data
        
        commits_path = self.data_dir / "commits.json"
        labels_path = self.data_dir / "labels.csv"
        
        if not commits_path.exists() or not labels_path.exists():
            raise FileNotFoundError(
                f"BigVul dataset not found at {self.data_dir}. "
                "Expected: commits.json, labels.csv"
            )
        
        # Load CVE information
        with open(commits_path, 'r') as f:
            commits = json.load(f)
        
        # Load labels
        self.data = pd.read_csv(labels_path)
        
        # Merge CVE information
        commits_df = pd.DataFrame(commits)
        self.data = self.data.merge(commits_df, on='commit_hash', how='left')
        
        return self.data
    
    def parse_sample(self, sample: Dict) -> Tuple[str, int, Dict]:
        """Parse BigVul sample.
        
        Returns:
            Tuple of (code_str, vulnerability_label, cve_info)
        """
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
        """Get list of projects in dataset."""
        commits_path = self.data_dir / "commits.json"
        with open(commits_path, 'r') as f:
            commits = json.load(f)
        return list(set(c.get('project') for c in commits))


class DSieveDataLoader(BaseDataLoader):
    """Loader for D-Sieve dataset (binary obfuscation robustness).
    
    D-Sieve contains the same code compiled with different optimization levels
    and obfuscation techniques. Used to test robustness.
    
    Format expected:
    - source/: Original C/C++ source files
    - binaries/: Compiled binaries with different optimization levels
    - obfuscated/: Obfuscated binaries
    - labels.csv: Labels
    """
    
    def load(self) -> pd.DataFrame:
        """Load D-Sieve dataset."""
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
        """Parse D-Sieve sample.
        
        Returns:
            Tuple of (binary_content, vulnerability_label, binary_info)
        """
        binary_path = self.data_dir / "binaries" / sample['binary_filename']
        
        if not binary_path.exists():
            raise FileNotFoundError(f"Binary file not found: {binary_path}")
        
        # Read binary as bytes (for disassembly later)
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
        """Get list of optimization levels in dataset."""
        if self.data is None:
            self.load()
        return list(self.data['optimization_level'].unique())


class DataLoaderFactory:
    """Factory for creating appropriate data loaders."""
    
    _loaders = {
        'SARD': SARDDataLoader,
        'BigVul': BigVulDataLoader,
        'D-Sieve': DSieveDataLoader,
    }
    
    @classmethod
    def create(cls, dataset_name: str, data_dir: Optional[Path] = None) -> BaseDataLoader:
        """Create data loader for specified dataset.
        
        Args:
            dataset_name: Name of dataset ('SARD', 'BigVul', 'D-Sieve')
            data_dir: Path to dataset directory
            
        Returns:
            Appropriate data loader instance
        """
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
        """Register a custom data loader."""
        cls._loaders[name] = loader_class


# Example usage function
def load_multimodal_dataset(
    dataset_names: List[str],
    split_ratio: Tuple[float, float, float] = (0.7, 0.15, 0.15)
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load and combine multiple datasets.
    
    Args:
        dataset_names: List of dataset names to load
        split_ratio: (train_ratio, val_ratio, test_ratio)
        
    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    all_data = []
    
    for dataset_name in dataset_names:
        loader = DataLoaderFactory.create(dataset_name)
        data = loader.load()
        data['dataset_source'] = dataset_name
        all_data.append(data)
    
    combined_data = pd.concat(all_data, ignore_index=True)
    
    # Split data
    n_total = len(combined_data)
    n_train = int(n_total * split_ratio[0])
    n_val = int(n_total * split_ratio[1])
    
    train_df = combined_data[:n_train]
    val_df = combined_data[n_train:n_train+n_val]
    test_df = combined_data[n_train+n_val:]
    
    return train_df, val_df, test_df
