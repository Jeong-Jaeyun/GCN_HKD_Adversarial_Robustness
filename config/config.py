"""Global configuration for GCN-HKD Adversarial Robustness system.

This module defines all hyperparameters, model configurations, and training settings
for the vulnerability detection system with adversarial robustness.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import yaml


@dataclass
class DataConfig:
    """Configuration for data loading and preprocessing."""
    
    # Dataset selection
    datasets: List[str] = field(default_factory=lambda: ["SARD", "BigVul"])
    
    # Graph construction
    graph_types: List[str] = field(default_factory=lambda: ["CFG", "DFG"])
    max_nodes: int = 10000
    node_feature_dims: int = 768
    
    # Data splits
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    
    # Preprocessing
    batch_size: int = 32
    num_workers: int = 4
    prefetch_factor: int = 2


@dataclass
class AdversarialConfig:
    """Configuration for adversarial transformation module."""
    
    # Transformation techniques
    transformations: List[str] = field(
        default_factory=lambda: [
            "variable_renaming",
            "dead_code_insertion",
            "control_flow_flattening"
        ]
    )
    
    # Perturbation parameters
    variable_rename_prob: float = 0.3
    dead_code_ratio: float = 0.2
    control_flow_complexity: float = 0.4
    
    # Attack strength (epsilon for adversarial examples)
    perturbation_budget: float = 0.1
    num_perturbation_samples: int = 5


@dataclass
class ModelConfig:
    """Configuration for model architecture."""
    
    # GAT (Graph Attention Network) parameters
    num_gat_layers: int = 3
    num_attention_heads: int = 8
    hidden_dim: int = 256
    output_dim: int = 128
    dropout: float = 0.2
    
    # Knowledge Distillation parameters
    temperature: float = 4.0
    alpha_kd: float = 0.5  # Weight for KD loss vs task loss
    
    # Teacher model (clean code)
    teacher_hidden_dim: int = 512
    teacher_num_layers: int = 4
    
    # Student model (perturbed code)
    student_hidden_dim: int = 256
    student_num_layers: int = 3


@dataclass
class TrainingConfig:
    """Configuration for training procedure."""
    
    # Optimization
    optimizer: str = "adamw"
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    warmup_steps: int = 1000
    
    # Scheduling
    scheduler: str = "cosine"
    num_epochs: int = 100
    patience: int = 15
    
    # Loss weights
    vulnerability_loss_weight: float = 1.0
    kd_loss_weight: float = 0.5
    robustness_loss_weight: float = 0.3
    
    # Training details
    gradient_clip: float = 1.0
    accumulation_steps: int = 4
    mixed_precision: bool = True
    
    # Validation & testing
    eval_every_n_steps: int = 100
    save_best: bool = True


@dataclass  
class EvaluationConfig:
    """Configuration for evaluation and robustness testing."""
    
    # Metrics
    metrics: List[str] = field(
        default_factory=lambda: [
            "accuracy", "precision", "recall", "f1",
            "auc_roc", "auc_pr"
        ]
    )
    
    # Adversarial robustness
    evaluate_robustness: bool = True
    perturbation_levels: List[float] = field(
        default_factory=lambda: [0.05, 0.1, 0.15, 0.2, 0.3]
    )
    
    # Real-world validation
    test_on_real_cves: bool = True
    test_on_kernel_code: bool = True  # Linux kernel validation
    test_on_openssl: bool = True       # OpenSSL validation


@dataclass
class Config:
    """Master configuration class combining all sub-configs."""
    
    # Sub-configurations
    data: DataConfig = field(default_factory=DataConfig)
    adversarial: AdversarialConfig = field(default_factory=AdversarialConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    
    # System
    seed: int = 42
    device: str = "cuda"  # or "cpu"
    distributed: bool = False
    num_gpus: int = 1
    
    # Logging
    log_level: str = "INFO"
    log_interval: int = 10
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'Config':
        """Load configuration from YAML file."""
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)
    
    def to_yaml(self, yaml_path: str) -> None:
        """Save configuration to YAML file."""
        with open(yaml_path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)
    
    def to_dict(self) -> Dict:
        """Convert configuration to dictionary."""
        return {
            'data': self.data.__dict__,
            'adversarial': self.adversarial.__dict__,
            'model': self.model.__dict__,
            'training': self.training.__dict__,
            'evaluation': self.evaluation.__dict__,
            'seed': self.seed,
            'device': self.device,
            'distributed': self.distributed,
            'num_gpus': self.num_gpus,
            'log_level': self.log_level,
            'log_interval': self.log_interval,
        }


# Default configuration instance
default_config = Config()
