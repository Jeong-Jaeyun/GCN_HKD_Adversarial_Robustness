
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional
import yaml


@dataclass
class DataConfig:


    datasets: List[str] = field(default_factory=lambda: ["SARD", "BigVul"])


    graph_types: List[str] = field(default_factory=lambda: ["CFG", "DFG"])
    max_nodes: int = 10000
    node_feature_dims: int = 768


    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15


    batch_size: int = 32
    num_workers: int = 4
    prefetch_factor: int = 2


@dataclass
class AdversarialConfig:


    transformations: List[str] = field(
        default_factory=lambda: [
            "variable_renaming",
            "dead_code_insertion",
            "control_flow_flattening"
        ]
    )


    variable_rename_prob: float = 0.3
    dead_code_ratio: float = 0.2
    control_flow_complexity: float = 0.4


    perturbation_budget: float = 0.1
    num_perturbation_samples: int = 5


@dataclass
class ModelConfig:


    num_gat_layers: int = 3
    num_attention_heads: int = 8
    hidden_dim: int = 256
    output_dim: int = 128
    dropout: float = 0.2


    temperature: float = 4.0
    alpha_kd: float = 0.5


    teacher_hidden_dim: int = 512
    teacher_num_layers: int = 4


    student_hidden_dim: int = 256
    student_num_layers: int = 3


@dataclass
class TrainingConfig:


    optimizer: str = "adamw"
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    warmup_steps: int = 1000


    scheduler: str = "cosine"
    num_epochs: int = 100
    patience: int = 15


    vulnerability_loss_weight: float = 1.0
    kd_loss_weight: float = 0.5
    robustness_loss_weight: float = 0.3


    gradient_clip: float = 1.0
    accumulation_steps: int = 4
    mixed_precision: bool = True


    eval_every_n_steps: int = 100
    save_best: bool = True


@dataclass
class EvaluationConfig:


    metrics: List[str] = field(
        default_factory=lambda: [
            "accuracy", "precision", "recall", "f1",
            "auc_roc", "auc_pr"
        ]
    )


    evaluate_robustness: bool = True
    perturbation_levels: List[float] = field(
        default_factory=lambda: [0.05, 0.1, 0.15, 0.2, 0.3]
    )


    test_on_real_cves: bool = True
    test_on_kernel_code: bool = True
    test_on_openssl: bool = True


@dataclass
class Config:


    data: DataConfig = field(default_factory=DataConfig)
    adversarial: AdversarialConfig = field(default_factory=AdversarialConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)


    seed: int = 42
    device: str = "cuda"
    distributed: bool = False
    num_gpus: int = 1


    log_level: str = "INFO"
    log_interval: int = 10

    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'Config':
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f) or {}

        if not isinstance(config_dict, dict):
            raise ValueError("YAML config must be a mapping at the top level.")

        return cls(
            data=DataConfig(**config_dict.get('data', {})),
            adversarial=AdversarialConfig(**config_dict.get('adversarial', {})),
            model=ModelConfig(**config_dict.get('model', {})),
            training=TrainingConfig(**config_dict.get('training', {})),
            evaluation=EvaluationConfig(**config_dict.get('evaluation', {})),
            seed=config_dict.get('seed', 42),
            device=config_dict.get('device', 'cuda'),
            distributed=config_dict.get('distributed', False),
            num_gpus=config_dict.get('num_gpus', 1),
            log_level=config_dict.get('log_level', 'INFO'),
            log_interval=config_dict.get('log_interval', 10),
        )

    def to_yaml(self, yaml_path: str) -> None:
        with open(yaml_path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)

    def to_dict(self) -> Dict:
        return asdict(self)


default_config = Config()
