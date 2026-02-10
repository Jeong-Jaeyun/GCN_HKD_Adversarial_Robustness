# GCN-HKD Adversarial Robustness for Vulnerability Detection

A professional-grade deep learning system for detecting software vulnerabilities with adversarial robustness, targeting Q1 tier conferences (NDSS, CCS, Usenix Security, etc.).

## Overview

This system implements a **three-phase architecture** for robust vulnerability detection:

### **Phase 1: Adversarial Transformation Module**
- Variable renaming, dead code insertion, control flow flattening
- Semantic-preserving code transformations
- Creates adversarially perturbed training samples

### **Phase 2: Robust Feature Extraction (Graph Attention Network)**
- **CFG+DFG** hybrid graphs (Control Flow + Data Flow)
- **Graph Attention Network (GAT)** with multi-head attention
- Learns to focus on semantic edges while ignoring adversarial noise
- Better than GCN for robustness (attention-based)

### **Phase 3: Hierarchical Knowledge Distillation**
- **Teacher Model**: Trained on clean source code graphs
- **Student Model**: Trained on perturbed code graphs
- **Knowledge Transfer**: Student learns high-level vulnerability patterns from teacher
- Makes predictions robust despite code transformations

---

## Project Structure

```
GCN_HKD_Adversarial_Robustness/
│
├── config/                    # Configuration management
│   ├── config.py             # All hyperparameters (dataclass-based)
│   └── paths.py              # Path management
│
├── data/                      # Data loading and preprocessing
│   ├── loaders.py            # SARD, BigVul, D-Sieve dataset loaders
│   ├── preprocessors.py      # CFG/DFG graph construction
│   └── datasets.py           # PyTorch Dataset classes
│
├── models/                    # Deep learning models
│   ├── base.py              # Base model class
│   ├── gat.py               # Graph Attention Network (main feature extractor)
│   ├── teacher.py           # Teacher model (clean code)
│   └── student.py           # Student model (perturbed code)
│
├── adversarial/              # Phase 1: Adversarial transformations
│   ├── transformations.py   # Code transformations (variable renaming, dead code, etc.)
│   └── perturbations.py     # Graph-level perturbations (node/edge attacks)
│
├── distillation/             # Phase 3: Knowledge distillation
│   ├── hkd.py               # Hierarchical knowledge distillation orchestrator
│   └── losses.py            # Distillation losses (task + KD + robustness)
│
├── training/                 # Training orchestration
│   ├── trainer.py           # Main training loop with KD
│   ├── validator.py         # Validation utilities
│   └── callbacks.py         # Checkpointing, logging, evaluation callbacks
│
├── evaluation/               # Robustness evaluation
│   ├── metrics.py           # Accuracy, precision, recall, F1, ROC-AUC, PR-AUC
│   └── robustness.py        # Adversarial robustness evaluation
│
├── utils/                    # Utility functions
│   ├── logger.py            # Logging setup
│   ├── graph_utils.py       # Graph conversion and normalization
│   └── visualization.py     # Plots and figures
│
├── main.py                   # Main entry point (orchestrates entire pipeline)
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

---

## Key Features

### 📊 **Multi-Dataset Support**
- **SARD (NIST)**: Diverse CWE categories for learning CWE hierarchy
- **BigVul (CVE)**: Real vulnerabilities from open-source projects (C/C++)
- **D-Sieve**: Binary robustness testing across optimization levels

### 🛡️ **Adversarial Robustness**
- Evaluates robustness to:
  - **Node feature perturbations** (input noise)
  - **Edge perturbations** (structural changes)
  - **Structural transformations** (code obfuscation)
- Certified robustness evaluation (Randomized smoothing ready)

### 🎓 **Hierarchical Knowledge Distillation**
- Multi-level feature transfer
- Temperature-scaled softmax for soft targets
- Layer-wise distillation loss

### 📈 **Comprehensive Evaluation**
- Accuracy, Precision, Recall, F1, MCC
- ROC-AUC and PR-AUC curves
- Robustness curves across perturbation budgets
- Per-class metrics

---

## Installation

### Prerequisites
- Python 3.10+
- CUDA 11.8+ (for GPU support)

### Setup
```bash
# Clone repository
cd GCN_HKD_Adversarial_Robustness

# Install dependencies
pip install -r requirements.txt

# Optional: For binary analysis
# pip install capstone pwntools
```

---

## Quick Start

### 1. Prepare Datasets
```bash
# Download and organize datasets
mkdir -p data/raw/{SARD,BigVul,D-Sieve}

# Place data following the structure expected by loaders:
# data/raw/SARD/
#   ├── metadata.json
#   ├── labels.csv
#   └── source_code/
#
# data/raw/BigVul/
#   ├── commits.json
#   ├── labels.csv
#   └── source/
```

### 2. Run Training Pipeline
```bash
# Default configuration
python main.py --output-dir ./results

# With custom parameters
python main.py \
    --output-dir ./results \
    --device cuda \
    --num-epochs 100 \
    --batch-size 32

# With custom config file
python main.py --config config_custom.yaml
```

### 3. Evaluate Robustness
```python
from evaluation.robustness import RobustnessEvaluator
from torch.utils.data import DataLoader

evaluator = RobustnessEvaluator(device='cuda')
results = evaluator.evaluate_robustness(
    model=student_model,
    test_loader=test_loader,
    perturbation_budgets=[0.05, 0.1, 0.15, 0.2, 0.3]
)
```

---

## Training Pipeline

### Phase 0: Data Loading
```python
from data.datasets import MultiDatasetVulnerabilityDataset

train_dataset = MultiDatasetVulnerabilityDataset(
    dataset_names=['SARD', 'BigVul'],
    split='train'
)
```

### Phase 1: Initialize Models
```python
from models.teacher import TeacherModel
from models.student import StudentModel

teacher = TeacherModel(input_dim=768, hidden_dim=512, num_layers=4)
student = StudentModel(input_dim=768, hidden_dim=256, num_layers=3)
```

### Phase 2: Knowledge Distillation Training
```python
from distillation.hkd import HierarchicalKnowledgeDistillation
from training.trainer import Trainer
from config.config import Config

config = Config()
trainer = Trainer(config, teacher, student, device='cuda')

# Training loop
for epoch in range(config.training.num_epochs):
    metrics = trainer.train_epoch(train_loader, val_loader)
    if trainer.should_stop():
        break
```

### Phase 3: Adversarial Robustness Evaluation
```python
from evaluation.robustness import RobustnessEvaluator

evaluator = RobustnessEvaluator()
results = evaluator.evaluate_robustness(
    student,
    test_loader,
    perturbation_budgets=[0.05, 0.1, 0.15, 0.2]
)
```

---

## Configuration

All hyperparameters are managed in `config/config.py`:

```python
from config.config import Config

config = Config(
    # Data
    data=DataConfig(datasets=['SARD', 'BigVul']),
    
    # Adversarial
    adversarial=AdversarialConfig(
        transformations=['variable_renaming', 'dead_code_insertion'],
        variable_rename_prob=0.3,
        dead_code_ratio=0.2
    ),
    
    # Model
    model=ModelConfig(
        num_gat_layers=3,
        num_attention_heads=8,
        temperature=4.0
    ),
    
    # Training
    training=TrainingConfig(
        optimizer='adamw',
        learning_rate=1e-3,
        num_epochs=100
    )
)
```

Or load from YAML:
```python
config = Config.from_yaml('config.yaml')
```

---

## Datasets for Q1 Journal Submission

### Recommended Evaluation
1. **SARD**: Synthetic, well-organized, good for CWE hierarchy learning
2. **BigVul**: Real CVEs, proves "works in real-world"
3. **D-Sieve + Real code**: Test on Linux Kernel / OpenSSL to show robustness

### Tips for Q1 Acceptance
✅ Include direct adversarial modifications to Linux Kernel / OpenSSL code
✅ Show robustness curves with multiple attack types
✅ Compare against baselines (GCN, CNN-based detectors)
✅ Ablation study: impact of each phase (Phase 1, 2, 3)
✅ Statistical significance testing

---

## Example: Real-World CVE Evaluation

```python
import subprocess

# Download and analyze Linux Kernel source
subprocess.run(['git', 'clone', 'https://github.com/torvalds/linux.git'])

# Apply adversarial transformations
from adversarial.transformations import AdversarialTransformer

transformer = AdversarialTransformer()

with open('linux/drivers/usb/core/hub.c', 'r') as f:
    original_code = f.read()

# Transform with variable renaming
result = transformer.transform(original_code, 'variable_renaming')

# Evaluate detection
logits, probs = student_model(graph_x, edge_index, edge_attr)
prediction = probs.argmax() # 0=safe, 1=vulnerable
```

---

## Module Documentation

### `config/`
- **`config.py`**: Centralized hyperparameter management
- **`paths.py`**: Path handling for data, models, logs

### `data/`
- **`loaders.py`**: Dataset loaders for SARD, BigVul, D-Sieve
- **`preprocessors.py`**: CFG and DFG construction from code
- **`datasets.py`**: PyTorch Dataset and DataLoader integration

### `models/`
- **`gat.py`**: Graph Attention Network (Phase 2 core)
- **`teacher.py`**: Teacher model trained on clean code
- **`student.py`**: Student model trained with knowledge distillation

### `adversarial/`
- **`transformations.py`**: Code-level transformations (Phase 1)
- **`perturbations.py`**: Graph-level perturbation attacks

### `distillation/`
- **`hkd.py`**: Knowledge distillation orchestrator (Phase 3)
- **`losses.py`**: Multi-level distillation losses

### `training/`
- **`trainer.py`**: Main training loop with distillation
- **`validator.py`**: Model evaluation utilities
- **`callbacks.py`**: Training callbacks

### `evaluation/`
- **`metrics.py`**: Comprehensive metric computation
- **`robustness.py`**: Adversarial robustness evaluation

---

## Citation

If you use this system, please cite:

```bibtex
@software{gcn_hkd_2024,
  title={GCN-HKD: Graph Neural Networks with Hierarchical Knowledge Distillation for Adversarially Robust Vulnerability Detection},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/GCN_HKD_Adversarial_Robustness}
}
```

---

## Future Work

- [ ] Binary code analysis module (using Ghidra/IDA APIs)
- [ ] More sophisticated CFG extraction (using LLVM IR)
- [ ] Randomized smoothing for certified robustness
- [ ] Graph neural network baselines (GraphSAGE, GIN)
- [ ] Attention visualization tools
- [ ] Real-time detection server

---

## License

MIT License - See LICENSE file

---

## Contact

For questions or collaborations, please open an issue or contact the authors.
