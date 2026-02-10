# Project Structure Summary

Complete overview of the GCN-HKD Adversarial Robustness system architecture.

```
GCN_HKD_Adversarial_Robustness/
│
├──  config/                              # Configuration Management
│   ├── __init__.py
│   ├── config.py                           # [CORE] Hyperparameter management (dataclass-based)
│   └── paths.py                            # [CORE] Path management for datasets, models, logs
│
├──  data/                                # Data Loading & Preprocessing
│   ├── __init__.py
│   ├── loaders.py                          # [CORE] Dataset loaders (SARD, BigVul, D-Sieve)
│   ├── preprocessors.py                    # [CORE] CFG/DFG graph construction
│   └── datasets.py                         # [CORE] PyTorch Dataset classes
│
├──  models/                              # Deep Learning Models
│   ├── __init__.py
│   ├── base.py                             # [CORE] Base model class
│   ├── gat.py                              # [CORE] Graph Attention Network (Phase 2)
│   ├── teacher.py                          # [CORE] Teacher model (clean code)
│   └── student.py                          # [CORE] Student model (perturbed code)
│
├──  adversarial/                         # Phase 1: Adversarial Transformations
│   ├── __init__.py
│   ├── transformations.py                  # [CORE] Code transformations (variable renaming, dead code, CFG flattening)
│   └── perturbations.py                    # [CORE] Graph-level attacks (node/edge perturbations)
│
├──  distillation/                        # Phase 3: Knowledge Distillation
│   ├── __init__.py
│   ├── hkd.py                              # [CORE] Hierarchical knowledge distillation orchestrator
│   └── losses.py                           # [CORE] Distillation losses (task + KD + robustness)
│
├──  training/                            # Training Orchestration
│   ├── __init__.py
│   ├── trainer.py                          # [CORE] Main training loop with KD
│   ├── validator.py                        # [CORE] Validation utilities
│   └── callbacks.py                        # [CORE] Checkpointing, logging callbacks
│
├──  evaluation/                          # Robustness Evaluation
│   ├── __init__.py
│   ├── metrics.py                          # [CORE] Comprehensive metrics (Acc, P, R, F1, AUC-ROC, AUC-PR)
│   └── robustness.py                       # [CORE] Adversarial robustness evaluation
│
├──  utils/                               # Utility Functions
│   ├── __init__.py
│   ├── logger.py                           # Logging setup
│   ├── graph_utils.py                      # Graph conversion and normalization
│   └── visualization.py                    # Plotting utilities (robustness curves, training history)
│
├──  main.py                              # [ENTRY POINT] Main pipeline orchestrator
├──  example_usage.py                     # Usage examples and demonstrations
├──  requirements.txt                     # Python dependencies
├──  config_example.yaml                  # Example configuration file
├──  README.md                            # Project overview and quick start
├──  USAGE_GUIDE.md                       # Comprehensive usage guide
├──  .gitignore                           # Git ignore patterns
└──  PROJECT_STRUCTURE.md                 # This file

```

---

## Module Relationships

```
┌─────────────────────────────────────────────────────────────┐
│                       MAIN.PY (Entry Point)                  │
└────────────────┬────────────────────────────────────────────┘
                 │
        ┌────────┴────────┐
        │                 │
        ▼                 ▼
    ┌────────┐        ┌──────────┐
    │ Config │        │ PathCfg  │
    └────────┘        └──────────┘
        │                 │
        └────────┬────────┘
                 │
    ┌────────────▼──────────────┐
    │   Phase 0: Data Loading   │
    ├───────────────────────────┤
    │ - DataLoader (SARD, etc)  │
    │ - GraphConstructor        │
    │ - VulnerabilityDataset    │
    └────────────┬──────────────┘
                 │
    ┌────────────▼──────────────────┐
    │  Phase 1: Transformations     │
    ├────────────────────────────────┤
    │ - AdversarialTransformer       │
    │ - Code mutations (adversarial) │
    └────────────┬───────────────────┘
                 │
    ┌────────────▼──────────────┐
    │  Phase 2: Feature Extract  │
    ├───────────────────────────┤
    │ - GAT (Graph Attention)   │
    │ - TeacherModel            │
    │ - StudentModel            │
    └────────────┬──────────────┘
                 │
    ┌────────────▼──────────────────────┐
    │  Phase 3: Knowledge Distillation  │
    ├──────────────────────────────────┤
    │ - HierarchicalKD                  │
    │ - DistillationLoss                │
    │ - Feature-level + Logit-level KD  │
    └────────────┬──────────────────────┘
                 │
    ┌────────────▼──────────────────┐
    │    Training                    │
    ├──────────────────────────────┤
    │ - Trainer (main training loop)│
    │ - Callbacks (checkpointing)   │
    │ - Validation                  │
    └────────────┬──────────────────┘
                 │
    ┌────────────▼──────────────────┐
    │   Evaluation                   │
    ├──────────────────────────────┤
    │ - RobustnessEvaluator        │
    │ - MetricsComputer            │
    │ - Adversarial attacks        │
    └──────────────────────────────┘
```

---

## Data Flow

```
Raw Code (Source/Binary)
        │
        ▼
Graph Construction (CFG + DFG)
        │
    ┌───┴───┐
    │       │
    ▼       ▼
 Clean   Adversarial
  Code   Transform
    │       │
    └───┬───┘
        │
    ┌───▼────┐
    │ Graphs │
    └───┬────┘
        │
    ┌───┴───────────┐
    │               │
    ▼               ▼
  Teacher        Student
  Model          Model
    │               │
    └───┬───────────┘
        │
     Knowledge
    Distillation
        │
    ┌───▼──────────┐
    │ Trained      │
    │ Student      │
    │ Model        │
    └───┬──────────┘
        │
    ┌───▼────────────────┐
    │    Evaluation      │
    ├────────────────────┤
    │ - Clean accuracy   │
    │ - Robustness       │
    │ - Metrics          │
    └────────────────────┘
```

---

## Key Components

### 1. Configuration System (config/)
- **Purpose**: Centralized hyperparameter management
- **Files**: 
  - `config.py`: Dataclass-based configuration with all model/training/data parameters
  - `paths.py`: Automatic path creation for datasets, models, logs
- **Usage**: 
  ```python
  from config.config import Config
  config = Config()
  config = Config.from_yaml('my_config.yaml')
  ```

### 2. Data Processing (data/)
- **Purpose**: Load and preprocess vulnerability datasets
- **Files**:
  - `loaders.py`: Loaders for SARD, BigVul, D-Sieve
  - `preprocessors.py`: CFG and DFG construction from code
  - `datasets.py`: PyTorch Dataset wrappers
- **Output**: Graph-based representations of code (node features + edge indices)

### 3. Models (models/)
- **Purpose**: Neural network architectures for vulnerability detection
- **Files**:
  - `gat.py`: Graph Attention Network (recommended over standard GCN)
  - `teacher.py`: Larger model trained on clean code
  - `student.py`: Smaller model trained via knowledge distillation
- **Key Advantage**: Attention mechanism ignores adversarial noise

### 4. Adversarial Transformations (adversarial/)
- **Purpose**: Phase 1 - Create code variations that preserve semantics
- **Techniques**:
  - Variable renaming
  - Dead code insertion
  - Control flow flattening
- **Effect**: Changes graph structure without changing vulnerability status

### 5. Knowledge Distillation (distillation/)
- **Purpose**: Phase 3 - Transfer knowledge from teacher to student
- **Loss Components**:
  - Task loss (cross-entropy)
  - Logit-level KD loss (soft targets)
  - Feature-level KD loss (intermediate representations)
- **Result**: Student learns semantic features despite graph noise

### 6. Training (training/)
- **Purpose**: Orchestrate training loop with knowledge distillation
- **Components**:
  - Trainer: Main loop with forward/backward passes
  - Validator: Validation on held-out set
  - Callbacks: Checkpointing, early stopping, logging

### 7. Evaluation (evaluation/)
- **Purpose**: Comprehensive robustness and metrics evaluation
- **Metrics**:
  - Standard: Accuracy, Precision, Recall, F1
  - Advanced: AUC-ROC, AUC-PR, Matthews Correlation Coefficient
  - Robustness: Performance across perturbation budgets
- **Attacks**: Node feature, edge, and structural perturbations

### 8. Utilities (utils/)
- **Purpose**: Helper functions for logging, visualization, graph processing
- **Tools**:
  - Logger: Structured logging to file and console
  - GraphUtils: NetworkX to tensor conversion
  - Visualization: Robustness curves, training history, confusion matrices

---

## Execution Flow

### Standard Training
```python
python main.py
```

Flow:
1. Load config
2. Load datasets (SARD, BigVul, etc.)
3. Construct graphs (CFG+DFG)
4. Initialize teacher and student models
5. Training loop (with KD):
   - Forward pass on teacher (no grad)
   - Forward pass on student (with grad)
   - Compute distillation loss
   - Backward and optimize student
6. Evaluate robustness on test set
7. Save results and checkpoints

### Custom Execution
```python
# Full control over pipeline
from config.config import Config
from data.datasets import VulnerabilityDataset
from models.teacher import TeacherModel
from models.student import StudentModel
from training.trainer import Trainer

config = Config()
train_dataset = VulnerabilityDataset('SARD', split='train')
teacher = TeacherModel(input_dim=768, ...)
student = StudentModel(input_dim=768, ...)
trainer = Trainer(config, teacher, student)

for epoch in range(config.training.num_epochs):
    trainer.train_epoch(train_loader, val_loader)
```

---

## File Dependencies

```
main.py
  ├─ config/config.py
  ├─ data/loaders.py
  ├─ data/preprocessors.py
  ├─ data/datasets.py
  ├─ models/teacher.py
  │   └─ models/gat.py
  ├─ models/student.py
  │   └─ models/gat.py
  ├─ training/trainer.py
  │   ├─ training/validator.py
  │   ├─ distillation/hkd.py
  │   │   └─ distillation/losses.py
  │   └─ training/callbacks.py
  ├─ evaluation/robustness.py
  │   ├─ evaluation/metrics.py
  │   └─ adversarial/perturbations.py
  └─ utils/logger.py
```

---

## Dataset Format Requirements

### SARD
```
data/raw/SARD/
├── metadata.json (CWE info)
├── labels.csv (file, cwe_id, is_vulnerable)
└── source_code/ (C/C++ files)
```

### BigVul
```
data/raw/BigVul/
├── commits.json (CVE metadata)
├── labels.csv (commit_hash, cve_id, is_vulnerable)
└── source/ (C/C++ files)
```

### D-Sieve
```
data/raw/D-Sieve/
├── source/ (original C/C++)
├── binaries/ (O0, O1, O2, O3)
├── obfuscated/ (obfuscated binaries)
└── labels.csv
```

---

## Configuration Hierarchy

```
Default Config (in code)
        ↓
Config File (YAML)
        ↓
Command Line Arguments
        ↓
Final Configuration
```

Priority: Command line > YAML > Default

---

## Important Classes

### Core Models
- `TeacherModel`: Larger model for clean code (teacher role)
- `StudentModel`: Smaller model for perturbed code (student role)
- `GraphAttentionNetwork`: GAT feature extractor (Phase 2)

### Data Classes
- `VulnerabilityDataset`: Single-dataset PyTorch dataset
- `MultiDatasetVulnerabilityDataset`: Multi-dataset support

### Training Classes
- `Trainer`: Main training loop with KD
- `HierarchicalKnowledgeDistillation`: KD orchestrator

### Evaluation Classes
- `RobustnessEvaluator`: Adversarial robustness testing
- `MetricsComputer`: Metric computation

### Adversarial Classes
- `AdversarialTransformer`: Code transformations
- `AdversarialAttackGenerator`: Graph perturbations

---

## Testing & Validation

Run example usage:
```bash
python example_usage.py
```

This demonstrates:
- Phase 1: Code transformations
- Phase 2: Model initialization
- Phase 3: Knowledge distillation
- Robustness evaluation

---

## Publication Checklist for Q1 Journals

 Three phases clearly separated
 Multiple datasets (SARD, BigVul, D-Sieve)
 Real-world validation (Linux Kernel, OpenSSL)
 Comprehensive metrics (accuracy, precision, recall, F1, AUC-ROC, AUC-PR)
 Adversarial robustness evaluation
 Comparison with baselines (GCN, CNN-based)
 Ablation studies
 Code and data reproducibility
 Detailed architecture documentation
 Professional code structure and documentation

---

## Future Extension Points

1. **Binary Analysis Module**: Add Ghidra/IDA API support
2. **LLVM Passes**: Better CFG extraction
3. **Additional Baselines**: GraphSAGE, GIN comparisons
4. **Certified Robustness**: Randomized smoothing
5. **Attention Visualization**: Interpretability tools
6. **Real-time Detection**: Server deployment

---

For detailed usage, see [USAGE_GUIDE.md](USAGE_GUIDE.md)
For quick start, see [README.md](README.md)
