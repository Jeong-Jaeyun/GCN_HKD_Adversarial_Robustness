# GCN-HKD Usage Guide

Complete guide for using the GCN-HKD Adversarial Robustness system.

## Table of Contents
1. [Installation](#installation)
2. [Data Preparation](#data-preparation)
3. [Basic Training](#basic-training)
4. [Advanced Configuration](#advanced-configuration)
5. [Evaluation](#evaluation)
6. [Real-World Examples](#real-world-examples)

---

## Installation

### Step 1: Clone Repository
```bash
cd /path/to/workspace
git clone https://github.com/yourusername/GCN_HKD_Adversarial_Robustness.git
cd GCN_HKD_Adversarial_Robustness
```

### Step 2: Create Virtual Environment
```bash
# Using venv
python3.10 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Or using conda
conda create -n gcn-hkd python=3.10
conda activate gcn-hkd
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Verify Installation
```python
# Test imports
python -c "import torch; import torch_geometric; print(' Installation successful')"
```

---

## Data Preparation

### SARD Dataset (NIST)

```bash
# 1. Download from https://samate.nist.gov/SRD/
# 2. Organize structure:

mkdir -p data/raw/SARD
cd data/raw/SARD

# Expected structure:
# SARD/
# ├── metadata.json          # CWE information
# ├── labels.csv             # Vulnerability labels
# └── source_code/           # C/C++ source files
```

**metadata.json format:**
```json
{
  "1": {
    "cwe_id": 79,
    "cwe_name": "Improper Neutralization of Input During Web Page Generation",
    "category": "Injection"
  }
}
```

**labels.csv format:**
```csv
filename,cwe_id,is_vulnerable,split
buffer_overflow_001.c,680,1,train
safe_string_001.c,1,0,train
```

### BigVul Dataset (CVE from Real Projects)

```bash
# 1. Download from https://github.com/sahibor/BigVul
# 2. Organize:

mkdir -p data/raw/BigVul
cd data/raw/BigVul

# Expected structure:
# BigVul/
# ├── commits.json           # CVE and commit metadata
# ├── labels.csv             # Labels
# └── source/                # C/C++ source files
```

**commits.json format:**
```json
[
  {
    "commit_hash": "abc123def456",
    "cve_id": "CVE-2021-1234",
    "project": "openssl",
    "is_vulnerable": 1
  }
]
```

### D-Sieve Dataset

```bash
# Download from: https://github.com/zhangxwww/D-Sieve

mkdir -p data/raw/D-Sieve
cd data/raw/D-Sieve

# Expected structure:
# D-Sieve/
# ├── source/                # Original C/C++ source
# ├── binaries/              # Compiled binaries (O0, O1, O2, O3)
# ├── obfuscated/            # Obfuscated binaries
# └── labels.csv             # Labels
```

---

## Basic Training

### Quick Start (5 minutes)

```bash
# 1. Run with default configuration
python main.py

# 2. Check results
ls results/
cat results/*/training.log
```

### With Custom Parameters

```bash
# Command line arguments
python main.py \
    --output-dir ./my_results \
    --device cuda \
    --num-epochs 50 \
    --batch-size 16
```

### With Custom Config File

```bash
# 1. Create config file
cp config_example.yaml my_config.yaml

# 2. Modify parameters
# Edit my_config.yaml...

# 3. Run with config
python main.py --config my_config.yaml
```

---

## Advanced Configuration

### Customizing Model Architecture

```python
from config.config import Config, ModelConfig
from models.teacher import TeacherModel
from models.student import StudentModel

# Create custom config
config = Config(
    model=ModelConfig(
        num_gat_layers=4,           # More layers for larger models
        num_attention_heads=12,      # More attention heads
        hidden_dim=512,              # Larger hidden dimension
        temperature=2.0,             # Lower temperature = harder targets
        alpha_kd=0.7                 # Higher weight on distillation loss
    )
)

# Initialize models with custom config
teacher = TeacherModel(
    input_dim=768,
    hidden_dim=config.model.teacher_hidden_dim,
    num_layers=config.model.teacher_num_layers
)

student = StudentModel(
    input_dim=768,
    hidden_dim=config.model.student_hidden_dim,
    num_layers=config.model.student_num_layers
)
```

### Custom Adversarial Transformations

```python
from adversarial.transformations import AdversarialTransformer

# More aggressive transformations
transformer = AdversarialTransformer(
    variable_rename_prob=0.8,        # Rename more variables
    dead_code_ratio=0.5,              # Larger dead code blocks
    control_flow_complexity=0.8       # More CF flattening
)

result = transformer.transform(code, 'all')
print(f"Changes: {result.changes_made}")
```

### Multi-Dataset Training

```python
from data.datasets import MultiDatasetVulnerabilityDataset
from torch.utils.data import DataLoader

# Combine multiple datasets
dataset = MultiDatasetVulnerabilityDataset(
    dataset_names=['SARD', 'BigVul', 'D-Sieve'],
    split='train',
    balance_datasets=True  # Balance contributions
)

dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
```

---

## Evaluation

### Adversarial Robustness Testing

```python
from evaluation.robustness import RobustnessEvaluator
from torch.utils.data import DataLoader

# Setup evaluator
evaluator = RobustnessEvaluator(device='cuda')

# Evaluate robustness
results = evaluator.evaluate_robustness(
    model=student_model,
    test_loader=test_loader,
    perturbation_budgets=[0.05, 0.1, 0.15, 0.2, 0.3],
    attack_types=['node_feature', 'edge', 'structural'],
    return_predictions=True
)

# Analyze results
print(f"Clean Accuracy: {results['summary']['clean_accuracy']:.4f}")
print(f"Avg Robust Accuracy: {results['summary']['avg_robustness_accuracy']:.4f}")
print(f"Robustness Gap: {results['summary']['avg_robustness_gap']:.4f}")
```

### Compute Comprehensive Metrics

```python
from evaluation.metrics import MetricsComputer
import numpy as np

metric_computer = MetricsComputer()

# Compute metrics
predictions = np.array([1, 0, 1, 1, 0])
targets = np.array([1, 0, 0, 1, 0])

metrics = metric_computer.compute_metrics(predictions, targets)
print(f"Accuracy: {metrics['accuracy']:.4f}")
print(f"Precision: {metrics['precision']:.4f}")
print(f"Recall: {metrics['recall']:.4f}")
print(f"F1: {metrics['f1']:.4f}")
```

### Generate Plots

```python
from utils.visualization import (
    plot_robustness_curve,
    plot_training_history,
    plot_confusion_matrix
)
from pathlib import Path

# Robustness curves
plot_robustness_curve(
    robustness_results,
    save_path=Path('results/robustness_curve.png')
)

# Training curves
plot_training_history(
    train_history,
    val_history,
    save_path=Path('results/training_history.png')
)

# Confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(targets, predictions)
plot_confusion_matrix(cm, save_path=Path('results/confusion_matrix.png'))
```

---

## Real-World Examples

### Example 1: Testing on Linux Kernel Code

```python
import subprocess
from pathlib import Path
from adversarial.transformations import AdversarialTransformer
from data.preprocessors import HybridGraphConstructor

# Clone Linux source
subprocess.run([
    'git', 'clone', '--depth=1',
    'https://github.com/torvalds/linux.git',
    'data/raw/linux'
])

# Select vulnerable function (example: CVE-2021-XXXXX)
kernel_file = Path('data/raw/linux/drivers/usb/core/hub.c')
with open(kernel_file, 'r') as f:
    original_code = f.read()

# Extract vulnerable function
vulnerable_func = original_code[5000:8000]  # Example range

# Transform with adversarial perturbations
transformer = AdversarialTransformer()
transformed_result = transformer.transform(
    vulnerable_func,
    transformation_type='variable_renaming'
)

# Construct graph
graph_constructor = HybridGraphConstructor()
graph_clean = graph_constructor.construct_from_source(vulnerable_func)
graph_perturbed = graph_constructor.construct_from_source(
    transformed_result.transformed_code
)

# Test model predictions
logits_clean, probs_clean = model(graph_x_clean, edge_index_clean)
logits_pert, probs_pert = model(graph_x_pert, edge_index_pert)

print(f"Clean prediction: {probs_clean.argmax().item()}")
print(f"Perturbed prediction: {probs_pert.argmax().item()}")
print(f"Robust: {probs_clean.argmax() == probs_pert.argmax()}")
```

### Example 2: OpenSSL Vulnerability Detection

```python
# Download OpenSSL source
subprocess.run([
    'git', 'clone', '--depth=1',
    'https://github.com/openssl/openssl.git',
    'data/raw/openssl'
])

# Find specific functions with known CVEs
openssl_files = [
    'data/raw/openssl/ssl/s3_lib.c',
    'data/raw/openssl/crypto/bn/bn_lib.c'
]

results_by_file = {}

for file_path in openssl_files:
    with open(file_path, 'r') as f:
        code = f.read()
    
    # Extract functions and analyze
    # (Use proper AST parsing in production)
    
    graph = graph_constructor.construct_from_source(code)
    logits, probs = model(graph_x, edge_index, edge_attr)
    
    results_by_file[file_path] = {
        'vulnerable_probability': probs[0, 1].item(),
        'prediction': 'VULNERABLE' if probs[0, 1] > 0.5 else 'SAFE'
    }

# Report findings
for file_path, result in results_by_file.items():
    print(f"{file_path}: {result['prediction']} ({result['vulnerable_probability']:.4f})")
```

### Example 3: Comparing Multiple Attack Types

```python
from adversarial.perturbations import AdversarialAttackGenerator
from evaluation.robustness import RobustnessEvaluator

# Setup
attack_gen = AdversarialAttackGenerator(
    perturbation_types=['node_feature', 'edge', 'structural']
)

evaluator = RobustnessEvaluator()

# Evaluate each attack type
attack_results = {}

for attack_type in ['node_feature', 'edge', 'structural']:
    results, _ = evaluator._evaluate_attack(
        model=student_model,
        test_loader=test_loader,
        attack_type=attack_type,
        perturbation_budget=0.2
    )
    attack_results[attack_type] = results

# Compare robustness
print("Attack Type Comparison:")
print("=" * 60)
for attack_type, metrics in attack_results.items():
    print(f"\n{attack_type.upper()}")
    print(f"  Clean:     {metrics['accuracy_clean']:.4f}")
    print(f"  Perturbed: {metrics['accuracy_perturbed']:.4f}")
    print(f"  Gap:       {metrics['robustness_gap']:.4f}")
    print(f"  Agreement: {metrics['agreement']:.4f}")
```

---

## Troubleshooting

### Issue: Out of Memory Error
```python
# Solution 1: Reduce batch size
config.data.batch_size = 8

# Solution 2: Reduce max nodes
config.data.max_nodes = 5000

# Solution 3: Enable mixed precision
config.training.mixed_precision = True
```

### Issue: Slow Data Loading
```python
# Solution: Increase workers
config.data.num_workers = 8
config.data.prefetch_factor = 4

# Or preprocess datasets
python scripts/preprocess_datasets.py  # Not included yet
```

### Issue: Model Not Converging
```python
# Solution 1: Adjust learning rate
config.training.learning_rate = 0.0005

# Solution 2: Reduce distillation weight
config.training.kd_loss_weight = 0.3

# Solution 3: Increase teacher training
# Pre-train teacher on clean data first
```

---

## Best Practices

 **Do:**
- Use multiple datasets for training (SARD + BigVul)
- Evaluate robustness across perturbation budgets
- Compare against baseline methods
- Perform ablation studies
- Save checkpoints regularly

 **Don't:**
- Mix shuffled and unshuffled batches
- Train without validation
- Ignore class imbalance issues
- Skip robustness evaluation
- Use tiny batch sizes (< 8)

---

## Citation

If you use this system, please cite:
```bibtex
@software{gcn_hkd_2024,
  title={GCN-HKD: Adversarially Robust Vulnerability Detection},
  author={Authors},
  year={2024}
}
```

---

For more help, see [README.md](README.md) or open an issue.
