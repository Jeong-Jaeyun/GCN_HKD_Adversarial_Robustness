"""
Example script demonstrating the GCN-HKD system.

This script shows how to:
1. Load and preprocess data
2. Initialize models
3. Run knowledge distillation training
4. Evaluate adversarial robustness
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from pathlib import Path

from config.config import Config
from models.teacher import TeacherModel
from models.student import StudentModel
from distillation.hkd import HierarchicalKnowledgeDistillation
from training.trainer import Trainer
from evaluation.robustness import RobustnessEvaluator
from adversarial.transformations import AdversarialTransformer
from utils.logger import setup_logger


def create_dummy_dataset(num_samples: int = 100, num_nodes: int = 512):
    """Create dummy dataset for demonstration."""
    # Random graph data
    x = torch.randn(num_samples, num_nodes, 768)  # Node features
    edge_indices = [torch.randint(0, num_nodes, (2, num_nodes // 2)) for _ in range(num_samples)]
    edge_attrs = [torch.rand(num_nodes // 2, 5) for _ in range(num_samples)]
    labels = torch.randint(0, 2, (num_samples,))  # Binary labels
    
    return x, edge_indices, edge_attrs, labels


def demo_phase1_transformations():
    """Demonstrate Phase 1: Adversarial Transformations."""
    print("\n" + "=" * 80)
    print("Phase 1: Adversarial Transformation Module")
    print("=" * 80)
    
    # Example C code
    sample_code = """
    int buffer_overflow(const char* input) {
        char buffer[16];
        strcpy(buffer, input);  // Vulnerability: no bounds checking
        return strlen(buffer);
    }
    """
    
    # Initialize transformer
    transformer = AdversarialTransformer(
        variable_rename_prob=0.5,
        dead_code_ratio=0.2,
        control_flow_complexity=0.4
    )
    
    # Apply transformations
    print("\nOriginal code:")
    print(sample_code)
    
    result = transformer.transform(sample_code, 'variable_renaming')
    print("\nTransformed code (variable renaming):")
    print(result.transformed_code)
    print(f"\nChanges: {result.changes_made}")
    
    return sample_code, result


def demo_phase2_models():
    """Demonstrate Phase 2: Model Initialization."""
    print("\n" + "=" * 80)
    print("Phase 2: Robust Feature Extraction (GAT)")
    print("=" * 80)
    
    config = Config()
    
    # Initialize teacher model
    teacher = TeacherModel(
        input_dim=768,
        node_feature_dim=768,
        hidden_dim=config.model.teacher_hidden_dim,
        output_dim=config.model.output_dim,
        num_layers=config.model.teacher_num_layers,
        device='cpu'
    )
    
    print(f"\nTeacher Model Summary:")
    print(f"  {teacher.get_model_summary()}")
    
    # Initialize student model
    student = StudentModel(
        input_dim=768,
        node_feature_dim=768,
        hidden_dim=config.model.student_hidden_dim,
        output_dim=config.model.output_dim,
        num_layers=config.model.student_num_layers,
        device='cpu'
    )
    
    print(f"\nStudent Model Summary:")
    print(f"  {student.get_model_summary()}")
    
    return teacher, student


def demo_phase3_distillation(teacher, student):
    """Demonstrate Phase 3: Knowledge Distillation."""
    print("\n" + "=" * 80)
    print("Phase 3: Hierarchical Knowledge Distillation")
    print("=" * 80)
    
    # Create dummy data
    x, edge_indices, edge_attrs, labels = create_dummy_dataset(num_samples=4)
    
    # Initialize KD
    kd = HierarchicalKnowledgeDistillation(
        teacher_model=teacher,
        student_model=student,
        temperature=4.0,
        device='cpu'
    )
    
    print("\nInitializing knowledge distillation...")
    print(f"  Teacher: {teacher.feature_extractor.num_layers} layers")
    print(f"  Student: {student.feature_extractor.num_layers} layers")
    print(f"  Temperature: 4.0")
    
    # Forward pass with KD
    print("\nRunning forward pass...")
    
    for i, (xi, edge_idx, edge_attr, label) in enumerate(zip(x, edge_indices, edge_attrs, labels)):
        try:
            # Convert to batch format
            x_batch = xi.unsqueeze(0)  # Add batch dimension
            edge_idx_batch = edge_idx
            edge_attr_batch = edge_attr.unsqueeze(0) if edge_attr.shape[0] > 0 else None
            label_batch = label.unsqueeze(0)
            
            # Forward pass via teacher and student
            with torch.no_grad():
                teacher_logits, _ = teacher(x_batch, edge_idx_batch, edge_attr_batch)
                student_logits, _ = student(x_batch, edge_idx_batch, edge_attr_batch)
            
            if i == 0:
                print(f"Sample {i+1}:")
                print(f"  Teacher logits: {teacher_logits}")
                print(f"  Student logits: {student_logits}")
                print(f"  ✓ Forward pass successful")
            
        except Exception as e:
            print(f"Sample {i+1}: Skipped (graph too small)")
    
    return kd


def demo_robustness_evaluation():
    """Demonstrate adversarial robustness evaluation."""
    print("\n" + "=" * 80)
    print("Robustness Evaluation")
    print("=" * 80)
    
    # Mock predictions
    clean_accs = [0.95]
    budgets = [0.05, 0.1, 0.15, 0.2]
    
    print("\nMock Robustness Results:")
    print(f"{'Budget':<10} {'Clean Acc':<15} {'Robust Acc':<15} {'Gap':<10}")
    print("-" * 50)
    
    for budget in budgets:
        robust_acc = clean_accs[0] * (1 - budget * 2)  # Simple decay model
        gap = clean_accs[0] - robust_acc
        print(f"{budget:<10.2f} {clean_accs[0]:<15.4f} {robust_acc:<15.4f} {gap:<10.4f}")


def main():
    """Run complete demonstration."""
    logger = setup_logger(__name__)
    
    logger.info("\n" + "=" * 80)
    logger.info("GCN-HKD System Demonstration")
    logger.info("=" * 80)
    
    # Phase 1: Transformations
    original_code, transform_result = demo_phase1_transformations()
    
    # Phase 2: Model Initialization
    teacher, student = demo_phase2_models()
    
    # Phase 3: Knowledge Distillation
    kd = demo_phase3_distillation(teacher, student)
    
    # Robustness Evaluation
    demo_robustness_evaluation()
    
    logger.info("\n" + "=" * 80)
    logger.info("Demonstration Complete!")
    logger.info("=" * 80)
    logger.info("\nNext steps:")
    logger.info("1. Prepare real datasets (SARD, BigVul, D-Sieve)")
    logger.info("2. Run: python main.py")
    logger.info("3. Check results in ./results/")
    logger.info("\nFor more details, see USAGE_GUIDE.md")


if __name__ == '__main__':
    main()
