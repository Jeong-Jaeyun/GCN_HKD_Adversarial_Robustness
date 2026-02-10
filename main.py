"""Main entry point for GCN-HKD Adversarial Robustness system.

Orchestrates the complete pipeline:
1. Load datasets (SARD, BigVul, D-Sieve)
2. Construct graphs (CFG+DFG)
3. Train teacher model on clean code
4. Train student model via knowledge distillation on perturbed code
5. Evaluate robustness against adversarial attacks
6. Test on real-world CVEs (Linux Kernel, OpenSSL)
"""

import argparse
import logging
import torch
from pathlib import Path
from datetime import datetime

from config.config import Config, default_config
from config.paths import PathConfig
from utils.logger import setup_logger


def main(config: Config, output_dir: Path):
    """Main training and evaluation pipeline.
    
    Args:
        config: Configuration object
        output_dir: Output directory for results
    """
    # Setup logging
    log_file = output_dir / "training.log"
    logger = setup_logger('GCN_HKD', log_file=log_file)
    
    logger.info("=" * 80)
    logger.info("GCN-HKD Adversarial Robustness for Vulnerability Detection")
    logger.info(f"Timestamp: {datetime.now().isoformat()}")
    logger.info("=" * 80)
    
    logger.info(f"\nConfiguration:")
    logger.info(f"  Datasets: {config.data.datasets}")
    logger.info(f"  Model: GAT with {config.model.num_gat_layers} layers")
    logger.info(f"  Training epochs: {config.training.num_epochs}")
    logger.info(f"  Device: {config.device}")
    
    # Phase 1: Load and preprocess data
    logger.info("\n" + "=" * 80)
    logger.info("Phase 0: Data Loading and Preprocessing")
    logger.info("=" * 80)
    
    try:
        from data.loaders import load_multimodal_dataset
        from data.datasets import VulnerabilityDataset, MultiDatasetVulnerabilityDataset
        
        logger.info(f"Loading datasets: {config.data.datasets}")
        train_df, val_df, test_df = load_multimodal_dataset(
            config.data.datasets,
            split_ratio=(config.data.train_ratio, config.data.val_ratio, config.data.test_ratio)
        )
        
        logger.info(f"  Train samples: {len(train_df)}")
        logger.info(f"  Val samples: {len(val_df)}")
        logger.info(f"  Test samples: {len(test_df)}")
        
        # Create datasets
        train_dataset = MultiDatasetVulnerabilityDataset(
            config.data.datasets,
            split='train',
            max_nodes=config.data.max_nodes,
            node_feature_dim=config.data.node_feature_dims
        )
        
        val_dataset = MultiDatasetVulnerabilityDataset(
            config.data.datasets,
            split='val',
            max_nodes=config.data.max_nodes,
            node_feature_dim=config.data.node_feature_dims
        )
        
        test_dataset = MultiDatasetVulnerabilityDataset(
            config.data.datasets,
            split='test',
            max_nodes=config.data.max_nodes,
            node_feature_dim=config.data.node_feature_dims
        )
        
        logger.info("✓ Data loading complete")
        
    except Exception as e:
        logger.error(f"Error in data loading: {e}")
        return
    
    # Phase 2: Initialize models
    logger.info("\n" + "=" * 80)
    logger.info("Phase 1: Model Initialization")
    logger.info("=" * 80)
    
    try:
        from models.teacher import TeacherModel
        from models.student import StudentModel
        
        # Initialize teacher model
        teacher_model = TeacherModel(
            input_dim=config.data.max_nodes,
            node_feature_dim=config.data.node_feature_dims,
            hidden_dim=config.model.teacher_hidden_dim,
            output_dim=config.model.output_dim,
            num_layers=config.model.teacher_num_layers,
            num_attention_heads=config.model.num_attention_heads,
            dropout=config.model.dropout,
            device=config.device
        )
        
        logger.info(f"Teacher model: {teacher_model.get_model_summary()}")
        
        # Initialize student model
        student_model = StudentModel(
            input_dim=config.data.max_nodes,
            node_feature_dim=config.data.node_feature_dims,
            hidden_dim=config.model.student_hidden_dim,
            output_dim=config.model.output_dim,
            num_layers=config.model.student_num_layers,
            num_attention_heads=config.model.num_attention_heads,
            dropout=config.model.dropout,
            device=config.device
        )
        
        logger.info(f"Student model: {student_model.get_model_summary()}")
        
    except Exception as e:
        logger.error(f"Error in model initialization: {e}")
        return
    
    # Phase 3: Knowledge Distillation Training
    logger.info("\n" + "=" * 80)
    logger.info("Phase 2: Model Training with Knowledge Distillation")
    logger.info("=" * 80)
    
    try:
        from training.trainer import Trainer
        from torch.utils.data import DataLoader
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.data.batch_size,
            shuffle=True,
            num_workers=config.data.num_workers
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.data.batch_size,
            shuffle=False,
            num_workers=config.data.num_workers
        )
        
        # Initialize trainer
        trainer = Trainer(config, teacher_model, student_model, device=config.device)
        
        # Training loop
        logger.info(f"Starting training for {config.training.num_epochs} epochs...")
        
        epoch = 0
        while epoch < config.training.num_epochs and not trainer.should_stop():
            epoch_metrics = trainer.train_epoch(train_loader, val_loader)
            epoch += 1
            
            if epoch % 5 == 0:
                logger.info(f"Checkpoint at epoch {epoch}")
                checkpoint_path = output_dir / f"student_epoch{epoch}.pt"
                trainer.save_checkpoint(checkpoint_path)
        
        logger.info("✓ Training complete")
        
    except Exception as e:
        logger.error(f"Error in training: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return
    
    # Phase 4: Robustness Evaluation
    logger.info("\n" + "=" * 80)
    logger.info("Phase 3: Adversarial Robustness Evaluation")
    logger.info("=" * 80)
    
    try:
        from evaluation.robustness import RobustnessEvaluator
        from torch.utils.data import DataLoader
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=config.data.batch_size,
            shuffle=False,
            num_workers=0
        )
        
        evaluator = RobustnessEvaluator(device=config.device)
        
        logger.info("Evaluating adversarial robustness...")
        robustness_results = evaluator.evaluate_robustness(
            student_model,
            test_loader,
            perturbation_budgets=config.evaluation.perturbation_levels,
            attack_types=['node_feature', 'edge', 'structural']
        )
        
        logger.info(f"\nRobustness Summary:")
        logger.info(f"  Clean Accuracy: {robustness_results['summary']['clean_accuracy']:.4f}")
        logger.info(f"  Avg Robust Accuracy: {robustness_results['summary']['avg_robustness_accuracy']:.4f}")
        logger.info(f"  Avg Robustness Gap: {robustness_results['summary']['avg_robustness_gap']:.4f}")
        
        logger.info("✓ Robustness evaluation complete")
        
        # Save results
        import json
        results_file = output_dir / "robustness_results.json"
        with open(results_file, 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            json.dump(str(robustness_results), f, indent=2)
        
    except Exception as e:
        logger.error(f"Error in robustness evaluation: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return
    
    logger.info("\n" + "=" * 80)
    logger.info("Pipeline Complete!")
    logger.info("=" * 80)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="GCN-HKD Adversarial Robustness for Vulnerability Detection"
    )
    
    parser.add_argument(
        '--config',
        type=str,
        help='Path to config YAML file'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./results',
        help='Output directory for results'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        choices=['cuda', 'cpu'],
        help='Computing device'
    )
    
    parser.add_argument(
        '--num-epochs',
        type=int,
        help='Number of training epochs'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        help='Batch size'
    )
    
    return parser.parse_args()


if __name__ == '__main__':
    import os
    
    # Parse arguments
    args = parse_args()
    
    # Load configuration
    if args.config:
        config = Config.from_yaml(args.config)
    else:
        config = default_config
    
    # Override with command line arguments
    if args.device:
        config.device = args.device
    if args.num_epochs:
        config.training.num_epochs = args.num_epochs
    if args.batch_size:
        config.data.batch_size = args.batch_size
    
    # Create output directory
    output_dir = Path(args.output_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save configuration
    config.to_yaml(output_dir / "config.yaml")
    
    # Run pipeline
    main(config, output_dir)
