"""Main training loop and trainer."""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple, Callable
import logging
from pathlib import Path
import json
from datetime import datetime

from config.config import Config
from distillation.hkd import HierarchicalKnowledgeDistillation


class Trainer:
    """Trainer for the vulnerability detection system with knowledge distillation."""
    
    def __init__(
        self,
        config: Config,
        teacher_model: nn.Module,
        student_model: nn.Module,
        device: str = 'cuda'
    ):
        """Initialize trainer.
        
        Args:
            config: Configuration object
            teacher_model: Teacher model (pre-trained)
            student_model: Student model (to train)
            device: Computing device
        """
        self.config = config
        self.device = device
        
        # Initialize knowledge distillation
        self.kd = HierarchicalKnowledgeDistillation(
            teacher_model=teacher_model,
            student_model=student_model,
            temperature=config.model.temperature,
            feature_alpha=config.training.kd_loss_weight,
            device=device
        )
        
        # Setup optimizer
        self.optimizer = self._create_optimizer()
        
        # Setup scheduler
        self.scheduler = self._create_scheduler()
        
        # Training state
        self.current_epoch = 0
        self.current_step = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
        # History
        self.train_history = {'loss': [], 'accuracy': []}
        self.val_history = {'loss': [], 'accuracy': []}
        
        # Logger
        self.logger = logging.getLogger(__name__)
    
    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer."""
        params = self.kd.student.parameters()
        
        if self.config.training.optimizer == 'adamw':
            return optim.AdamW(
                params,
                lr=self.config.training.learning_rate,
                weight_decay=self.config.training.weight_decay
            )
        elif self.config.training.optimizer == 'adam':
            return optim.Adam(
                params,
                lr=self.config.training.learning_rate,
                weight_decay=self.config.training.weight_decay
            )
        elif self.config.training.optimizer == 'sgd':
            return optim.SGD(
                params,
                lr=self.config.training.learning_rate,
                weight_decay=self.config.training.weight_decay,
                momentum=0.9
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.config.training.optimizer}")
    
    def _create_scheduler(self) -> Optional[object]:
        """Create learning rate scheduler."""
        if self.config.training.scheduler == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.training.num_epochs
            )
        elif self.config.training.scheduler == 'step':
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=10,
                gamma=0.1
            )
        elif self.config.training.scheduler == 'linear':
            return optim.lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=1.0,
                end_factor=0.1,
                total_iters=self.config.training.num_epochs
            )
        else:
            return None
    
    def train_epoch(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None
    ) -> Dict[str, float]:
        """Train for one epoch.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
            
        Returns:
            Dict with epoch metrics
        """
        self.current_epoch += 1
        
        # Training phase
        train_metrics = self._train_step(train_loader)
        
        # Validation phase  
        val_metrics = {}
        if val_loader is not None:
            val_metrics = self._val_step(val_loader)
        
        # Update scheduler
        if self.scheduler is not None:
            self.scheduler.step()
        
        # Log and check stopping criteria
        epoch_metrics = {**train_metrics, **val_metrics}
        self._log_epoch(epoch_metrics)
        self._check_stopping(val_metrics)
        
        return epoch_metrics
    
    def _train_step(self, train_loader: DataLoader) -> Dict[str, float]:
        """Single training step."""
        self.kd.student.train()
        
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        num_batches = 0
        
        for batch_idx, batch in enumerate(train_loader):
            # Move to device
            batch = self._move_batch_to_device(batch)
            
            # Forward pass
            loss, student_out, loss_components = self.kd.forward(
                graph_x=batch['graph_x'],
                edge_index=batch['graph_edge_index'],
                targets=batch['label'],
                edge_attr=batch['graph_edge_attr'],
                return_features=True
            )
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            if self.config.training.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.kd.student.parameters(),
                    self.config.training.gradient_clip
                )
            
            self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            logits = student_out['logits']
            preds = logits.argmax(dim=-1)
            targets = batch['label']
            
            total_correct += (preds == targets).sum().item()
            total_samples += targets.size(0)
            num_batches += 1
            self.current_step += 1
            
            # Log every N steps
            if batch_idx % self.config.training.log_interval == 0:
                self.logger.info(
                    f"Epoch {self.current_epoch} Batch {batch_idx}/{len(train_loader)} "
                    f"Loss: {loss.item():.4f} "
                    f"Acc: {total_correct / total_samples:.4f}"
                )
        
        return {
            'train_loss': total_loss / num_batches,
            'train_accuracy': total_correct / total_samples
        }
    
    def _val_step(self, val_loader: DataLoader) -> Dict[str, float]:
        """Single validation step."""
        self.kd.student.eval()
        
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                # Move to device
                batch = self._move_batch_to_device(batch)
                
                # Forward pass
                loss, student_out, _ = self.kd.forward(
                    graph_x=batch['graph_x'],
                    edge_index=batch['graph_edge_index'],
                    targets=batch['label'],
                    edge_attr=batch['graph_edge_attr']
                )
                
                total_loss += loss.item()
                logits = student_out['logits']
                preds = logits.argmax(dim=-1)
                targets = batch['label']
                
                total_correct += (preds == targets).sum().item()
                total_samples += targets.size(0)
                num_batches += 1
        
        return {
            'val_loss': total_loss / num_batches,
            'val_accuracy': total_correct / total_samples
        }
    
    def _move_batch_to_device(self, batch: Dict) -> Dict:
        """Move batch tensors to device."""
        return {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }
    
    def _log_epoch(self, metrics: Dict[str, float]) -> None:
        """Log epoch metrics."""
        log_msg = f"Epoch {self.current_epoch}: "
        for key, value in metrics.items():
            log_msg += f"{key}={value:.4f} "
            
            # Save to history
            if 'train' in key:
                metric_name = key.replace('train_', '')
                if metric_name not in self.train_history:
                    self.train_history[metric_name] = []
                self.train_history[metric_name].append(value)
            
            if 'val' in key:
                metric_name = key.replace('val_', '')
                if metric_name not in self.val_history:
                    self.val_history[metric_name] = []
                self.val_history[metric_name].append(value)
        
        self.logger.info(log_msg)
    
    def _check_stopping(self, val_metrics: Dict[str, float]) -> None:
        """Check early stopping criteria."""
        if 'val_loss' in val_metrics:
            val_loss = val_metrics['val_loss']
            
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
            else:
                self.patience_counter += 1
    
    def should_stop(self) -> bool:
        """Check if training should stop."""
        if self.patience_counter >= self.config.training.patience:
            self.logger.info(
                f"Early stopping triggered after {self.config.training.patience} epochs"
            )
            return True
        
        if self.current_epoch >= self.config.training.num_epochs:
            return True
        
        return False
    
    def save_checkpoint(self, path: Path) -> None:
        """Save training checkpoint."""
        checkpoint = {
            'epoch': self.current_epoch,
            'student_state_dict': self.kd.student.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_history': self.train_history,
            'val_history': self.val_history,
            'best_val_loss': self.best_val_loss,
        }
        
        torch.save(checkpoint, path)
        self.logger.info(f"Checkpoint saved to {path}")
    
    def load_checkpoint(self, path: Path) -> None:
        """Load training checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.kd.student.load_state_dict(checkpoint['student_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.train_history = checkpoint['train_history']
        self.val_history = checkpoint['val_history']
        self.best_val_loss = checkpoint['best_val_loss']
        
        self.logger.info(f"Checkpoint loaded from {path}")
