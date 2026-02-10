"""Training callbacks for monitoring and checkpointing."""

import logging
from pathlib import Path
from typing import Dict, Optional
import json
from datetime import datetime


class Callback:
    """Base callback class."""
    
    def on_epoch_end(self, epoch: int, metrics: Dict) -> None:
        """Called at end of epoch."""
        pass


class CheckpointCallback(Callback):
    """Callback for saving model checkpoints."""
    
    def __init__(
        self,
        save_dir: Path,
        monitor: str = 'val_loss',
        save_best_only: bool = True,
        mode: str = 'min'
    ):
        """Initialize checkpoint callback.
        
        Args:
            save_dir: Directory to save checkpoints
            monitor: Metric to monitor
            save_best_only: Save only best checkpoint
            mode: 'min' or 'max'
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.monitor = monitor
        self.save_best_only = save_best_only
        self.mode = mode
        
        if mode == 'min':
            self.best_value = float('inf')
            self.is_better = lambda x, y: x < y
        else:
            self.best_value = -float('inf')
            self.is_better = lambda x, y: x > y
        
        self.logger = logging.getLogger(__name__)
    
    def on_epoch_end(self, epoch: int, metrics: Dict, model=None) -> None:
        """Save checkpoint if needed."""
        if model is None:
            return
        
        if self.monitor not in metrics:
            return
        
        current_value = metrics[self.monitor]
        
        if self.is_better(current_value, self.best_value):
            self.best_value = current_value
            checkpoint_path = self.save_dir / f"best_model.pt"
            
            # Save checkpoint
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'metrics': metrics,
                'timestamp': datetime.now().isoformat()
            }
            
            import torch
            torch.save(checkpoint, checkpoint_path)
            self.logger.info(
                f"Saved best model to {checkpoint_path} "
                f"({self.monitor}={current_value:.4f})"
            )
        elif not self.save_best_only:
            checkpoint_path = self.save_dir / f"checkpoint_epoch{epoch}.pt"
            import torch
            torch.save(model.state_dict(), checkpoint_path)


class LoggingCallback(Callback):
    """Callback for logging metrics."""
    
    def __init__(
        self,
        log_file: Optional[Path] = None,
        log_every: int = 1
    ):
        """Initialize logging callback.
        
        Args:
            log_file: File to write logs to
            log_every: Log every N epochs
        """
        self.log_file = log_file
        self.log_every = log_every
        self.logger = logging.getLogger(__name__)
        
        if log_file is not None:
            self.log_file = Path(log_file)
            self.log_file.parent.mkdir(parents=True, exist_ok=True)
    
    def on_epoch_end(self, epoch: int, metrics: Dict) -> None:
        """Log metrics."""
        if epoch % self.log_every != 0:
            return
        
        log_msg = f"Epoch {epoch}: "
        for key, value in metrics.items():
            log_msg += f"{key}={value:.4f} "
        
        self.logger.info(log_msg)
        
        # Write to file
        if self.log_file is not None:
            with open(self.log_file, 'a') as f:
                f.write(log_msg + '\n')


class EvaluationCallback(Callback):
    """Callback for evaluation on test set."""
    
    def __init__(
        self,
        test_loader,
        evaluator,
        eval_every: int = 5
    ):
        """Initialize evaluation callback.
        
        Args:
            test_loader: Test data loader
            evaluator: Evaluator object with evaluate() method
            eval_every: Evaluate every N epochs
        """
        self.test_loader = test_loader
        self.evaluator = evaluator
        self.eval_every = eval_every
        self.logger = logging.getLogger(__name__)
    
    def on_epoch_end(self, epoch: int, metrics: Dict, model=None) -> None:
        """Run evaluation if needed."""
        if model is None or epoch % self.eval_every != 0:
            return
        
        test_metrics = self.evaluator.evaluate(model, self.test_loader)
        
        log_msg = f"Test metrics at epoch {epoch}: "
        for key, value in test_metrics.items():
            log_msg += f"{key}={value:.4f} "
        
        self.logger.info(log_msg)
