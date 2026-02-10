"""Base model class with common functionality."""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple
from abc import ABC, abstractmethod


class BaseModel(nn.Module, ABC):
    """Abstract base class for all models in the system.
    
    Provides common functionality for model initialization, forward pass,
    and checkpoint management.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        dropout: float = 0.2,
        device: str = 'cuda'
    ):
        """Initialize base model.
        
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden layer dimension
            output_dim: Output dimension
            dropout: Dropout rate
            device: Device ('cuda' or 'cpu')
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.dropout = dropout
        self.device = device
        
        self.to(device)
    
    @abstractmethod
    def forward(self, *args, **kwargs) -> torch.Tensor:
        """Forward pass to be implemented by subclasses."""
        pass
    
    def extract_features(
        self,
        graph_x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Extract intermediate features from model.
        
        Useful for knowledge distillation - returns representations
        before final classification layer.
        """
        with torch.no_grad():
            # This will be overridden by subclasses
            features = self.forward(graph_x, edge_index, edge_attr)
        return features
    
    def get_attention_weights(
        self,
        graph_x: torch.Tensor,
        edge_index: torch.Tensor
    ) -> Dict[int, torch.Tensor]:
        """Get attention weights from each layer (for GAT models).
        
        Returns:
            Dict mapping layer index to attention weight matrix
        """
        # To be overridden by GAT implementations
        return {}
    
    def save_checkpoint(self, path: str) -> None:
        """Save model checkpoint."""
        torch.save({
            'model_state_dict': self.state_dict(),
            'model_config': {
                'input_dim': self.input_dim,
                'hidden_dim': self.hidden_dim,
                'output_dim': self.output_dim,
                'dropout': self.dropout,
            }
        }, path)
    
    def load_checkpoint(self, path: str) -> None:
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.load_state_dict(checkpoint['model_state_dict'])
    
    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_model_summary(self) -> Dict:
        """Get model information summary."""
        return {
            'total_params': self.count_parameters(),
            'input_dim': self.input_dim,
            'hidden_dim': self.hidden_dim,
            'output_dim': self.output_dim,
            'device': self.device,
            'model_type': self.__class__.__name__
        }


class ClassificationHead(nn.Module):
    """Vulnerability classification head."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_classes: int = 2,
        dropout: float = 0.2
    ):
        """Initialize classification head.
        
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden dimension
            num_classes: Number of output classes (default: vulnerable/safe)
            dropout: Dropout rate
        """
        super().__init__()
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.
        
        Returns:
            Tuple of (logits, probabilities)
        """
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        logits = self.fc2(x)
        probs = torch.softmax(logits, dim=-1)
        
        return logits, probs
