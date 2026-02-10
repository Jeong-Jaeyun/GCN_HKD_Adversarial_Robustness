"""Student model - trained on adversarially perturbed code graphs.

The student model learns to identify vulnerabilities even when code
has been transformed/obfuscated. It learns from the teacher via
knowledge distillation.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple
from .base import BaseModel, ClassificationHead
from .gat import GraphAttentionNetwork


class StudentModel(BaseModel):
    """Student model for adversarial robustness.
    
    Trained on adversarially perturbed (transformed) code graphs
    while learning 'high-level features' from the teacher model.
    
    Properties:
    - Smaller capacity than teacher (fewer parameters)
    - Trained on perturbed data via knowledge distillation
    - Learns to ignore adversarial noise in graph structure
    - Focuses on semantic features taught by teacher
    """
    
    def __init__(
        self,
        input_dim: int,
        node_feature_dim: int = 768,
        hidden_dim: int = 256,
        output_dim: int = 128,
        num_layers: int = 3,
        num_attention_heads: int = 8,
        dropout: float = 0.2,
        num_classes: int = 2,
        device: str = 'cuda'
    ):
        """Initialize student model.
        
        Args:
            input_dim: Input node feature dimension
            node_feature_dim: Node embedding dimension
            hidden_dim: Hidden layer dimension (smaller than teacher)
            output_dim: Output feature dimension
            num_layers: Number of GAT layers
            num_attention_heads: Number of attention heads
            dropout: Dropout rate
            num_classes: Number of classes
            device: Device
        """
        super().__init__(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            dropout=dropout,
            device=device
        )
        
        self.num_classes = num_classes
        self.node_feature_dim = node_feature_dim
        
        # Create node embedding layer if needed
        if input_dim != node_feature_dim:
            self.node_embedding = nn.Linear(input_dim, node_feature_dim)
        else:
            self.node_embedding = nn.Identity()
        
        # Core GAT feature extractor (smaller than teacher)
        self.feature_extractor = GraphAttentionNetwork(
            input_dim=node_feature_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_layers=num_layers,
            num_attention_heads=num_attention_heads,
            dropout=dropout,
            num_classes=num_classes,
            device=device
        )
    
    def forward(
        self,
        graph_x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None,
        return_features: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through student model.
        
        Args:
            graph_x: Node feature matrix (possibly from adversarially perturbed code)
            edge_index: Edge indices
            edge_attr: Edge attributes
            batch: Batch indices
            return_features: Whether to return intermediate features
            
        Returns:
            Tuple of (logits, probabilities) or (logits, features) if return_features=True
        """
        # Embed node features
        x = self.node_embedding(graph_x)
        
        # Forward through feature extractor
        logits, probs = self.feature_extractor(x, edge_index, edge_attr, batch)
        
        if return_features:
            # Return intermediate features for distillation
            features = self.feature_extractor.extract_features(
                x, edge_index, edge_attr, batch
            )
            return logits, features
        
        return logits, probs
    
    def get_intermediate_representations(
        self,
        graph_x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None,
        layer_indices: Optional[list] = None
    ) -> dict:
        """Get intermediate representations from multiple layers.
        
        Args:
            graph_x: Node features
            edge_index: Edge indices
            edge_attr: Edge attributes
            batch: Batch indices
            layer_indices: Which layers to extract from
            
        Returns:
            Dict mapping layer index to feature tensor
        """
        x = self.node_embedding(graph_x)
        
        representations = {}
        
        if layer_indices is None:
            layer_indices = list(range(self.feature_extractor.num_layers + 1))
        
        for layer_idx in range(self.feature_extractor.num_layers + 1):
            if layer_idx in layer_indices:
                features = self.feature_extractor.extract_features(
                    x, edge_index, edge_attr, batch, return_layer=layer_idx
                )
                representations[layer_idx] = features
        
        return representations
    
    def adapt_architecture(self, teacher_output_dim: int) -> None:
        """Adapt student architecture to match teacher output.
        
        Can be called to match the teacher model's output dimension
        for more effective knowledge distillation.
        
        Args:
            teacher_output_dim: Teacher model's output dimension
        """
        # Add adaptation layer if dimensions mismatch
        if self.output_dim != teacher_output_dim:
            self.adaptation_layer = nn.Linear(self.output_dim, teacher_output_dim)
        else:
            self.adaptation_layer = nn.Identity()
    
    def get_adaptation_layer(self) -> Optional[nn.Module]:
        """Get the adaptation layer if it exists."""
        return getattr(self, 'adaptation_layer', None)
