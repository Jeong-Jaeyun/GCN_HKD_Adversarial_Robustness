"""Teacher model - trained on clean source code graphs.

The teacher model learns to identify vulnerabilities in clean,
unmodified code. It serves as a knowledge source for the student model.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple
from .base import BaseModel, ClassificationHead
from .gat import GraphAttentionNetwork


class TeacherModel(BaseModel):
    """Teacher model for knowledge distillation.
    
    Trained on clean (non-adversarially perturbed) source code graphs.
    Learns robust, semantic-level vulnerability representations.
    
    Properties:
    - Larger capacity than student (more parameters)
    - Trained on clean data with full performance
    - Extracts 'high-level features' representing vulnerability patterns
    - Acts as knowledge source for student distillation
    """
    
    def __init__(
        self,
        input_dim: int,
        node_feature_dim: int = 768,
        hidden_dim: int = 512,
        output_dim: int = 256,
        num_layers: int = 4,
        num_attention_heads: int = 8,
        dropout: float = 0.2,
        num_classes: int = 2,
        device: str = 'cuda'
    ):
        """Initialize teacher model.
        
        Args:
            input_dim: Input node feature dimension
            node_feature_dim: Node embedding dimension
            hidden_dim: Hidden layer dimension (larger for teacher)
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
        
        # Core GAT feature extractor
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
        """Forward pass through teacher model.
        
        Args:
            graph_x: Node feature matrix
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
        
        Used for hierarchical knowledge distillation.
        
        Args:
            graph_x: Node features
            edge_index: Edge indices
            edge_attr: Edge attributes
            batch: Batch indices
            layer_indices: Which layers to extract from (None = all)
            
        Returns:
            Dict mapping layer index to feature tensor
        """
        x = self.node_embedding(graph_x)
        
        representations = {}
        
        if layer_indices is None:
            layer_indices = list(range(self.feature_extractor.num_layers + 1))
        
        # Extract features at different depths
        for layer_idx in range(self.feature_extractor.num_layers + 1):
            if layer_idx in layer_indices:
                features = self.feature_extractor.extract_features(
                    x, edge_index, edge_attr, batch, return_layer=layer_idx
                )
                representations[layer_idx] = features
        
        return representations
    
    def get_attention_maps(
        self,
        graph_x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None
    ) -> dict:
        """Get attention weights from all layers.
        
        Useful for interpretability and understanding what the model focuses on.
        
        Returns:
            Dict mapping layer index to attention weight tensor
        """
        x = self.node_embedding(graph_x)
        return self.feature_extractor.get_attention_weights(x, edge_index, edge_attr)
    
    def freeze_backbone(self) -> None:
        """Freeze feature extractor for transfer learning."""
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
    
    def unfreeze_backbone(self) -> None:
        """Unfreeze feature extractor."""
        for param in self.feature_extractor.parameters():
            param.requires_grad = True
