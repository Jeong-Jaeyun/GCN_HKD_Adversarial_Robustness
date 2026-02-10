"""Graph Attention Network (GAT) implementation for robust feature extraction.

GAT is recommended over GCN because attention mechanism helps the model
focus on semantically important edges while ignoring adversarial noise.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool
from typing import Dict, Optional, Tuple
from .base import BaseModel, ClassificationHead


class GATLayer(nn.Module):
    """Single Graph Attention Network layer with multi-head attention."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_heads: int = 8,
        dropout: float = 0.2,
        concat: bool = True
    ):
        """Initialize GAT layer.
        
        Args:
            in_channels: Input feature dimension
            out_channels: Output feature dimension per head
            num_heads: Number of attention heads
            dropout: Dropout rate
            concat: Whether to concatenate or average head outputs
        """
        super().__init__()
        
        self.num_heads = num_heads
        self.concat = concat
        
        # GAT convolution layer
        self.gat_conv = GATConv(
            in_channels=in_channels,
            out_channels=out_channels,
            heads=num_heads,
            dropout=dropout,
            concat=concat,
            add_self_loops=True
        )
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Node feature matrix (N, in_channels)
            edge_index: Edge indices (2, E)
            edge_attr: Edge features (E, edge_feature_dim)
            
        Returns:
            Updated node features (N, out_channels * num_heads or out_channels)
        """
        x = self.gat_conv(x, edge_index, edge_attr)
        return x


class GraphAttentionNetwork(BaseModel):
    """Multi-layer Graph Attention Network for robust feature extraction.
    
    Key properties:
    - Uses attention mechanism to learn edge importance
    - Robust to adversarial perturbations (ignores noise edges)
    - Extracts hierarchical graph features
    - Produces both logits and attention weights
    
    This is the main feature extractor in Phase 2 of the architecture.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 3,
        num_attention_heads: int = 8,
        dropout: float = 0.2,
        num_classes: int = 2,
        device: str = 'cuda'
    ):
        """Initialize Graph Attention Network.
        
        Args:
            input_dim: Input node feature dimension
            hidden_dim: Hidden layer dimension
            output_dim: Output feature dimension
            num_layers: Number of GAT layers
            num_attention_heads: Number of attention heads per layer
            dropout: Dropout rate
            num_classes: Number of output classes
            device: Device ('cuda' or 'cpu')
        """
        super().__init__(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            dropout=dropout,
            device=device
        )
        
        self.num_layers = num_layers
        self.num_attention_heads = num_attention_heads
        self.num_classes = num_classes
        
        # Input projection layer
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # GAT layers
        self.gat_layers = nn.ModuleList()
        for layer_idx in range(num_layers):
            layer_in_dim = hidden_dim if layer_idx > 0 else hidden_dim
            layer_out_dim = hidden_dim if layer_idx < num_layers - 1 else output_dim
            
            self.gat_layers.append(
                GATLayer(
                    in_channels=layer_in_dim,
                    out_channels=layer_out_dim,
                    num_heads=num_attention_heads,
                    dropout=dropout,
                    concat=(layer_idx < num_layers - 1)  # Concat all but last layer
                )
            )
        
        # Batch normalization for each layer
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(
                hidden_dim if i < num_layers - 1 else output_dim
            )
            for i in range(num_layers)
        ])
        
        # Classification head
        self.classifier = ClassificationHead(
            input_dim=output_dim,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            dropout=dropout
        )
        
        # Dropout layer
        self.dropout_layer = nn.Dropout(dropout)
        
        # Store attention weights for visualization
        self.attention_weights = {}
    
    def forward(
        self,
        graph_x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through GAT.
        
        Args:
            graph_x: Node feature matrix (N, input_dim)
            edge_index: Edge indices (2, E)
            edge_attr: Edge features (E, edge_feature_dim)
            batch: Batch indices for graphs (for batched processing)
            return_attention: Whether to return attention weights
            
        Returns:
            Tuple of (logits, probabilities)
        """
        # Project input features
        x = self.input_proj(graph_x)
        x = F.relu(x)
        x = self.dropout_layer(x)
        
        # Pass through GAT layers
        for layer_idx, (gat_layer, bn_layer) in enumerate(
            zip(self.gat_layers, self.batch_norms)
        ):
            # GAT forward pass
            x = gat_layer(x, edge_index, edge_attr)
            
            # Batch normalization
            x = bn_layer(x)
            
            # Activation and dropout
            if layer_idx < self.num_layers - 1:
                x = F.relu(x)
                x = self.dropout_layer(x)
        
        # Global pooling (graph-level representation)
        if batch is not None:
            graph_embedding = global_mean_pool(x, batch)
        else:
            graph_embedding = x.mean(dim=0, keepdim=True)
        
        # Classification
        logits, probs = self.classifier(graph_embedding)
        
        return logits, probs
    
    def extract_features(
        self,
        graph_x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None,
        return_layer: int = -1
    ) -> torch.Tensor:
        """Extract intermediate features from specified layer.
        
        Args:
            graph_x: Node features
            edge_index: Edge indices
            edge_attr: Edge features
            batch: Batch indices
            return_layer: Which layer to return features from (-1 = last)
            
        Returns:
            Intermediate features from specified layer
        """
        x = self.input_proj(graph_x)
        x = F.relu(x)
        
        if return_layer == 0 or return_layer == -self.num_layers:
            return x
        
        for layer_idx, (gat_layer, bn_layer) in enumerate(
            zip(self.gat_layers, self.batch_norms)
        ):
            x = gat_layer(x, edge_index, edge_attr)
            x = bn_layer(x)
            
            if layer_idx == return_layer - 1 or layer_idx == self.num_layers + return_layer - 1:
                return x
            
            if layer_idx < self.num_layers - 1:
                x = F.relu(x)
        
        # Global pooling
        if batch is not None:
            x = global_mean_pool(x, batch)
        else:
            x = x.mean(dim=0, keepdim=True)
        
        return x
    
    def get_attention_weights(
        self,
        graph_x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None
    ) -> Dict[int, torch.Tensor]:
        """Get attention weights from all GAT layers.
        
        Returns:
            Dict mapping layer index to attention weight tensor
        """
        attention_weights = {}
        
        x = self.input_proj(graph_x)
        x = F.relu(x)
        
        for layer_idx, (gat_layer, bn_layer) in enumerate(
            zip(self.gat_layers, self.batch_norms)
        ):
            # Get attention weights from GATConv layer
            _ = gat_layer(x, edge_index, edge_attr)
            
            # Store attention weights
            if hasattr(gat_layer.gat_conv, 'att'):
                attention_weights[layer_idx] = gat_layer.gat_conv.att.detach()
            
            x = bn_layer(x)
            if layer_idx < self.num_layers - 1:
                x = F.relu(x)
        
        return attention_weights
