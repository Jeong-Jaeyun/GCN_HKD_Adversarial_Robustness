
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool
from typing import Dict, Optional, Tuple
from .base import BaseModel, ClassificationHead


class GATLayer(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_heads: int = 8,
        dropout: float = 0.2,
        concat: bool = True
    ):
        super().__init__()

        self.num_heads = num_heads
        self.concat = concat


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


        x = self.gat_conv(x, edge_index)
        return x


class GraphAttentionNetwork(BaseModel):

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


        self.input_proj = nn.Linear(input_dim, hidden_dim)


        self.gat_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        current_dim = hidden_dim
        for layer_idx in range(num_layers):
            is_last = layer_idx == num_layers - 1
            layer_out_dim = output_dim if is_last else hidden_dim


            self.gat_layers.append(
                GATLayer(
                    in_channels=current_dim,
                    out_channels=layer_out_dim,
                    num_heads=num_attention_heads,
                    dropout=dropout,
                    concat=False
                )
            )
            self.batch_norms.append(nn.BatchNorm1d(layer_out_dim))
            current_dim = layer_out_dim


        self.classifier = ClassificationHead(
            input_dim=output_dim,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            dropout=dropout
        )


        self.dropout_layer = nn.Dropout(dropout)


        self.attention_weights = {}

    def forward(
        self,
        graph_x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        x = self.input_proj(graph_x)
        x = F.relu(x)
        x = self.dropout_layer(x)


        for layer_idx, (gat_layer, bn_layer) in enumerate(
            zip(self.gat_layers, self.batch_norms)
        ):

            x = gat_layer(x, edge_index, edge_attr)


            x = bn_layer(x)


            if layer_idx < self.num_layers - 1:
                x = F.relu(x)
                x = self.dropout_layer(x)


        if batch is not None:
            graph_embedding = global_mean_pool(x, batch)
        else:
            graph_embedding = x.mean(dim=0, keepdim=True)


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
        attention_weights = {}

        x = self.input_proj(graph_x)
        x = F.relu(x)

        for layer_idx, (gat_layer, bn_layer) in enumerate(
            zip(self.gat_layers, self.batch_norms)
        ):
            x, (_, attn_alpha) = gat_layer.gat_conv(
                x,
                edge_index,
                return_attention_weights=True
            )
            attention_weights[layer_idx] = attn_alpha.detach()

            x = bn_layer(x)
            if layer_idx < self.num_layers - 1:
                x = F.relu(x)

        return attention_weights
