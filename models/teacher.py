
import torch
import torch.nn as nn
from typing import Optional, Tuple
from .base import BaseModel, ClassificationHead
from .gat import GraphAttentionNetwork


class TeacherModel(BaseModel):

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
        super().__init__(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            dropout=dropout,
            device=device
        )

        self.num_classes = num_classes
        self.node_feature_dim = node_feature_dim


        if input_dim != node_feature_dim:
            self.node_embedding = nn.Linear(input_dim, node_feature_dim)
        else:
            self.node_embedding = nn.Identity()


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

        x = self.node_embedding(graph_x)


        logits, probs = self.feature_extractor(x, edge_index, edge_attr, batch)

        if return_features:

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

    def get_attention_maps(
        self,
        graph_x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None
    ) -> dict:
        x = self.node_embedding(graph_x)
        return self.feature_extractor.get_attention_weights(x, edge_index, edge_attr)

    def freeze_backbone(self) -> None:
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self) -> None:
        for param in self.feature_extractor.parameters():
            param.requires_grad = True
