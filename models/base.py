
import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple
from abc import ABC, abstractmethod


class BaseModel(nn.Module, ABC):

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        dropout: float = 0.2,
        device: str = 'cuda'
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.dropout = dropout
        self.device = device

        self.to(device)

    @abstractmethod
    def forward(self, *args, **kwargs) -> torch.Tensor:
        pass

    def extract_features(
        self,
        graph_x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        with torch.no_grad():

            features = self.forward(graph_x, edge_index, edge_attr)
        return features

    def get_attention_weights(
        self,
        graph_x: torch.Tensor,
        edge_index: torch.Tensor
    ) -> Dict[int, torch.Tensor]:

        return {}

    def save_checkpoint(self, path: str) -> None:
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
        checkpoint = torch.load(path, map_location=self.device)
        self.load_state_dict(checkpoint['model_state_dict'])

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_model_summary(self) -> Dict:
        return {
            'total_params': self.count_parameters(),
            'input_dim': self.input_dim,
            'hidden_dim': self.hidden_dim,
            'output_dim': self.output_dim,
            'device': self.device,
            'model_type': self.__class__.__name__
        }


class ClassificationHead(nn.Module):

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_classes: int = 2,
        dropout: float = 0.2
    ):
        super().__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        logits = self.fc2(x)
        probs = torch.softmax(logits, dim=-1)

        return logits, probs
